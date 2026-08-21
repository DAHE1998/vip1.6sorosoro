#!/usr/bin/env python3
"""audio/merge_speaker.py — 句子按时间归属shot，嵌入对话。

用法: python3 audio/merge_speaker.py <vn>
依赖: dedup 骨架 visual/dedup/<vh>_skeleton.json + srt 字幕 或 transcribe/raw_segments.json
      （srt 直读时另需 speaker/speakers.json 对齐说话人）
产物: audio/dialogue/<vh>_dialogue.json、<vh>_dialogue_text.txt

句子来源二选一（有字幕优先）：
  1. srt 字幕直读（两级查找: 1.视频同名 .srt  2.input/subtitles/<video_hash>.srt）
     说话人由声纹对齐: 句子起点落在哪个 pyannote 段就用哪个 speaker
  2. 无字幕: 读 transcribe/raw_segments.json（ASR 产物，带 pyannote speaker）
"""
import json, os, sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srt_to_segments import find_srt, parse_srt

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def overlap(a0, a1, b0, b1):
    """两段时间区间重叠长度（ms）。"""
    return max(0, min(a1, b1) - max(a0, b0))

def frame_to_ms(frame, fps):
    """帧号 → 毫秒（fps 换算，供台词时间归属）。"""
    return int(frame / fps * 1000)

def main():
    """按首字帧号把台词落进 shot/scene，写 dialogue.json 与 dialogue_text.txt。"""
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <vn> [<project> <vid_name>]"); sys.exit(1)

    vn = sys.argv[1]
    mode_b = (len(sys.argv) == 4)
    project_name = sys.argv[2] if mode_b else None
    vid_name = sys.argv[3] if mode_b else vn
    sfx = f"_{vid_name}"

    # 2026-08-14 修复：batch_pipeline 以 OUT_ROOT 调用，子脚本需与其布局一致（见 asr_pipeline.py）
    OUT_ROOT = os.environ.get("OUT_ROOT")
    if OUT_ROOT:
        base_dir = OUT_ROOT
    elif mode_b:
        base_dir = os.path.join(PROJECT_DIR, "output", project_name)
    else:
        base_dir = os.path.join(PROJECT_DIR, "output", vn)

    # 读 dedup 骨架（2026-08-19 大名：每个视频的终点就到 dedup；ASR 输出物按哈希
    # 命名对齐，video_hash 透传 dedup。graph_merge/onion 已废弃不进）
    skel_path = os.path.join(base_dir, "visual", "dedup", f"{vid_name}_skeleton.json")
    if not os.path.isfile(skel_path):
        print(f"[merge] 找不到 dedup 骨架: {skel_path}（先跑视觉流水线至 dedup）")
        sys.exit(1)
    dia_dir   = os.path.join(base_dir, "audio", "dialogue")
    os.makedirs(dia_dir, exist_ok=True)

    skeleton     = json.load(open(skel_path))
    shots        = skeleton["shots"]
    fps          = skeleton["fps"]

    # 句子来源: 有字幕直读 srt（跳过 raw_segments），无字幕读 ASR 产物
    srt_path = find_srt(skeleton.get("video", ""), skeleton.get("video_hash", ""))
    if srt_path:
        raw_segments = parse_srt(srt_path)
        # 说话人对齐: 句子起点落在哪个 pyannote 段就用哪个说话人（跨段取起点段）
        spk_path = os.path.join(base_dir, "audio", "speaker", f"{vid_name}_speakers.json")
        if os.path.isfile(spk_path):
            spk_segs = json.load(open(spk_path))
            for seg in raw_segments:
                seg["speaker"] = -1
                for ps in spk_segs:
                    if ps["start_ms"] <= seg["start_ms"] <= ps["end_ms"]:
                        seg["speaker"] = ps["speaker"]
                        break
            n_unk = sum(1 for s in raw_segments if s["speaker"] == -1)
            print(f"[merge] srt 直读: {os.path.basename(srt_path)} ({len(raw_segments)} 段) "
                  f"+ 声纹对齐 {len(spk_segs)} 段, 未覆盖 {n_unk} 句")
        else:
            print(f"[merge] srt 直读: {os.path.basename(srt_path)} ({len(raw_segments)} 段, 无 speakers.json 说话人未知)")
    else:
        raw_path = os.path.join(base_dir, "audio", "transcribe", f"{vid_name}_raw_segments.json")
        raw_segments = [r for r in json.load(open(raw_path)) if r["text"].strip()]
    print(f"[merge] {len(shots)} shots, {len(raw_segments)} segments")

    # 重复段压缩连续 ≥3 条相同文本 → 合并为 1 条）
    # 例：scene96 含糊段 ASR 输出「去年」×256（8.5s）→ 1 条「去年」；
    #     「はい」×2 等正常应答重复不压（不够 3 条）
    raw_segments = sorted(raw_segments, key=lambda x: x["start_ms"])
    squashed = []
    i = 0
    while i < len(raw_segments):
        j = i + 1
        while j < len(raw_segments) and raw_segments[j]["text"] == raw_segments[i]["text"] \
                and raw_segments[j].get("speaker") == raw_segments[i].get("speaker"):
            j += 1
        if j - i >= 3:
            first = dict(raw_segments[i])
            first["end_ms"] = raw_segments[j - 1]["end_ms"]
            squashed.append(first)
        else:
            squashed.extend(raw_segments[i:j])
        i = j
    n_squash = len(raw_segments) - len(squashed)
    raw_segments = squashed
    if n_squash:
        print(f"[merge] 重复段压缩: 并入 {n_squash} 条重复（剩 {len(raw_segments)} 条）")

    # 说话人整数ID → A/B/C（按首次出现时间）
    label_map = OrderedDict()
    for seg in raw_segments:
        spk = seg.get("speaker", -1)
        if spk != -1 and spk not in label_map:
            label_map[spk] = LETTERS[len(label_map)] if len(label_map) < len(LETTERS) else f"S{spk}"
    print(f"[merge] speakers: {list(label_map.values())}")

    shot_ranges = [(s["id"],
                    frame_to_ms(s["range"]["start"], fps),
                    frame_to_ms(s["range"]["end"],   fps)) for s in shots]

    shot_dialogue = {s["id"]: [] for s in shots}

    for seg in raw_segments:
        ms0, ms1  = seg["start_ms"], seg["end_ms"]
        spk_label = label_map.get(seg.get("speaker", -1), "?")
        lang      = seg.get("lang", "none")
        # 用句子起点（第一个字的时间戳）落到对应shot，整句不拆
        target_sid = None
        for sid, s0, s1 in shot_ranges:
            if s0 <= ms0 <= s1:
                target_sid = sid
                break
        if target_sid is None:
            # 起点不在任何shot内，取边界最近的shot
            target_sid = min(shot_ranges,
                             key=lambda x: min(abs(x[1]-ms0), abs(x[2]-ms0)))[0]

        shot_dialogue[target_sid].append({
            "speaker":  spk_label,
            "lang":     lang,
            "text":     seg["text"].strip(),
            "start_ms": ms0,
            "end_ms":   ms1,
        })

    n_shots_with_dia = 0
    n_sentences      = 0
    all_text_lines   = []
    for s in shots:
        dia = sorted(shot_dialogue[s["id"]], key=lambda x: x["start_ms"])
        # pyannote 段碎 → 语气词/残词独立行：相邻同 speaker 同 lang 的 ≤2 字句并入前一句
        # （并入不丢内容；「啊」「嗯」这类语气词与前后句连读才符合真实说话节奏）
        merged = []
        for d in dia:
            if len(d["text"]) <= 2 and merged \
                    and merged[-1]["speaker"] == d["speaker"] \
                    and merged[-1]["lang"] == d["lang"]:
                merged[-1]["text"] += d["text"]
                merged[-1]["end_ms"] = d["end_ms"]
            else:
                merged.append(d)
        dia = merged
        s["dialogue"] = dia
        if dia:
            n_shots_with_dia += 1
            n_sentences += len(dia)
            for sent in dia:
                lt = f"[{sent['lang']}]" if sent["lang"] != "none" else ""
                all_text_lines.append(f"{lt}{sent['speaker']}: {sent['text']}")

    # ASR 自己的骨架（2026-08-19 大名范例）：scene 级聚合台词，透传视频信息+哈希。
    # 每 scene：scene_id（= dedup scene 列表序，0-based，全链路统一）+ shot_range
    #（shot 编号范围，0-based，对应 <哈希>_shot_id）+ asr（['说话人|台词', ...]，
    # 说话人=声纹字母，按 shot 序 + 台词 start_ms 先后排）。台词按 scene 覆盖的 shot 聚合。
    # 台词归属：每句按首字帧号唯一落一个 scene（2026-08-19 大名：句级台词只属一个
    # scene，绝不横跨）。黑帧 scene 与相邻正常 scene 共享 shot_range（黑帧段把同 shot
    # 拆开）——不能按「scene 覆盖的 shot 拉台词」（会一句重复进两个 scene），必须按
    # 台词首字帧号落在本 scene 的帧范围内才归属。
    # 2026-08-19 大名修正：scene = shot 范围，shot = 帧范围，联合 = 本 scene 覆盖的
    # 帧区间 [fstart, fend]，区间内任意帧号都对应本 scene。不可用 scene 的稀疏代表帧
    # frames（每 scene 仅 2 帧）当范围——否则 268 句只落进 5 句（bug 根因）。
    # 2026-08-19 大名定稿：黑帧的 ASR 向后归纳（黑帧 scene 自身 asr 空，其区间内台词
    # 归其后第一个正常 scene；最后的黑帧无后 scene，无法归纳）。
    # 归属不依赖 shot 归属的 ms（int(frame/fps*1000) 截断会让边界帧台词 start_ms 略超
    # 本 shot ms 上界而错落相邻 shot → 只遍历 scene 覆盖的 shot 就丢句，如「腾飞/啊」），
    # 改为直接遍历全部台词按 fn0 落 scene 帧区间。
    fps = skeleton["fps"]
    scenes = skeleton.get("scenes", [])
    shot_frames = {s["id"]: (s["range"]["start"], s["range"]["end"]) for s in shots}
    spans = []
    for sc in scenes:
        rng = sc["shot_range"]
        fr = shot_frames.get(rng["start"]), shot_frames.get(rng["end"])
        spans.append((fr[0][0], fr[1][1]) if all(fr) else None)
    fwd_target = [None] * len(scenes)      # 黑帧 scene → 其后第一个正常 scene
    nxt = None
    for i in range(len(scenes) - 1, -1, -1):
        if scenes[i].get("black"):
            fwd_target[i] = nxt
        else:
            nxt = i
    lines = []                              # 全部台词（shot dialogue 已 ≤2 字合并）
    for sh in shots:
        for d in sh.get("dialogue", []):
            lines.append((int(d["start_ms"] / 1000.0 * fps), f"{d['speaker']}|{d['text']}"))
    buckets = [[] for _ in scenes]
    for fn0, line in lines:
        tgt = None
        for i, (span, is_black) in enumerate(zip(spans, [s.get("black") for s in scenes])):
            if span and span[0] <= fn0 <= span[1]:
                tgt = fwd_target[i] if is_black else i   # 黑帧 → 向后归纳
                break
        if tgt is not None:
            buckets[tgt].append(line)
    asr_scenes = []
    for idx, sc in enumerate(scenes):
        asr_scenes.append({"scene_id": idx, "shot_range": sc["shot_range"], "asr": buckets[idx]})
    out_sk = {"video": skeleton.get("video"), "video_id": skeleton.get("video_id"),
              "video_hash": skeleton.get("video_hash"), "fps": fps,
              "width": skeleton.get("width"), "height": skeleton.get("height"),
              "total_frames": skeleton.get("total_frames"), "scenes": asr_scenes}
    out_sk = {k: v for k, v in out_sk.items() if v is not None}

    dia_skel_path = os.path.join(dia_dir, f"{vid_name}_dialogue.json")
    with open(dia_skel_path, "w", encoding="utf-8") as f:
        json.dump(out_sk, f, ensure_ascii=False, indent=2)
    txt_path = os.path.join(dia_dir, f"{vid_name}_dialogue_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text_lines) + "\n")

    print(f"[merge] -> {dia_skel_path} ({n_shots_with_dia}/{len(shots)} shots)")
    print(f"\n  --- preview (first 20) ---")
    for l in all_text_lines[:20]: print(f"  {l}")
    print(f"\n[merge] done: {n_shots_with_dia} shots, {n_sentences} sentences")

if __name__ == "__main__":
    main()
