#!/usr/bin/env python3
"""chapter/assemble.py — 章节划分 ① 组装脚本（2026-08-20 大名：自包含）。

用法: OUT_ROOT=<输出目录> python3 chapter/assemble.py <video_hash>
     或 python3 chapter/assemble.py <video_hash> <输出目录>
依赖: visual/dedup/<hash>_skeleton.json、vlm/<hash>_desc.json、
     audio/dialogue/<hash>_dialogue.json、
     visual/face_head_fusion/<hash>_person_timeline.json（A 逻辑直接产出；B 逻辑缺此文件时
     回退读全局 person_timeline.json，按视频哈希在 dedup 骨架排序序列中定位本集 offset 重映射）
产物: <out_root>/chapter/<video_hash>_assembled.json（两段式：透传区 + n_scenes/scenes）
     {"video": <hash>, "n_scenes": N, "scenes": [
        {"scene_id", "frames_range": [s,e], "persons": ["P{n}",...],
         "frames": ["F1|desc",...], "asr": ["A|text",...]}]}

说明: 读 dedup 骨架 + VLM desc + ASR 台词 + 人物链 → 组装成「组装好的 json」（嵌入骨架，
逐 scene 的 画面 F 描述 / 对白 ASR / 登场人物），供 submit_api.py 送 LLM 划章。
自包含：不依赖外部 prompt 文件（prompt 在 submit_api.py 内嵌）。
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def resolve_out_root(vh, argv):
    return os.environ["OUT_ROOT"]   # 必须由 shikoto 设置，禁止单跑、禁止回退


def load_skeleton(vh, video_out):
    """dedup 骨架（内容哈希前缀）：visual/dedup/<hash>_skeleton.json"""
    p = Path(video_out) / "visual" / "dedup" / f"{vh}_skeleton.json"
    if not p.is_file():
        cands = list((Path(video_out) / "visual" / "dedup").glob(f"{vh}_skeleton.json"))  # Path.glob：锚点字面，方括号文件夹名不当通配符
        cands += list((Path(video_out) / "visual" / "dedup").glob(f"{vh}_*.json"))
        if cands:
            p = Path(cands[0])
    if not p.is_file():
        sys.exit(f"[assemble] ❌ 无 dedup 骨架 {p}（先跑 visual/dedup.py）")
    return json.loads(p.read_text(encoding="utf-8"))


def load_vlm_desc(vh, video_out):
    """vlm/<hash>_desc.json → {帧号int: 描述}。只读本视频 <hash>_desc.json，B 路线不混他集。"""
    p = Path(video_out) / "vlm" / f"{vh}_desc.json"
    if not p.is_file():
        print(f"[assemble] ⚠️ 无 vlm desc {p}，帧描述为空")
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    m = {}
    for seg in d.get("segments", []):
        for f in seg.get("frames", []):
            mm = re.match(r"^[0-9a-f]+_f(\d+)(?:.*?[：:]\s*(.*))?$", f)
            if mm:
                m[int(mm.group(1))] = (mm.group(2) or "").strip()
    print(f"[assemble] vlm desc: {len(m)} 帧有描述")
    return m


def load_audio_asr(vh, video_out):
    """audio/dialogue/<hash>_dialogue.json → {scene_id: [speaker|text, ...]}
    2026-08-19 大名：ASR 骨架 = scene 级 asr，scene_id = dedup 列表序。"""
    p = Path(video_out) / "audio" / "dialogue" / f"{vh}_dialogue.json"
    if not p.is_file():
        print(f"[assemble] ⚠️ 无 dialogue {p}，台词为空")
        return {}
    scenes = json.loads(p.read_text(encoding="utf-8")).get("scenes", [])
    return {sc["scene_id"]: sc.get("asr") or [] for sc in scenes}


def _timeline_to_map(tl):
    """person_timeline 的 timeline[] → {scene_idx: [pid, ...]}（本地 scene 编号）。"""
    m = {}
    for t in tl:
        for iv in t.get("intervals", []):
            for si in range(iv["start_scene"], iv["end_scene"] + 1):
                m.setdefault(si, []).append(t["person_id"])
    return m


def _load_global_person_map(vh, video_out):
    """B 逻辑回退：全局 person_timeline.json 的 scene_id 跨视频连续编号。
    用视频哈希在 dedup 骨架排序序列（顺序须与 global_cos 一致）中定位本集 offset，
    把全局段重映射回本集本地 scene，使哈希贯穿全局、无需冗余切分文件。"""
    import glob as _glob
    fhf = Path(video_out) / "visual" / "face_head_fusion"
    g = fhf / "person_timeline.json"
    if not g.is_file():
        print(f"[assemble] ⚠️ 无 person_timeline（全局/per-video 均无）{g}，登场人物为空")
        return {}
    offset = 0
    n_scenes = None
    for skp in sorted((Path(video_out) / "visual" / "dedup").glob("*_skeleton.json")):
        sk = json.loads(Path(skp).read_text(encoding="utf-8"))
        sk_vh = sk.get("video_hash") or sk.get("video_id")
        cnt = len(sk.get("scenes", []))
        if sk_vh == vh:
            n_scenes = cnt
            break
        offset += cnt
    if n_scenes is None:
        print(f"[assemble] ⚠️ 全局 timeline 回退：dedup 中找不到本集 {vh}，登场人物为空")
        return {}
    gtl = json.loads(g.read_text(encoding="utf-8")).get("timeline", [])
    local_tl = []
    for t in gtl:
        ivs = []
        for iv in t.get("intervals", []):
            s = iv["start_scene"] - offset
            e = iv["end_scene"] - offset
            if e < 0 or s >= n_scenes:
                continue
            s = max(0, s)
            e = min(n_scenes - 1, e)
            ivs.append({"start_scene": s, "end_scene": e, "n_scenes": e - s + 1})
        if ivs:
            local_tl.append({"person_id": t["person_id"], "intervals": ivs})
    return _timeline_to_map(local_tl)


def _parse_frame_mark(mark):
    """帧标记 → (视频哈希, 计算源帧号)。与 visual/face_recognition.py 的 parse_frame_mark 一致：
    簇成员取冒号后簇 key 帧名（共用），其余取自身。标记：
    <hash>_f<fn> / <hash>_f<fn>_cXXXkey / <hash>_f<fn>_cXXX：<hash>_f<src>_cXXXkey"""
    body = mark.split("：")[-1]
    m = re.match(r"(.+?)_f(\d+)", body)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _gc_frame_scene_map(vh, video_out, frame_map):
    """吃 GC 骨架 + 簇内共享识别结果：每个 gc scene 的分析帧 = 簇 key 帧（成员共用），
    frame_map[<hash>_f<帧号>] 的识别结果即该 scene 登场人物（簇内共享）。
    scene 本地序 = 同 video_hash 的 gc scene 序（与 dedup 骨架一致，零偏移）。"""
    gc_p = Path(video_out) / "visual" / "global_cos" / "gc_skeleton.json"
    if not gc_p.is_file():
        return None
    scenes = json.loads(gc_p.read_text(encoding="utf-8")).get("scenes", [])
    m = {}
    i = 0
    for sc in scenes:
        if sc.get("video_hash") != vh:
            continue
        marks = sc.get("frames") or []
        if marks:
            vh2, fn = _parse_frame_mark(marks[0])
            if vh2 is not None:
                pids = frame_map.get(f"{vh2}_f{fn}")
                if pids:
                    m[i] = sorted(pids)
        i += 1
    return m


def load_person_map(vh, video_out):
    """哈希贯穿全局：优先 frame_map（<hash>_<帧号> → 识别结果，A/B 均产出）+ GC 骨架
    （簇内共享）逐 scene 归属；旧数据无 frame_map 时回退 scene 区间逻辑。"""
    fhf = Path(video_out) / "visual" / "face_head_fusion"
    for name, fn in (("per-video", fhf / f"{vh}_person_timeline.json"),
                     ("全局", fhf / "person_timeline.json")):
        if fn.is_file():
            d = json.loads(fn.read_text(encoding="utf-8"))
            fm = d.get("frame_map")
            if fm:
                m = _gc_frame_scene_map(vh, video_out, fm)
                if m is not None:
                    print(f"[assemble] person_timeline({name}, GC骨架+簇共享): {len(m)} scenes 有人物")
                    return m
    p = fhf / f"{vh}_person_timeline.json"
    if p.is_file():
        tl = json.loads(p.read_text(encoding="utf-8")).get("timeline", [])
        m = _timeline_to_map(tl)
        print(f"[assemble] person_timeline(per-video, 区间): {len(m)} scenes 有人物")
        return m
    m = _load_global_person_map(vh, video_out)
    print(f"[assemble] person_timeline(全局, 区间回退): {len(m)} scenes 有人物")
    return m


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <video_hash> [out_root]")
        sys.exit(1)
    vh = sys.argv[1]
    video_out = resolve_out_root(vh, sys.argv)

    sk = load_skeleton(vh, video_out)
    scenes = sk["scenes"]
    shots = load_audio_asr(vh, video_out)
    vlm_desc = load_vlm_desc(vh, video_out)
    person_map = load_person_map(vh, video_out)

    embedded = []
    # 「向前借描述」：已送检帧号按时间升序；无描述帧向前找最近有描述的帧号取描述
    desc_fns = sorted(vlm_desc.keys())
    for i, sc in enumerate(scenes):
        sid = i  # scene_id = dedup scene 列表序
        sr = sc.get("frames_range") or {}
        st, ed = int(sr.get("start", 0)), int(sr.get("end", 0))
        persons = person_map.get(sid, [])
        # frames：scene 内帧号 → vlm desc；无描述帧向前借最近有描述帧（2026-08-30 大名：
        # VLM 5 筛 3 只是去掉语义相近没必要描述的图，落骨架仍是 dedup 后全部代表帧）
        fr_lines = []
        # 黑帧/转场 scene：无画面内容，写特效名（2026-08-30 大名）；不遍历 frames
        if sc.get("black"):
            fr_lines.append("F1|黑帧/转场")
        else:
            for fn in sc.get("frames", []):
                fn = int(fn)
                dsc = vlm_desc.get(fn)
                if not dsc:
                    prev = [x for x in desc_fns if x < fn]
                    if prev:
                        dsc = vlm_desc.get(prev[-1], "")
                if dsc:
                    fr_lines.append(f"F{len(fr_lines) + 1}|{dsc}")
        # asr：scene 级 asr 按落盘序（speaker|text），去空
        asr_lines = []
        for line in shots.get(sid) or []:
            txt = line.split("|", 1)[1] if "|" in line else line
            if txt.strip() and txt.strip() not in ("None", "无"):
                asr_lines.append(line)
        embedded.append({
            "scene_id": sid,
            "frames_range": [st, ed],
            "persons": [f"P{x}" for x in persons],
            "frames": fr_lines,
            "asr": asr_lines,
        })

    out_dir = Path(video_out) / "chapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{vh}_assembled.json"
    # 两段式（定稿）：① 透传区 = 上游骨架头无脑拷贝，所有下游产物必须带这一大段；
    # ② 本产物自己的输出段（n_scenes + scenes）
    HEADER_KEYS = ("video", "video_hash", "video_id", "project_id",
                   "fps", "width", "height", "total_frames")
    out_obj = {k: sk[k] for k in HEADER_KEYS if k in sk}
    out_obj["n_scenes"] = len(scenes)
    out_obj["scenes"] = embedded
    json.dump(out_obj, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[assemble] -> {out} ({len(embedded)} scenes)")


if __name__ == "__main__":
    main()
