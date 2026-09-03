#!/usr/bin/env python3
"""onion_model — 洋葱模型 v4：以 dedup scene 为原子，人物链切点切分。
照 git 原版逻辑，仅把原子从 shot 换成 scene。

只读人脸识别的 person_timeline（per-scene），不依赖 graph_merge。
短时长人物的 interval 边界定义切点，短人物主导（切点落在更短人物区间内
且由更长人物产生 → 抑制）。按存活切点把 scene 序列切成 proto_scenes（可切可并）。

数据流:
  dedup/<vn>/skeleton.json (scenes)  +  <vn>/face_head_fusion/person_timeline.json
  → <vn>/onion_model/skeleton.json (proto_scenes)

B 路线（多视频连续剧）：person_timeline 为全局产物（scene 号跨集连续），
PERSON_SCENE_OFFSET 环境变量把全局号映射回本视频本地号（照 graph_merge B 分支）。

用法:  python3 visual/onion_model.py <视频名>  [<project_name> <vid_name>]
"""
import json, os, sys
from collections import defaultdict

OUT_ROOT = os.environ["OUT_ROOT"]   # 必须由 shikoto 设置，禁止单跑、禁止回退

# project_mode: mode_b, project_name, vid_name（照 face_recognition.py 模式）
_mode_b = False
_project_name = None
_vid_name = None

MIN_PERSON_SCENES = 3


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <视频名> [<project_name> <vid_name>]"); sys.exit(1)

    video_name = sys.argv[1]
    mode_b = (len(sys.argv) == 4)
    project_name = sys.argv[2] if mode_b else None
    vid_name = sys.argv[3] if mode_b else video_name
    global _mode_b, _project_name, _vid_name
    _mode_b = mode_b
    _project_name = project_name
    _vid_name = vid_name

    visual_dir = os.path.join(OUT_ROOT, "visual")
    sfx = f"_{vid_name}"
    sk_path = os.path.join(visual_dir, "dedup", f"{vid_name}_skeleton.json")
    if not os.path.isfile(sk_path):
        sk_path = os.path.join(visual_dir, "dino", f"{vid_name}_skeleton.json")
    with open(sk_path) as f:
        sk = json.load(f)
    scenes = sk["scenes"]
    shots = sk["shots"]
    fps = sk["fps"]
    M = len(scenes)

    # 读 person_timeline（人脸识别的人物链，per-scene；per-video 优先，fallback 全局）
    pt_path = os.path.join(visual_dir, "face_head_fusion", f"{vid_name}_person_timeline.json")
    if not os.path.isfile(pt_path):
        pt_path = os.path.join(visual_dir, "face_head_fusion", "person_timeline.json")
    if not os.path.isfile(pt_path):
        print("  [onion] no person_timeline, all fragment")
        _emit_all_fragment(sk, scenes, shots, fps, video_name); return

    pt = json.load(open(pt_path))
    # B 路线：全局 person_timeline 的 scene 号跨集连续，偏移映射回本视频本地号（照 graph_merge B 分支）
    off = int(os.environ.get("PERSON_SCENE_OFFSET", "0"))
    raw_intervals = defaultdict(list)
    for tl in pt["timeline"]:
        pid = tl["person_id"]
        for ival in tl["intervals"]:
            s, e = ival["start_scene"] - off, ival["end_scene"] - off
            if e < 0 or s >= M:
                continue                        # 非本集（全局号偏移出界）
            raw_intervals[pid].append((max(s, 0), min(e, M - 1)))

    person_total = {pid: sum(e - s + 1 for s, e in ivs) for pid, ivs in raw_intervals.items()}

    # solo 过滤：< MIN_PERSON_SCENES 的人物不参与洋葱
    active = {pid: t for pid, t in person_total.items() if t >= MIN_PERSON_SCENES}
    skipped = len(person_total) - len(active)
    if skipped:
        print(f"  [onion] filtered {skipped} solo persons (< {MIN_PERSON_SCENES} scenes)")
    raw_intervals = {pid: ivs for pid, ivs in raw_intervals.items() if pid in active}
    person_total = {pid: t for pid, t in person_total.items() if pid in active}

    if not active:
        print("  [onion] no active persons, all fragment")
        _emit_all_fragment(sk, scenes, shots, fps, video_name); return

    # === 步骤1: 排序人物链长度，短→长 ===
    person_order = sorted(active.keys(), key=lambda p: active[p])
    print(f"  person order (shortest first): {[f'P{p}({active[p]}sc)' for p in person_order]}")

    # === 步骤2: 收集切点（每个 interval 的 start 和 end+1）===
    all_cuts = {}
    for pid in person_order:
        for s, e in raw_intervals[pid]:
            all_cuts.setdefault(s, set()).add(pid)
            all_cuts.setdefault(e + 1, set()).add(pid)

    # === 步骤3: 抑制 — 切点落在更短人物 interval 内部且由更长人物产生 → 删 ===
    for spid in person_order:
        for s_start, s_end in raw_intervals[spid]:
            for pos in list(all_cuts.keys()):
                if s_start < pos <= s_end:
                    cutters = all_cuts[pos]
                    if any(person_total.get(c, 0) > person_total[spid] for c in cutters):
                        del all_cuts[pos]

    # === 步骤4: 加全局边界，按存活切点切分 [0..M) ===
    cuts = set(all_cuts.keys())
    cuts.add(0)
    cuts.add(M)
    cuts = sorted(c for c in cuts if 0 <= c <= M)

    proto = []
    for i in range(len(cuts) - 1):
        seg_start = cuts[i]
        seg_end = cuts[i + 1] - 1
        if seg_end < seg_start:
            continue
        persons = set()
        for pid, ivs in raw_intervals.items():
            for s, e in ivs:
                if not (e < seg_start or s > seg_end):
                    persons.add(pid)
        proto.append(_make_scene(seg_start, seg_end, scenes, shots, persons, fps))

    # === 步骤5: 输出 ===
    sk["proto_scenes"] = proto
    _write(sk, proto, video_name)
    n_p = sum(1 for p in proto if p["persons"])
    n_f = len(proto) - n_p
    print(f"  {M} scenes -> {len(proto)} proto ({n_p} person, {n_f} fragment)")


def _make_scene(sa, sb, scenes, shots, persons, fps):
    """sa..sb 为 dedup scene 下标范围。frames_range / shot_frame / asr 由组成 scene 聚合。"""
    seg = scenes[sa:sb + 1]
    sf = scenes[sa]["frames_range"]["start"]
    ef = scenes[sb]["frames_range"]["end"]
    shot_frame = {}
    frames = []
    asr = []
    for sc in seg:
        shot_frame.update(sc.get("shot_frame", {}))
        frames.extend(sc.get("frames", [sc.get("key_frame")]))
        asr.extend(sc.get("asr", []))
    return {
        "id": -1,  # _write 重新编号
        "scene_range": {"start": sa, "end": sb},
        "frames_range": {"start": scenes[sa]["frames_range"]["start"], "end": scenes[sb]["frames_range"]["end"]},
        "shot_frame": shot_frame,
        "frames": frames,
        "asr": asr,
        "n_scenes": sb - sa + 1,
        "duration_s": round((ef - sf + 1) / fps, 1),   # 2026-09-02：范围一律 frames_range，range 字段已删
        "persons": sorted(persons),
    }


def _emit_all_fragment(sk, scenes, shots, fps, video_name):
    proto = [_make_scene(i, i, scenes, shots, set(), fps) for i in range(len(scenes))]
    sk["proto_scenes"] = proto
    _write(sk, proto, video_name)
    print(f"  {len(scenes)} scenes -> {len(proto)} fragment scenes")


def _write(sk, proto, video_name):
    for i, sc in enumerate(proto):
        sc["id"] = i
    out_dir = os.path.join(OUT_ROOT, "visual", "onion_model")
    out_path = os.path.join(out_dir, f"{_vid_name}_skeleton.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sk, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
