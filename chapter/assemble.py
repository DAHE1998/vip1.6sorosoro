#!/usr/bin/env python3
"""chapter/assemble.py — 章节划分 ① 组装脚本（2026-08-20 大名：自包含）。

用法: OUT_ROOT=<输出目录> python3 chapter/assemble.py <video_hash>
     或 python3 chapter/assemble.py <video_hash> <输出目录>
依赖: visual/dedup/<hash>_skeleton.json、vlm/<hash>_desc.json、
     audio/dialogue/<hash>_dialogue.json、visual/face_head_fusion/<hash>_person_timeline.json
产物: <out_root>/chapter/<video_hash>_assembled.json
     {"video": <hash>, "n_scenes": N, "scenes": [
        {"scene_id", "shot_range": [s,e], "persons": ["P{n}",...],
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
    if os.environ.get("OUT_ROOT"):
        return os.environ["OUT_ROOT"]
    if len(argv) >= 3:
        return argv[2]
    return str(BASE / "output" / vh)


def load_skeleton(vh, video_out):
    """dedup 骨架（内容哈希前缀）：visual/dedup/<hash>_skeleton.json"""
    p = Path(video_out) / "visual" / "dedup" / f"{vh}_skeleton.json"
    if not p.is_file():
        cands = glob.glob(str(Path(video_out) / "visual" / "dedup" / f"{vh}_skeleton.json"))
        cands += glob.glob(str(Path(video_out) / "visual" / "dedup" / f"{vh}_*.json"))
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


def load_person_map(vh, video_out):
    """visual/face_head_fusion/<hash>_person_timeline.json → {scene_idx: [pid, ...]}
    scene_idx = dedup scene 列表序。"""
    p = Path(video_out) / "visual" / "face_head_fusion" / f"{vh}_person_timeline.json"
    if not p.is_file():
        print(f"[assemble] ⚠️ 无 person_timeline {p}，登场人物为空")
        return {}
    tl = json.loads(p.read_text(encoding="utf-8")).get("timeline", [])
    m = {}
    for t in tl:
        for iv in t.get("intervals", []):
            for si in range(iv["start_scene"], iv["end_scene"] + 1):
                m.setdefault(si, []).append(t["person_id"])
    print(f"[assemble] person_timeline: {len(m)} scenes 有人物")
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
    for i, sc in enumerate(scenes):
        sid = i  # scene_id = dedup scene 列表序
        sr = sc.get("shot_range") or {}
        st, ed = int(sr.get("start", 0)), int(sr.get("end", 0))
        persons = person_map.get(sid, [])
        # frames：scene 内帧号 → vlm desc，F+序号 与 intro 的「F18=该 scene 第18帧」一致
        fr_lines = []
        for fn in sc.get("frames", []):
            dsc = vlm_desc.get(int(fn))
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
            "shot_range": [st, ed],
            "persons": [f"P{x}" for x in persons],
            "frames": fr_lines,
            "asr": asr_lines,
        })

    out_dir = Path(video_out) / "chapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{vh}_assembled.json"
    json.dump({"video": vh, "n_scenes": len(scenes), "scenes": embedded},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[assemble] -> {out} ({len(embedded)} scenes)")


if __name__ == "__main__":
    main()
