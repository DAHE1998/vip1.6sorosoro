#!/usr/bin/env python3
"""chapter/postprocess_boundaries.py — 后处理：强制每个 chapter >= 5 scenes。

用法: python3 chapter/postprocess_boundaries.py <视频名> [<项目名> <视频输出目录>]（B 路线带后两参）
依赖: chapter/chapter_boundaries/chapter_boundaries.json（手工优先，serve_viz 页面保存）
     或 chapter_boundaries_vlm_tokens{sfx}.json（API 划章产物，fallback）
产物: chapter/chapter_boundaries/chapter_boundaries.json（合并小 chapter 后回写）

说明: 计算各 chapter 大小，把 < MIN_SIZE=5 的小 chapter 向前合并（首章过小与下一章合并），
重新生成 boundaries 回写。2026-08-20：chapter_seg_text_api.py 输出到 chapter/chapter_boundaries/，
此处必须带 chapter/ 段；2026-08-12：手工边界优先的规则。
"""
import json, os, sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V = sys.argv[1] if len(sys.argv) > 1 else "video1"
mode_b = (len(sys.argv) >= 4)
project_name = sys.argv[2] if mode_b else None
video_out_dir = sys.argv[3] if mode_b and len(sys.argv) > 3 else None

if mode_b and video_out_dir:
    # 2026-08-20：chapter_seg_text_api.py 输出到 chapter/chapter_boundaries/，此处必须带 chapter/ 段
    d = os.path.join(video_out_dir, "chapter", "chapter_boundaries")
else:
    d = os.path.join(BASE, "output", V, "chapter", "chapter_boundaries")
# 手工边界优先（serve_viz 页面保存），无手工边界 fallback API 划章产物（2026-08-12）
manual = os.path.join(d, "chapter_boundaries.json")
sfx = os.environ.get("TOKEN_SEG_OUT_SUFFIX", "")
api = os.path.join(d, f"chapter_boundaries_vlm_tokens{sfx}.json")
path = manual if os.path.isfile(manual) else api
if not os.path.isfile(path):
    print(f"[postprocess] 警告: 无手工边界也无 API 划章产物 ({manual} / {api})", file=sys.stderr)
MIN_SIZE = 5
d = json.load(open(path))

boundaries = d["boundaries"]
n_scenes = d["n_scenes"]

# 计算每个 chapter 的 [start, end, size]
chapters = []
prev = 0
for b in boundaries:
    chapters.append({"start": prev, "end": b - 1, "size": b - prev})
    prev = b
chapters.append({"start": prev, "end": n_scenes - 1, "size": n_scenes - prev})

print(f"Before: {len(chapters)} chapters")
for i, ch in enumerate(chapters):
    print(f"  C{i}: scene {ch['start']}~{ch['end']} ({ch['size']} scenes)")

# 合并小 chapter（向前合并）
merged = []
i = 0
while i < len(chapters):
    if chapters[i]["size"] >= MIN_SIZE:
        merged.append(chapters[i])
        i += 1
    else:
        # 小 chapter 合并到前一个
        if merged:
            merged[-1]["end"] = chapters[i]["end"]
            merged[-1]["size"] = merged[-1]["end"] - merged[-1]["start"] + 1
        else:
            # 第一个 chapter 太小，和下一个合并
            chapters[i]["start"] = chapters[i]["start"]
            chapters[i]["size"] = chapters[i]["size"] + chapters[i+1]["size"] if i+1 < len(chapters) else chapters[i]["size"]
            merged.append(chapters[i])
        i += 1

# 重新生成 boundaries
new_boundaries = []
for ch in merged[:-1]:  # 最后一个 chapter 不需要 boundary
    new_boundaries.append(ch["end"] + 1)

print(f"\nAfter: {len(merged)} chapters")
for i, ch in enumerate(merged):
    print(f"  C{i}: scene {ch['start']}~{ch['end']} ({ch['size']} scenes)")
print(f"  small chapters: {sum(1 for c in merged if c['size'] < MIN_SIZE)}")

d["boundaries"] = new_boundaries
d["n_chapters"] = len(merged)

if mode_b and video_out_dir:
    out_path = os.path.join(video_out_dir, "chapter", "chapter_boundaries", "chapter_boundaries.json")
else:
    out_path = os.path.join(BASE, "output", V, "chapter", "chapter_boundaries", "chapter_boundaries.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"\n[post] -> {out_path}")
