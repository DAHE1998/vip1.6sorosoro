#!/usr/bin/env python3
"""chapter/global_index.py — B 路线项目全局索引（纯拼装，无 API）。

读取: <out_root>/chapter/*_blocks.json（每个视频一份，index.py 产物）
输出: <out_root>/<pname>_global.json
结构: {project: <pname>, videos: [{video_id, file, n_blocks, blocks}]}

用法:
    python3 global_index.py <out_root> <pname>
"""
import glob, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(out_root, pname):
    root = Path(out_root)
    chapter_dir = root / "chapter"
    if not chapter_dir.is_dir():
        print(f"❌ chapter 目录不存在: {chapter_dir}"); sys.exit(1)

    blk_files = sorted(chapter_dir.glob("*_blocks.json"))   # Path.glob：锚点字面，方括号文件夹名不当通配符
    if not blk_files:
        print(f"⚠ 无 *_blocks.json（先跑 index.py），跳过"); sys.exit(2)

    videos = []
    for bf in blk_files:
        try:
            d = json.load(open(bf, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 无法解析 {os.path.basename(bf)}: {e}"); continue
        blocks = d.get("blocks") or []
        videos.append({
            "video_id": d.get("video_id", os.path.basename(bf).split("_")[0]),
            "file": os.path.basename(bf),
            "n_blocks": len(blocks),
            "blocks": blocks,
        })

    if not videos:
        print("⚠ 无可合并的 blocks"); sys.exit(3)
    videos.sort(key=lambda v: v["video_id"])

    out = {
        "project": pname,
        "n_videos": len(videos),
        "n_blocks_total": sum(v["n_blocks"] for v in videos),
        "videos": videos,
    }
    out_path = root / f"{pname}_global.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ 全局索引 → {out_path}（{len(videos)} 视频, {out['n_blocks_total']} 块）")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    build(sys.argv[1], sys.argv[2])
