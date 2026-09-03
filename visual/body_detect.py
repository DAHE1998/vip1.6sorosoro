#!/usr/bin/env python3
"""body_detect — YOLOv8 person 检测（2026-08-19 大名：只算簇 key 帧，簇内共享）
读 global_cos 骨架（新规矩，不读 DINO 骨架）→ 每 scene 计算源帧
（簇 key + 单帧，黑帧跳过，簇成员共用簇 key 帧）→ YOLOv8n 只对这些帧检测。
产物: <video_dir>/visual/body_detect/body_bbox.json（每项目全局一份，2026-08-21 大名：
    身体骨架 = 全局一份，键对齐 gc 帧标记 <hash>_f<fn>，下游拿帧名直接查）
    {"videos": [...], "<hash>_f<fn>": [[x1,y1,x2,y2,conf], ...]}  原始帧像素坐标
"""
import json, os, sys, re
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    print(f"用法: {sys.argv[0]} <项目名>（每项目全局跑一次，产出 body_bbox.json）"); sys.exit(1)
project_name = sys.argv[1]

OUT_ROOT = os.environ["OUT_ROOT"]   # 必须由 shikoto 设置，禁止单跑、禁止回退
video_dir = Path(OUT_ROOT)
frames_dir = video_dir / "shikomi" / "frames"
out_dir = video_dir / "visual" / "body_detect"
out_dir.mkdir(parents=True, exist_ok=True)

# 读 global_cos 骨架（唯一产物 gc_skeleton.json，帧处理目录，2026-08-19 大名）：
# 定「检测哪些帧」，不读 DINO 骨架。全项目所有视频一起算（身体骨架每项目全局一份）
gc_path = video_dir / "visual" / "global_cos" / "gc_skeleton.json"
if not gc_path.is_file():
    raise SystemExit(f"❌ 无 global_cos 骨架 {gc_path}（先跑 visual/global_cos.py）")
sk = json.loads(gc_path.read_text(encoding="utf-8"))

# 每视频每 scene 计算源帧；簇成员共用簇 key 帧（只算一次）；黑帧跳过
compute = {}            # video_hash -> [fn, ...]
seen = set()
for sc in sk["scenes"]:
    if sc.get("black"):
        continue
    vh = sc.get("video_hash")
    marks = sc.get("frames") or []
    if not marks:
        continue
    body = marks[0].split("：")[-1]
    m = re.search(r"_f(\d+)", body)
    fn = int(m.group(1))
    key = (vh, fn)
    if key in seen:
        continue
    seen.add(key)
    compute.setdefault(vh, []).append(fn)
for fns in compute.values():
    fns.sort()
total = sum(len(fns) for fns in compute.values())
print(f"compute frames: {total}（簇 key + 单帧，簇内共享，全项目 {len(compute)} 视频）")

from ultralytics import YOLO
import cv2

model = YOLO(os.path.join(os.environ.get("MODELS_ROOT", str(Path(__file__).resolve().parent.parent.parent / "models")), "yolo/yolov8n.pt"))

def detect_one(vh, fn):
    p = frames_dir / f"{vh}_f{fn}.jpg"
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    res = model.predict(img, conf=0.25, classes=[0], verbose=False, device="cuda")
    boxes = []
    for r in res:
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            boxes.append([x1, y1, x2, y2, float(b.conf)])
    return boxes

out_path = out_dir / "body_bbox.json"
if out_path.exists() and os.environ.get("FORCE") != "1":
    print(f"✔ 已存在（FORCE=1 重跑）: {out_path}")
    sys.exit(0)

result = {}
done = 0
for vh, fns in compute.items():
    for fn in fns:
        boxes = detect_one(vh, fn)
        if boxes:
            result[f"{vh}_f{fn}"] = boxes
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{total} …", flush=True)

# 头部透传 gc 的 videos 列表（上游信息不断）；帧键 = gc 同款帧标记 <hash>_f<fn>
# （2026-08-21 大名：身体骨架每项目一份，键对齐 gc 骨架，下游拿帧名直接查）
out_path.write_text(json.dumps({"videos": sk.get("videos", []), **result}), encoding="utf-8")
print(f"✔ {out_path} — {len(result)} 帧有人（{total} 帧检测完，簇 key + 单帧）")
