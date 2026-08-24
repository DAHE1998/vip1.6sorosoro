#!/usr/bin/env python3
"""face_detect — 读DINO key_frames + SCRFD ONNX → 每帧有脸/无脸。"""
import json, os, sys, time
import numpy as np
import cv2
import onnxruntime as ort

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    print(f"用法: {sys.argv[0]} <视频名> [<project_name> <vid_name>]"); sys.exit(1)

video_name = sys.argv[1]
mode_b = (len(sys.argv) == 4)
project_name = sys.argv[2] if mode_b else None
vid_name = sys.argv[3] if mode_b else video_name

OUT_ROOT = os.environ.get("OUT_ROOT")
if OUT_ROOT:
    video_dir = OUT_ROOT
elif mode_b:
    video_dir = os.path.join(PROJECT_DIR, "output", project_name)
else:
    video_dir = os.path.join(PROJECT_DIR, "output", video_name)
vis_dir = os.path.join(video_dir, "visual", "face_detect")
frames_dir = os.path.join(video_dir, "shikomi", "frames")
os.makedirs(vis_dir, exist_ok=True)

def load_video_hash(vid_name, skel_dir):
    """取视频内容指纹 video_hash：vid_name 即内容哈希（batch_pipeline 算好传入，
    下游禁止二次计算哈希/文件名匹配，直接继承）"""
    return vid_name
    raise SystemExit(f"❌ 无 dino 骨架匹配 {vid_name}（先跑 dino_cluster）")


# 只处理DINO key_frames（帧前缀 = video_hash 内容指纹）
vh = load_video_hash(vid_name, os.path.join(video_dir, "visual", "dino"))
dino_sk = json.load(open(os.path.join(video_dir, "visual", "dino", f"{vh}_skeleton.json")))
key_frames = []
for s in dino_sk["shots"]:
    kfs = s.get("key_frames") or [(s["range"]["start"] + s["range"]["end"]) // 2]
    key_frames.extend(int(f) for f in kfs)
key_frames = sorted(set(key_frames))

def _locate_model(env_names, *candidates):
    """模型目录定位：显式 env 优先 → HF_HOME 布局推导 → 报错提示（零硬编码，2026-08-07 铁律）"""
    import glob
    for n in env_names:
        v = os.environ.get(n)
        if v and os.path.isdir(v):
            return v
    hf = os.environ.get("HF_HOME", "/models/hf")
    for c in candidates:
        if c.startswith("glob:"):
            m = glob.glob(c[5:].format(hf=hf))
            if m:
                return m[0]
        else:
            p = os.path.expanduser(c.format(hf=hf))
            if os.path.isdir(p):
                return p
    print(f"[ERR] 模型未找到：请设置 {'/'.join(env_names)}（HF_HOME={hf} 下也未找到）", file=sys.stderr)
    sys.exit(1)


INSIGHTFACE_DIR = _locate_model(("INSIGHTFACE_DIR",),
                                "{hf}/insightface/models/buffalo_l")
sess = ort.InferenceSession(os.path.join(INSIGHTFACE_DIR, "det_10g.onnx"),
                            providers=["CUDAExecutionProvider"])

out, fc = {}, 0
t0 = time.time()
for fn_num in key_frames:
    img = cv2.imread(os.path.join(frames_dir, f"{vh}_f{fn_num}.jpg"))
    if img is None: continue
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (640,640)), 1/128, (640,640),
                                  (127.5,127.5,127.5), swapRB=True)
    o = sess.run(None, {"input.1": blob})
    hf = any((x.flatten() > 0.5).any() for x in o[:3])
    out[str(fn_num)] = bool(hf)
    if hf: fc += 1

# 头部对齐 dino 骨架输出格式（哈希值贯穿全场：video_hash 等透传，防自造格式）
meta = {k: dino_sk.get(k) for k in ("video_id", "video", "video_hash",
                                    "fps", "width", "height", "total_frames")}
json.dump({**meta, "n": len(out), "n_face": fc, "face_map": out},
          open(os.path.join(vis_dir, f"{vid_name}_face_map.json"), "w"))
print(f"[facedet] {vid_name}: {len(out)}fr {time.time()-t0:.0f}s {fc} faces")
