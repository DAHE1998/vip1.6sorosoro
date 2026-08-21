#!/usr/bin/env python3
"""visual/dedup.py — 视觉 scene 骨架：单视频内去重，形成 scene。
相邻 key_frame cos≥0.75 合并，段内有脸帧优先选代表，跨 shot 去重组 = scene；
保留原始 shot 编号（scene.shot_range 用原始号区间，含被吞的 shot）；
下游（人脸识别 / onion / graph_merge）以 dedup scene 为原子，scene_range 用原始号；
ASR 注入不在 dedup 做（由 chapter 步骤 assemble.py 负责）。

用法: python3 visual/dedup.py <视频名> [<project_name> <vid_name>]
依赖: dino/<vhash>_key_frame_embeddings.npz（DINO 只出向量，矩阵自算）；dino/<vhash>_skeleton.json（shots/key_frames 来源）；face_detect/<vid>_face_map.json（有脸帧优先）；黑帧段读 preproc/events black_segments 与 features yavg 兜底（2026-08-17 大名）
产物: visual/dedup/<vid>_skeleton.json（scenes 骨架）、<vid>_scene_visual_graph.npy、<vid>_removed_frames.json、<vid>_model_meta.json
"""
import json, os, sys, glob
from pathlib import Path
import numpy as np

def load_video_hash(vid_name, skel_dir):
    """取视频内容指纹 video_hash：vid_name 即内容哈希（batch_pipeline 算好传入，
    下游禁止二次计算哈希/文件名匹配，直接继承）"""
    return vid_name
    raise SystemExit(f"❌ 无 dino 骨架匹配 {vid_name}（先跑 dino_cluster）")


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
    video_dir = os.path.join("output", project_name)
else:
    video_dir = os.path.join("output", video_name)
visual_dir = os.path.join(video_dir, "visual")
dino_dir = os.path.join(visual_dir, "dino")
out_dir = os.path.join(visual_dir, "dedup")
os.makedirs(out_dir, exist_ok=True)

vh = load_video_hash(vid_name, dino_dir)
dino_file = f"{vh}_skeleton.json"
face_detect_file = f"{vid_name}_face_map.json"

sk = json.load(open(os.path.join(dino_dir, dino_file)))
shots = sk["shots"]

# 两两余弦矩阵（DINO 只出视觉向量，矩阵 dedup 自算）
emb = np.load(os.path.join(dino_dir, f"{vh}_key_frame_embeddings.npz"))["embeddings"].astype(np.float32)
kfn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
sim = kfn @ kfn.T
if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
    raise ValueError(f"invalid similarity matrix: {sim.shape}")

# 帧号 → 矩阵行号（与 DINO 收集顺序一致：逐 shot key_frames，去重保序）
fn_order = []
for s in shots:
    kfs = s.get("key_frames") or [(s["range"]["start"] + s["range"]["end"]) // 2]
    for fn in kfs:
        if int(fn) not in fn_order:
            fn_order.append(int(fn))
f2idx = {fn: i for i, fn in enumerate(fn_order)}
if sim.shape[0] != len(fn_order):
    raise ValueError(f"similarity matrix {sim.shape[0]}x{sim.shape[1]} vs {len(fn_order)} key_frames")

dedup_thr = float(os.environ.get("DEDUP_COS_THR", "0.75"))
dino_meta = {}
meta_path = os.path.join(dino_dir, f"{vh}_model_meta.json")
if os.path.isfile(meta_path):
    dino_meta = json.load(open(meta_path))
print(f"[05] DINO model: {dino_meta.get('model_id', 'unknown')} "
      f"matrix={sim.shape} threshold={dedup_thr}")

# 人脸检测结果（第一轮：抽的帧是否有脸）
fm = {}
fp = os.path.join(visual_dir, "face_detect", face_detect_file)
if not os.path.isfile(fp):
    raise FileNotFoundError(f"人脸检测结果 missing: {fp}; run face_detect before dedup")
fm = json.load(open(fp)).get("face_map", {})
print(f"[05] face priority: {len(fm)} frames in face_detect")

# 每 shot 取关键帧（有 key_frames 用全部，否则用中点帧）
ffn, fsid = [], []
for si, s in enumerate(shots):
    kfs = s.get("key_frames", [])
    if not kfs:
        kfs = [(s["range"]["start"] + s["range"]["end"]) // 2]
    for fn in kfs:
        if fn in f2idx:
            ffn.append(int(fn)); fsid.append(si)

orig_ffn = list(ffn)
all_removed = set()

# ── 黑帧帧（合并约束 + 黑帧独立 scene）──
def _load_black_frames(video_dir, vhash):
    """加载黑帧帧号集合（2026-08-17 大名：黑帧帧为合并约束 + 独立成黑帧 scene）。
    长黑段：preproc/events.json black_segments（blackdetect_vulkan，≥2s）；
    短黑段补检：features.json 逐帧 yavg<20 连续 ≥0.5s（blackdetect 漏检兜底）。"""
    black_frames = set()
    ev_path = os.path.join(video_dir, "preproc", "events", f"{vhash}_events.json")
    if os.path.isfile(ev_path):
        ev = json.load(open(ev_path))
        for b in ev.get("black_segments", []):
            black_frames.update(range(int(b["start_frame"]), int(b["end_frame"]) + 1))
    fe_path = os.path.join(video_dir, "preproc", "features", f"{vhash}_features.json")
    if os.path.isfile(fe_path):
        fe = json.load(open(fe_path))
        yavg = [x["yavg"] for x in fe.get("quality", [])]
        fps = sk.get("fps") or 25.0
        min_len = int(0.5 * fps)
        run = []
        for fn, ya in enumerate(yavg):
            if ya < 20:
                run.append(fn)
            elif run:
                if len(run) >= min_len:
                    black_frames.update(run)
                run = []
        if run and len(run) >= min_len:
            black_frames.update(run)
    return black_frames

black_frames = _load_black_frames(video_dir, vh)


def _black_runs(fr):
    """黑帧帧集合 → 连续黑帧段 [(start, end), ...]（黑帧独立 scene 的依据）"""
    if not fr:
        return []
    s = sorted(fr)
    runs, a, b = [], s[0], s[0]
    for x in s[1:]:
        if x == b + 1:
            b = x
        else:
            runs.append((a, b))
            a = b = x
    runs.append((a, b))
    return runs


black_runs = _black_runs(black_frames)

# scene_shots: 存活帧号 -> 归并进来的原始 shot id 集合
scene_shots = {fn: {sid} for fn, sid in zip(ffn, fsid)}

while True:
    I = np.array([f2idx[f] for f in ffn])
    adj = sim[I[:-1], I[1:]]
    new_fn, new_sid, round_removed = [], [], set()
    fi = 0
    while fi < len(ffn):
        fj = fi
        while fj < len(ffn) - 1 and adj[fj] >= dedup_thr \
                and fsid[fj] == fsid[fj + 1] \
                and (ffn[fj] in black_frames) == (ffn[fj + 1] in black_frames):
            fj += 1      # 只在 shot 内部去重（2026-08-18 大名：去重后的 shot 变成 scene，
                         # 不跨 shot 合并——否则代表帧的 scene_shots 吞进邻 shot 的 id，
                         # scene.shot_range 就跨多 shot 了）；黑帧帧不与非黑帧帧合并
        if fj - fi >= 1:
            cand = list(range(fi, fj + 1))
            face_c = [k for k in cand if str(ffn[k]) in fm and fm[str(ffn[k])]]
            if face_c:
                cand = face_c
            if len(cand) < 2:
                best = cand[0]
            else:
                cand_idx = [f2idx[ffn[k]] for k in cand]
                sub = sim[np.ix_(cand_idx, cand_idx)]
                cm = sub.sum(axis=1) - 1.0
                cm /= sub.shape[0] - 1
                best = cand[int(np.argmax(cm))]
            for k in range(fi, fj + 1):
                if k != best:
                    scene_shots[ffn[best]].update(scene_shots[ffn[k]])
                    round_removed.add(ffn[k])
            new_fn.append(ffn[best]); new_sid.append(fsid[best])
        else:
            new_fn.append(ffn[fi]); new_sid.append(fsid[fi])
        fi = fj + 1
    if not round_removed:
        break
    all_removed.update(round_removed)
    ffn, fsid = new_fn, new_sid

# 重建 scene 骨架：先定黑帧 scene，再合并剩余 shot（2026-08-17 大名）。
# 黑帧段（black_runs）独立成黑帧 scene（black=True，frame_range=精确帧区间），
# 是合并的硬边界：普通帧归并 scene 以黑帧段为界，黑帧段两侧即使同一原始 shot
# 也分属不同 scene（黑帧 scene 前后停止合并）。不做事后补建/剔除/拆分。

def _black_run_of(fn):
    """帧号 → 所在黑帧段 (ws, we)；不在黑帧段 = None"""
    for ws, we in black_runs:
        if ws <= fn <= we:
            return ws, we
    return None


def _cross_black(lo, hi):
    """帧区间 [lo, hi] 是否跨黑帧段（存在黑帧段在 lo 与 hi 之间）"""
    return any(lo < ws and we < hi for ws, we in black_runs)


# ① 先定黑帧 scene：黑帧段独立成 scene，作为合并剩余 shot 的边界
black_scenes = []
for ws, we in black_runs:
    bshots = [s["id"] for s in shots
              if not (s["range"]["end"] < ws or s["range"]["start"] > we)]
    if not bshots:
        continue
    bfn = sorted(f for f in black_frames if ws <= f <= we)
    black_scenes.append({
        "id": len(black_scenes),
        "shot_range": {"start": min(bshots), "end": max(bshots)},
        "shot_frame": {},
        "frames": [f for f in bfn if f in ffn],      # 黑帧段存活帧（可能空）
        "n_shots": len(bshots),
        "key_frame": bfn[0] if bfn else None,
        "black": True,
        "frame_range": [ws, we],                     # 黑帧段精确帧区间（选帧判中断依据）
        "black_frames": bfn,
    })

# ② 再合并剩余 shot：普通帧归并 scene，黑帧段为界（黑帧 scene 前后停止合并）
scenes = []
i = 0
n = len(ffn)
while i < n:
    if _black_run_of(ffn[i]) is not None:            # 黑帧帧跳过（归黑帧 scene）
        i += 1
        continue
    sid = fsid[i]
    j = i
    while j < n and fsid[j] == sid and _black_run_of(ffn[j]) is None \
            and not _cross_black(ffn[i], ffn[j]):
        j += 1                                       # 跨黑帧段即停，同一 shot 也开新 scene
    merged = set()
    for k in range(i, j):
        merged.update(scene_shots[ffn[k]])
    sids = sorted(merged)
    scenes.append({
        "id": len(scenes),
        "shot_range": {"start": sids[0], "end": sids[-1]},
        "shot_frame": {str(sid): ffn[i]},
        "frames": ffn[i:j],                          # scene 内全部存活帧（保留帧全集，展示用）
        "n_shots": len(sids),
        "key_frame": ffn[i],
    })
    i = j

# ③ 黑帧 scene 按时间序归位：黑帧按 frame_range 起点，普通按实际首帧 frames[0]。
# 普通 scene 的 shot_range 可能覆盖黑帧段（shot 粒度，帧被切到黑帧段后仍指向原 shot），
# 作排序键会把黑帧段后的 scene 排到黑帧 scene 前，黑帧隔断失效（2026-08-17 大名）。
scenes = scenes + black_scenes
scenes.sort(key=lambda sc: (sc.get("frame_range") or sc["frames"])[0])
n_black = sum(1 for sc in scenes if sc.get("black"))
print(f"[dedup] 黑帧帧 {len(black_frames)} / 黑帧段 {len(black_runs)} / 黑帧 scene {n_black}")

# scene.shot_range 用原始 shot 归属（sids[0]/sids[-1]，即 scene 存活帧的 shot id 集合）。
# 2026-08-05 曾有后处理把 shot_range 推挤成「连续无重叠区间」，末位 scene 空间被挤光时
# 造出越界 shot id（如只有 0..194 却出现 195），下游 shots[id] 索引 KeyError。
# 2026-08-18 大名：只能在 shot 内部去重、不可破坏 shot → 回归原始归属，不做任何推挤。
sk["scenes"] = scenes

# scene×scene 视觉图（graph_merge 消费）：每 scene 用其 key_frame 向量，取 sim 子矩阵；
# 补建黑帧 scene 可能无存活帧（key_frame 不在 f2idx）→ 不进视觉图（黑帧不参与合并）
kf_idx = [f2idx[sc["key_frame"]] for sc in scenes if sc.get("key_frame") in f2idx]
np.save(os.path.join(out_dir, f"{vid_name}_scene_visual_graph.npy"),
        sim[np.ix_(kf_idx, kf_idx)].astype(np.float32))

# 场景 id 命名（2026-08-18 大名定稿）：16 位 video_hash 内容指纹（与
# preproc/frames/ 帧前缀同源，贯穿全场唯一）+ Scene{序号}（Scene 与序号间无下划线）。
# 例：d4f40304e63203ee_Scene01；图片名 = 骨架 id + _ + 绝对帧号。
for i, sc in enumerate(scenes):
    sc["id"] = f"{vh}_Scene{i + 1:02d}"
# video_id 保持 dino 骨架原值（文件名）——不覆盖（大名 2026-08-19：文件名归
# video_id，内容哈希归 video_hash，两者是两码事；曾误用 video_hash 覆盖 video_id）
with open(os.path.join(out_dir, f"{vid_name}_skeleton.json"), "w") as f:
    json.dump(sk, f, ensure_ascii=False, indent=2)

json.dump({
    "n_scenes": len(scenes),
    "all_frames": orig_ffn,
    "removed_frames": sorted(all_removed),
}, open(os.path.join(out_dir, f"{vid_name}_removed_frames.json"), "w"))
json.dump({
    "dino_model_id": dino_meta.get("model_id"),
    "embedding_dim": int(sim.shape[1]),
    "dedup_cos_thr": dedup_thr,
    "face_map_used": True,
    "face_map_frames": len(fm),
    "n_scenes": len(scenes),
}, open(os.path.join(out_dir, f"{vid_name}_model_meta.json"), "w"), ensure_ascii=False, indent=2)

print(f"[05] {len(orig_ffn)} frames -> {len(scenes)} scenes")
