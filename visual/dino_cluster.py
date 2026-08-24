#!/usr/bin/env python3
"""visual/dino_cluster.py — DINO 视觉 shot 图: key_frames → DINOv3 → embedding 矩阵。

用法:
  Mode A（单视频）:        python3 visual/dino_cluster.py <视频名>
  Mode B 全局:            python3 visual/dino_cluster.py "" <project_name>
  Mode B 单视频（供 dedup）: python3 visual/dino_cluster.py <vid_name> <project_name> <vid_name>
依赖: shikomi/select_frames/<vhash>_select_frames.json（key_frames 来源）；shikomi/frames224/ 224×224 RGB bin（Preproc 已导出小图，零解码零 resize）；DINOv3 模型（DINO_MODEL_DIR）
产物: Mode A output/<视频名>/visual/dino/（<vhash>_key_frame_embeddings.npz、<vhash>_model_meta.json、<vhash>_skeleton.json）；Mode B 全局 output/<项目>/visual/dino/（key_frame_embeddings.npz 全局矩阵、frame_map.json、model_meta.json、global_similarity_matrix.npy、<vhash>_skeleton.json），并落 visual/dedup/ 骨架与 scene 视觉图
"""
import json, sys, os, glob
import numpy as np
import torch
from transformers import AutoModel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("OUT_ROOT")

if len(sys.argv) < 2:
    print(f"用法: {sys.argv[0]} <视频名> [<project_name> <vid_name>]")
    print(f"      {sys.argv[0]} '' <project_name>  (Mode B 全局)")
    sys.exit(1)

video_name = sys.argv[1]  # Mode A: 视频名; Mode B: 空字符串=全局, 或具体视频名
mode_b_global = (len(sys.argv) == 2 and video_name == "")
mode_b = (len(sys.argv) >= 3)
project_name = sys.argv[2] if mode_b else None
vid_name = sys.argv[3] if (mode_b and len(sys.argv) >= 4) else video_name

DINO_MODEL_DIR = os.environ.get(
    "DINO_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "dinov3-vitl16-pretrain-lvd1689m"),
)
DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
S = 224
BS2 = int(os.environ.get("DINO_BATCH_SIZE", "8"))

# ── 辅助函数 ──

def load_select_frames(skel_path):
    """从 select_frames 骨架中收集 key_frames，返回 (kf_list, shots)"""
    with open(skel_path) as f:
        sk = json.load(f)
    shots = sk["shots"]
    kf_list = []
    for s in shots:
        kf = s.get("key_frames")
        if not kf:
            mid = (s["range"]["start"] + s["range"]["end"]) // 2
            kf = [int(mid)]
        for fn in kf:
            kf_list.append({"local_frame": int(fn), "shot_id": s["id"]})
    return kf_list, shots

def load_video_hash(vid_name, skel_dir):
    """取视频内容指纹 video_hash：vid_name 即内容哈希（batch_pipeline 算好传入，
    下游禁止二次计算哈希/文件名匹配，直接继承）"""
    return vid_name


def read_frame(frames_dir, vh, local_frame):
    """读取 frames224 224×224×3 RGB bin（Preproc 已导出小图，零解码零 resize）"""
    p = os.path.join(frames_dir, f"{vh}_f{local_frame}.bin")
    if not os.path.isfile(p):
        return None
    return np.fromfile(p, dtype=np.uint8).reshape(S, S, 3)

def run_dino(frames_array):
    """跑 DINOv3，返回 embeddings (N, 1024) float16"""
    MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)

    ts = []
    for b in range(0, len(frames_array), 512):
        ch = frames_array[b:b + 512]
        t = torch.from_numpy(ch).to("cuda", dtype=torch.float32, non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous() / 255.0
        t = (t - MEAN) / STD
        ts.append(t)
    all_t = torch.cat(ts)
    del frames_array

    print(f"  loading DINOv3 from {DINO_MODEL_DIR}...")
    m0 = torch.cuda.memory_allocated() / 1024 ** 3
    model = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True, dtype=torch.float32).to("cuda").eval()
    print(f"  GPU: {m0:.1f} -> {torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB")

    embs = []
    for b in range(0, len(all_t), BS2):
        with torch.inference_mode():
            embs.append(model(pixel_values=all_t[b:b + BS2]).pooler_output)
    all_e = torch.cat(embs)
    del all_t
    torch.cuda.empty_cache()

    return all_e.cpu().numpy().astype(np.float16)

def save_global(out_dir, kf_emb, all_kf_entries, n_videos):
    """保存全局输出：key_frame_embeddings.npz + frame_map.json + model_meta.json"""
    # 全局 npz
    global_ids = np.arange(len(all_kf_entries), dtype=np.int32)
    np.savez(os.path.join(out_dir, "key_frame_embeddings.npz"),
             embeddings=kf_emb, frame_ids=global_ids)

    # frame_map
    frame_map = {}
    for i, kf in enumerate(all_kf_entries):
        frame_map[str(i)] = {"video_id": kf["video_id"], "local_frame": kf["local_frame"]}
    with open(os.path.join(out_dir, "frame_map.json"), "w") as f:
        json.dump(frame_map, f, ensure_ascii=False, indent=2)

    # meta
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump({
            "model_id": DINO_MODEL_ID, "model_dir": DINO_MODEL_DIR,
            "embedding_dim": int(kf_emb.shape[1]),
            "embedding_dtype": str(kf_emb.dtype), "inference_dtype": "float32",
            "input_size": S, "batch_size": BS2,
            "n_frames": len(all_kf_entries), "n_videos": n_videos,
        }, f, ensure_ascii=False, indent=2)

def save_per_video(out_dir, vid, shots, meta=None):
    """保存单视频骨架（供 dedup 读取）；meta 透传 select_frames 元数据（骨架传导不丢字段）"""
    vh = (meta or {}).get("video_hash")
    if not vh:
        raise SystemExit(f"❌ {vid} select_frames 缺 video_hash（C 端产物）")
    skel_path = os.path.join(out_dir, f"{vh}_skeleton.json")
    sk = {"shots": shots, "video_id": vid}
    if meta:
        for k in ("video", "video_hash", "fps", "width", "height", "total_frames"):
            if k in meta:
                sk[k] = meta[k]
    with open(skel_path, "w") as f:
        json.dump(sk, f, ensure_ascii=False, indent=2)

# ── 主逻辑 ──

if mode_b_global:
    # ═══ Mode B: 全局 DINO ═══
    # 收集所有视频的 select_frames
    sel_dir = os.path.join(OUT_ROOT if OUT_ROOT else os.path.join(PROJECT_DIR, "output", project_name), "shikomi", "select_frames")
    sel_files = sorted(glob.glob(os.path.join(sel_dir, "*_select_frames.json")))
    if not sel_files:
        print(f"[04] Mode B 全局: 没有 select_frames in {sel_dir}")
        sys.exit(1)

    out_dir = os.path.join(OUT_ROOT if OUT_ROOT else os.path.join(PROJECT_DIR, "output", project_name), "visual", "dino")
    os.makedirs(out_dir, exist_ok=True)

    # 检查是否已缓存
    global_npz = os.path.join(out_dir, "key_frame_embeddings.npz")
    global_meta = os.path.join(out_dir, "model_meta.json")
    if os.path.isfile(global_npz) and os.path.isfile(global_meta):
        print(f"[04] DINO 全局: cached ({len(sel_files)} videos)")
        sys.exit(0)

    # 收集所有视频的 key_frames
    all_kf = []   # [{video_id, local_frame, shot_id}, ...]
    all_video_shots = {}  # vid -> shots[]
    all_video_meta = {}  # vid -> select_frames 元数据（fps/video 透传）
    frames_base = os.path.join(OUT_ROOT if OUT_ROOT else os.path.join(PROJECT_DIR, "output", project_name), "shikomi", "frames224")

    for sf in sel_files:
        with open(sf) as f:
            _meta = json.load(f)
        vid = os.path.basename(_meta.get("video", "")).rsplit(".", 1)[0]
        kfs, shots = load_select_frames(sf)
        for kf in kfs:
            kf["video_id"] = vid
            all_kf.append(kf)
        all_video_shots[vid] = shots
        all_video_meta[vid] = _meta

    # 去重（同一视频同一帧只取一次）
    seen = set()
    unique_kf = []
    for kf in all_kf:
        key = (kf["video_id"], kf["local_frame"])
        if key not in seen:
            seen.add(key)
            unique_kf.append(kf)
    all_kf = unique_kf

    n_all = len(all_kf)
    print(f"[04] DINO 全局: {n_all} key_frames, {len(sel_files)} videos")

    # 读取帧（帧前缀 = video_hash 内容指纹）
    vid2vh = {vid: m.get("video_hash") for vid, m in all_video_meta.items()}
    arr = np.zeros((n_all, S, S, 3), dtype=np.uint8)
    missing = 0
    for i, kf in enumerate(all_kf):
        img = read_frame(frames_base, vid2vh.get(kf["video_id"], ""), kf["local_frame"])
        if img is None:
            missing += 1
        else:
            arr[i] = img
    print(f"  read {n_all - missing}/{n_all} frames")

    kf_emb = run_dino(arr)

    # 保存全局输出
    save_global(out_dir, kf_emb, all_kf, len(sel_files))
    print(f"[04] DINO 全局: {kf_emb.shape} -> {out_dir}/")

    # 保存全局输出
    save_global(out_dir, kf_emb, all_kf, len(sel_files))
    print(f"[04] DINO 全局: {kf_emb.shape} -> {out_dir}/")

    # 保存 per-video 骨架
    for vid, shots in all_video_shots.items():
        save_per_video(out_dir, vid, shots, all_video_meta[vid])

    # ── 全局矩阵（所有帧 × 所有帧 余弦相似度）──
    print(f"[04] 构建全局余弦矩阵 ({n_all}x{n_all})...")
    kfn = kf_emb / (np.linalg.norm(kf_emb, axis=1, keepdims=True) + 1e-10)
    global_sim = kfn @ kfn.T
    np.save(os.path.join(out_dir, "global_similarity_matrix.npy"), global_sim.astype(np.float16))
    print(f"[04] 全局矩阵: {global_sim.shape}")

    # ── Per-video 场景合并（各自视频内部矩阵 → dedup/scene 合并）──
    dedup_thr = float(os.environ.get("DEDUP_COS_THR", "0.75"))
    dedup_out = os.path.join(os.path.dirname(out_dir), "dedup")
    os.makedirs(dedup_out, exist_ok=True)

    # 建立 video_id → global index 范围
    vid_to_indices = {}
    for i, kf in enumerate(all_kf):
        vid = kf["video_id"]
        vid_to_indices.setdefault(vid, []).append(i)

    for vid, vid_shots in all_video_shots.items():
        vid_indices = vid_to_indices.get(vid, [])
        if len(vid_indices) == 0:
            continue

        # 提取该视频的 embeddings
        vid_emb = kf_emb[vid_indices]
        vid_frame_ids = [all_kf[i]["local_frame"] for i in vid_indices]

        # frame → global index 映射
        f2idx = {all_kf[i]["local_frame"]: i for i in vid_indices}

        # 每 shot 取关键帧
        ffn, fsid = [], []
        for si, s in enumerate(vid_shots):
            kfs = s.get("key_frames", [])
            if not kfs:
                mid = (s["range"]["start"] + s["range"]["end"]) // 2
                kfs = [int(mid)]
            for fn in kfs:
                if fn in f2idx:
                    ffn.append(int(fn))
                    fsid.append(si)

        if len(ffn) == 0:
            # 没有关键帧，直接跳过
            continue

        # 归一化
        vid_kfn = vid_emb / (np.linalg.norm(vid_emb, axis=1, keepdims=True) + 1e-10)

        # 迭代去重合并
        scene_shots = {fn: {sid} for fn, sid in zip(ffn, fsid)}
        while True:
            I = np.array([f2idx[f] for f in ffn])
            adj = np.array([float(np.dot(kfn[I[i]], kfn[I[i + 1]])) for i in range(len(ffn) - 1)])
            new_fn, new_sid, round_removed = [], [], set()
            fi = 0
            while fi < len(ffn):
                fj = fi
                while fj < len(ffn) - 1 and adj[fj] >= dedup_thr:
                    fj += 1
                if fj - fi >= 1:
                    cand = list(range(fi, fj + 1))
                    # 有脸的优先（face_map 已在 dedup.py 处理，这里简化）
                    if len(cand) < 2:
                        best = cand[0]
                    else:
                        sub = vid_kfn[[vid_indices.index(f2idx[ffn[k]]) for k in cand if f2idx[ffn[k]] in vid_indices]]
                        if len(sub) > 1:
                            cm = (sub @ sub.T).sum(axis=1) - 1.0
                            cm /= sub.shape[0] - 1
                            best = cand[int(np.argmax(cm))]
                        else:
                            best = cand[0]
                    for k in range(fi, fj + 1):
                        if k != best:
                            scene_shots[ffn[best]].update(scene_shots[ffn[k]])
                            round_removed.add(ffn[k])
                    new_fn.append(ffn[best])
                    new_sid.append(fsid[best])
                else:
                    new_fn.append(ffn[fi])
                    new_sid.append(fsid[fi])
                fi = fj + 1
            if not round_removed:
                break
            all_removed = round_removed
            ffn, fsid = new_fn, new_sid

        # 重建 scene 骨架
        scenes = []
        i = 0
        while i < len(ffn):
            sid = fsid[i]
            j = i
            while j < len(ffn) and fsid[j] == sid:
                j += 1
            merged = set()
            for k in range(i, j):
                merged.update(scene_shots[ffn[k]])
            sids = sorted(merged)
            scenes.append({
                "scene_id": len(scenes),
                "shot_range": {"start": sids[0], "end": sids[-1]},
                "shot_frame": {str(sid): ffn[i]},
                "n_shots": len(sids),
                "key_frame": ffn[i],
            })
            i = j

        # 边界修正
        for idx, sc in enumerate(scenes):
            start, end = sc["shot_range"]["start"], sc["shot_range"]["end"]
            if idx > 0:
                prev_end = scenes[idx - 1]["shot_range"]["end"]
                if start <= prev_end:
                    start = prev_end + 1
            if idx < len(scenes) - 1:
                nxt_start = scenes[idx + 1]["shot_range"]["start"]
                if end >= nxt_start:
                    end = nxt_start - 1
            if end < start:
                end = start
            sc["shot_range"]["start"], sc["shot_range"]["end"] = start, end
            sc["n_shots"] = end - start + 1

        # 输出 per-video dedup skeleton
        sk_out = {"shots": vid_shots, "scenes": scenes, "video_id": vid}
        out_path = os.path.join(dedup_out, f"{vid}_skeleton.json")
        with open(out_path, "w") as f:
            json.dump(sk_out, f, ensure_ascii=False, indent=2)
        print(f"[04] dedup [{vid}]: {len(vid_shots)} shots -> {len(scenes)} scenes")

    # scene 级视觉图（per-video，供 graph_merge 无人区 CC 用）
    for vid, vid_shots in all_video_shots.items():
        vid_indices = vid_to_indices.get(vid, [])
        if len(vid_indices) == 0:
            continue
        sk_path = os.path.join(dedup_out, f"{vid}_skeleton.json")
        if not os.path.isfile(sk_path):
            continue
        sk = json.load(open(sk_path))
        scenes = sk["scenes"]
        if len(scenes) == 0:
            continue

        scene_emb = np.zeros((len(scenes), kf_emb.shape[1]), dtype=np.float32)
        for i, sc in enumerate(scenes):
            embs = []
            for sid in range(sc["shot_range"]["start"], sc["shot_range"]["end"] + 1):
                for s in vid_shots:
                    if s["id"] == sid:
                        for fn in s.get("key_frames", []):
                            if fn in f2idx:
                                embs.append(kf_emb[f2idx[fn]])
            if embs:
                scene_emb[i] = np.mean(embs, axis=0)
        scene_emb /= (np.linalg.norm(scene_emb, axis=1, keepdims=True) + 1e-10)
        np.save(os.path.join(dedup_out, f"{vid}_scene_embedding_mean.npy"), scene_emb.astype(np.float16))
        np.save(os.path.join(dedup_out, f"{vid}_scene_visual_graph.npy"), (scene_emb @ scene_emb.T))

    # dedup meta
    json.dump({
        "dino_model_id": DINO_MODEL_ID,
        "embedding_dim": int(kf_emb.shape[1]),
        "dedup_cos_thr": dedup_thr,
        "face_map_used": False,
        "n_videos": len(sel_files),
    }, open(os.path.join(dedup_out, "model_meta.json"), "w"), ensure_ascii=False, indent=2)

    print(f"[04] DINO 全局: 完成（含 per-video dedup）")

else:
    # ═══ Mode A / 单视频文件夹（per-video，输出到 project 目录）═══
    base_dir = project_name if mode_b else video_name
    out_base = OUT_ROOT if OUT_ROOT else os.path.join(PROJECT_DIR, "output", base_dir)
    vh = load_video_hash(vid_name, os.path.join(out_base, "shikomi", "skeleton"))
    in_path = os.path.join(out_base, "shikomi", "select_frames", f"{vh}_select_frames.json")
    kfs, shots = load_select_frames(in_path)
    seen = set()
    unique_kf = []
    for kf in kfs:
        key = kf["local_frame"]
        if key not in seen:
            seen.add(key)
            unique_kf.append(kf)
    kfs = unique_kf

    n_all = len(kfs)
    print(f"[04] DINO [{vid_name}]: {n_all} key_frames")

    frames_base = os.path.join(out_base, "shikomi", "frames224")
    arr = np.zeros((n_all, S, S, 3), dtype=np.uint8)
    for i, kf in enumerate(kfs):
        img = read_frame(frames_base, vh, kf["local_frame"])
        if img is not None:
            arr[i] = img

    kf_emb = run_dino(arr)

    out_dir = os.path.join(out_base, "visual", "dino")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{vh}_key_frame_embeddings.npz"),
             embeddings=kf_emb, frame_ids=np.arange(n_all, dtype=np.int32))
    with open(os.path.join(out_dir, f"{vh}_model_meta.json"), "w") as f:
        json.dump({
            "model_id": DINO_MODEL_ID, "model_dir": DINO_MODEL_DIR,
            "embedding_dim": int(kf_emb.shape[1]),
            "embedding_dtype": str(kf_emb.dtype), "inference_dtype": "float32",
            "input_size": S, "batch_size": BS2,
        }, f, ensure_ascii=False, indent=2)
    meta = json.load(open(in_path))
    # 2026-08-24 fix: 保证 skeleton 中的 key_frames 与 npz 行数一致。
    # load_select_frames 对缺 key_frames 的 shot 会 fallback 到中点帧并写入 npz，
    # 但原始 shots 仍缺该字段，导致下游 fuse 报 "key_frames != npz 行数"。
    # 这里同步给 skeleton 补上同样的 fallback key_frames。
    for s in shots:
        if not s.get("key_frames"):
            mid = int((s["range"]["start"] + s["range"]["end"]) // 2)
            s["key_frames"] = [mid]
    save_per_video(out_dir, vid_name, shots, meta)
    print(f"[04] DINO [{vid_name}]: {kf_emb.shape} -> {out_dir}/")
