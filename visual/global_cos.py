#!/usr/bin/env python3
"""visual/global_cos.py — 全局 DINO cos 矩阵 + 相似簇计算 + Cluster 内代表帧选择（仿 vlm/select_segments.py 的 compute_clusters）。
流程：读 dedup scene 骨架 → 提取非黑帧 scene 代表帧 → 读 DINO 向量 → 全局余弦矩阵 → 并查集聚类
      → 每 Cluster 选代表帧（视觉中心性 + 基础坏帧过滤 + face_present 优先，禁按时间选帧）→ 输出帧处理目录。

用法（sorosoro env）: python visual/global_cos.py <video_dir> [--thr 0.9] [--save-cos]
依赖: visual/dedup/<prefix>_skeleton.json（dedup scene 骨架，含 scene 代表帧）；visual/dino/<prefix>_key_frame_embeddings.npz（DINO 向量）；visual/face_detect/<name>_face_map.json（face_present，dedup 前全帧检测）；shikomi/features/<prefix>_features.json（sharpness/yavg，坏帧轻量判定）
产物: visual/global_cos/gc_skeleton.json（唯一产物：scene 骨架帧处理目录，含完整 N×N 全局 cos 矩阵）；--save-cos 另存 gc_global_cos.npy
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
CLUSTER_THR = float(os.environ.get("CLUSTER_THR", "0.9"))

# 坏帧轻量判定阈值（2026-08-19 施工文档：不新增 AI 质量模型，只做已有/轻量判断；默认保守值待领导定稿，env 可覆盖）
BLUR_THR = float(os.environ.get("GLOBAL_COS_BLUR_THR", "100"))              # sharpness 低于此 = 严重模糊
UNDEREXPOSE_THR = float(os.environ.get("GLOBAL_COS_UNDEREXPOSE_THR", "15"))  # yavg 低于此 = 严重欠曝
OVEREXPOSE_THR = float(os.environ.get("GLOBAL_COS_OVEREXPOSE_THR", "245"))   # yavg 高于此 = 严重过曝


def load_all_dedup_skeletons(video_dir):
    """读目录下**所有** dedup scene 骨架（全局：1 个视频也全局、N 个视频也全局）；
    prefix = 骨架 video_hash（内容指纹，改名/重封装不变）"""
    out = (Path(os.environ["OUT_ROOT"]) if os.environ.get("OUT_ROOT")
           else BASE / "output" / video_dir)
    skels = sorted(glob.glob(str(out / "visual/dedup/*_skeleton.json")))
    if not skels:
        raise SystemExit(f"❌ 无 dedup 骨架（先跑 visual/dedup.py）")
    infos = []
    for p in skels:
        sk = json.load(open(p))
        prefix = sk.get("video_hash")
        if not prefix:
            raise SystemExit(f"❌ dedup 骨架 {p} 缺 video_hash")
        infos.append((Path(sk.get("video", "")).stem, sk, prefix, out))
    print(f"✔ dedup 骨架 {len(infos)} 个（全局聚簇）: {[i[2] for i in infos]}")
    return infos


def load_dino_vectors_for_scene_reps(out, prefix, scene_reps):
    """读 DINO 向量 + 为非黑帧 scene 代表帧构建索引映射 + 归一化 emb。
    返回 (idx_of, emb_n, rep_frames, rep_indices, black_scenes)：黑帧 scene 单独输出，不走 cos 矩阵。"""
    dino_dir = out / "visual" / "dino"
    npz_path = dino_dir / f"{prefix}_key_frame_embeddings.npz"
    skel_path = dino_dir / f"{prefix}_skeleton.json"

    if not npz_path.exists():
        raise SystemExit(f"❌ 无 DINO 向量：{npz_path}（先跑 visual/dino_cluster.py）")
    if not skel_path.exists():
        raise SystemExit(f"❌ 无 DINO skeleton: {skel_path}")

    d = np.load(str(npz_path))
    emb = d["embeddings"].astype(np.float32)

    # 读 DINO skeleton 获取 key_frames 列表（帧号 → npz 行号映射）
    dsk = json.loads(skel_path.read_text(encoding="utf-8"))
    kfs = [kf for s in dsk["shots"] for kf in s.get("key_frames", [])]
    # npz 行号 ↔ key_frames 展平序
    idx_of = {kf: i for i, kf in enumerate(kfs)}

    # 分离黑帧 scene 和非黑帧 scene（黑帧 scene 有专门输出，不走 cos 矩阵）
    black_scenes = [sc for sc in scene_reps if sc.get("black")]
    non_black_scenes = [sc for sc in scene_reps if not sc.get("black")]

    # 提取**非黑帧 scene**代表帧的 npz 行号
    rep_frames = [sc["key_frame"] for sc in non_black_scenes]
    rep_indices = []
    for fn in rep_frames:
        if fn not in idx_of:
            raise SystemExit(f"❌ 非黑帧 scene 代表帧 {fn} 不在 DINO key_frames 中（数据不一致）")
        rep_indices.append(idx_of[fn])

    # 归一化：cos(a,b) = dot(a_norm, b_norm)
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    print(f"✔ DINO 向量：{emb.shape} = {len(kfs)} 帧 × {emb.shape[1]} 维")
    print(f"✔ Scene: {len(scene_reps)} 个（非黑帧 {len(non_black_scenes)} / 黑帧 {len(black_scenes)}）")
    print(f"✔ 非黑帧 scene 代表帧：{len(rep_frames)} 帧（用于 cos 矩阵）")
    return idx_of, emb_n, rep_frames, rep_indices, black_scenes


def compute_global_cos_matrix(emb_n):
    """全局余弦矩阵（GPU 加速）：N×N 对称矩阵，diag=1.0。返回 (N, N) float32 余弦相似度矩阵。"""
    N = emb_n.shape[0]
    print(f"[cos] 构建 {N}×{N} 余弦矩阵...")

    # GPU 矩阵乘法：cos(a,b) = dot(a_norm, b_norm)
    emb_t = torch.from_numpy(emb_n).to("cuda")
    with torch.inference_mode():
        cos_t = emb_t @ emb_t.T  # (N, N)
    cos_matrix = cos_t.cpu().numpy().astype(np.float32)

    print(f"[cos] 矩阵完成：{cos_matrix.shape}, diag=1.0, range=[{cos_matrix.min():.3f}, {cos_matrix.max():.3f}]")
    return cos_matrix


def compute_clusters_from_cos(cos_matrix, frames, thr=CLUSTER_THR):
    """从余弦矩阵计算相似簇（并查集连通分量，cos≥thr 视为同簇）。
    返回 (cluster_of, clusters)：{帧号: 簇 id(c001...)} 与 [(簇 id, [帧号...])...] 按簇大小降序。"""
    N = cos_matrix.shape[0]
    adj = cos_matrix >= thr

    # 并查集
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 上三角遍历（对称矩阵）
    for a in range(N):
        for b in range(a + 1, N):
            if adj[a, b]:
                union(a, b)

    # 收集连通分量
    comp = {}
    for i in range(N):
        comp.setdefault(find(i), []).append(i)

    # 过滤单元素簇（只保留真正的"簇"=多帧相似）
    clusters = [(i, members) for i, members in comp.items() if len(members) > 1]
    clusters.sort(key=lambda x: -len(x[1]))  # 大簇在前

    # 帧号 → 簇 id 映射
    cluster_of = {}
    result_clusters = []
    for ci, (_, members) in enumerate(clusters, 1):
        cid = f"c{ci:03d}"
        member_frames = [frames[i] for i in members]
        result_clusters.append((cid, member_frames))
        for fn in member_frames:
            cluster_of[fn] = cid

    print(f"✔ 簇映射：{len(cluster_of)} 帧 / {len(clusters)} 簇（thr={thr}）")
    return cluster_of, result_clusters


def load_face_map(out, prefix):
    """读 face_detect 产物 face_map（dedup 前 DINO key_frames 全帧检测，代表帧 face_present 数据源）。
    返回 {帧号: bool}（键为 int）。"""
    p = out / "visual" / "face_detect" / f"{prefix}_face_map.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 face_map：{p}（先跑 visual/face_detect.py）")
    fm = json.loads(p.read_text(encoding="utf-8"))
    face_map = {int(k): bool(v) for k, v in fm.get("face_map", {}).items()}
    print(f"✔ face_map：{len(face_map)} 帧（face_present={sum(face_map.values())}）")
    return face_map


def load_features(out, prefix):
    """读 shikomi features（sharpness / quality[yavg]），坏帧轻量判定数据源。
    features 按帧序号（0 基）索引，长度 = total_frames。"""
    p = out / "shikomi" / "features" / f"{prefix}_features.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 features：{p}（先跑 shikomi/kaiseki）")
    ft = json.loads(p.read_text(encoding="utf-8"))
    sharpness = np.array(ft["sharpness"], dtype=np.float32)
    quality = ft["quality"]
    print(f"✔ features：{len(sharpness)} 帧（sharpness / yavg 轻量质量）")
    return sharpness, quality


def is_bad_frame(fn, sharpness, quality):
    """基础画面可用性（轻量，无 AI 质量模型）：解码异常 / 严重模糊 / 严重欠曝 / 严重过曝"""
    if fn < 0 or fn >= len(sharpness):
        return True  # 解码异常/索引越界
    q = quality[fn]
    if not isinstance(q, dict) or "yavg" not in q:
        return True  # 数据异常（缺质量条目）
    if sharpness[fn] < BLUR_THR:
        return True  # 严重模糊
    if q["yavg"] < UNDEREXPOSE_THR:
        return True  # 严重欠曝
    if q["yavg"] > OVEREXPOSE_THR:
        return True  # 严重过曝
    return False


def select_representatives(emb_norm, global_frames, clusters, face_map_by_vh, feat_by_vh):
    """全局 Cluster 内代表帧选择（2026-08-19 领导施工文档：禁止按时间顺序选帧）。
    新标准：1) Cluster 视觉中心性（centroid = normalize(mean(emb))，centrality 余弦最高）；
    2) 基础画面可用性（先剔坏帧：严重模糊/过曝/欠曝/解码异常，逐帧查自己视频 features）；
    3) face_present 优先（复用 dedup 前 face_map，按视频查，不重检测）。全局聚簇可跨视频。
    返回 {帧标识 (vh, fn): {"representative": (rvh, rfn), "centrality": float}}，覆盖全部非黑帧 scene 帧。"""
    emb_of = {fid: emb_norm[i] for i, fid in enumerate(global_frames)}
    reps = {}

    def pick(frames):
        # 1) 基础画面可用性过滤（逐帧查自己视频 features）
        cand = [f for f in frames if not is_bad_frame(f[1], *feat_by_vh[f[0]])]
        if not cand:
            cand = frames  # 全坏兜底：退化为原集
        # 2) face_present 优先（按视频查 face_map，不重检测）
        face_cand = [f for f in cand if face_map_by_vh[f[0]].get(f[1])]
        pool = face_cand if face_cand else cand
        # 3) Cluster 视觉中心性：centroid = normalize(mean(emb))
        centroid = np.mean([emb_of[f] for f in pool], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        best = max(pool, key=lambda f: float(emb_of[f] @ centroid))
        return best, float(emb_of[best] @ centroid)

    for cid, frames in clusters:
        rep, cent = pick(frames)
        for f in frames:
            reps[f] = {"representative": rep, "centrality": round(cent, 6)}
    # 单帧（未进多帧簇）：代表 = 自身，centrality = 1.0
    for f in global_frames:
        reps.setdefault(f, {"representative": f, "centrality": 1.0})
    return reps


def build_skeleton(infos, cluster_of, reps, thr, cos_matrix, global_frames):
    """全局骨架输出（唯一产物 gc_skeleton.json，**帧处理目录**，2026-08-19 大名）。
    scenes = 所有视频 scene 按序一条，每 scene 带 video_hash + frames 帧标记（下游读此目录定要处理的帧：
    簇 key + 单帧，簇成员共用簇 key 帧，黑帧不处理）。帧标记三种：
      正常帧 <hash>_f<fn>；簇代表帧 <hash>_f<fn>_<cXXX>key；
      簇成员 <hash>_f<fn>_<cXXX>：<hash>_f<代表帧>_<cXXX>key（可跨视频共用）。
    黑帧 scene：frames 空数组 + black=true 显式标注。
    cos（2026-08-21 大名定稿 A 方案：完整 N×N 全局 cos 矩阵落进骨架，下游从骨架读、禁二次计算）：
    顶层 cos_matrix = N×N（对称 diag=1，行/列 = cos_frames 非黑 scene 代表帧），cos_frames = [[video_hash, 帧号]...]。
    返回骨架 dict（videos meta + threshold / n_clusters / scenes + cos_matrix/cos_frames）。"""
    rep_of = {fid: r["representative"] for fid, r in reps.items()}
    videos = []
    scenes = []
    meta_keys = ("video", "video_id", "video_hash", "fps", "width", "height", "total_frames")
    for name, sk, prefix, out in infos:
        videos.append({k: sk.get(k) for k in meta_keys if k in sk})
        for sc in sk["scenes"]:
            # scene_id：从 dedup id "32145b_Scene20" 解析序号 → 20
            sid = sc["id"]
            try:
                sid = int(str(sid).rsplit("Scene", 1)[1])
            except Exception:
                pass  # 解析失败退化为完整 id
            entry = {"scene_id": sid, "video_hash": prefix,
                     "shot_range": sc.get("shot_range")}
            if sc.get("black"):
                entry.update({"frames": [], "black": True})
                scenes.append(entry)
                continue
            fn = sc["key_frame"]
            fid = (prefix, fn)
            c = cluster_of.get(fid)
            if c is None:
                mark = f"{prefix}_f{fn}"                                # 正常帧（不在簇内）
            else:
                rep = rep_of[fid]
                if rep == fid:
                    mark = f"{prefix}_f{fn}_{c}key"                     # 簇代表帧
                else:
                    rvh, rfn = rep
                    mark = f"{prefix}_f{fn}_{c}：{rvh}_f{rfn}_{c}key"    # 簇成员：共用簇代表帧（可跨视频）
            entry["frames"] = [mark]
            scenes.append(entry)
    return {
        "videos": videos,          # 每视频一份 meta（下游按 video_hash 取，供 HTML 渲染等）
        "threshold": thr,
        "n_clusters": len(set(cluster_of.values())),
        "scenes": scenes,          # 帧处理目录：所有视频 scene 按序一条，带 video_hash + frames 标记
        # 完整 N×N 全局 cos（2026-08-21 大名定稿 A）：行/列 = cos_frames 非黑 scene 代表帧
        "cos_matrix": cos_matrix.tolist(),
        "cos_frames": [[vh, fn] for vh, fn in global_frames],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--thr", type=float, default=CLUSTER_THR, help="簇判定阈值（默认 0.9）")
    ap.add_argument("--save-cos", action="store_true", help="保存全局 cos 矩阵到 npy")
    args = ap.parse_args()

    # 挨个读目录下**所有**视频的单视频产物（dedup 骨架 / DINO 向量 / face_map / features），直到读完。
    # A 路线 1 个视频 = 读 1 份，B 路线 N 个视频 = 读 N 份；单视频也走同一全局逻辑。
    infos = load_all_dedup_skeletons(args.video_dir)
    if not infos:
        raise SystemExit("❌ 无 dedup 骨架")

    all_emb = []          # 各视频非黑帧代表帧归一化 emb，全局合并
    global_frames = []    # 全局帧标识 [(vh, fn), ...]，与 emb_all 行序对应
    face_map_by_vh = {}
    feat_by_vh = {}
    for name, sk, prefix, out in infos:
        scene_reps = sk["scenes"]
        idx_of, emb_n, rep_frames, rep_indices, black_scenes = load_dino_vectors_for_scene_reps(out, prefix, scene_reps)
        all_emb.append(emb_n[rep_indices])
        global_frames.extend((prefix, fn) for fn in rep_frames)
        face_map_by_vh[prefix] = load_face_map(out, prefix)
        feat_by_vh[prefix] = load_features(out, prefix)

    emb_all = np.concatenate(all_emb, axis=0)
    print(f"✔ 全局合并：{len(global_frames)} 帧（{len(infos)} 个视频）")

    # 全局 cos 矩阵（非黑帧代表帧 × 非黑帧代表帧，GPU）
    cos_matrix = compute_global_cos_matrix(emb_all)

    # 输出目录：output/<项目>/visual/global_cos/
    out = infos[0][3]
    global_cos_dir = out / "visual" / "global_cos"
    global_cos_dir.mkdir(parents=True, exist_ok=True)

    # 保存 cos 矩阵（可选）
    if args.save_cos:
        cos_path = global_cos_dir / "gc_global_cos.npy"
        np.save(str(cos_path), cos_matrix)
        print(f"✔ Cos 矩阵已保存：{cos_path}")

    # 全局相似簇（输入是全局非黑帧代表帧）
    cluster_of, clusters = compute_clusters_from_cos(cos_matrix, global_frames, thr=args.thr)

    # Cluster 内代表帧选择（视觉中心性 + 基础坏帧过滤 + face_present 优先，禁按时间选帧）
    reps = select_representatives(emb_all, global_frames, clusters, face_map_by_vh, feat_by_vh)

    # 骨架输出（唯一产物）：帧处理目录 + 完整全局 cos 矩阵，含所有视频 scenes
    skeleton = build_skeleton(infos, cluster_of, reps, args.thr, cos_matrix, global_frames)
    skel_path = global_cos_dir / "gc_skeleton.json"
    # 完整 N×N cos 矩阵值量大（vivant 2287² ≈ 520 万），indent 徒增 2-3 倍体积、
    # 对可读性毫无意义 → 整文件紧凑写（下游 json.load 消费，2026-08-21 大名 A 方案）
    skel_path.write_text(json.dumps(skeleton, ensure_ascii=False), encoding="utf-8")
    print(f"✔ 骨架（唯一输出）：{skel_path}（{len(skeleton['scenes'])} scene / {skeleton['n_clusters']} 簇 / thr={skeleton['threshold']}）")
    print("  标记样例（前 8 条非黑帧 scene）:")
    shown = 0
    for s in skeleton["scenes"]:
        if s["frames"]:
            print(f"    scene {s['scene_id']} [{s['video_hash']}]: {s['frames'][0]}")
            shown += 1
        if shown >= 8:
            break

    # 打印最大几个簇（含代表帧）
    if clusters:
        print("\n最大 5 簇:")
        for cid, frames in clusters[:5]:
            rep = reps[frames[0]]["representative"]
            print(f"  {cid}: {len(frames)} 帧 → 代表帧 {rep[0]}_f{rep[1]}")

    # 打印代表帧选择统计
    print(f"✔ 代表帧选择完成：覆盖 {len(reps)} 帧（簇 key + 单帧 = 下游要处理的帧）")


if __name__ == "__main__":
    main()
