#!/usr/bin/env python3
"""visual/face_recognition.py — 人脸检测 + ArcFace 识别 + 全局匹配 → person_timeline。

用法:
  Mode A（单视频）:   python3 visual/face_recognition.py <视频名>
  Mode B 单视频:     python3 visual/face_recognition.py <视频名> --project <project_name> --video <vid_name>
  Mode B 全局:       python3 visual/face_recognition.py --project <project_name>
  可选: [--dino-filter --dino-cos 0.95]（DINO 均值过滤，生成 flat_dino HTML）
依赖: visual/global_cos/gc_skeleton.json（帧处理目录，先跑 visual/global_cos.py）；shikomi/frames/ 帧图；insightface 本地模型 buffalo_l（一切本地，禁联网下载）；--dino-filter 需 visual/dino/ DINO 向量
产物: face_head_fusion/person_timeline.json(+_meta)（canonical thr=0.40）、sweep_records.json 缓存；v01_visual_group/exp_sweep/ 各阈值 HTML
"""
import json, os, sys, time, warnings, re, argparse, glob
from pathlib import Path
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
OUT_ROOT = os.environ.get("OUT_ROOT")
EXPECTED_DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# project_mode: mode_b, project_name, vid_name
_mode_b = False
_project_name = None
_vid_name = None

def visual_dir(vn):
    """返回 visual 目录（OUT_ROOT 优先；否则按 Mode A/B 分支）"""
    if OUT_ROOT:
        return os.path.join(OUT_ROOT, "visual")
    if _mode_b:
        return os.path.join(OUTPUT_DIR, _project_name, "visual")
    return os.path.join(OUTPUT_DIR, vn, "visual")

def load_video_hash(vid_name, skel_dir):
    """取视频内容指纹 video_hash：vid_name 即内容哈希（batch_pipeline 算好传入，
    下游禁止二次计算哈希/文件名匹配，直接继承）"""
    return vid_name

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

def _insightface_local_only(sub_dir, name, force=False, root="~/.insightface"):
    """一切本地（2026-08-18 大名）：insightface 下载源是 GitHub（国内直连不通会 hang）。
    模型已落标准路径 <root>/models/<name>；本地命中即返回，缺失快速报错，绝不联网下载。"""
    dir_path = os.path.join(os.path.expanduser(root), sub_dir, name)
    if os.path.exists(dir_path):
        return dir_path
    raise FileNotFoundError(
        f"❌ insightface 本地模型缺失: {dir_path}\n"
        "（联网下载已禁用——模型必须已放标准路径，勿让程序连网）")

def _patch_insightface_offline():
    """把 insightface 下载入口全替换成本地-only（缺失即报错，防任何路径拼错触发联网 hang）"""
    try:
        from insightface.utils import storage as _st
        _st.download = _insightface_local_only
        _st.ensure_available = _insightface_local_only
        _st.download_onnx = lambda sub_dir, model_file, force=False, root="~/.insightface", download_zip=False: \
            _insightface_local_only(sub_dir, model_file, force=force, root=root)
    except Exception:
        pass

def frames_dir(vn):
    """帧图目录：统一 <out>/<项目>/shikomi/frames/（2026-08-19 大名定稿新格式）"""
    if OUT_ROOT:
        return os.path.join(OUT_ROOT, "shikomi", "frames")
    if _mode_b:
        return os.path.join(OUTPUT_DIR, _project_name, "shikomi", "frames")
    return os.path.join(OUTPUT_DIR, vn, "shikomi", "frames")

def get_suffix(name):
    """返回文件名后缀 _<name>"""
    return f"_{name}"

def load_global_cos_skeleton(vn):
    """读 global_cos 骨架（唯一产物 gc_skeleton.json，**帧处理目录**，2026-08-19 大名）：
    下游只读它确定要处理的帧。缺失 → 报错停（不兜底降级，先跑 visual/global_cos.py）。
    返回 (sk, scenes, vh)：scenes 即目录里的帧标记（A 单视频即该视频数据）；
    vh = 视频哈希（帧路径 / 输出命名用，A 模式 vn 即哈希）"""
    gc_path = os.path.join(visual_dir(vn), "global_cos", "gc_skeleton.json")
    if not os.path.isfile(gc_path):
        raise SystemExit(f"❌ 无 global_cos 骨架 {gc_path}（先跑 python visual/global_cos.py）")
    sk = json.load(open(gc_path))
    scenes = sk["scenes"]
    # gc 骨架的 videos 含每视频 meta（width/height/fps 等，供 HTML 渲染）
    meta = next((v for v in sk.get("videos", []) if v.get("video_hash") == vn), {})
    for k in ("video_id", "video", "video_hash", "fps", "width", "height", "total_frames"):
        if k in meta:
            sk[k] = meta[k]
    return sk, scenes, vn

def parse_compute_src(mark):
    """解析 global_cos 骨架 frames 标记 → 计算源帧号（簇内共用簇 key 帧）。
    标记三种（全角冒号 ：分隔）：
      - 正常帧 <hash>_f<fn>                  → 源帧 = fn（自身）
      - 簇 key  <hash>_f<fn>_cXXXkey         → 源帧 = fn（自身）
      - 簇成员  <hash>_f<fn>_cXXX：<hash>_f<src>_cXXXkey → 源帧 = src（簇 key 帧）
    无法解析 → 返回 None（调用方报错停）"""
    body = mark.split("：")[-1]               # 簇成员取冒号后（簇 key 帧名），其余取自身
    m = re.search(r"_f(\d+)", body)
    return int(m.group(1)) if m else None

def parse_frame_mark(mark):
    """解析帧标记 → (视频哈希, 计算源帧号)。簇成员取冒号后簇 key 帧名（共用）。
    标记：<hash>_f<fn> / <hash>_f<fn>_cXXXkey / <hash>_f<fn>_cXXX：<hash>_f<src>_cXXXkey
    无法解析 → (None, None)（调用方报错停）"""
    body = mark.split("：")[-1]               # 簇成员取冒号后（簇 key 帧名），其余取自身
    m = re.match(r"(.+?)_f(\d+)", body)
    return (m.group(1), int(m.group(2))) if m else (None, None)

QUALITY_THR = 0.30
MIN_FACE_AREA_PERCENT = 0.001
MATCH_THRS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c",
    "#e67e22", "#f1c40f", "#e91e63", "#00bcd4", "#ff5722", "#8bc34a",
    "#3f51b5", "#ff9800", "#795548", "#607d8b", "#cddc39", "#ff6f00",
]

def detect_faces(vn, force=False):
    """单视频人脸检测 + 特征抽取：读 gc_skeleton 帧处理目录，逐计算源帧检测（簇内共用），
    有缓存则复用。返回 (recs, scenes, sk, scene_frames)。"""
    out_dir = os.path.join(visual_dir(vn), "face_head_fusion")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "sweep_records.json")

    if not force and os.path.isfile(cache_path):
        data = json.load(open(cache_path))
        scene_frames = {int(k): v for k, v in data["scene_frames"].items()}
        print(f"[sweep] reuse {len(data['recs'])} records")
        return data["recs"], data["scenes"], data["sk"], scene_frames

    from insightface.app import FaceAnalysis
    import cv2
    _patch_insightface_offline()          # 一切本地：禁用 GitHub 模型下载（缺失即报错）
    INSIGHTFACE_DIR = _locate_model(("INSIGHTFACE_DIR",),
                                    "{hf}/insightface/models/buffalo_l")
    # root 必须是 <root>/models/<name> 的父目录：FaceAnalysis 内部 ensure_available 拼
    # root/models/<name>（storage.py），传浅一层会触发 GitHub 下载 hang
    fa = FaceAnalysis(name=os.path.basename(INSIGHTFACE_DIR),
                      root=os.path.dirname(os.path.dirname(INSIGHTFACE_DIR)),
                      providers=["CUDAExecutionProvider"],
                      allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=0, det_size=(960, 960))
    fa.det_model.det_thresh = 0.3

    # 读 global_cos 骨架（新规矩 2026-08-19）：定「识别哪些帧」，不读 dedup/dino 骨架
    sk, scenes, vh = load_global_cos_skeleton(vn)
    fd = frames_dir(vn)

    # 解析每 scene frames 标记 → 计算源帧；簇成员共用簇 key 帧（只算一次）。
    # 黑帧 scene（black=true）：不识别，但作为独立 scene 保留槽位、绝对顺序不乱
    source_scenes = defaultdict(list)     # 计算源帧号 → [scene_idx...]（0-based）
    scene_src = {}                        # scene_idx → 计算源帧号（黑帧 scene 无）
    for idx, sc in enumerate(scenes):
        if sc.get("black"):
            continue
        marks = sc.get("frames") or []
        if not marks:
            continue
        src = parse_compute_src(marks[0])
        if src is None:
            raise SystemExit(
                f"❌ scene {sc.get('scene_id')} 帧标记无法解析: {marks[0]}")
        source_scenes[src].append(idx)
        scene_src[idx] = src

    compute_frames = sorted(source_scenes)
    print(f"[sweep] {vn}: {len(scenes)} scenes, "
          f"{len(compute_frames)} compute frames（簇 key + 单帧，簇内共用）")

    recs = []
    n_face = 0
    scene_frames = defaultdict(list)
    t0 = time.time()
    for fn in compute_frames:
        fp = os.path.join(fd, f"{vh}_f{fn}.jpg")
        sids = source_scenes[fn]
        for sid in sids:
            scene_frames[sid].append(fn)

        img = cv2.imread(fp)
        if img is None:
            continue

        pfaces = fa.get(img)
        face_dets = []
        if pfaces:
            frame_area = img.shape[0] * img.shape[1]
            raw = [f for f in pfaces if f.det_score >= QUALITY_THR]
            n_raw = len(raw)
            dyn_thr = MIN_FACE_AREA_PERCENT + 0.002 * n_raw
            for f in raw:
                bbox = [int(v) for v in f.bbox.tolist()]
                fw = bbox[2] - bbox[0]
                fh = bbox[3] - bbox[1]
                if fw * fh < frame_area * dyn_thr:
                    continue
                face_emb = f.normed_embedding.tolist()
                face_dets.append({
                    "frame": fn, "video_id": vh,
                    "has_face": True,
                    "face_emb": face_emb,
                    "face_bbox": bbox,
                    "det_score": float(f.det_score),
                })

        # 识别结果簇内共用：同一源帧检测一次，复制到其服务的每个 scene
        if face_dets:
            for d in face_dets:
                for sid in sids:
                    recs.append({**d, "scene_id": sid})
            n_face += len(face_dets) * len(sids)
        else:
            for sid in sids:
                recs.append({
                    "scene_id": sid, "frame": fn, "video_id": vh,
                    "has_face": False,
                    "face_bbox": None,
                    "face_emb": None,
                    "det_score": 0.0,
                })

    # scenes 补 key_frame（= 计算源帧），兼容 build_html/build_flat_dino_html 渲染；
    # 黑帧 scene 无 key_frame（渲染安全跳过），但仍是独立 scene 保留原位
    for idx, sc in enumerate(scenes):
        if idx in scene_src:
            sc.setdefault("key_frame", scene_src[idx])

    dur = time.time() - t0
    n_face = sum(1 for r in recs if r["has_face"])
    print(f"[sweep] done: {n_face}/{len(recs)} faces in {dur:.1f}s")

    json.dump({
        "recs": recs, "scenes": scenes, "sk": sk,
        "scene_frames": {int(k): v for k, v in scene_frames.items()},
    }, open(cache_path, "w"), ensure_ascii=False)

    return recs, scenes, sk, scene_frames

def detect_faces_global(project_name, force=False):
    """B 路线全局人脸检测：读 gc_skeleton.json（帧处理目录）→ 处理目录里所有要处理的帧
    （簇 key + 单帧，跨全部视频，簇成员共用簇 key 帧，黑帧跳过）。返回 (recs, scenes, sk, scene_frames)：
    recs 各条 video_id = 帧标记哈希；scenes/sk 来自 gc_skeleton（sk 注入首视频 meta 供 HTML 渲染）。"""
    out_dir = os.path.join(visual_dir(project_name), "face_head_fusion")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "sweep_records_global.json")

    if not force and os.path.isfile(cache_path):
        data = json.load(open(cache_path))
        sf = {int(k): v for k, v in data["scene_frames"].items()}
        print(f"[sweep] 全局 reuse {len(data['recs'])} records")
        return data["recs"], data["scenes"], data["sk"], sf

    from insightface.app import FaceAnalysis
    import cv2
    _patch_insightface_offline()          # 一切本地：禁用 GitHub 模型下载（缺失即报错）
    INSIGHTFACE_DIR = _locate_model(("INSIGHTFACE_DIR",),
                                    "{hf}/insightface/models/buffalo_l")
    # root 必须是 <root>/models/<name> 的父目录：FaceAnalysis 内部 ensure_available 拼
    # root/models/<name>（storage.py），传浅一层会触发 GitHub 下载 hang
    fa = FaceAnalysis(name=os.path.basename(INSIGHTFACE_DIR),
                      root=os.path.dirname(os.path.dirname(INSIGHTFACE_DIR)),
                      providers=["CUDAExecutionProvider"],
                      allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=0, det_size=(960, 960))
    fa.det_model.det_thresh = 0.3

    # 读全局 gc_skeleton.json（帧处理目录：所有视频的帧标记，哈希贯穿）
    gc_path = os.path.join(visual_dir(project_name), "global_cos", "gc_skeleton.json")
    if not os.path.isfile(gc_path):
        raise SystemExit(f"❌ 无 global_cos 骨架 {gc_path}（先跑 visual/global_cos.py）")
    sk = json.load(open(gc_path))
    scenes = sk["scenes"]
    # 首视频 meta 供 HTML 渲染（width/height/fps；真实数据，非兜底）
    vids = sk.get("videos", [])
    v0 = vids[0] if vids else {}
    for k in ("fps", "width", "height", "total_frames", "video"):
        if k in v0:
            sk[k] = v0[k]
    sk["video_id"] = project_name

    if OUT_ROOT:
        frames_base = os.path.join(OUT_ROOT, "shikomi", "frames")
    else:
        frames_base = os.path.join(OUTPUT_DIR, project_name, "shikomi", "frames")

    # 解析每 scene frames 标记 → 计算源帧（簇 key + 单帧；黑帧跳过；簇成员共用簇 key 帧）
    source_scenes = defaultdict(list)     # (vh, fn) → [scene_idx...]（0-based 全局下标）
    scene_src = {}                        # scene_idx → (vh, fn)
    for idx, sc in enumerate(scenes):
        if sc.get("black"):
            continue
        marks = sc.get("frames") or []
        if not marks:
            continue
        vh, fn = parse_frame_mark(marks[0])
        if fn is None:
            raise SystemExit(f"❌ scene {sc.get('scene_id')} 帧标记无法解析: {marks[0]}")
        source_scenes[(vh, fn)].append(idx)
        scene_src[idx] = (vh, fn)

    compute_frames = sorted(source_scenes)
    print(f"[sweep] 全局: {len(scenes)} scenes, {len(compute_frames)} compute frames"
          f"（簇 key + 单帧，跨 {len(vids)} 视频）")

    recs = []
    scene_frames = defaultdict(list)
    for (vh, fn) in compute_frames:
        fp = os.path.join(frames_base, f"{vh}_f{fn}.jpg")
        sids = source_scenes[(vh, fn)]
        for sid in sids:
            scene_frames[sid].append(fn)
        if not os.path.isfile(fp):
            continue
        img = cv2.imread(fp)
        if img is None:
            continue

        pfaces = fa.get(img)
        face_dets = []
        if pfaces:
            frame_area = img.shape[0] * img.shape[1]
            raw = [f for f in pfaces if f.det_score >= QUALITY_THR]
            n_raw = len(raw)
            dyn_thr = MIN_FACE_AREA_PERCENT + 0.002 * n_raw
            for f in raw:
                bbox = [int(v) for v in f.bbox.tolist()]
                fw = bbox[2] - bbox[0]
                fh = bbox[3] - bbox[1]
                if fw * fh < frame_area * dyn_thr:
                    continue
                face_dets.append({
                    "frame": fn, "video_id": vh,
                    "has_face": True,
                    "face_emb": f.normed_embedding.tolist(),
                    "face_bbox": bbox,
                    "det_score": float(f.det_score),
                })

        # 识别结果簇内共用：同一源帧检测一次，复制到其服务的每个 scene
        if face_dets:
            for d in face_dets:
                for sid in sids:
                    recs.append({**d, "scene_id": sid})
        else:
            for sid in sids:
                recs.append({"scene_id": sid, "frame": fn, "video_id": vh,
                             "has_face": False, "face_emb": None,
                             "face_bbox": None, "det_score": 0.0})

    # scenes 补 key_frame（= 计算源帧），兼容 build_flat_dino_html 渲染
    for idx, sc in enumerate(scenes):
        if idx in scene_src:
            sc.setdefault("key_frame", scene_src[idx][1])

    n_face = sum(1 for r in recs if r["has_face"])
    print(f"[sweep] 全局: {n_face}/{len(recs)} faces, {len(scenes)} scenes")

    json.dump({
        "recs": recs, "scenes": scenes, "sk": sk,
        "scene_frames": {int(k): v for k, v in scene_frames.items()},
    }, open(cache_path, "w"), ensure_ascii=False)

    return recs, scenes, sk, scene_frames

def compute_sim_matrix(recs):
    """预计算余弦相似度矩阵（一次，所有阈值和方法共用）。返回 (face_indices, sim_matrix)：
    face_indices[i] = recs 中的下标，sim_matrix = n×n 余弦相似度。"""
    face_indices = [i for i, r in enumerate(recs) if r["has_face"] and r["face_emb"] is not None]
    n = len(face_indices)
    if n < 2:
        return face_indices, None
    emb = np.stack([np.array(recs[i]["face_emb"], dtype=np.float32) for i in face_indices])
    sim_matrix = emb @ emb.T
    print(f"[sweep] sim matrix: {n}×{n} faces")
    return face_indices, sim_matrix

def chinese_whispers(recs, face_indices, sim_matrix, match_thr):
    """Chinese Whispers 社区检测，使用预计算相似度矩阵。
    直接写 pid 到 recs 每条记录上，支持一帧多人。
    """
    n = len(face_indices)
    if n < 2:
        return 0, []

    # 构建邻接表（从预计算矩阵读取）
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if recs[face_indices[i]]["frame"] == recs[face_indices[j]]["frame"]:
                continue
            sim = float(sim_matrix[i, j])
            if sim >= match_thr:
                adj[i].append((j, sim))
                adj[j].append((i, sim))

    if not any(adj):
        # 没有边连接——每个 face 独立
        for i in range(n):
            recs[face_indices[i]]["pid"] = i
        return n, [1] * n

    # CW 迭代
    labels = list(range(n))
    for _iter in range(20):
        order = list(range(n))
        np.random.shuffle(order)
        changed = 0
        for i in order:
            if not adj[i]:
                continue
            votes = {}
            for j, w in adj[i]:
                lbl = labels[j]
                votes[lbl] = votes.get(lbl, 0.0) + w
            if not votes:
                continue
            best_label = max(votes, key=votes.get)
            if labels[i] != best_label:
                labels[i] = best_label
                changed += 1
        if changed == 0:
            break

    # 重编号
    unique = {}
    for i in range(n):
        lab = labels[i]
        if lab not in unique:
            unique[lab] = len(unique)
    for i in range(n):
        labels[i] = unique[labels[i]]

    # 收集结果并写 pid 到 recs
    label_groups = defaultdict(list)
    for i in range(n):
        lab = labels[i]
        label_groups[lab].append(i)

    for pid, idxs in label_groups.items():
        for i in idxs:
            recs[face_indices[i]]["pid"] = pid

    chain_sizes = sorted([len(v) for v in label_groups.values()], reverse=True)
    return len(label_groups), chain_sizes

def global_match(recs, face_indices, sim_matrix, match_thr):
    """并查集全局匹配，使用预计算相似度矩阵。
    直接写 pid 到 recs 每条记录上，支持一帧多人。
    """
    n = len(face_indices)
    if n < 2:
        return 0, []

    # 并查集
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if recs[face_indices[i]]["frame"] == recs[face_indices[j]]["frame"]:
                continue
            if float(sim_matrix[i, j]) >= match_thr:
                union(i, j)
                pairs += 1

    # 收集结果并写 pid 到 recs
    label_groups = defaultdict(list)
    for i in range(n):
        label_groups[find(i)].append(i)

    for pid, key in enumerate(sorted(label_groups.keys())):
        for i in label_groups[key]:
            recs[face_indices[i]]["pid"] = pid

    n_persons = len(label_groups)
    chain_sizes = sorted([len(v) for v in label_groups.values()], reverse=True)
    return n_persons, chain_sizes

def build_html(vn, recs, scenes, sk, scene_frames, match_thr, out_path, chain_sizes):
    """小网格版 HTML（索引页看）：grid 每 scene 一列，帧 + 人脸框 + P{id} 标注"""
    frame_w = sk["width"]
    frame_h = sk["height"]
    n_det = sum(1 for r in recs if r.get("pid") is not None)
    persons = set(r["pid"] for r in recs if r.get("pid") is not None)

    # 预构建 frame → set(pid) 映射
    frame_pids = defaultdict(set)
    for r in recs:
        pid = r.get("pid")
        if pid is not None:
            frame_pids[r["frame"]].add(pid)

    h = [
        '<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>',
        '<style>',
        '*{margin:0;padding:0;box-sizing:border-box}',
        'body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:16px}',
        'h1{font-size:18px;margin-bottom:2px}',
        '.sub{font-size:12px;color:#888;margin-bottom:12px}',
        '.stats{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}',
        '.stat-box{background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:8px 14px}',
        '.stat-box .num{font-size:24px;font-weight:700;color:#8af}',
        '.stat-box .lbl{font-size:10px;color:#666}',
        '.grid{display:flex;gap:3px;overflow-x:auto;padding-bottom:16px}',
        '.col{min-width:70px;max-width:100px;flex-shrink:0;border:1px solid #333;border-radius:3px;overflow:hidden;background:#1a1a2e}',
        '.col .head{padding:3px 5px;font-size:9px;font-weight:600;background:#16213e;border-bottom:1px solid #333}',
        '.col .head .n{color:#8af}',
        '.col .head .t{font-size:7px;color:#666;display:block}',
        '.col .head .persons{margin-top:1px;display:flex;gap:1px;flex-wrap:wrap}',
        '.col .head .persons span{padding:0 3px;border-radius:2px;font-size:7px;color:#fff}',
        '.frame-wrap{position:relative;width:100%;aspect-ratio:' + str(frame_w) + '/' + str(frame_h) + ';overflow:hidden;background:#000}',
        '.frame-wrap img{width:100%;height:100%;object-fit:contain;display:block}',
        '.frame-wrap.no-face img{opacity:0.20}',
        '.face-box{position:absolute;border:1.5px solid;border-radius:1px;pointer-events:none;box-sizing:border-box}',
        '.face-label{position:absolute;bottom:0;left:0;font-weight:700;color:#fff;'
        'line-height:1;padding:0 2px;white-space:nowrap;font-size:7px}',
        '</style></head><body>',
    ]
    h.append(f'<h1>MATCH_THR={match_thr} — {vn}</h1>')
    h.append(f'<p class=sub>{n_det} faces, {len(persons)} persons · 链长分布: {chain_sizes[:10]}</p>')
    h.append(f'<div style="margin-bottom:12px"><a href=index.html style="color:#8af">← 返回对比</a></div>')
    h.append('<div class=grid>')

    for si in range(len(scenes)):
        frames = scene_frames.get(si, [])
        if not frames:
            continue
        persons_seen = set()
        for fn in frames:
            persons_seen.update(frame_pids.get(fn, set()))
        persons = sorted(persons_seen)
        h.append(f'<div class=col><div class=head><span class=n>SC{si}</span>'
                 f'<span class=t>{len(frames)}f</span>')
        if persons:
            h.append('<div class=persons>')
            for pid in persons:
                c = COLORS[pid % len(COLORS)]
                h.append(f'<span style=background:{c}>P{pid}</span>')
            h.append('</div>')
        h.append('</div>')
        for fn in sorted(frames):
            # 取该帧所有带 pid 的记录（支持一帧多人）
            frame_dets = [r for r in recs if r["frame"] == fn and r.get("pid") is not None]
            has_face = len(frame_dets) > 0
            cls = "frame-wrap"
            if not has_face:
                cls += " no-face"
            h.append(f'<div class={cls}>')
            h.append(f'<img src=frame_viz/frames/f{fn}.jpg onerror="this.parentElement.style.display=\'none\'" loading=lazy>')
            # 为每个人脸各画一个框
            for r in frame_dets:
                pid = r["pid"]
                bx, by, bx2, by2 = r["face_bbox"]
                pct_l = bx / frame_w * 100
                pct_t = by / frame_h * 100
                pct_w = (bx2 - bx) / frame_w * 100
                pct_h = (by2 - by) / frame_h * 100
                c = COLORS[pid % len(COLORS)]
                h.append(f'<div class=face-box style="left:{pct_l:.1f}%;top:{pct_t:.1f}%;'
                         f'width:{pct_w:.1f}%;height:{pct_h:.1f}%;border-color:{c}">'
                         f'<div class=face-label style="background:{c}">P{pid}</div></div>')
            h.append('</div>')
        h.append('</div>')
    h.append('</div></body></html>')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(h))

    base = os.path.dirname(os.path.dirname(out_path))
    fv_src = os.path.join(base, "frame_viz")
    fv_dst = os.path.join(os.path.dirname(out_path), "frame_viz")
    if not os.path.exists(fv_src):
        fs = frames_dir(vn)
        if os.path.isdir(fs):
            os.makedirs(fv_src, exist_ok=True)
            if not os.path.exists(os.path.join(fv_src, "frames")):
                rel = os.path.relpath(fs, fv_src)
                os.symlink(rel, os.path.join(fv_src, "frames"))
    if os.path.exists(fv_src) and not os.path.exists(fv_dst):
        os.symlink(os.path.relpath(fv_src, os.path.dirname(fv_dst)), fv_dst)

def build_detailed_html(vn, recs, scenes, sk, match_thr, chain_sizes, out_path):
    """大图版：每人链一行，只看有脸的帧，框清楚。"""
    frame_w = sk["width"]
    frame_h = sk["height"]

    # 按 pid 分组（每人一条链）
    pid_recs = defaultdict(list)
    for r in recs:
        pid = r.get("pid")
        if pid is not None:
            pid_recs[pid].append(r)

    n_det = sum(len(v) for v in pid_recs.values())
    persons = set(pid_recs.keys())

    COLORS = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c",
        "#e67e22", "#f1c40f", "#e91e63", "#00bcd4", "#ff5722", "#8bc34a",
        "#3f51b5", "#ff9800", "#795548", "#607d8b", "#cddc39", "#ff6f00",
    ]

    h = [
        '<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>',
        '<style>',
        '*{margin:0;padding:0;box-sizing:border-box}',
        'body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:20px}',
        'h1{font-size:22px;margin-bottom:4px}',
        '.sub{font-size:14px;color:#888;margin-bottom:16px}',
        '.stats{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap}',
        '.stat-box{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:10px 18px}',
        '.stat-box .num{font-size:28px;font-weight:700;color:#8af}',
        '.stat-box .lbl{font-size:11px;color:#666}',
        '.chain{margin-bottom:24px;border:1px solid #333;border-radius:8px;overflow:hidden}',
        '.chain .head{padding:8px 14px;font-size:14px;font-weight:600;display:flex;align-items:center;gap:10px}',
        '.chain .head .pid{font-size:18px}',
        '.chain .head .info{font-size:12px;color:#888}',
        '.chain .frames{display:flex;gap:4px;overflow-x:auto;padding:8px}',
        '.frame-wrap{position:relative;flex-shrink:0;width:240px;aspect-ratio:' + str(frame_w) + '/' + str(frame_h) + ';overflow:hidden;background:#000;border-radius:4px;border:1px solid #333}',
        '.frame-wrap img{width:100%;height:100%;object-fit:contain;display:block}',
        '.frame-label{position:absolute;top:2px;left:2px;font-size:10px;color:#fff;background:rgba(0,0,0,0.6);padding:1px 4px;border-radius:2px}',
        '.face-box{position:absolute;border:2.5px solid;border-radius:3px;pointer-events:none;box-sizing:border-box}',
        '.face-label{position:absolute;bottom:0;left:0;font-weight:700;color:#fff;'
        'line-height:1.2;padding:2px 5px;white-space:nowrap;font-size:13px}',
        '</style></head><body>',
    ]
    h.append(f'<h1>MATCH_THR={match_thr} — {vn} 人脸详情</h1>')
    h.append(f'<p class=sub>{n_det} faces, {len(persons)} persons · 链分布: {chain_sizes[:10]}</p>')
    h.append(f'<div style="margin-bottom:16px"><a href=index.html style="color:#8af">← 返回对比</a></div>')

    # 按链长降序排列
    sorted_pids = sorted(pid_recs.keys(), key=lambda p: len(pid_recs[p]), reverse=True)

    # 显示链长 >= 3 的，单帧/双帧不足3shot跳过
    for pid in sorted_pids:
        rec_list = sorted(pid_recs[pid], key=lambda x: x["frame"])
        if len(rec_list) < 3:
            continue
        c = COLORS[pid % len(COLORS)]
        h.append(f'<div class=chain><div class=head style="background:{c}22;border-bottom:2px solid {c}">'
                 f'<span class=pid style="color:{c}">P{pid}</span>'
                 f'<span class=info>{len(rec_list)} 帧</span></div>'
                 f'<div class=frames>')
        for r in rec_list:
            fn = r["frame"]
            h.append(f'<div class=frame-wrap>')
            h.append(f'<div class=frame-label>SC{r["scene_id"]} f{fn}</div>')
            h.append(f'<img src=frame_viz/frames/f{fn}.jpg onerror="this.parentElement.style.display=\'none\'" loading=lazy>')
            if r.get("face_bbox"):
                bx, by, bx2, by2 = r["face_bbox"]
                pct_l = bx / frame_w * 100
                pct_t = by / frame_h * 100
                pct_w = (bx2 - bx) / frame_w * 100
                pct_h = (by2 - by) / frame_h * 100
                h.append(f'<div class=face-box style="left:{pct_l:.1f}%;top:{pct_t:.1f}%;'
                         f'width:{pct_w:.1f}%;height:{pct_h:.1f}%;border-color:{c}">'
                         f'<div class=face-label style="background:{c}">P{pid}</div></div>')
            h.append('</div>')
        h.append('</div></div>')

    h.append('</body></html>')

    base_dir = os.path.dirname(out_path)
    os.makedirs(base_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(h))

    # frame_viz 链接
    base = os.path.dirname(base_dir)
    fv_src = os.path.join(base, "frame_viz")
    fv_dst = os.path.join(base_dir, "frame_viz")
    if not os.path.exists(fv_src):
        fs = frames_dir(vn)
        if os.path.isdir(fs):
            os.makedirs(fv_src, exist_ok=True)
            if not os.path.exists(os.path.join(fv_src, "frames")):
                rel = os.path.relpath(fs, fv_src)
                os.symlink(rel, os.path.join(fv_src, "frames"))
    if os.path.exists(fv_src) and not os.path.exists(fv_dst):
        os.symlink(os.path.relpath(fv_src, os.path.dirname(fv_dst)), fv_dst)
    print(f"[sweep] detailed HTML -> {out_path}")

def build_index(vn, results, out_dir):
    """阈值扫描索引页：列出各 MATCH_THR 结果卡片（人物数 / 最长链 / 链分布）"""
    h = [
        '<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>',
        '<style>',
        '*{margin:0;padding:0;box-sizing:border-box}',
        'body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:40px}',
        'h1{font-size:24px;margin-bottom:8px}',
        'h2{font-size:14px;color:#888;margin-bottom:30px}',
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}',
        '.card{background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:20px;text-decoration:none;color:#8af;display:block}',
        '.card:hover{background:#16213e;border-color:#8af}',
        '.card .thr{font-size:28px;font-weight:700}',
        '.card .stat{font-size:13px;color:#888;margin-top:8px}',
        '.card .stat span{color:#ddd;font-weight:600}',
        '</style></head><body>',
        f'<h1>阈值扫描 — {vn}</h1>',
        f'<h2>全局余弦匹配，MATCH_THR 从 0.35 到 0.65</h2>',
        '<div class=grid>',
    ]
    for thr, data in sorted(results.items()):
        cs = data.get("chain_sizes", [])
        cs_str = ", ".join(str(s) for s in cs[:8])
        if len(cs) > 8:
            cs_str += "…"
        h.append(f'<a class=card href=thr_{thr:.2f}.html>'
                 f'<div class=thr>{thr:.2f}</div>'
                 f'<div class=stat>人物: <span>{data["persons"]}</span></div>'
                 f'<div class=stat>最长链: <span>{cs[0] if cs else 0}</span> shots</div>'
                 f'<div class=stat>链分布: <span style="font-size:11px">{cs_str}</span></div>'
                 f'</a>')
    h.append('</div></body></html>')
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("\n".join(h))
    print(f"[sweep] index -> {out_dir}/index.html")

def run_method(method_name, match_fn, recs, face_indices, sim_matrix, scenes, sk, scene_frames, vn, out_base):
    """跑一种匹配方法，生成 HTML。所有阈值共用预计算矩阵。"""
    results = {}
    for thr in MATCH_THRS:
        # 清除上一轮 pid 标注
        for r in recs:
            r.pop("pid", None)
        n_persons, chain_sizes = match_fn(recs, face_indices, sim_matrix, thr)
        print(f"  [{method_name}] thr={thr:.2f}: {n_persons} persons, chains={chain_sizes[:6]}")

        # 小网格版（索引页看）
        html_path = os.path.join(out_base, f"{method_name}_thr_{thr:.2f}.html")
        build_html(vn, recs, scenes, sk, scene_frames, thr, html_path, chain_sizes)

        # 大图详情版（只看有脸的帧，每人链一行）
        det_path = os.path.join(out_base, f"{method_name}_detail_{thr:.2f}.html")
        build_detailed_html(vn, recs, scenes, sk, thr, chain_sizes, det_path)

        results[thr] = {
            "persons": n_persons,
            "chain_sizes": chain_sizes,
        }
    return results

def build_compare_index(vn, results_union, results_cw, out_base):
    """索引页：只选阈值，直接看大图画框。"""
    h = [
        '<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>',
        '<style>',
        '*{margin:0;padding:0;box-sizing:border-box}',
        'body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:30px}',
        'h1{font-size:22px;margin-bottom:4px}',
        '.sub{font-size:13px;color:#888;margin-bottom:20px}',
        '.group{margin-bottom:20px}',
        '.group h3{font-size:14px;color:#8af;margin-bottom:8px}',
        '.row{display:flex;gap:8px;flex-wrap:wrap}',
        '.btn{display:inline-block;padding:10px 20px;background:#1a1a2e;border:1px solid #333;border-radius:6px;color:#8af;text-decoration:none;font-size:18px;font-weight:700}',
        '.btn:hover{background:#16213e;border-color:#8af}',
        '.btn .m{font-size:11px;font-weight:400;color:#666;display:block}',
        '</style></head><body>',
        f'<h1>{vn}</h1>',
        f'<p class=sub>点阈值看大图画框结果</p>',
    ]
    # 并查集
    h.append('<div class=group><h3>并查集 (Union-Find)</h3><div class=row>')
    for thr in MATCH_THRS:
        h.append(f'<a class=btn href=union_detail_{thr:.2f}.html>{thr:.2f}<span class=m>并查集</span></a>')
    h.append('</div></div>')
    # CW
    h.append('<div class=group><h3>Chinese Whispers</h3><div class=row>')
    for thr in MATCH_THRS:
        h.append(f'<a class=btn href=cw_detail_{thr:.2f}.html>{thr:.2f}<span class=m>CW</span></a>')
    h.append('</div></div>')

    h.append('</body></html>')
    with open(os.path.join(out_base, "index.html"), "w") as f:
        f.write("\n".join(h))
    print(f"[sweep] index -> {out_base}/index.html")

def build_flat_dino_html(vn, recs, scenes, sk, match_thr, dino_cos_thr, valid_pids, short_pids, merged_pids_set, dino_chains, out_path):
    """DINO 过滤后的平铺 HTML — grid 每行5帧，不足3scene的 pid 不画框。"""
    frame_w = sk["width"]
    frame_h = sk["height"]

    all_frames = []
    for sc in scenes:
        kf = sc.get("key_frame")
        if kf is not None:
            all_frames.append(int(kf))

    merged_frames = set()
    for pid_str, ch in dino_chains.items():
        pid = int(pid_str)
        if pid in valid_pids and ch["dino_shots"] == 1 and ch["raw_shots"] > 1:
            for fn in ch["frames"][1:]:
                merged_frames.add(fn)

    h = [
        '<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>',
        '<style>',
        '*{margin:0;padding:0;box-sizing:border-box}',
        'body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:12px}',
        'h1{font-size:15px;margin-bottom:1px}',
        '.sub{font-size:11px;color:#888;margin-bottom:8px}',
        '.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:2px;max-width:100%}',
        '.frame-wrap{position:relative;aspect-ratio:' + str(frame_w) + '/' + str(frame_h) + ';overflow:hidden;background:#000;border-radius:2px}',
        '.frame-wrap img{width:100%;height:100%;object-fit:contain;display:block}',
        '.frame-wrap.no-face img{opacity:0.08}',
        '.frame-wrap.merged img{opacity:0.20}',
        '.fn{position:absolute;top:1px;left:1px;font-size:8px;color:#fff;background:rgba(0,0,0,0.6);padding:0 3px;border-radius:2px;line-height:14px;z-index:5;pointer-events:none}',
        '.merge-badge{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#ff9800;font-size:8px;font-weight:600;background:rgba(0,0,0,0.7);padding:1px 4px;border-radius:2px;z-index:5;pointer-events:none}',
        '.face-box{position:absolute;border:2px solid;border-radius:2px;pointer-events:none;box-sizing:border-box}',
        '.face-label{position:absolute;bottom:0;left:0;font-weight:700;color:#fff;'
        'line-height:1;padding:0 2px;white-space:nowrap;font-size:9px}',
        '.legend{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;font-size:10px}',
        '.legend-item{display:flex;align-items:center;gap:3px}',
        '.legend-dot{width:8px;height:8px;border-radius:50%}',
        '</style></head><body>',
    ]
    h.append(f'<h1>{vn} — DINO 平铺 (match={match_thr}, dino={dino_cos_thr})</h1>')
    h.append(f'<p class=sub>{len(all_frames)} 帧 · {len(valid_pids)} 有效人物 · '
             f'{len(short_pids)} 不足3shot已剔除 · {len(merged_pids_set)} 链合并</p>')
    h.append('<div class=legend>')
    for pid in sorted(valid_pids):
        c = COLORS[pid % len(COLORS)]
        lbl = f'P{pid}'
        if pid in merged_pids_set:
            lbl += '(合并)'
        h.append(f'<div class=legend-item><span class=legend-dot style=background:{c}></span>{lbl}</div>')
    h.append('</div>')

    h.append('<div class=grid>')
    for fn in all_frames:
        frame_dets = [r for r in recs if r["frame"] == fn and r.get("pid") is not None and r["pid"] in valid_pids]
        has_face = len(frame_dets) > 0
        is_merged = fn in merged_frames
        cls = "frame-wrap"
        if not has_face and not is_merged:
            cls += " no-face"
        elif is_merged:
            cls += " merged"
        h.append(f'<div class={cls}>')
        h.append(f'<div class=fn>f{fn}</div>')
        h.append(f'<img src=frame_viz/frames/f{fn}.jpg onerror="this.parentElement.style.display=\'none\'" loading=lazy>')
        if is_merged:
            h.append('<div class=merge-badge>合并</div>')
        if not is_merged:
            for r in frame_dets:
                pid = r["pid"]
                bx, by, bx2, by2 = r["face_bbox"]
                pct_l = bx / frame_w * 100
                pct_t = by / frame_h * 100
                pct_w = (bx2 - bx) / frame_w * 100
                pct_h = (by2 - by) / frame_h * 100
                c = COLORS[pid % len(COLORS)]
                h.append(f'<div class=face-box style="left:{pct_l:.1f}%;top:{pct_t:.1f}%;'
                         f'width:{pct_w:.1f}%;height:{pct_h:.1f}%;border-color:{c}">'
                         f'<div class=face-label style="background:{c}">P{pid}</div></div>')
        h.append('</div>')
    h.append('</div></body></html>')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(h))
    base = os.path.dirname(os.path.dirname(out_path))
    fv_src = os.path.join(base, "frame_viz")
    fv_dst = os.path.join(os.path.dirname(out_path), "frame_viz")
    if not os.path.exists(fv_src):
        fs = frames_dir(vn)
        if os.path.isdir(fs):
            os.makedirs(fv_src, exist_ok=True)
            if not os.path.exists(os.path.join(fv_src, "frames")):
                rel = os.path.relpath(fs, fv_src)
                os.symlink(rel, os.path.join(fv_src, "frames"))
    if os.path.exists(fv_src) and not os.path.exists(fv_dst):
        os.symlink(os.path.relpath(fv_src, os.path.dirname(fv_dst)), fv_dst)
    print(f"[sweep] flat dino HTML -> {out_path}")

def load_dino_embeddings(vn):
    """加载 DINO embeddings。全局模式读全局 npz，单视频模式读 per-video npz。"""
    if _mode_b and not _vid_name:
        # 全局模式：读全局 key_frame_embeddings.npz
        dino_path = os.path.join(visual_dir(vn), "dino", "key_frame_embeddings.npz")
    else:
        name = _vid_name if _mode_b else vn
        vh = load_video_hash(name, os.path.join(visual_dir(vn), "dino"))
        dino_path = os.path.join(visual_dir(vn), "dino", f"{vh}_key_frame_embeddings.npz")
    if not os.path.isfile(dino_path):
        print(f"[dino] WARNING: {dino_path} not found")
        return None
    name = _vid_name if _mode_b else vn
    vh = load_video_hash(name, os.path.join(visual_dir(vn), "dino"))
    meta_path = os.path.join(visual_dir(vn), "dino", f"{vh}_model_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"[dino] model metadata missing: {meta_path}")
    meta = json.load(open(meta_path))
    model_id = meta.get("model_id")
    if model_id != EXPECTED_DINO_MODEL_ID:
        raise ValueError(f"[dino] unexpected model: {model_id!r}, expected {EXPECTED_DINO_MODEL_ID!r}")
    data = np.load(dino_path)
    embeddings = data["embeddings"].astype(np.float32)
    frame_ids = data["frame_ids"].astype(np.int32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(frame_ids):
        raise ValueError(f"[dino] invalid embeddings: {embeddings.shape} vs {frame_ids.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("[dino] embeddings contain non-finite values")
    if len(set(map(int, frame_ids))) != len(frame_ids):
        raise ValueError("[dino] frame_ids contain duplicates")
    emb_dict = {int(frame_id): embeddings[i] for i, frame_id in enumerate(frame_ids)}
    print(f"[dino] model={model_id} dim={embeddings.shape[1]} loaded={len(emb_dict)}")
    return emb_dict

def evaluate_chain_mean(frames, dino_emb, cos_thr):
    """用角色平均 DINO 判断是否同一画面。返回 (raw, dino_cnt, details)."""
    if len(frames) < 2:
        return len(frames), len(frames), []
    embs = {}
    for fn in frames:
        e = dino_emb.get(fn)
        if e is not None:
            embs[fn] = e
    if len(embs) < 2:
        return len(frames), len(frames), []
    all_embs = np.stack(list(embs.values()))
    mean_emb = all_embs.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
    for fn in embs:
        embs[fn] = embs[fn] / (np.linalg.norm(embs[fn]) + 1e-10)
    cos_vals = {fn: float(np.dot(e, mean_emb)) for fn, e in embs.items()}
    all_same = all(cos_vals.get(fn, 0) > cos_thr for fn in frames if fn in cos_vals)
    if all_same:
        details = [{"frame": fn, "cos_to_mean": round(cos_vals.get(fn, 0), 4), "merged": True} for fn in frames]
        details[0]["merged"] = False
        return len(frames), 1, details
    unique_shots = 1
    details = []
    last_close = True
    for i, fn in enumerate(frames):
        cv = cos_vals.get(fn)
        close = cv is not None and cv > cos_thr if cv is not None else False
        merged = False
        if i > 0 and close and last_close:
            merged = True
        elif i > 0 and not close:
            unique_shots += 1
        elif i > 0 and close and not last_close:
            unique_shots += 1
        details.append({"frame": fn, "cos_to_mean": round(cv, 4) if cv is not None else None, "merged": merged})
        last_close = close
    return len(frames), unique_shots, details

def dino_filter(recs, dino_emb, dino_cos_thr):
    """DINO 均值过滤（同画面判定）。返回 (valid_pids, short_pids, merged_pids_set, dino_chains)。"""
    pid_frames = defaultdict(set)
    for r in recs:
        pid = r.get("pid")
        if pid is not None and r.get("has_face"):
            pid_frames[pid].add(r["frame"])
    chains = {pid: sorted(frames) for pid, frames in pid_frames.items()}
    dino_chains = {}
    valid_pids = set()
    short_pids = set()
    merged_pids_set = set()
    for pid, frames in chains.items():
        raw, dino_cnt, details = evaluate_chain_mean(frames, dino_emb, dino_cos_thr)
        dino_chains[str(pid)] = {"raw_shots": raw, "dino_shots": dino_cnt, "frames": frames, "details": details}
        if dino_cnt >= 3:
            valid_pids.add(pid)
        else:
            short_pids.add(pid)
        if dino_cnt == 1 and raw > 1:
            merged_pids_set.add(pid)
    print(f"[dino-filter] {len(valid_pids)} valid, {len(short_pids)} short (<3shot), {len(merged_pids_set)} merged")
    return valid_pids, short_pids, merged_pids_set, dino_chains

def build_person_timeline(recs, scenes, gap_max=3):
    """将 pid-assigned recs 转为 per-scene person_timeline + 帧级 frame_map。
    frame_map 键 = <video_hash>_f<帧号>（哈希贯穿全局：跨视频/跨 A/B 逻辑全局唯一），
    值 = 该帧识别出的 pid 列表 —— 满足「无论 A/B，人脸识别记录 <哈希>_<帧号>: 识别结果」。"""
    pid_scenes = defaultdict(set)
    frame_map = defaultdict(list)
    for r in recs:
        if r.get("pid") is not None:
            pid_scenes[r["pid"]].add(r["scene_id"])
            frame_map[f"{r['video_id']}_f{r['frame']}"].append(r["pid"])
    frame_map = {k: sorted(set(v)) for k, v in frame_map.items()}
    timeline = []
    for pid in sorted(pid_scenes):
        scene_ids = sorted(pid_scenes[pid])
        if len(scene_ids) < 3:  # 过滤不足3scene的solo
            continue
        intervals = []
        start = scene_ids[0]
        prev = scene_ids[0]
        for s in scene_ids[1:]:
            if s - prev <= gap_max + 1:
                prev = s
            else:
                intervals.append({"start_scene": int(start), "end_scene": int(prev), "n_scenes": prev - start + 1})
                start = prev = s
        intervals.append({"start_scene": int(start), "end_scene": int(prev), "n_scenes": prev - start + 1})
        timeline.append({"person_id": int(pid), "intervals": intervals})
    return {"n_tracks": len(timeline), "timeline": timeline, "frame_map": frame_map}

def main():
    """命令行入口：解析参数 → 人脸检测（A/B/全局）→ 阈值扫描（并查集 / CW）→ 可选 DINO 过滤 → person_timeline 输出"""
    parser = argparse.ArgumentParser(description="exp_thresh_sweep")
    parser.add_argument("vn", help="video name (empty string for global mode)")
    parser.add_argument("--dino-filter", action="store_true", help="启用 DINO 均值过滤，生成 flat_dino HTML")
    parser.add_argument("--dino-cos", type=float, default=0.95, help="DINO 均值 cos 阈值 (默认0.95)")
    parser.add_argument("--project", help="project name (Mode B)")
    parser.add_argument("--video", help="video name in project (Mode B)")
    args = parser.parse_args()

    global _mode_b, _project_name, _vid_name
    global_mode = False
    if args.project and not args.video:
        # Mode B 全局模式
        global_mode = True
        _mode_b = True
        _project_name = args.project
        _vid_name = ""
    elif args.project and args.video:
        _mode_b = True
        _project_name = args.project
        _vid_name = args.video
        vn = args.video
    else:
        vn = args.vn

    if global_mode:
        recs, scenes, sk, scene_frames = detect_faces_global(args.project)
        vn = args.project
    else:
        recs, scenes, sk, scene_frames = detect_faces(vn)

    out_base = os.path.join(visual_dir(vn), "v01_visual_group", "exp_sweep")
    os.makedirs(out_base, exist_ok=True)

    t0 = time.time()

    # 预计算余弦相似度矩阵（一次，所有阈值和方法共用）
    face_indices, sim_matrix = compute_sim_matrix(recs)

    # 方法1: 并查集
    print("\n── 并查集 (Union-Find) ──")
    results_union = run_method("union", global_match, recs, face_indices, sim_matrix, scenes, sk, scene_frames, vn, out_base)

    # 方法2: Chinese Whispers
    print("\n── Chinese Whispers ──")
    results_cw = run_method("cw", chinese_whispers, recs, face_indices, sim_matrix, scenes, sk, scene_frames, vn, out_base)

    # 对比索引
    build_compare_index(vn, results_union, results_cw, out_base)

    # DINO 过滤（可选）
    if args.dino_filter:
        print(f"\n── DINO 均值过滤 (cos_thr={args.dino_cos}) ──")
        dino_emb = load_dino_embeddings(vn)
        if dino_emb is not None:
            # 用最佳阈值 0.40 重新跑一次匹配
            for r in recs:
                r.pop("pid", None)
            global_match(recs, face_indices, sim_matrix, 0.40)

            valid_pids, short_pids, merged_pids_set, dino_chains = dino_filter(recs, dino_emb, args.dino_cos)

            html_path = os.path.join(out_base, f"flat_dino_0.40.html")
            build_flat_dino_html(vn, recs, scenes, sk, 0.40, args.dino_cos,
                                 valid_pids, short_pids, merged_pids_set, dino_chains, html_path)

    # 标准 JSON 输出: person_timeline.json (canonical thr=0.40)
    for r in recs:
        r.pop("pid", None)
    global_match(recs, face_indices, sim_matrix, 0.40)
    pt = build_person_timeline(recs, scenes)
    pt["video_id"] = (_vid_name if _mode_b else vn) or "global"
    pt_dir = os.path.join(visual_dir(vn), "face_head_fusion")
    os.makedirs(pt_dir, exist_ok=True)

    # 全局模式输出无前缀文件，其他模式输出 <vid>_person_timeline.json
    if global_mode:
        pt_path = os.path.join(pt_dir, "person_timeline.json")
        pt_meta_path = os.path.join(pt_dir, "person_timeline_meta.json")
    else:
        pvn = _vid_name if _mode_b else vn
        pt_path = os.path.join(pt_dir, f"{pvn}_person_timeline.json")
        pt_meta_path = os.path.join(pt_dir, f"{pvn}_person_timeline_meta.json")

    with open(pt_path, "w") as f:
        json.dump(pt, f, ensure_ascii=False, indent=2)
    n_tracks = pt["n_tracks"]
    with open(pt_meta_path, "w") as f:
        json.dump({
            "dino_model_id": EXPECTED_DINO_MODEL_ID if args.dino_filter else None,
            "dino_filter": bool(args.dino_filter),
            "dino_cos_thr": args.dino_cos if args.dino_filter else None,
            "n_tracks": n_tracks,
        }, f, ensure_ascii=False, indent=2)
    print(f"[sweep] person_timeline -> {pt_path} ({n_tracks} tracks)")
    dur = time.time() - t0
    print(f"\n[sweep] {dur:.1f}s total -> {out_base}/")

if __name__ == "__main__":
    main()
