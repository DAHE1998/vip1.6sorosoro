#!/usr/bin/env python3
"""vlm/fuse_segments.py — 融合脚本（vlm 三件事之二：1.选帧 2.融合 3.送检）：自包含版
（2026-08-18 大名），读选帧骨架段划分把每段帧拼成一张段图，全 GPU 不二次计算。

用法（sorosoro env）: python vlm/fuse_segments.py vivant [--ep EP01]
依赖: output/<项目>/vlm/*_skeleton.json（选帧骨架）、shikomi/frames/<prefix>_f<fn>.jpg、
      visual/dino/<prefix>_skeleton.json + npz、visual/face_head_fusion/sweep_records、
      visual/body_detect/body_bbox.json、audio/dialogue/<hash>_dialogue.json
产物: output/<项目>/vlm/segments/（段图 {seg}.jpg + 验收页）、
      output/<项目>/vlm/<prefix>_fused.json（融合骨架）

说明：
  岗位定位（2026-08-18 定稿）：三脚本各自自包含，不依赖 vlm/visual_ribbon 库（已整体预删除
  进 vlm/trash/）。融合只读上游产物，不二次计算：
    ① 选帧产物 output/<项目>/vlm/*_skeleton.json（段划分，select_segments 落盘）
    ② 簇共享忽略：fuse_frames 内 DINO cos≥0.9 连通簇（GPU 并入计算，mark_clusters 已删
       2026-08-18），段内帧带簇且簇已在此前 seg 出现 → 忽略只拼剩余
    ③ DINO 向量 visual/dino/<prefix>_skeleton.json + npz —— 只构造一次（禁止每段重读重算
       = 重复计算），cos 全 torch GPU 批量
    ④ face_idx（face_head_fusion sweep_records）/ body_bbox（body_detect）/ dialogue
       （audio，台词判定）——全部只读，不重检测

  融合逻辑（build_ribbon，与 novelize.py 定稿逐字一致）：读骨架 segments → 帧号 →
  shikomi/frames → build_ribbon（RIB_CFG 定稿，零改动）。像素计算全部 torch GPU：
  importance（阶段1 GPU 批量预算一次搬显存）/ compress_cold（列重采样）/ rasterize（画布
  拼接）/ 锁列 mask / face 保护 mask / DINO cos。无任何 CPU 兜底（CUDA 不可用直接报错，
  禁 try/except 回退）。阶段2 串行（禁止多进程 fork 大缓存——2026-08-17 假死教训）。

  编号（2026-08-17 定稿）：段号 = 骨架 segments[].seg（哈希+段首帧所在 shot id）；段图 tag =
  {prefix}_novel_seg{f段首帧号}；段子图 {tag}_seg{区段id+1}.jpg；单帧段不建段图，走
  「检测→锁主体列→压背景→压完打标（左1/4）」：无人帧 3:4 竖版、有人帧压背景锁主体列。
  2026-08-18 大名定稿：vlm/ = 选帧骨架 + 融合图片文件夹 + 融合骨架 + VLM描述。
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as tio

BASE = Path(__file__).resolve().parent.parent


def resolve_out(video_dir):
    """输出根定位：OUT_ROOT 必须由 shikoto 设置，禁止单跑、禁止回退（2026-09-01 大名）。"""
    return Path(os.environ["OUT_ROOT"])

# RIB_CFG 定稿（2026-08-17，novelize.py 逐字一致；max_frames=0 为 2026-08-18 大名补订：
# 一段所有帧拼一张横向长条，DINO 视觉跳变/黑帧边界画黑线隔开，禁按 run 拆子图）
RIB_CFG = dict(max_overlap=0.0, curve="smoothstep", face_protect=True,
               seam_mode="dp", dynamic_overlap=False, compress_cold_weight=0.5,
               compress_bg_frac=0.25, body_lock=True, asr_split=True,
               compress_cold_weight_no_asr=0.4, max_frames=0)

CLUSTER_THR = 0.9   # 相似簇判定阈值（2026-08-17 大名：0.9 以上才算数，mark_clusters 原值）


# ═══════════════════════════════════ config（内联 visual_ribbon/config.py）═══════════════════════════════════
@dataclass(frozen=True)
class FaceBox:
    """人脸 bbox，原始帧像素坐标"""
    x1: float
    y1: float
    x2: float
    y2: float
    det_score: float = 0.0


@dataclass(frozen=True)
class FrameSpec:
    frame_id: int
    scene_id: int
    path: str
    ts: float
    faces: tuple = ()
    bodies: tuple = ()
    has_asr: bool = False
    black_before: bool = False


@dataclass
class RibbonConfig:
    height: int = 540
    width: int = 0
    max_overlap: float = 0.40
    curve: str = "smoothstep"
    face_protect: bool = True
    seam_mode: str = "fixed"
    dynamic_overlap: bool = True
    face_pad_factor: float = 1.0
    face_weight: float = 1.0
    max_frames: int = 5
    compress_cold_weight: float = 0.5
    compress_bg_only: bool = True
    compress_bg_frac: float = 0.25
    body_lock: bool = False
    min_body_ratio: float = 0.2
    body_subject_ratio: float = 1.4
    asr_split: bool = False
    compress_cold_weight_asr: float = 0.3
    compress_cold_weight_no_asr: float = 0.7
    min_dino_merge: float = 0.15

    def __post_init__(self):
        assert self.max_overlap in (0.0, 0.10, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5), f"max_overlap={self.max_overlap} 不在 V0.1 扫描表"
        assert self.curve in ("linear", "smoothstep")
        assert self.seam_mode in ("fixed", "dp")
        assert 0.5 <= self.face_pad_factor <= 1.5
        assert 0.2 <= self.compress_cold_weight <= 1.0


# ═══════════════════════════════════ importance GPU（内联，vlm 禁 CPU，无兜底）═══════════════════════════════════
_TORCH = None


def _torch():
    """torch CUDA 句柄（vlm 禁 CPU：CUDA 不可用直接报错，无兜底）"""
    global _TORCH
    if _TORCH is None:
        _TORCH = torch if torch.cuda.is_available() else False
    if _TORCH is False:
        raise SystemExit("❌ vlm 禁 CPU：CUDA 不可用（torch.cuda.is_available()=False）")
    return _TORCH


def _blur_gpu(x, sigma, dev, torch, F):
    """cv2 GaussianBlur(ksize=(0,0), sigmaX=sigma) 同款：reflect101 边界 + 1D 分离核"""
    ks = int(round(sigma * 3)) * 2 + 1
    ax = torch.arange(ks, dtype=torch.float32, device=dev) - (ks - 1) / 2.0
    k = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    k = k / k.sum()
    p = ks // 2
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k.view(1, 1, 1, ks))
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, ks, 1))


def _importance_gpu_batch_tensor(t, dev, torch, F):
    """GPU 张量 (B,3,H,W) float32 → 批量 kernels，返回同序 [H×W float32 GPU tensor]
    （2026-08-18 自包含：imp 全 GPU 驻留，face 合并/压缩直接吃，不来回拷 numpy）"""
    wg = torch.tensor([0.114, 0.587, 0.299], dtype=torch.float32, device=dev).view(3, 1, 1)
    gray = (t * wg).sum(1, keepdim=True).round()

    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                      dtype=torch.float32, device=dev)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                      dtype=torch.float32, device=dev)
    gp = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(gp, kx)
    gy = F.conv2d(gp, ky)

    mag = torch.sqrt(gx * gx + gy * gy)
    mag_max = mag.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
    edge = mag / mag_max
    edge_density = _blur_gpu(edge, 8.0, dev, torch, F)

    e2 = _blur_gpu(mag * mag, 4.0, dev, torch, F)
    em = _blur_gpu(mag, 4.0, dev, torch, F)
    texture = torch.clamp((e2 - em * em) / (mag_max * mag_max), 0, None)

    lin = torch.where(t / 255.0 <= 0.04045, t / 255.0 / 12.92,
                      ((t / 255.0 + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 2], lin[:, 1], lin[:, 0]
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    ff = lambda v: torch.where(v > 0.008856, v ** (1.0 / 3.0), 7.787 * v + 16.0 / 116.0)
    L = 116.0 * ff(Y / 1.0) - 16.0
    lum = torch.clamp(L * 255.0 / 100.0, 0, 255).round()
    l2 = _blur_gpu(lum * lum, 8.0, dev, torch, F)
    lm = _blur_gpu(lum, 8.0, dev, torch, F)
    color_struct = torch.clamp((l2 - lm * lm) / (255.0 * 255.0), 0, None)

    imp = 0.5 * edge_density + 0.3 * texture + 0.2 * color_struct
    imp = torch.clamp(imp / (imp.amax(dim=(1, 2, 3), keepdim=True) + 1e-6), 0, 1)
    return [imp[i, 0] for i in range(imp.shape[0])]     # (B,1,H,W) → 同序 [H×W GPU]


def compute_importance(frame) -> torch.Tensor:
    """单帧 importance（(H,W,3) float32 CUDA BGR → H×W float32 GPU）。全 GPU，零往返。"""
    torch = _torch()
    t = frame.permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W) CUDA
    return _importance_gpu_batch_tensor(t, torch.device("cuda"), torch, torch.nn.functional)[0]


# ═══════════════════════════════════ face 保护 mask（GPU，vlm 禁 CPU）═══════════════════════════════════
def face_protection_mask(shape, faces, pad_factor: float = 1.0) -> torch.Tensor:
    """(H,W)；faces: 已缩放到 ribbon 像素的 FaceBox 列表 → H×W float32 GPU mask"""
    torch = _torch()
    H, W = shape
    m = torch.zeros((H, W), dtype=torch.float32, device=torch.device("cuda"))
    for f in faces:
        w = f.x2 - f.x1
        h = f.y2 - f.y1
        if w <= 0 or h <= 0:
            continue
        pad = max(w, h) * pad_factor
        x1 = int(round(max(0, f.x1 - pad)))
        y1 = int(round(max(0, f.y1 - pad)))
        x2 = int(round(min(W, f.x2 + pad)))
        y2 = int(round(min(H, f.y2 + pad)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1.0
    return m


def combine_importance(imp: torch.Tensor, face_mask: torch.Tensor, face_weight: float = 1.0) -> torch.Tensor:
    """face_weight=1 → bbox 内重要度地板为 1.0（完全保护）。GPU elementwise。"""
    return torch.clamp(imp + face_mask * face_weight, 0, 1)


# ═══════════════════════════════════ DINO GPU 索引（一次构造，禁每段重载）═══════════════════════════════════
class DinoIndex:
    """帧号 → 归一化 DINO 向量（GPU tensor）；cos 全 torch GPU 批量。
    2026-08-18 大名否掉重复计算：嵌入只 from_out 一次（fuse main 构造传入），
    绝不允许每个 segment 重新 from_specs（每段重读整张 npz + 重算归一化）。"""

    def __init__(self, idx_of: dict, emb_n: torch.Tensor):
        self.idx_of = idx_of
        self.emb_n = emb_n

    @classmethod
    def from_out(cls, out, prefix):
        """读 visual/dino/<prefix>_skeleton.json + npz；npz 行号 ↔ key_frames 展平序。
        np.load 仅是读盘 IO，此后全部 torch GPU。"""
        visual_dir = out / "visual" / "dino"
        sk_path = visual_dir / f"{prefix}_skeleton.json"
        npz_path = visual_dir / f"{prefix}_key_frame_embeddings.npz"
        if not (sk_path.exists() and npz_path.exists()):
            raise SystemExit(f"❌ 无 DINO 向量: {visual_dir}（skeleton/npz 缺失，管线必出）")
        dsk = json.loads(sk_path.read_text(encoding="utf-8"))
        kfs = [kf for s in dsk["shots"] for kf in s["key_frames"]]
        emb = np.load(str(npz_path))["embeddings"].astype(np.float32)
        if len(kfs) != emb.shape[0]:
            raise SystemExit(f"❌ DINO key_frames {len(kfs)} != npz {emb.shape[0]} 行（上游不一致）")
        idx_of = {kf: i for i, kf in enumerate(kfs)}
        torch = _torch()
        t = torch.from_numpy(np.ascontiguousarray(emb)).to("cuda")
        emb_n = t / (t.norm(dim=1, keepdim=True) + 1e-9)
        return cls(idx_of, emb_n.contiguous())

    def splits(self, frames, thr: float):
        """specs 序帧号列表 → 切点下标集合（缝隙 k = frames[k]|frames[k+1]）：
        缝隙自身 DINO 值 < thr → 切。一次 GPU 批量算全部缝隙 cos。"""
        if len(frames) < 2:
            return set()
        ok, ia_l, ib_l = [], [], []
        for k in range(len(frames) - 1):
            ia, ib = self.idx_of.get(frames[k]), self.idx_of.get(frames[k + 1])
            if ia is None or ib is None:
                raise SystemExit(f"❌ 帧 {frames[k]}/{frames[k+1]} 不在 DINO key_frames（上游不完整）")
            ok.append(k); ia_l.append(ia); ib_l.append(ib)
        dev = self.emb_n.device
        ia_t = torch.tensor(ia_l, device=dev, dtype=torch.long)
        ib_t = torch.tensor(ib_l, device=dev, dtype=torch.long)
        cs = (self.emb_n.index_select(0, ia_t) * self.emb_n.index_select(0, ib_t)).sum(dim=1)
        return {k for k, c in zip(ok, cs.tolist()) if c < thr}


# ═══════════════════════════════════ 帧对合成（内联 compositor.py，max_overlap=0 纯硬切）═══════════════════════════════════
@dataclass
class Transition:
    from_frame: int
    to_frame: int
    overlap_px: int
    transition_width_px: int
    band_start_x: float
    band_end_x: float
    seam_x: float
    overlap_frac: float = None


class Layer:
    frame_id: int
    scene_id: int
    x: float
    width: int
    effective_width: int


def _main_subject_bodies(bodies, cfg):
    """主体/路人分群（2026-08-16 大名定稿）：bbox 按高排序，最大相邻高度比 ≥
    body_subject_ratio → 小群=路人丢弃、大群=主体保留；差距不显著 → 全留。"""
    if len(bodies) <= 2:
        return bodies
    hs = sorted((y2 - y1, i) for i, (x1, y1, x2, y2) in enumerate(bodies))
    best_gap, cut = 0.0, None
    for k in range(len(hs) - 1):
        if hs[k][0] <= 0:
            continue
        r = hs[k + 1][0] / hs[k][0]
        if r > best_gap:
            best_gap, cut = r, k
    if best_gap >= cfg.body_subject_ratio and cut is not None:
        keep = {i for _, i in hs[cut + 1:]}
        return [b for idx, b in enumerate(bodies) if idx in keep]
    return bodies


def _body_cols_force(H, W, bodies) -> torch.Tensor:
    """身体 bbox 列表 → H×W bool GPU tensor（bbox 内整列区域，供锁列压缩）"""
    torch = _torch()
    m = torch.zeros((H, W), dtype=torch.bool, device=torch.device("cuda"))
    for (x1, y1, x2, y2) in bodies:
        x1, y1 = int(round(max(0, x1))), int(round(max(0, y1)))
        x2, y2 = int(round(min(W, x2))), int(round(min(H, y2)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def _face_cols_force(H, W, faces) -> torch.Tensor:
    """缩放后 FaceBox 列表 → 本帧内 H×W bool GPU tensor（人脸区域，含 pad 放大）"""
    torch = _torch()
    m = torch.zeros((H, W), dtype=torch.bool, device=torch.device("cuda"))
    for f in faces:
        w = f.x2 - f.x1
        h = f.y2 - f.y1
        if w <= 0 or h <= 0:
            continue
        pad = max(w, h)
        x1 = int(round(max(0, f.x1 - pad)))
        y1 = int(round(max(0, f.y1 - pad)))
        x2 = int(round(min(W, f.x2 + pad)))
        y2 = int(round(min(H, f.y2 + pad)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def _importance_with_faces(frame, faces, cfg) -> torch.Tensor:
    """基础重要度（GPU）→ face 保护合并（GPU）"""
    imp = compute_importance(frame)
    if cfg.face_protect and faces:
        mask = face_protection_mask((frame.shape[0], frame.shape[1]), faces, cfg.face_pad_factor)
        imp = combine_importance(imp, mask, cfg.face_weight)
    return imp


def _resize_t(t, out_h, out_w):
    """GPU 尺寸缩放，返回 GPU tensor（压缩链内用，避免来回拷）。
    相对分辨率（2026-08-18 大名定稿）：out_w<=0 时按原宽高比推导。"""
    H, W = t.shape[0], t.shape[1]
    if out_w <= 0:
        out_w = max(1, int(round(out_h * W / H)))
    if (H, W) == (out_h, out_w):
        return t
    x = t.permute(2, 0, 1).unsqueeze(0) if t.dim() == 3 else t.view(1, 1, H, W)
    out = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode="area")
    return out[0].permute(1, 2, 0) if t.dim() == 3 else out[0, 0]


def compress_cold(frame, imp, a, lock_cols=None, bg_only=True, bg_frac=0.25, min_frac=0.0,
                  out_h=0, out_w=0):
    """冷区水平压缩 + 统一输出尺寸（全 GPU，零往返）。输入 frame: (H,W,3) float32 CUDA BGR；
    imp: (H,W) float32 GPU；lock_cols: bool GPU/numpy → 权重强制 1.0 不压缩；out_h/out_w>0 时
    输出前 GPU 缩放到目标尺寸。
    返回 (压缩帧 (H,target_w,3) CUDA, 压缩 imp CUDA, (cum, target) CUDA)。"""
    torch = _torch()
    dev = frame.device
    W = frame.shape[1]
    ft = frame                       # 已是 GPU float32 HWC（零往返）
    it = imp                         # 已是 GPU HW
    col_imp = it.mean(dim=0)
    if bg_only:
        q = torch.quantile(col_imp, 1.0 - bg_frac)
        w = torch.where(col_imp >= q, torch.tensor(1.0, device=dev), torch.tensor(a, device=dev))
    else:
        w = a + (1.0 - a) * col_imp
    if lock_cols is not None:
        lt = lock_cols.to(dev) if torch.is_tensor(lock_cols) \
            else torch.from_numpy(np.ascontiguousarray(lock_cols)).to(dev)
        w[lt] = 1.0
    cum = torch.cumsum(w, dim=0)
    target_w = max(64, int(round(float(cum[-1]))))
    if min_frac > 0:
        target_w = max(target_w, int(round(min_frac * W)))
    target = torch.linspace(0.0, float(cum[-1]), target_w, device=dev)
    cols = torch.searchsorted(cum, target, right=True).clamp(0, W - 1)
    fr = ft.index_select(1, cols)
    im = it.index_select(1, cols)
    if out_h > 0:
        fr = _resize_t(fr, out_h, out_w)
        im = _resize_t(im, out_h, out_w)
    return fr, im, (cum, target)


def _mark_frame_label(img: np.ndarray, label: str, cx: int = None) -> np.ndarray:
    """把画面序号标在图宽左 1/4 处（黑底包数字，防 VLM 当长条全景）。cx = 整条 canvas 绝对 x
    （多帧段延后到输出端一次画，2026-08-21 大名：全 GPU 数据流，cv2 只在输出端回 CPU）；
    None = 本图宽左 1/4。字号/线宽/顶部偏移/黑底边距 = 相对高度比例（2026-08-18 定稿）。
    cv2 前强制连续可写 uint8。"""
    fr = np.clip(img, 0, 255).astype(np.uint8)       # 先转 uint8（新数组，可写连续）
    if not fr.flags.writeable or not fr.flags.c_contiguous:
        fr = np.ascontiguousarray(fr).copy()
    h, w = fr.shape[:2]
    fscale = max(1.4, min(4.5, h / 280.0))          # 字号相对高度（2026-08-18 大名：标嫌小调大）
    thick = max(2, int(round(h / 110)))             # 线宽相对高度
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
    px = max((w // 4 if cx is None else cx) - tw // 2, 4)   # 宽左 1/4 居中（大名定稿）
    y_top = max(6, h // 80)                          # 顶部偏移相对高度
    pad = max(3, h // 140)                           # 黑底边距相对高度
    cv2.rectangle(fr, (px - pad, max(0, y_top - pad)),
                  (px + tw + pad, y_top + th + pad), (0, 0, 0), -1)
    cv2.putText(fr, label, (px, y_top + th), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                (0, 215, 255), thick, cv2.LINE_AA)
    return fr


def load_skeleton(video_dir, ep=None):
    """读选帧骨架 output/<项目>/vlm/*_skeleton.json（select 落盘；2026-08-18 大名定稿：
    output/<项目名字>/vlm/<选帧骨架>.json，不再读 visual/dedup）。--ep 过滤 + 唯一性选集；
    prefix = 骨架 video_hash（帧前缀内容指纹）"""
    out = resolve_out(video_dir)
    skels = sorted((out / "vlm").glob("*_skeleton.json"))   # Path.glob：锚点路径字面，方括号文件夹名不当通配符
    picks = []
    for p in skels:
        sk = json.load(open(p))
        name = Path(sk["video"]).stem                       # 去 .mp4
        if ep and (sk.get("video_hash") or "").lower() != ep.lower():
            continue
        picks.append((name, sk))
    if not picks:
        raise SystemExit(f"❌ 无选帧骨架匹配 {ep or '全部'}——先跑 vlm/select_segments.py（选帧）")
    if len(picks) != 1:
        raise SystemExit(f"❌ 选帧骨架不唯一: {[n for n, _ in picks]}（加 --ep 过滤）")
    name, sk = picks[0]
    prefix = sk.get("video_hash")
    if not prefix:
        raise SystemExit(f"❌ 选帧骨架 {name} 缺 video_hash")
    print(f"✔ 选中: {name}（video_hash={prefix}，segments={len(sk.get('segments', []))}）")
    return name, sk, prefix, out


def load_face_index(sk, out):
    """sweep_records_global.json（无则单视频版）→ {frame: FaceBox|None}。
    video_id 匹配：video_hash（现役贯穿）"""
    path = out / "visual/face_head_fusion/sweep_records_global.json"
    if not path.exists():
        path = out / "visual/face_head_fusion/sweep_records.json"
    if not path.exists():
        raise SystemExit(f"❌ 无人脸索引 sweep_records: {out / 'visual/face_head_fusion'}（管线必出）")
    want = sk.get("video_hash")
    recs = json.load(open(path))["recs"]
    idx = {}
    for r in recs:
        rid = str(r.get("video_id"))
        if rid != want:
            continue
        if r.get("has_face") and r.get("face_bbox"):
            b = r["face_bbox"]
            idx[r["frame"]] = FaceBox(b[0], b[1], b[2], b[3], r["det_score"])
        else:
            idx.setdefault(r["frame"], None)
    nf = sum(1 for v in idx.values() if v)
    print(f"✔ 人脸索引: {len(idx)} 帧 / {nf} 有脸")
    return idx


def load_dialogue(out, sk):
    """audio/dialogue 产物（哈希贯穿，与选帧 load_dialogue 同源）→ {scene_id: [speaker|text, ...]}
    2026-08-19 大名：ASR 骨架 = scene 级 asr，has_asr 按帧所在 scene 查，不再时间相交"""
    vid = sk.get("video_hash")
    p = out / "audio" / "dialogue" / f"{vid}_dialogue.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 dialogue: {p}（上游 ASR 未跑，管线必出）")
    dj = json.load(open(p))
    _check_hash(dj, "dialogue", sk)
    dls = {sc["scene_id"]: sc.get("asr") or [] for sc in dj.get("scenes", [])}
    n = sum(len(v) for v in dls.values())
    print(f"✔ 台词段: {n} 条（dialogue json scene 级 asr，video_hash 校验通过）")
    return dls


def _check_hash(d, label, sk):
    """video_hash 交叉校验（施工表哈希贯穿：骨架头部对齐 dino 格式，读时必须校验）"""
    if d["video_hash"] != sk["video_hash"]:
        raise SystemExit(f"❌ {label} video_hash={d['video_hash']} != 骨架 "
                         f"{sk['video_hash']}（跨模块哈希不一致，禁止融合）")


def _body(out, sk):
    """YOLO 身体 bbox（body_detect 全局产物 body_bbox.json，必出；键 = gc 同款帧标记
    <hash>_f<fn>，2026-08-21 大名：身体骨架每项目全局一份，下游拿帧名直接查）"""
    vh = sk["video_hash"]
    p = out / "visual/body_detect" / "body_bbox.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 body_bbox {vh}: {out / 'visual/body_detect'}（上游 body_detect 未跑）")
    d = json.load(open(p, encoding="utf-8"))
    # 2026-09-02 修：body_bbox 可合法为空（无人物视频检测不到任何 frame key），
    # single_img 空 bodies 照常出图。仅当 body_detect 根本没覆盖本视频才报错（按 videos 列表判断）
    covered = any(isinstance(v, dict) and (v.get("video_hash") == vh or v.get("video_id") == vh)
                  for v in d.get("videos", []))
    if not covered:
        raise SystemExit(f"❌ body_bbox 未覆盖 {vh}（上游 body_detect 未含本视频，或跑错项目）")
    return d


# ═══════════════════════════════════ 融合主流程 ═══════════════════════════════════
def _decode_gpu_batch(paths):
    """路径列表 → (N,H,W,3) uint8 CUDA BGR tensor。GPU 解码（NVJpeg）+ GPU 驻留，
    零 CPU 往返（2026-08-21 大名：帧留 GPU，直到输出端才回 CPU）。"""
    bufs = [torch.from_numpy(np.fromfile(p, dtype=np.uint8)) for p in paths]
    ts = tio.decode_jpeg(bufs, device="cuda")           # list of (3,H,W) uint8 CUDA RGB
    return torch.stack(ts).permute(0, 2, 3, 1).flip(-1)   # (N,H,W,3) uint8 CUDA BGR


def _read_frame_bgr(path):
    """单帧 GPU 解码（NVJpeg）→ (H,W,3) float32 CUDA BGR（与 load_frames 同解码器）。"""
    return _decode_gpu_batch([path])[0].to(torch.float32)


def frame_no(fn):
    """'9d5dab_f40409_c008key' → 40409（帧名带簇后缀，取 _f 后帧号）"""
    return int(fn.split("_f", 1)[1].split("_", 1)[0])


def _is_key_frame(fn):
    """帧名后缀含 key（_key / _c{id}key，2026-08-21 大名：与 submit _is_key 同判定）；
    簇成员 _c{id} 无 key 后缀。纯簇成员段（段内无 key 帧）不出图，簇复用省送检。"""
    return fn.rsplit("_", 1)[-1].endswith("key")


def seg_tag(prefix, seg):
    """段图 tag = 段号（大名 2026-08-18：定了 seg 就用 seg 号命名，不再 novel_s{shot}/
    novel_seg{帧号} 混用）。seg = 9d5dab_s23 → jpg/json = 9d5dab_s23.jpg/.json"""
    return seg["seg"]


def compute_clusters(segments, dino, thr=CLUSTER_THR):
    """全局相似簇：已前移至 select_segments（簇标记落帧名，2026-08-18 大名）"""
    raise SystemExit("❌ 簇计算已前移 select_segments，fuse 只读帧名后缀（_key/_c{id}key）")


def single_img(seg, fn, face_idx, body_bbox, prefix, out_dir, out_name=None):
    """逐帧出图（2026-08-17 定稿单帧逻辑；2026-08-30 大名：多帧段逐帧调用）：读帧→检测→
    锁主体列→压背景→统一缩 cfg.height→压缩完打标（左1/4）。无人帧压 3:4 竖版；有人帧压背景锁主体列。
    out_name：输出文件名（None 时 = {seg['seg']}.jpg；多帧段传 {seg}{n}.jpg 逐帧）。"""
    frame = _read_frame_bgr(out_dir.parent.parent / "shikomi/frames" / f"{prefix}_f{fn}.jpg")
    faces = (face_idx[fn],) if face_idx.get(fn) is not None else ()
    cfg = RibbonConfig(**RIB_CFG)
    H0, W0 = frame.shape[:2]
    bodies = tuple(tuple(b[:4]) for b in body_bbox.get(f"{prefix}_f{fn}", ()))
    bodies = _main_subject_bodies(
        [b for b in bodies if (b[3] - b[1]) >= H0 * cfg.min_body_ratio], cfg)
    has_subject = bool(faces) or bool(bodies)
    if not has_subject:
        imp = compute_importance(frame)
        min_frac = 0.75 * frame.shape[0] / frame.shape[1]
        fr, _, _ = compress_cold(frame, imp, 0.2, None, True, 0.25, min_frac,
                                 out_h=cfg.height)
        img = out_name or f"{seg['seg']}.jpg"
    else:
        imp = _importance_with_faces(frame, faces, cfg)
        lock = None
        if faces:
            lock = _face_cols_force(frame.shape[0], frame.shape[1], faces).any(axis=0)
        if bodies:
            b = _body_cols_force(frame.shape[0], frame.shape[1], bodies).any(axis=0)
            lock = b if lock is None else torch.logical_or(lock, b)
        cc = (cfg.compress_cold_weight_asr if seg["has_asr"]
              else cfg.compress_cold_weight_no_asr) if cfg.asr_split else cfg.compress_cold_weight
        fr, _, _ = compress_cold(frame, imp, cc, lock, cfg.compress_bg_only,
                                 cfg.compress_bg_frac, 0.0, out_h=cfg.height)
        img = out_name or f"{seg['seg']}.jpg"
    fr = _mark_frame_label(fr.clamp(0, 255).to(torch.uint8).cpu().numpy(), "1")
    p = out_dir / img
    if not p.exists():
        cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img, "vlm/segments", has_subject


def _process_seg(job, ctx):
    """阶段2 串行 worker：逐帧出图。job = (kind, seg, fns, tag, specs)。
    specs 在 main 一次算好传入（不二次计算）。共享数据从 ctx 读（单进程串行）。
    2026-08-30 大名：删拼图，多帧段逐帧出图 {seg}{n}.jpg（n = 帧号在 fns 的序号）。"""
    kind, seg, fns, tag, specs = job
    # 无 try/except 兜底（2026-08-18 大名：出错直接抛，脚本即停，禁吞错误继续跑）
    if len(fns) < 2:
        img, sub, has_subject = single_img(seg, fns[0], ctx["face_idx"], ctx["body_bbox"],
                                           ctx["prefix"], ctx["out_dir"])
        return "single", (seg["seg"], img, sub, has_subject)
    # 多帧段：逐帧 single_img，命名 {seg}{n}.jpg（原在帧内序号）
    imgs = []
    for n, fn in enumerate(fns):
        out_name = f"{seg['seg']}{n}.jpg"
        img, sub, has_subject = single_img(seg, fn, ctx["face_idx"], ctx["body_bbox"],
                                           ctx["prefix"], ctx["out_dir"], out_name=out_name)
        imgs.append(img)
    return "multi", (seg["seg"], len(fns), "", imgs)


def main():
    """主流程：读选帧骨架 → 划出图清单（纯簇成员段跳过）→ 逐帧出图 → 落验收页。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--ep", default=None)
    args = ap.parse_args()

    name, sk, prefix, out = load_skeleton(args.video_dir, args.ep)
    fps = sk["fps"]

    segments = sk["segments"]                         # 选帧骨架 segments 键（段来源）
    face_idx = load_face_index(sk, out)
    body_bbox = _body(out, sk)
    out_dir = out / "vlm" / "segments"                # 融合图片文件夹（大名 2026-08-18 定稿）
    out_dir.mkdir(parents=True, exist_ok=True)

    # 跑前清空本视频旧段图/验收页（2026-08-18 大名：每次重跑都清空缓存，防新旧混杂）
    n_old = sum(1 for p in out_dir.glob(f"{prefix}_s*.jpg")) \
        + sum(1 for p in out_dir.glob("验收_帧融合_*.html"))
    for p in list(out_dir.glob(f"{prefix}_s*.jpg")) \
            + list(out_dir.glob("验收_帧融合_*.html")):
        p.unlink()
    if n_old:
        print(f"🧹 清空 {n_old} 个旧段图/验收页（{out_dir} 下）", flush=True)

    print(f"[fuse] {name}: {len(segments)} 段（选帧产物），fps={fps}", flush=True)

    # ── 出图清单：select 已选好每段帧（帧名带簇标记，2026-08-21 大名），fuse 逐帧出图 ──
    jobs = []
    n_skip_cluster = 0              # 纯簇成员段（簇复用省送检，不出图）
    for seg in segments:
        # 纯簇成员段：段内无 key 帧（全 _c{id} 簇成员，簇 rep 已在上游标好，submit 复用
        # desc 不送检）→ 不出图（2026-08-21 大名：跳过不拼但记录好）
        if seg["frames"] and not any(_is_key_frame(f) for f in seg["frames"]):
            n_skip_cluster += 1
            continue
        # 只出 key 帧图（2026-08-21 大名：簇共享帧 _c{id} 不送检，共享 rep desc）。
        keys = [f for f in seg["frames"] if _is_key_frame(f)]
        fns = [frame_no(f) for f in keys]
        if len(fns) < 2:
            jobs.append(("single", seg, fns, None, None))
            continue
        tag = seg_tag(prefix, seg)
        jobs.append(("multi", seg, fns, tag, None))
    print(f"✔ 出图清单: 段 {len(segments)} → 出图 {len(jobs)}（单帧段 "
          f"{sum(1 for j in jobs if j[0] == 'single')}，纯簇成员跳过 {n_skip_cluster}）",
          flush=True)

    # ══ 逐段融合（串行，GPU 像素计算，无多进程无 fork）══
    ctx = dict(face_idx=face_idx, body_bbox=body_bbox, prefix=prefix, out_dir=out_dir)
    del face_idx, body_bbox
    n_built = n_skip = 0
    rows, single_rows = [], []
    t_all = time.time()
    for i, job in enumerate(jobs):      # 无 errs 吞错：_process_seg 异常直接抛，脚本即停
        kind, data = _process_seg(job, ctx)
        if kind == "multi":
            rows.append(data)
            n_built += 1
        else:
            single_rows.append(data)
            n_skip += 1
        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            print(f"  [{i + 1}/{len(jobs)}] 建 {n_built} 张，跳过 {n_skip}，"
                  f"{time.time() - t_all:.0f}s", flush=True)
    del jobs

    # ── 融合验收页 ──
    ver = int(time.time())   # 图片缓存版本：每次跑刷新，浏览器不缓存旧图（2026-08-18 大名：黑线粗细调不动 = ?v={ver} 缓存）
    src = f"/{args.video_dir}/vlm/segments/"
    rows_html = "".join(
        '<div class=row>'
        + "".join(f'<img loading=lazy src={src}{s}.jpg?v={ver}>' for s in subs)
        + f'<div class=cap>seg{seg} — {nf}帧 ov={ov}</div></div>'
        for seg, nf, ov, subs in rows)
    single_html = "".join(
        f'<div class=row><img loading=lazy src=/{args.video_dir}/{sub}/{img}?v={ver}>'
        f'<div class=cap>seg{s} — 单帧（{"无人帧压3:4竖版" if not has_subject else "有人帧压背景"}）</div></div>'
        for s, img, sub, has_subject in single_rows)
    html = ("<!doctype html><html><head><meta charset=utf-8>"
            f"<title>帧融合全量验收 v3 自包含 — {name}</title>"
            "<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:20px}"
            "h2{color:#ffd700}.sub{color:#888;font-size:13px;margin-bottom:16px}"
            ".row{margin-bottom:14px;border-bottom:1px solid #2a2a2a;padding-bottom:8px}"
            "img{max-width:100%;border-radius:6px}.cap{color:#888;font-size:12px;margin-top:4px}</style>"
            "</head><body><h2>帧融合全量验收 v3 自包含 — " + name + "</h2>"
            f"<div class=sub>{n_built} 张段图（max_overlap=0 纯硬切；GPU 融合；DINO 一次加载）；"
            f"{len(single_rows)} 段单帧</div>"
            + rows_html + f"<h2>单帧段（{len(single_rows)}）</h2>" + single_html
            + "</body></html>")
    (out_dir / "验收_帧融合_v3_自包含_全量.html").write_text(html, encoding="utf-8")
    print(f"✔ 出图 {n_built} 张（{len(single_rows)} 单帧段）→ {out_dir}", flush=True)

    # 骨架头：只落透传区 + 段图清单（2026-08-30 大名：fused.json 无人消费，删除）
    print(f"✔ 验收页: {out_dir / '验收_帧融合_v3_自包含_全量.html'}（总耗时 {time.time() - t_all:.0f}s）",
          flush=True)


if __name__ == "__main__":
    main()
