#!/usr/bin/env python3
"""vlm/select_segments.py — 选帧脚本（vlm 三件事之一：1.选帧 2.融合 3.送检）：按簇共享方案
（2026-08-21 定稿）直接读上游产物筛选 scene 融合成段，不二次计算。

用法（amaterasu env）: python vlm/select_segments.py vivant [--ep EP01]
依赖: visual/global_cos/gc_skeleton.json、audio/dialogue/<hash>_dialogue.json、
      visual/face_detect/<hash>_face_map.json、visual/body_detect/body_bbox.json
产物: output/<项目>/vlm/<hash>_skeleton.json（选帧骨架）

说明：
  岗位定位（2026-08-21 大名定稿）：整合上游信息筛选，直接读上游输出，不二次计算。
  选帧 = 读 gc 骨架（visual/global_cos/gc_skeleton.json = 帧处理目录 + 簇标记 + 完整 N×N
  全局 cos 矩阵）→ 每 scene 提取代表帧（帧名后缀 gc 已标：单帧 / 簇 key _c{id}key / 簇成员
  _c{id}）+ scene 全局 cos 相似度（cos 一律从 gc 骨架矩阵读，禁读 visual/dino/*.npz 二次计算
  ——2026-08-21 大名暴怒踩坑）→ 按有无台词是否连续选出要融合的 scene → 连续的无台词/有台词
  scene 融合成批，单批 ≤5 scene → 批内相邻 Scene cos ≥ 0.90 跳过后面 scene（与前一个保留的
  比，停止 = cos < 0.90 或帧类型不连续）→ 无台词段 ≤5 帧即可、有台词段 ≤5 帧提炼 ≤3 帧
  （人脸→身体→质量）。台词读 ASR 骨架（audio/dialogue/<hash>_dialogue.json，scene_id = 列表
  序 0-based，与 gc 骨架 scenes 列表序对齐——2026-08-19 大名：台词按 scene 聚合，has_asr
  直接查）。

  铁律（2026-08-18 大名）：下游只读上游，不做计算；禁止任何兜底/回退逻辑。缺文件/字段不一致/
  帧不在 cos 矩阵 = 上游没跑完或异常，直接报错，绝不静默降级。

  数据源（全部直接读上游产物；哈希贯穿全场）：
    ① gc 骨架   visual/global_cos/gc_skeleton.json（每项目全局一份）：scenes 按视频分组、时间
                序，每 scene = {scene_id(SceneNN 1-based 显示), video_hash, shot_range, black,
                frames: 帧标记}；顶层 cos_matrix = 完整 N×N 全局 cos + cos_frames =
                [[video_hash, 帧号], ...]（行/列一一对应，非黑 scene 代表帧）。选图结构与 cos
                的唯一来源（2026-08-21 大名定稿 A）。
    ② ASR 台词段 audio/dialogue/<hash>_dialogue.json → scenes[].asr（scene 级，scene_id = 列表
                序 0-based 对齐，has_asr 直接查）
    ③ 有脸 map   visual/face_detect/<hash>_face_map.json → {fn: 有脸}（有台词段提炼优先级 人脸；
                缺帧 = 无脸，2026-08-21 大名 L20）
    ④ 身体骨架   visual/body_detect/body_bbox.json（每项目全局一份，键对齐 gc 帧名 <hash>_f<fn>）
                → {fn: 有身体}（有台词段提炼优先级 身体）

  筛选逻辑（段 = 融合批，2026-08-21 大名定稿 L20）：
    1. scene 有/无台词：直接查 ASR 骨架 scene 级 asr（scene_id = gc scenes 列表序 0-based）
    2. 黑帧 scene（black: true）跳过且视为中断——隔断两侧即使同类也分属不同批
    3. 批划分：只有连续的无台词片段/连续的有台词片段才能同一批（跨类必分）；批 ≤5 scene
    4. 批内相邻 Scene 全局 cos ≥ 0.90 → 后一个 Scene 跳过（后一个补进的仍与前一个保留的 ≥0.90
       → 继续跳过；停止 = cos < 0.90 或帧类型不连续）；scene 已上游去重
    5. 每 seg 帧 = 保留 scene 代表帧（帧名透传 gc 标记）：无台词 seg ≤5 帧即可；有台词 seg ≤5
       帧 → 提炼 ≤3 帧，保留优先级 有人脸 → 有身体 → 质量好（分数 = 段内平均全局 cos + 人脸
       +1 + 身体 +1，去相似补位）
    6. 每 seg 独立 shot_range：start = seg 首 scene 的 shot_range.start、end = seg 尾 scene 的
       shot_range.end（照 gc 骨架 scene shot_range 透传）

  输出契约（3 脚本 3 骨架各管各的，大名 2026-08-18）：select→选帧骨架，落
  output/<项目>/vlm/<hash>_skeleton.json，只带消费方字段：
    - 顶层：video / video_hash / prefix / fps / scenes（fuse 消费：{id, frame} 源帧→scene id）/
      segments
    - 段字段：seg_id / seg / scenes / shot_range / frames（帧名透传 gc 标记）/ has_asr / t0 / t1
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# 阈值（现役定稿，锁死不动）
MAX_SCENES = 5        # 单批 scene 数上限（大名 2026-08-18）
COS_SKIP = 0.90       # 批内相邻 Scene 全局 cos ≥ 此值 → 后一个跳过（2026-08-21 大名 L20）
KEEP_ASR = 3          # 有台词段提炼后帧数上限（人脸→身体→质量，2026-08-21 大名 L20）


# ─────────────────────────────── 骨架 IO（video_hash 内容指纹命名贯穿全场）
def load_gc(out, ep=None):
    """读 gc 骨架（visual/global_cos/gc_skeleton.json，每项目全局一份；2026-08-21 大名：
    选图结构与 cos 的唯一来源，禁读 dedup / DINO npz 二次计算）。--ep 选该集（video_hash），
    A 单视频可省；prefix = 视频 video_hash。scene_id 重赋列表序 0-based（2026-08-19 大名：
    与 dialogue/chapter 对齐）"""
    p = out / "visual" / "global_cos" / "gc_skeleton.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 gc 骨架: {p}（先跑 visual/global_cos.py）")
    gc = json.load(open(p))
    vids = [v for v in gc["videos"] if v.get("video_hash")]
    if ep:
        vids = [v for v in vids if v["video_hash"].lower() == ep.lower()]
    if not vids:
        raise SystemExit(f"❌ gc 骨架无 {ep or '视频'}（先跑 visual/global_cos.py）")
    if len(vids) != 1:
        raise SystemExit(f"❌ gc 骨架视频不唯一: {[v['video_hash'] for v in vids]}（加 --ep 过滤）")
    vh = vids[0]["video_hash"]
    name = Path(vids[0].get("video", "")).stem or vh
    scenes = [s for s in gc["scenes"] if s.get("video_hash") == vh]
    for idx, s in enumerate(scenes):
        s["scene_id"] = idx
    row_of = {tuple(v): i for i, v in enumerate(gc.get("cos_frames", []))}
    cm = gc.get("cos_matrix")
    n = len(cm[0]) if cm else 0
    print(f"✔ 选中: {name}（video_hash={vh}，{len(scenes)} scene，cos 矩阵 "
          f"{len(cm) if cm else 0}×{n}）")
    return name, gc, vh, scenes, row_of, cm, out


def load_dialogue(out, vh):
    """ASR 骨架 → {scene_id: [speaker|text, ...]}（scene_id = 列表序 0-based，2026-08-19
    大名：台词按 scene 聚合，不做时间窗相交）"""
    p = out / "audio" / "dialogue" / f"{vh}_dialogue.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 dialogue: {p}（上游 ASR 未跑，管线必出）")
    dj = json.load(open(p))
    dls = {sc["scene_id"]: sc.get("asr") or [] for sc in dj.get("scenes", [])}
    n = sum(len(v) for v in dls.values())
    print(f"✔ 台词段: {n} 条（{len(dls)} scene 有台词）")
    return dls


def load_face(out, vh):
    """有脸 map（visual/face_detect/<hash>_face_map.json，上游必出）→ {fn: 有脸}。
    缺文件/哈希不一致 = 上游异常报错停（铁律）；缺某帧 = 无脸（提炼不加分）。"""
    p = out / "visual" / "face_detect" / f"{vh}_face_map.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 face map: {p}（上游 face_detect 未跑，管线必出）")
    d = json.load(open(p))
    if d.get("video_hash") != vh:
        raise SystemExit(f"❌ face map video_hash={d.get('video_hash')} != 骨架 {vh}")
    return {int(k): bool(v) for k, v in d.get("face_map", {}).items()}


def load_body(out, vh):
    """身体骨架（visual/body_detect/body_bbox.json 每项目全局一份，键对齐 gc 帧名
    <hash>_f<fn>）→ {fn: 有身体}。缺文件 = 上游 body_detect 未跑，报错停（铁律）；
    缺某帧 = 无身体（提炼不加分）"""
    p = out / "visual" / "body_detect" / "body_bbox.json"
    if not p.exists():
        raise SystemExit(f"❌ 无 body_bbox: {p}（上游 body_detect 未跑，管线必出）")
    d = json.load(open(p))
    return {int(k.split("_f", 1)[1].split("_", 1)[0]): True
            for k in d if k.startswith(f"{vh}_f") and d[k]}


# ─────────────────────────────── gc 帧标记 → 帧名/帧号（透传 gc，不重算）
def _frame_no(mark):
    """gc 帧标记 → 代表帧号（<hash>_f<fn>[后缀][：...]，全角冒号后是簇 rep 引用）"""
    head = mark.split("：", 1)[0]
    return int(head.split("_f", 1)[1].split("_", 1)[0])


def _frame_name(mark):
    """gc 帧标记 → 帧名（2026-08-21 大名 L9 帧名后缀约定，透传 gc 不重算）：单帧补 _key；
    簇 key _c{id}key 原样；簇成员 _c{id}：去冒号引用只留簇成员名（簇 key 帧由 submit 靠
    骨架内 _c{id}key 帧解析 rep）"""
    head = mark.split("：", 1)[0]
    if "：" not in mark:
        tail = head.rsplit("_", 1)[-1]
        if tail.startswith("c"):
            return head
        return head + "_key"
    return head


# ─────────────────────────────── 全局 cos（读 gc 骨架矩阵，禁二次计算）
def _cos(cm, row_of, a, b):
    """两 scene 代表帧全局 cos（读 gc 骨架完整 N×N 矩阵，2026-08-21 定稿 A，禁读 DINO npz
    现算——踩坑）。帧不在矩阵 = 上游异常报错停（铁律）"""
    ra, rb = row_of.get(a), row_of.get(b)
    if ra is None or rb is None:
        raise SystemExit(f"❌ 代表帧 {a}/{b} 不在 gc cos 矩阵（上游 global_cos 未含）")
    return float(cm[ra][rb])


# ─────────────────────────────── 批划分（大名 2026-08-18：同类连续才同一批，黑帧=中断）
def scene_has_asr(sc, dlg):
    """scene 有/无台词：直接查 ASR 骨架 scene 级 asr（scene_id = 列表序 0-based；2026-08-19
    大名：台词按 scene 聚合，不自己算时间窗相交）"""
    return bool(dlg.get(sc["scene_id"]))


def build_batches(scenes, dlg):
    """批划分（段 = 融合批）：只有连续的无台词片段/连续的有台词片段才能同一批
    （跨类必分）；黑帧 scene 跳过且视为中断——隔断两侧即使同类也分属不同批；
    批 ≤5 个 scene（超 5 开新批）。返回 [(has_asr, [scene, ...]), ...]"""
    batches = []
    cur = None
    for sc in scenes:
        if sc.get("black"):               # 黑帧 = 中断（两侧即使同类也分属不同批）
            cur = None
            continue
        a = scene_has_asr(sc, dlg)
        if cur is None or cur[0] != a or len(cur[1]) >= MAX_SCENES:
            batches.append([a, [sc]])
            cur = batches[-1]
        else:
            cur[1].append(sc)
    return batches


def _is_cluster_key(mark):
    """gc 帧标记是否为簇 key 帧（_cXXXkey，簇共享 desc 的源头）。簇成员标记带
    `：<key帧>` 引用，先取冒号前段再判后缀，避免把成员误判成 key"""
    head = mark.split("：", 1)[0]
    tail = head.rsplit("_", 1)[-1]
    return tail.startswith("c") and tail.endswith("key")


def cos_skip(bsc, cosf, thr=COS_SKIP):
    """批内 Scene 按相邻全局 cos 去重（2026-08-21 L20：相邻 Scene cos ≥ 0.90 → 后一个跳过，
    补进的仍与前一个保留的比；停止 = cos < 0.90 或帧类型不连续）。簇 key 帧必保留（2026-08-21
    大名原则：key 帧豁免相似跳过并作新的比较参考——否则簇成员无 desc 可复用）。cos 读 gc
    骨架矩阵；scene 已上游去重。返回保留的 [scene, ...]"""
    if not bsc:
        return []
    kept, last = [bsc[0]], 0
    for i in range(1, len(bsc)):
        if _is_cluster_key(bsc[i]["frames"][0]):
            kept.append(bsc[i])             # 簇 key 帧必保留
            last = i
            continue
        if cosf(bsc[last], bsc[i]) >= thr:
            continue                        # 与保留的相似 → 跳过
        kept.append(bsc[i])
        last = i
    return kept


def refine_frames(bsc, cosf, face_of, body_of, keep=KEEP_ASR, thr=COS_SKIP):
    """有台词批（≤5 scene）→ 提炼 ≤keep 帧（2026-08-21 L20：优先级 有人脸 → 有身体 → 质量好；
    分数 = 段内平均全局 cos + 人脸 +1 + 身体 +1，与已选 cos ≥ 0.90 相似跳过——去相似补位）。
    簇 key 帧必保留先锁，剩余名额照原逻辑补足（否则簇成员无 desc 可复用）。返回时间序
    [scene, ...]"""
    n = len(bsc)
    forced = [j for j in range(n) if _is_cluster_key(bsc[j]["frames"][0])]
    if len(bsc) <= keep:
        return bsc
    if len(forced) >= keep:
        return [bsc[j] for j in sorted(set(forced))]
    avg = {}
    for j in range(n):
        s = sum(cosf(bsc[j], bsc[i]) for i in range(n) if i != j)
        avg[j] = s / (n - 1) if n > 1 else 1.0

    def score(j):
        fn = _frame_no(bsc[j]["frames"][0])
        return avg[j] + (1.0 if face_of.get(fn) else 0.0) \
                      + (1.0 if body_of.get(fn) else 0.0)

    picked = sorted(set(forced))           # 簇 key 帧必保留，先锁
    rest = [j for j in range(n) if j not in forced]
    while len(picked) < keep and rest:
        order = sorted(rest, key=score, reverse=True)
        for t, j in enumerate(order):
            if any(cosf(bsc[j], bsc[p]) >= thr for p in picked):
                continue
            picked.append(j)
            rest.remove(j)
            break
        else:
            picked.append(order[0])        # 与已选全相似 → 补位取最高分
            rest.remove(order[0])
    return [bsc[j] for j in sorted(picked)]


# ─────────────────────────────────────────────── main
def main():
    """主流程：读 gc/ASR/人脸/身体上游 → 划批 → 去重提炼 → 落选帧骨架"""
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--ep", default=None)
    args = ap.parse_args()

    out = BASE / "output" / args.video_dir
    name, gc, vh, scenes, row_of, cm, out = load_gc(out, args.ep)
    vmeta = next(v for v in gc["videos"] if v["video_hash"] == vh)
    fps = vmeta["fps"]

    dlg = load_dialogue(out, vh)
    face_of = load_face(out, vh)
    body_of = load_body(out, vh)

    def cosf(a, b):                         # 两 scene 全局 cos（gc 骨架矩阵）
        return _cos(cm, row_of, (vh, _frame_no(a["frames"][0])),
                    (vh, _frame_no(b["frames"][0])))

    batches = build_batches(scenes, dlg)
    n_asr = sum(1 for b in batches if b[0])
    print(f"[select] {name}: scenes={len(scenes)} fps={fps} → {len(batches)} 批"
          f"（有台词 {n_asr} / 无台词 {len(batches) - n_asr}）")

    # ── 选帧（2026-08-21 大名 L20：批内相邻 Scene cos ≥ 0.90 跳过；无台词 ≤5 帧、
    # 有台词提炼 ≤3 帧（人脸→身体→质量）；段 = 融合批）──
    results = []
    seg_ord = 0   # 独立段序号（大名 2026-08-18：seg_id 不绑定 shot id）
    for has_asr, bsc in batches:
        kept = cos_skip(bsc, cosf)
        if has_asr:
            kept = refine_frames(kept, cosf, face_of, body_of)
        if not kept:
            raise SystemExit(f"❌ 批内 scene 全被跳过: {[sc.get('scene_id') for sc in bsc]}"
                             "（上游异常）")
        seg_id, seg_ord = seg_ord, seg_ord + 1
        results.append({
            "seg_id": seg_id,
            "seg": f"{vh}_s{seg_id}",       # 段名 = 哈希 + 独立段序号（不绑定 shot id）
            "scenes": [sc["scene_id"] for sc in kept],
            "shot_range": {"start": kept[0]["shot_range"]["start"],
                           "end": kept[-1]["shot_range"]["end"]},
            "frames": [_frame_name(sc["frames"][0]) for sc in kept],  # 帧名透传 gc 标记
            "has_asr": has_asr,
            "t0": round(_frame_no(kept[0]["frames"][0]) / fps, 1),
            "t1": round(_frame_no(kept[-1]["frames"][0]) / fps, 1),
        })
    print(f"[select] 段: {len(results)}", flush=True)

    # ── 落盘：成品骨架 = 消费方字段 + segments，直落 output/<项目>/vlm/
    # （大名 2026-08-18 定稿：output/<项目名字>/vlm/<选帧骨架>.json；主链中间产物不落，
    # 多的都是垃圾）──
    vlm_dir = out / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    seg_path = vlm_dir / f"{vh}_skeleton.json"   # 标准名：<哈希>_<产物>.json（产物名英文）
    out_skel = {"video": vmeta.get("video"), "video_hash": vh, "prefix": vh, "fps": fps,
                "scenes": [{"id": sc["scene_id"], "frame": _frame_no(sc["frames"][0])}
                           for sc in scenes if not sc.get("black")],   # fuse: 源帧→scene id
                "segments": results}
    seg_path.write_text(json.dumps(out_skel, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"✔ 选帧骨架: {seg_path}（{len(results)} 段）", flush=True)


if __name__ == "__main__":
    main()
