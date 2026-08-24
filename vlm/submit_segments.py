#!/usr/bin/env python3
"""vlm/submit_segments.py — 送检脚本（vlm 三件事之三：1.选帧 2.融合 3.送检）：自包含版
（2026-08-18 大名），读选帧骨架 segments + 融合段图，按段类型送 VLM 生成 desc 写回骨架。

用法（sorosoro env）: python vlm/submit_segments.py vivant [--ep EP01] [--limit N] [--force] [--fill-back]
依赖: output/<项目>/vlm/*_skeleton.json（选帧骨架）、vlm/segments/{seg}.jpg（融合段图）、
      vlm/vlm_prompts/ 提示词、vlm/desc_segments_template.json
产物: output/<项目>/vlm/<prefix>_desc.json（VLM描述骨架）

说明：
  岗位定位（2026-08-18 定稿）：三脚本各自自包含，不依赖 vlm/visual_ribbon 库（已整体预删除
  进 vlm/trash/）。送检读骨架 segments 键（select_segments 写回）+ 段图（fuse_segments 产物，
  文件名 = 段号 {seg}.jpg），按段类型送 VLM → desc 写回骨架。只读不重算，不重新切段不重新融合。

  送检规则：
    - 多帧段（fuse_frames ≥ 2）→ 段图送 PROMPT_RIB（连环画，200 tok）
    - 单帧段（fuse_frames 1 张）→ 段图送 PROMPT_SINGLE（单帧，150 tok）
    - 关思考（enable_thinking=False，左 pad 防空 desc）
  输出：帧级 desc 回写落 output/<项目>/vlm/<prefix>_desc.json（VLM描述骨架，2026-08-18 大名：
  3 脚本 3 骨架各管各的——select→选帧骨架、fuse→融合骨架、submit→desc.json；选帧骨架是
  select 的产物，submit 只读不写）。
  resume：已写回 desc 的段跳过（防重复 API 消耗）；--force 重送。
  desc.json 段结构（2026-08-18 范例）：{seg_id, scenes, shot_range, frames: {帧名: 描述}}——
  帧名:描述 dict，不带 ASR（vlm 是视觉识别工作目录，对话属 audio/ 的活）。
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# 送检提示词（vlm/vlm_prompts/ 定稿文件，2026-08-15/19.9 领导验收版）
PROMPT_SINGLE = (BASE / "vlm/vlm_prompts/视觉识别_单图.txt").read_text(encoding="utf-8").strip()
PROMPT_RIB = (BASE / "vlm/vlm_prompts/视觉识别_连环画.txt").read_text(encoding="utf-8").strip()

QWEN_MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "/models/hf/hub/Qwen/Qwen3-VL-4B-Instruct")


def load_skeleton(video_dir, ep=None):
    """读选帧骨架 output/<项目>/vlm/*_skeleton.json（select 落盘，与 fuse 同源；2026-08-18
    大名定稿：output/<项目名字>/vlm/<选帧骨架>.json，不再读 visual/dedup）。--ep 过滤 +
    唯一性选集；prefix = 骨架 video_hash（帧前缀内容指纹）"""
    out = BASE / "output" / video_dir
    skels = glob.glob(str(out / "vlm" / "*_skeleton.json"))
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


def _split_f_entries(desc):
    """模型直接输出 1|… 2|…（可能同行粘连）→ {格序: 描述}。
    2026-08-21 大名（混合段错位修复）：编号→帧名精确回写——格子 N ↔ 段图
    帧名序 frames[N-1]（fuse 记录 order，格序 = frames 序），不再 zip 丢位。
    描述里的数字不算条目（前瞻必须 `数字|` 才算下一格）。"""
    out = {}
    for m in re.finditer(r"(\d+)\|(.*?)(?=\d+\||$)", desc, re.S):
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        text = m.group(2).strip()
        if text:
            out[num] = text
    return out


def _batch_submit(model, processor, img_paths, prompts, max_new_tokens):
    """批量送检：N 张段图（同 prompt 同 max_new_tokens）拼批一次 generate → [desc]。
    拼批逻辑（2026-08-08 定稿方案）：apply_chat_template(tokenize=False) 逐条 +
    processor(text, images, padding=True) 对齐 → 一次 model.generate 出全部
    （串行逐段 → 同档拼批，VLM_BATCH 倍提速）。"""
    texts, images = [], []
    for p, prompt in zip(img_paths, prompts):
        texts.append(processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "url": p},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False))
        images.append(Image.open(p).convert("RGB"))
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id)
    in_len = inputs["input_ids"].shape[-1]
    # 2026-08-17 OOM 修复：批后清空 caching allocator reserved——批次大小不一
    # （1~16 张、图宽不一）会碎片化累积，后期同大批也 OOM（全量崩 2 次教训）
    torch.cuda.empty_cache()
    return [processor.decode(o[in_len:], skip_special_tokens=True).strip()
            for o in out_ids]


def _load_hf_model():
    """hf 后端：直接加载 QWEN_MODEL_DIR 本地模型（自包含，2026-08-18 大名，
    绝不 import 别的目录脚本）。目录 config 自带 quantization_config → 模型
    自己说了算；hub 原版（无量化配置）→ 4bit NF4，跳过 lm_head + visual 塔"""
    import json as _json
    from transformers import (AutoModelForMultimodalLM, AutoProcessor,
                              BitsAndBytesConfig)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    _cfg = _json.load(open(os.path.join(QWEN_MODEL_DIR, "config.json")))
    _kw = {}
    if not _cfg.get("quantization_config"):
        _kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "model.visual"])
    model = AutoModelForMultimodalLM.from_pretrained(
        QWEN_MODEL_DIR, device_map={"": 0}, trust_remote_code=True, **_kw)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)
    return model, processor


def _fill_back(video_dir):
    """跨集簇回填（2026-08-21 大名定稿：送检时跨集簇成员留空，全部集送完后再从
    簇 key 帧复制 desc 回填）。读项目 vlm/ 下全部 <hash>_desc.json，收集 _cXXXkey
    帧 desc → 项目级 {簇id: (key帧名, desc)} 映射；对每个 desc 为空的簇成员帧
    （_cXXX 无 key 后缀）找同簇 key 帧 desc 回填，写回。只补空帧、不动已有描述。
    纯 JSON 无 GPU 计算，不加载引擎、不读骨架、忽略 --ep（跨集簇映射需全项目）。"""
    out = BASE / "output" / video_dir
    descs = sorted(Path(p) for p in glob.glob(str(out / "vlm" / "*_desc.json")))
    if not descs:
        raise SystemExit(f"❌ 无 desc.json（先跑 vlm/submit_segments.py 送检）")
    key_desc = {}                                        # 簇id → (key帧名, desc)
    for p in descs:
        for s in json.load(open(p, encoding="utf-8")).get("segments", []):
            for item in s.get("frames") or []:
                m = re.match(r"^(.*_c(\d+)key)：(.*)$", item, re.S)
                if m and m.group(3).strip():
                    key_desc.setdefault(m.group(2), (m.group(1), m.group(3).strip()))
    if not key_desc:
        print("✔ 无簇 key 帧描述，无需回填")
        return
    total = 0
    for p in descs:
        d = json.load(open(p, encoding="utf-8"))
        filled = 0
        for s in d.get("segments", []):
            frames = s.get("frames") or []
            for i, item in enumerate(frames):
                fname, sep, txt = item.partition("：")
                if txt.strip() or not sep:              # 已有描述 / 无冒号 → 不动
                    continue
                m = re.search(r"_c(\d+)", fname)
                if not m:                               # 留空却无簇 id = 模板异常
                    raise SystemExit(f"❌ {Path(p).name} seg{s.get('seg_id')} {fname}"
                                     " desc 空且无簇 id（模板异常）")
                kv = key_desc.get(m.group(1))
                if kv is None:                          # 簇 key 未送检 = 上游异常
                    raise SystemExit(f"❌ {Path(p).name} {fname} 簇 c{m.group(1)}"
                                     " 无 key 帧 desc（簇 key 帧所在集未送检？）")
                frames[i] = f"{fname}：{kv[1]}"
                filled += 1
        if filled:
            p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"✔ {Path(p).name}: 回填 {filled} 帧", flush=True)
        total += filled
    if total:
        print(f"✔ 跨集簇回填完成：{len(descs)} 个 desc.json，共回填 {total} 帧")
    else:
        print("✔ 已无留空簇成员帧，无需回填")


def main():
    """主流程：读选帧骨架 → resume 迁移 → 按类型送 VLM（vllm/hf）→ 簇复用 → 帧级 desc 回写落盘"""
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--ep", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只送前 N 段（验证用）")
    ap.add_argument("--force", action="store_true", help="已写回 desc 的段也重送")
    ap.add_argument("--fill-back", action="store_true",
                    help="跨集簇回填：全部集送完后，把留空的簇成员帧从簇 key 帧 desc 回填"
                         "（全项目扫描，忽略 --ep，纯 JSON 不加载引擎）")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="vllm=continuous batching 全量一次提交（vllm env 跑，默认）；hf=按 token 组批")
    args = ap.parse_args()

    if args.fill_back:
        _fill_back(args.video_dir)
        return

    name, sk, prefix, out = load_skeleton(args.video_dir, args.ep)
    segments = sk.get("segments")
    if not segments:
        raise SystemExit("❌ 骨架无 segments 键——先跑 vlm/select_segments.py（选帧）")
    out_dir = out / "vlm" / "segments"                   # 融合图片文件夹（段图）
    fps = sk["fps"]

    # resume：已送段跳过（防重复 API 消耗）；--force 重送。
    # 新结构（frames = 帧名:描述 dict）→ 合并回内存；旧结构（段级 desc）→
    # 合并 desc，帧级回写时从 desc 现切 F（旧 F 按行切粘连是坏数据）
    prev_frames, prev_desc = {}, {}
    desc_json = out / "vlm" / f"{prefix}_desc.json"
    if desc_json.exists():
        for s in json.load(open(desc_json)).get("segments", []):
            fr = s.get("frames")
            if isinstance(fr, dict) and fr:              # 旧 dict 版
                prev_frames[s["seg_id"]] = fr
            elif isinstance(fr, list):                   # 模板数组版（「帧名：描述」）
                fd = {}
                for item in fr:
                    if isinstance(item, str) and "：" in item:
                        fn, d = item.split("：", 1)
                        fd[fn] = d
                if fd:
                    prev_frames[s["seg_id"]] = fd
            elif s.get("desc"):
                prev_desc[s["seg_id"]] = s
    for s in segments:
        pf = prev_frames.get(s["seg_id"])
        if pf:
            s["frames"] = pf
        else:
            pd = prev_desc.get(s["seg_id"])
            if pd:
                s["desc"] = pd["desc"]
                s["F"] = pd.get("F")
    todo = [s for s in segments if args.force or not (
        (isinstance(s["frames"], dict) and s["frames"]) or s.get("desc"))]
    n_done = len(segments) - len(todo)
    print(f"[submit_segments] {len(segments)} 段，已写回 {n_done}，待送 {len(todo)}",
          flush=True)
    if args.limit:
        todo = todo[:args.limit]

    # ── 帧级 desc 回写（大名 2026-08-18：vlm 返回的 123 要对应上绝对帧号；
    # 对应完了才是照抄环节，逻辑错了全崩）──
    def _is_key(fn):
        return fn.rsplit("_", 1)[-1].endswith("key")

    def _abs_no(fn):
        return int(fn.split("_f", 1)[1].split("_", 1)[0])

    def _frames_desc():
        """段 frames → 帧名:描述 dict（resume 迁移路径同走；已回写段跳过重建）"""
        # 簇首帧映射（帧级、跨段，大名 2026-08-18 批准）：全部段里 _cXXXkey 帧
        cluster_first = {}
        for s in segments:
            for f in s["frames"]:
                sfx = f.rsplit("_", 1)[-1]
                if sfx.startswith("c") and sfx.endswith("key"):
                    cluster_first[sfx[1:-3]] = f        # 'c008key' → '008'
        # 遍1：送检段 desc 落位——格序 = key 帧名序（fuse 只拼 key 帧，order 即 keys，
        # 格子数 = key 帧数），VLM 编号 N ↔ 段图格序[N-1] 对位回写（2026-08-21 大名：
        # 混合段不按 frames 全量对位，否则格子数 ≠ frames 数编号错位）。单 key 帧段
        # 无编号，纯描述给唯一 key 帧。无 key 帧段（簇复用/留空段，fuse 未产图）遍1
        # 不填，遍2 从簇 key 分发。
        all_fd = {}                                     # 帧名 → desc（全部帧）
        for seg in segments:
            if isinstance(seg["frames"], dict):
                all_fd.update(seg["frames"])
                continue
            fd = {f: "" for f in seg["frames"]}         # 段内全部帧占位
            keys = [f for f in seg["frames"] if _is_key(f)]
            if keys:                                    # 有 key 帧 = 段图送检过
                fds = _split_f_entries(seg["desc"]) if seg.get("desc") else {}
                if len(keys) == 1:                      # 单 key 帧段：纯描述给唯一 key 帧
                    fd[keys[0]] = next(iter(fds.values()), seg.get("desc") or "")
                else:                                   # 连环画：编号 N ↔ 格序[N-1]
                    for i, f in enumerate(keys, 1):     # 格序 = key 帧名序
                        fd[f] = fds.get(i, "")
            all_fd.update(fd)
            seg["frames"] = fd
        # 遍2：分发——簇非首帧共用簇 key 帧 desc、无后缀帧抄段内最近 key 帧（遍1 只填
        # key 帧，簇成员/无后缀留空在这分发，2026-08-21 大名）；已送检有真实 desc 的帧
        # 不覆盖；跨视频簇 rep 在本骨架查不到 → all_fd.get(None,"") 留空，等 rep 送检。
        for seg in segments:
            fd = seg["frames"]
            if not isinstance(fd, dict) or not fd:
                continue
            keys = [f for f in fd if _is_key(f)]
            for f in list(fd):
                if _is_key(f):
                    continue
                if fd.get(f):                           # 已送检真实 desc → 不覆盖
                    continue
                sfx = f.rsplit("_", 1)[-1]
                if sfx.startswith("c"):                 # 簇非首帧：共用簇首帧 desc
                    src = cluster_first.get(sfx[1:])    # 'c008' → '008'
                    fd[f] = all_fd.get(src, "")
                else:                                   # 无后缀帧：段内最近 key 帧
                    prev = [k for k in keys if _abs_no(k) < _abs_no(f)]
                    nxt = [k for k in keys if _abs_no(k) > _abs_no(f)]
                    src = prev[-1] if prev else (nxt[0] if nxt else None)
                    fd[f] = fd.get(src, "")

    # ── VLM描述骨架：帧级 desc 只落 desc.json（大名 2026-08-18：vlm 返回的 123
    # 对应绝对帧号，回写完结果保存为 desc.json；选帧骨架是 select 的产物，
    # submit 只读不写，3 脚本 3 骨架各管各的）──
    desc_path = out / "vlm" / f"{prefix}_desc.json"   # 标准名：<哈希>_<产物>.json
    tpl_path = BASE / "vlm" / "desc_segments_template.json"   # 模板存档（大名 2026-08-18）

    def _check_template(segs):
        """落盘前对照 vlm/desc_segments_template.json 校验（大名：模板放好，
        每次看一遍对应上——键名/frames 数组/条目格式，对不上报错停，不静默）"""
        tpl = json.load(open(tpl_path, encoding="utf-8"))
        want = [k for k in ("seg_id", "scene id", "shot range", "frames") if k in tpl]
        for s in segs:
            have = [k for k in want if k in s]
            if have != want:
                raise SystemExit(f"❌ 段 {s.get('seg_id')} 键 {have} != 模板 {want}"
                                 "（对照 vlm/desc_segments_template.json）")
            fr = s["frames"]
            if not isinstance(fr, list) or not fr:
                raise SystemExit(f"❌ 段 {s['seg_id']} frames 非数组/空（模板要求数组）")
            for item in fr:
                if not (isinstance(item, str) and "：" in item):
                    raise SystemExit(f"❌ 段 {s['seg_id']} frames 条目非法: {item!r}"
                                     "（模板：<哈希>_<绝对帧号>[后缀]：<描述>）")
                fn = item.split("：", 1)[0]
                if not re.fullmatch(r"\w+_f\d+(?:_\w+)?", fn):
                    raise SystemExit(f"❌ 段 {s['seg_id']} 帧名非法: {fn!r}")
        print(f"✔ desc 对照模板校验通过（{len(segs)} 段）", flush=True)

    def _write_desc():
        """desc.json：每段 seg_id/「scene id」/「shot range」/frames——frames 是
        数组，每条 = 「帧名：描述」字符串，按段内时间序（大名 2026-08-18 模板
        原样：<哈希>_<绝对帧号>[<后缀>]：<描述>）。中途增量（HF 崩可续跑）与
        最终共用，不碰骨架"""
        out_desc = {"video": sk["video"], "video_hash": sk["video_hash"],
                    "prefix": prefix,
                    "segments": [{"seg_id": seg["seg_id"],
                                  "scene id": seg["scenes"],
                                  "shot range": seg["shot_range"],
                                  "frames": [f"{f}：{seg['frames'][f]}"
                                             for f in seg["frames"]]}
                                 for seg in segments]}
        _check_template(out_desc["segments"])
        desc_path.write_text(json.dumps(out_desc, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    if not todo:
        # 全部已送/已迁移：只刷新帧级 desc.json，不加载引擎、不碰骨架
        _frames_desc()
        _write_desc()
        print(f"✔ 全部已回写（{len(segments)} 段），desc.json 已刷新（0 送检）",
              flush=True)
        return

    if args.backend == "vllm":
        # vllm 后端（2026-08-18 大名：已有 vllm 为什么还慢——vllm continuous
        # batching 并发多段，HF generate 单段 ~16s 的瓶颈；vllm env 跑，bnb 4bit
        # 3060 12G 目标 4-6 并发）
        # vllm 0.27.1 + Python 3.10 bug（novelize 2026-08-16 绕法）：kernel_warmup
        # 无条件 import minimax_m3 → flashinfer.comm fd_exchange 用 array.array[int]
        # 注解（需 Py3.12）→ TypeError。Qwen3-VL 不依赖 M3/comm，预注册 no-op
        # 假模块绕过（不改 site-packages）
        import types
        _m = types.ModuleType("vllm.model_executor.warmup.minimax_m3_msa_warmup")
        _m.minimax_m3_msa_warmup = lambda worker, **kw: None
        sys.modules[_m.__name__] = _m
        from vllm import LLM, SamplingParams
        # 显存实测（2026-08-20 定稿，8G 卡可跑）：util=0.60 + ml=4096 → 峰值 6757 MiB
        # （省 30%，0.92 时曾 9643）；单图/拼图推理 token 远低于 4096，ml 是上限安全值。
        llm = LLM(model=QWEN_MODEL_DIR, quantization="bitsandbytes", dtype="bfloat16",
                  max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")),
                  gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.82")),
                  max_num_batched_tokens=int(os.environ.get("VLLM_MAX_BATCHED_TOKENS", "4096")),
                  max_num_seqs=int(os.environ.get("VLLM_MAX_NUM_SEQS", "64")),
                  enforce_eager=True)
        print("✔ vllm 就绪（Qwen3-VL-4B bnb4bit，continuous batching）", flush=True)
    else:
        model, processor = _load_hf_model()
        # 2026-08-17 空 desc 修复：decoder-only 生成必须左 pad（默认右 pad 批内
        # 非最长条生成错位 → 61 段空输出；主链 _batch_from_cache 是显式左 pad，没这问题）
        processor.tokenizer.padding_side = "left"
        print("✔ 模型就绪（Qwen3-VL，关思考，左 pad）", flush=True)

    vlm_batch = int(os.environ.get("VLM_BATCH", "16"))   # 批张数上限
    vlm_budget = int(os.environ.get("VLM_BUDGET", "8192"))  # 批视觉 token 预算
    # 12G 安全参考：主链 batch4 × 1080p 原帧（~2176 tok/帧）≈ 8.7k tok 验证过

    def _vis_tok(p):
        w, h = Image.open(p).size
        return math.ceil(h / 32) * math.ceil(w / 32)    # Qwen3-VL patch16/merge2

    def _groups(items):
        """按视觉 token 预算贪心组批——段图宽不一（3:4 的 405 宽 ~220 tok，
        宽融合段图可达 2k+ tok），固定张数批会 OOM；同主链『同档拼批』按 token 档"""
        toks = [_vis_tok(it[1]) for it in items]
        groups, cur, cur_t = [], [], 0
        for it, t in zip(items, toks):
            if cur and (cur_t + t > vlm_budget or len(cur) >= vlm_batch):
                groups.append(cur)
                cur, cur_t = [], 0
            cur.append(it)
            cur_t += t
        if cur:
            groups.append(cur)
        return groups

    def _seg_img(seg):
        """段图定位：文件名 = 段号（fuse_segments 2026-08-18 定稿，{seg}.jpg）"""
        img = out_dir / f"{seg['seg']}.jpg"
        if not img.exists():
            raise SystemExit(f"❌ 缺段图 {img}——先跑 vlm/fuse_segments.py（融合）")
        return img

    def _key_frames(seg):
        """拼图/送检帧 = 帧名后缀含 key（_key / _c{id}key，2026-08-18 大名）；
        frames 已回写为 dict（帧名:描述）时取键"""
        fr = seg["frames"]
        frs = list(fr) if isinstance(fr, dict) else fr
        return [f for f in frs if f.rsplit("_", 1)[-1].endswith("key")]

    def _frame_no(fn):
        """'9d5dab_f40409_c020key' → 40409（帧名带 key/簇后缀）"""
        return int(fn.split("_f", 1)[1].split("_", 1)[0])

    # 簇首段映射：含 _c{id}key 帧的段（segments 序首个，2026-08-18 大名：簇首帧
    # 标记为该簇 key 帧，送检后复用给簇内其余段）
    cluster_key_seg = {}
    for s in segments:
        for f in s["frames"]:
            sfx = f.rsplit("_", 1)[-1]
            if sfx.endswith("key") and sfx.startswith("c"):
                cluster_key_seg.setdefault(sfx[:-3], s)   # 'c020key' → c020

    # 按类型分两路（批内同 prompt 同 max_new_tokens）：连环画 200 / 单帧 150；
    # 无 key 帧段（帧全为簇非首 _c{id}，fuse 不产图）→ 簇复用，不送检
    tagged, reuse_pending = [], []
    for seg in todo:
        keys = _key_frames(seg)
        if not keys:
            reuse_pending.append(seg)
            continue
        img = _seg_img(seg)
        multi = len(keys) >= 2
        tagged.append((seg, img, "连环画" if multi else "单帧",
                       PROMPT_RIB if multi else PROMPT_SINGLE,
                       200 if multi else 150))

    t_all = time.time()
    n_multi = n_single = 0

    def _apply(seg, desc, typ, mnt):
        """desc/F/类型落段（两后端共用；VLM描述 = 纯视觉转文本，不带 ASR——
        2026-08-18 大名：vlm 是视觉识别工作目录，对话属 audio/ 的活）"""
        nonlocal n_multi, n_single
        seg["desc"] = desc
        seg["F"] = _split_f_entries(desc)
        seg["tokens"] = mnt
        seg["type"], seg["prompt"] = "scene", typ
        seg["desc_source"] = f"vlm/submit_segments（{prefix}）"
        if typ == "连环画":
            n_multi += 1
        else:
            n_single += 1

    if args.backend == "vllm":
        # 全量一次提交，vllm continuous batching 自调度（每条带自己的
        # SamplingParams：多帧 200 / 单帧 150 tok，关思考，temperature=0 确定性）
        messages, sps = [], []
        for _seg, img, _typ, prompt, mnt in tagged:
            messages.append([{"role": "user", "content": [
                # vllm 0.27.1 多模态格式：image_pil（PIL 直传，不走 URL fetch）
                {"type": "image_pil", "image_pil": Image.open(str(img))},
                {"type": "text", "text": prompt}]}])
            sps.append(SamplingParams(max_tokens=mnt, temperature=0.0))
        outs = llm.chat(messages=messages, sampling_params=sps)
        for (seg, _img, typ, _p, mnt), o in zip(tagged, outs):
            _apply(seg, o.outputs[0].text.strip(), typ, mnt)
        print(f"  [{len(tagged)}/{len(todo)}] 连环画 {n_multi} / 单帧 {n_single} / "
              f"{time.time() - t_all:.0f}s", flush=True)
    else:
        for typ, prompt, mnt in (("连环画", PROMPT_RIB, 200), ("单帧", PROMPT_SINGLE, 150)):
            group = [t for t in tagged if t[2] == typ]
            for batch in _groups(group):
                imgs = [str(t[1]) for t in batch]
                outs = _batch_submit(model, processor, imgs, [prompt] * len(batch), mnt)
                for (seg, _img, _typ, _p, _m), desc in zip(batch, outs):
                    _apply(seg, desc, _typ, _m)
                done = n_multi + n_single
                _write_desc()                      # 每批增量落盘（OOM 崩溃可 resume 续跑）
                if done % 20 == 0 or done == len(todo):
                    print(f"  [{done}/{len(todo)}] 连环画 {n_multi} / 单帧 {n_single} / "
                          f"对话 {n_dial} / {time.time() - t_all:.0f}s", flush=True)

    # ── 簇复用：无 key 帧段（帧全为簇非首，fuse 不产图）直接复用簇首段 desc
    # （2026-08-18 大名：有簇标记且前面有送 → 不检测，复用结果）──
    # 2026-08-21 大名：簇 key 未送检（跨视频 rep 在本骨架查不到）→ 成员 desc 直接留空，
    # 不补送不 raise——什么时候簇 key 帧送检，什么时候遍2 贴描述
    n_reuse = n_pending = 0
    for seg in reuse_pending:
        cids = [f.rsplit("_", 1)[-1] for f in seg["frames"]
                if f.rsplit("_", 1)[-1].startswith("c")]
        src = next((cluster_key_seg[c] for c in cids
                    if cluster_key_seg.get(c, {}).get("desc")), None)
        if src is None:
            seg["desc"] = ""
            seg["F"] = []
            seg["type"], seg["prompt"] = "scene", None
            seg["desc_source"] = f"cluster_pending:{cids[0]}（簇 key 未送检，留空）"
            n_pending += 1
            continue
        seg["desc"] = src["desc"]
        seg["F"] = src.get("F")
        seg["tokens"] = src.get("tokens")
        seg["type"], seg["prompt"] = "scene", src.get("prompt")
        seg["desc_source"] = f"cluster_reuse:{cids[0]}（{prefix}）"
        n_reuse += 1
    if n_reuse or n_pending:
        print(f"✔ 簇复用 {n_reuse} 段 / 留空 {n_pending} 段（簇 key 未送检，未送 VLM）",
              flush=True)

    _frames_desc()
    _write_desc()
    print(f"✔ VLM描述骨架: {desc_path}")
    print(f"✔ 帧级 desc 回写 {len(segments)} 段（送 VLM {len(tagged)} / 簇复用 "
          f"{n_reuse} / 留空 {n_pending} / 照抄填帧，总耗时 {time.time() - t_all:.0f}s）",
          flush=True)


if __name__ == "__main__":
    main()
