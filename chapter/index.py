#!/usr/bin/env python3
"""视频目录生成器 — 三段式拼装（各段独立来源，py 不内嵌内容）。

拼装顺序（头→中→下）:
  ① 格式说明  ← format.txt（外部文档）
  ② 提示词    ← prompt.txt（外部文档）
  ③ 骨架内容  ← 骨架 json（build_transcript 生成）
  → 一起送给 ModelScope API → 输出 blocks JSON

用法:
    python3 index.py [骨架.json] [输出名.json]
    默认: 骨架=25ca10_assembled.json  输出=<骨架名>_blocks.json
    格式说明/提示词固定读 format.txt / prompt.txt，改动不用动脚本。

依赖: 同目录 api_key.txt（第1行 base_url 第2行 key）；仅标准库。
"""
import json, os, ssl, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None


def load_api():
    base = os.environ.get("LLM_API_BASE", "").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    if not base or not key:
        lines = (HERE / "api_key.txt").read_text().splitlines()
        vals = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        base, key = vals[0], vals[1]
    return base, key


def api_chat(prompt_full):
    base, key = load_api()
    model = os.environ.get("LLM_API_MODEL", "gpt-5.6-sol")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_full}],
        "temperature": 0,
        # 2026-09-02: 换 gpt-5.6-sol（重推理模型，回"OK"都烧 4k+ token）。
        # 4096 不够推理+长块 JSON，调大到 16384，防输出被截断丢视频。
        "max_tokens": 16384,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    kw = {"timeout": 600}
    if _SSL_CTX is not None:
        kw["context"] = _SSL_CTX
    for attempt in (1, 2, 3, 4, 5):
        try:
            with urllib.request.urlopen(req, **kw) as r:
                resp = json.loads(r.read())
            if resp.get("choices"):
                break
            print(f"[api] attempt={attempt} choices=null（服务端偶发空返回），重试...")
            time.sleep(5 * attempt)
        except Exception as e:
            print(f"[api] attempt={attempt} 异常: {e}，重试...")
            time.sleep(5 * attempt)
    else:
        print("[api] 5 次全失败"); sys.exit(1)
    # 397B 有时返回 reasoning_content 在 message 里，content 可能 None
    msg = resp["choices"][0]["message"] or {}
    out = msg.get("content") or msg.get("reasoning_content") or ""
    if not out and "reasoning" in msg:
        out = msg["reasoning"]
    if not out:
        print("[api] 调试：response =", json.dumps(resp, ensure_ascii=False)[:500]); sys.exit(1)
    u = resp.get("usage", {})
    print(f"[api] {model} {time.time()-t0:.0f}s prompt_tokens={u.get('prompt_tokens')} "
          f"completion_tokens={u.get('completion_tokens')}")
    return out


def build_transcript(scenes):
    """③ 骨架内容段：逐 scene 拼文本（画面 F 行 + 对白行），与 format.txt 格式说明一致。"""
    parts = []
    for sc in scenes:
        sid = sc["scene_id"]
        persons = ",".join(sc.get("persons", [])) or "无"
        desc_str = ""
        if sc.get("frames"):
            desc_str = "\n  画面:\n" + "\n".join("    " + ln for ln in sc["frames"])
        asr_str = ""
        if sc.get("asr"):
            lines = []
            for line in sc["asr"]:
                p2 = line.split("|", 1)
                lines.append(f"    {p2[0]}: {p2[1]}" if len(p2) == 2 else f"    {line}")
            asr_str = "\n  对白:\n" + "\n".join(lines)
        body = f"Scene{sid} [persons: {persons}]" + desc_str + asr_str
        if not (desc_str or asr_str):
            body = f"Scene{sid} [persons: {persons}]\n  (None)"
        parts.append(body)
    return "\n\n".join(parts)


def run_blocks():
    """模式1（默认）：骨架 → blocks.json。三段式拼装。"""
    sk_name = sys.argv[1] if len(sys.argv) > 1 else "25ca10_assembled.json"
    out_name = sys.argv[2] if len(sys.argv) > 2 else None
    sk_path = Path(sk_name) if os.path.isabs(sk_name) else HERE / sk_name

    format_doc = (HERE / "format.txt").read_text(encoding="utf-8")
    prompt_doc = (HERE / "prompt.txt").read_text(encoding="utf-8")

    data = json.loads(sk_path.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    vh = sk_path.stem.split("_")[0]
    transcript = build_transcript(scenes)

    prompt_full = format_doc + "\n\n" + prompt_doc + "\n\n====================\n以下为待分析的全部 scene：\n====================\n\n" + transcript

    print(f"▶ 骨架={sk_path.name} | {len(scenes)} scenes | 总长 {len(prompt_full)} 字符")
    # 解析放重试循环内：模型偶发输出坏 JSON（如 f48c87 那次），发一次解析崩就漏产物
    blocks = None
    for attempt in (1, 2, 3):
        resp = api_chat(prompt_full)
        m = resp.find("{")
        if m < 0:
            print(f"[parse] attempt={attempt} 响应无 JSON 输出，重发重试...")
            continue
        try:
            blocks = json.loads(resp[m:resp.rfind("}") + 1])
            break
        except json.JSONDecodeError as e:
            print(f"[parse] attempt={attempt} JSON 解析失败: {e}，重发重试...")
    if blocks is None:
        print("❌ 响应 3 次无有效 JSON，响应片段:")
        print(resp[m:m + 600] if m >= 0 else resp[:600])
        sys.exit(1)
    blocks["video_id"] = vh
    # 2026-09-02 大名：每块补绝对帧范围 frames_range（由骨架 scene frames_range 算，
    # 不信 LLM 帧号；block 覆盖 scene_start..scene_end 的连续帧区间）
    scene_fr = {sc["scene_id"]: sc.get("frames_range") for sc in scenes}
    for blk in blocks.get("blocks", []):
        if "frames_range" in blk:
            continue
        try:
            s, e = int(blk.get("scene_start")), int(blk.get("scene_end"))
        except (TypeError, ValueError):
            continue
        f0, f1 = scene_fr.get(s), scene_fr.get(e)
        if f0 and f1:
            a0 = f0[0] if isinstance(f0, list) else f0["start"]
            a1 = f1[1] if isinstance(f1, list) else f1["end"]
            blk["frames_range"] = {"start": a0, "end": a1}

    out = HERE / (out_name or f"{sk_path.stem}_blocks.json")
    out.write_text(json.dumps(blocks, ensure_ascii=False, indent=2))
    print(f"✅ → {out.name}（{len(blocks['blocks'])} 块）")


def run_director():
    """模式2：director 模式 — 两个 JSON 一起给模型，产出剪辑方案 edit_plan.json。

    输入① blocks.json（叙事目录，第一遍概览用）
    输入② 骨架 assembled.json（scene 细节，第二遍选镜头用）
    """
    blocks_name = sys.argv[2] if len(sys.argv) > 2 else "25ca10_assembled_blocks.json"
    sk_name = sys.argv[3] if len(sys.argv) > 3 else "25ca10_assembled.json"
    out_name = sys.argv[4] if len(sys.argv) > 4 else None

    blocks_path = Path(blocks_name) if os.path.isabs(blocks_name) else HERE / blocks_name
    sk_path = Path(sk_name) if os.path.isabs(sk_name) else HERE / sk_name

    director_doc = (HERE / "director_prompt.txt").read_text(encoding="utf-8")

    # 第一遍素材：block 目录（小，全量给）
    blocks_json = blocks_path.read_text(encoding="utf-8")
    # 第二遍素材：骨架 scene 细节（transcript 全量）
    data = json.loads(sk_path.read_text(encoding="utf-8"))
    transcript = build_transcript(data["scenes"])

    prompt_full = (
        director_doc
        + "\n\n====================\n【第一遍素材】block 目录（叙事结构）：\n====================\n\n" + blocks_json
        + "\n\n====================\n【第二遍素材】全部 scene 明细：\n====================\n\n" + transcript
    )

    print(f"▶ blocks={blocks_path.name} | 骨架={sk_path.name} | 总长 {len(prompt_full)} 字符")
    resp = api_chat(prompt_full)
    m = resp.find("{")
    if m < 0:
        print("❌ 响应无 JSON 输出"); sys.exit(1)
    plan = json.loads(resp[m:resp.rfind("}") + 1])

    out = HERE / (out_name or f"{blocks_path.stem.split('_blocks')[0]}_editplan.json")
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    n_sel = sum(len(bp.get("selected_scenes", [])) for bp in plan.get("block_plans", []))
    print(f"✅ → {out.name}（{len(plan.get('block_plans', []))} 个 block 计划, {n_sel} 个选中镜头）")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "director":
        run_director()
    else:
        run_blocks()


if __name__ == "__main__":
    main()