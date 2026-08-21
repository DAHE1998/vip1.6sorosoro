#!/usr/bin/env python3
"""chapter/submit_api.py — 章节划分 ② 送 API 脚本（2026-08-20 大名：自包含）。

用法: OUT_ROOT=<输出目录> python3 chapter/submit_api.py <video_hash>
     或 python3 chapter/submit_api.py <video_hash> <输出目录>
依赖: chapter/<hash>_assembled.json（assemble.py 产物）
产物: <out_root>/chapter/<video_hash>_chapter_boundaries.json
     {"n_scenes": N, "boundaries": [新章节起始 scene 号...], "n_chapters": K,
      "api_model": ..., "api_base": ...}
     boundaries = 除第一个章节外各章节的起始 scene 号（0 无效已滤，首章从 0 隐含）。

说明: 读 assemble.py 产出的 <hash>_assembled.json → 内嵌 prompt（intro + rules，自包含
不读外部 .prompt）拼装逐 scene 文本 → 送 LLM chat/completions → 解析 boundaries → 落盘。
API 配置: 优先环境变量 LLM_API_BASE / LLM_API_KEY（或 MODELSCOPE_API_KEY），否则读本目录
api_key.txt（第 1 行 base_url，第 2 行 key）；env: LLM_API_MODEL 默认 Qwen/Qwen3.5-122B-A10B。
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MAX_NEW = 4096


def load_api():
    """API 配置：优先环境变量，否则读本目录 api_key.txt（base_url 行 + key 行）。"""
    d = Path(__file__).resolve().parent
    base = os.environ.get("LLM_API_BASE", "").rstrip("/")
    key = (os.environ.get("LLM_API_KEY") or os.environ.get("MODELSCOPE_API_KEY", "")).strip()
    if not base or not key:
        kf = d / "api_key.txt"
        if kf.is_file():
            lines = [ln.strip() for ln in kf.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
            if len(lines) >= 2:
                if not base:
                    base = lines[0].rstrip("/")
                if not key:
                    key = lines[1].strip()
    if not base:
        base = "https://api-inference.modelscope.cn/v1"
    if not key:
        sys.exit("[chapter] API key 未设置：LLM_API_KEY / MODELSCOPE_API_KEY / chapter/api_key.txt")
    return base, key

# ── 内嵌 prompt（自包含，不依赖外部 .prompt 文件）──
INTRO = """以下是视频素材的全部{n}个scene的文本表达，按时间顺序排列。

格式说明：
- 出现人物：P+数字编号来自视觉识别系统，表示视频画面出现的人物face id
- F+数字编号来自VLM，如：F18代表该scene的第18张视频帧的视觉内容
- 说话人A/B/C...来自声纹识别，ABC表示句子的说话人，但不和face id挂钩，需推测说话人身份。
- (none)=无对话，if （text=none） scene=空镜
"""

RULES = """任务：识别章节边界（把连续 scene 合并成章节 chapter）。

判断依据：
[P0]独立叙事：不包含两件及以上可单独讲述的事件，如探店，采访某一对象。
[P1]在哪里（ASR中提到的地点名称）
[P2]谁在画面里（句子人物标记）

粒度要求（重要，优先遵守）：
- 连续边界=error（不要出现相邻 scene 编号都当边界，如 78,79,80）
- 单 scene 章节=error；单个无台词 scene 必须并入前一个章节
- if距离靠近且有台词的scene之间夹杂了无对话镜头，无对话镜头属于前一章节，如：1[有]+2[无]+3[有]则={[1+2]+[3]}

硬性约束：
- 边界只能插在 scene 与 scene 之间，scene 内部不拆分
- 切点应该落在scene的尾部而不是开头
- 输出的编号是新章节起始的 scene 编号，必须在 0 到 {n_max}（共 {n} 个 scene）

输出要求：仅输出JSON：
{{"boundaries": [sceneid1, sceneid2, sceneid3, ...]}}
"""


def api_chat(prompt_full):
    """OpenAI 兼容 chat/completions 调用（国内 API 直连，不走代理）。
    temperature=0 对应本地 do_sample=False；3 次重试（指数退避）。"""
    base, key = load_api()
    model = os.environ.get("LLM_API_MODEL", "Qwen/Qwen3.5-122B-A10B")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_full}],
        "temperature": 0,
        "max_tokens": MAX_NEW,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"]["content"]
            u = resp.get("usage", {})
            print(f"[chapter] api {model} attempt={attempt} {time.time() - t0:.0f}s, "
                  f"prompt_tokens={u.get('prompt_tokens')}, "
                  f"completion_tokens={u.get('completion_tokens')}")
            return content
        except Exception as e:
            print(f"[chapter] api attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)
    sys.exit("[chapter] API 调用 3 次失败")


def parse_boundaries(response):
    """只解析格式，零过滤（同旧 chapter_seg_text_api）。"""
    m = re.search(r'\{[^{}]*"boundaries"[^{}]*\}', response, re.DOTALL)
    if not m:
        return []
    json_str = m.group()
    if json_str.count('[') > json_str.count(']'):
        json_str = json_str.rstrip().rstrip(',') + ']}'
    if json_str.count('"') % 2 != 0:
        json_str = json_str + '"'
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        nums = re.findall(r'"(\d+),(\d+)"', response)
        return sorted(set(int(b) for _, b in nums))
    raw_b = result.get("boundaries", [])
    boundaries = []
    for item in raw_b:
        if isinstance(item, int):
            boundaries.append(item)
        elif isinstance(item, str):
            item = item.strip()
            if item.startswith("Scene"):
                try:
                    boundaries.append(int(item[5:]))
                except ValueError:
                    pass
            elif "," in item:
                parts = item.split(",")
                if len(parts) == 2:
                    try:
                        boundaries.append(int(parts[1]))
                    except ValueError:
                        pass
            else:
                try:
                    boundaries.append(int(item))
                except ValueError:
                    pass
    return sorted(set(boundaries))


def build_transcript(embedded):
    """逐 scene 文本（画面 F 行 + 对白行），与 intro 格式说明一致。"""
    parts = []
    for sc in embedded:
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
        if desc_str or asr_str:
            body = f"Scene{sid} [persons: {persons}]" + desc_str + asr_str
        else:
            body = f"Scene{sid} [persons: {persons}]\n  (None)"
        parts.append(body)
    return "\n\n".join(parts)


def resolve_out_root(vh, argv):
    if os.environ.get("OUT_ROOT"):
        return os.environ["OUT_ROOT"]
    if len(argv) >= 3:
        return argv[2]
    return str(BASE / "output" / vh)


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <video_hash> [out_root]")
        sys.exit(1)
    vh = sys.argv[1]
    video_out = resolve_out_root(vh, sys.argv)

    embed_p = Path(video_out) / "chapter" / f"{vh}_assembled.json"
    if not embed_p.is_file():
        sys.exit(f"[chapter] ❌ 无组装产物 {embed_p}（先跑 chapter/assemble.py）")
    data = json.loads(embed_p.read_text(encoding="utf-8"))
    scenes = data["scenes"]
    n_scenes = len(scenes)

    intro = INTRO.replace("{n}", str(n_scenes)).replace("{n_max}", str(n_scenes - 1))
    rules = RULES.replace("{n}", str(n_scenes)).replace("{n_max}", str(n_scenes - 1))
    transcript = build_transcript(scenes)
    prompt_full = intro + "\n\n" + rules + "\n\n" + transcript
    print(f"[chapter] {n_scenes} scenes, transcript chars: {len(transcript)}")
    print(f"[chapter] api: {os.environ.get('LLM_API_MODEL', 'Qwen/Qwen3.5-122B-A10B')}")

    response = api_chat(prompt_full)
    print(f"[chapter] raw: {response}")
    final = parse_boundaries(response)
    # 0=首章起点无语义（首章从 0 隐含），>=n_scenes 越界，均滤
    final = [b for b in final if 0 < b < n_scenes]

    out_dir = Path(video_out) / "chapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{vh}_chapter_boundaries.json"
    api_base, _ = load_api()
    json.dump({"n_scenes": n_scenes,
               "boundaries": final,
               "n_chapters": len(final) + 1,
               "api_model": os.environ.get("LLM_API_MODEL", "Qwen/Qwen3.5-122B-A10B"),
               "api_base": api_base},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[chapter] boundaries ({len(final)}): {final}")
    print(f"[chapter] -> {out}")


if __name__ == "__main__":
    main()
