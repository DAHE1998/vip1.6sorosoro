#!/usr/bin/env python3
"""chapter/chapter_seg_text_api.py — 纯文本划章 API 推理版（2026-08-18 拼装重写）。

用法: python3 chapter/chapter_seg_text_api.py <视频名> [视频输出目录]
依赖: dedup scene 骨架（SCENE_SKELETON > dedup > graph_merge）、audio/dialogue/<V>_dialogue.json、
     vlm/<hash>_desc.json、face_head_fusion/<V>_person_timeline.json、chapter/ 下 rules.prompt + intro.prompt
产物: output/<视频>/chapter/scene_embedded/<V>_skeleton.json、
     chapter/chapter_boundaries/chapter_boundaries_vlm_tokens{sfx}.json

说明: chapter 定稿两件事（大名 2026-08-18）——
① 嵌入骨架：读 dedup scene 骨架 + 上游 vlm 产出 + ASR 台词 + 登场人物，嵌入成参考格式骨架
   {scene_id, scene范围, 登场人物, frames, asr}；
② 调用 API 划章：三段式（intro → rules → 逐 scene 文本）送 ModelScope chat/completions，边界落盘。
边界语义：边界 = 新 chapter 起始 scene 编号。本脚本不再读 vlm/scene_pack.json
（export_scene_pack.py 已删，ASR 直读 audio/dialogue）。
API 配置: 优先环境变量 LLM_API_BASE / LLM_API_KEY（或 MODELSCOPE_API_KEY），否则读本目录
api_key.txt（第 1 行 base_url，第 2 行 key）。LLM_API_MODEL 默认 Qwen/Qwen3.5-122B-A10B
（魔塔大模型）；TOKEN_SEG_PROMPT / INTRO / MAX_F / MAX_ASR_LINES / USE_ASR / OUT_SUFFIX 同旧版。
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request

MAX_NEW = 4096


def load_api():
    """API 配置：优先环境变量，否则读本目录 api_key.txt（base_url 行 + key 行）。"""
    d = os.path.dirname(os.path.abspath(__file__))
    base = os.environ.get("LLM_API_BASE", "").rstrip("/")
    key = (os.environ.get("LLM_API_KEY") or os.environ.get("MODELSCOPE_API_KEY", "")).strip()
    if not base or not key:
        kf = os.path.join(d, "api_key.txt")
        if os.path.isfile(kf):
            lines = [ln.strip() for ln in open(kf, encoding="utf-8").read().splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
            if len(lines) >= 2:
                if not base:
                    base = lines[0].rstrip("/")
                if not key:
                    key = lines[1].strip()
    if not base:
        base = "https://api-inference.modelscope.cn/v1"
    if not key:
        sys.exit("[seg] API key 未设置：LLM_API_KEY / MODELSCOPE_API_KEY / chapter/api_key.txt")
    return base, key


def api_chat(prompt_full):
    """② OpenAI 兼容 chat/completions 调用（国内 API 直连，不走代理）。
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
            print(f"[seg] api {model} attempt={attempt} {time.time()-t0:.0f}s, "
                  f"prompt_tokens={u.get('prompt_tokens')}, "
                  f"completion_tokens={u.get('completion_tokens')}")
            return content
        except Exception as e:
            print(f"[seg] api attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)
    sys.exit("[seg] API 调用 3 次失败")


def load_prompt(n_scenes):
    """① 规则文档：读 chapter/ 下 .prompt 全文，{n}/{n_max} 占位符替换（同旧版拼装）"""
    name = os.environ.get("TOKEN_SEG_PROMPT", "rules.prompt")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.isfile(p):
        sys.exit(f"[seg] 规则文档不存在: {p}")
    txt = open(p, encoding="utf-8").read()
    return (txt.replace("{n_max}", str(n_scenes - 1))
               .replace("{n}", str(n_scenes)))


def load_intro(n_scenes):
    """② 导语+格式说明（三段式第二段，同旧版拼装）"""
    name = os.environ.get("TOKEN_SEG_INTRO", "intro.prompt")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.isfile(p):
        return ""
    txt = open(p, encoding="utf-8").read()
    return (txt.replace("{n_max}", str(n_scenes - 1))
               .replace("{n}", str(n_scenes)))


def load_skeleton(video_out, V):
    """③ 骨架：SCENE_SKELETON > dedup/{V}_skeleton.json > graph_merge/{V}_skeleton.json >
    graph_merge/skeleton.json（2026-08-18：dedup 是 chapter 定稿底座，graph_merge 仅旧路线兜底）"""
    cands = []
    if os.environ.get("SCENE_SKELETON"):
        cands.append(os.environ["SCENE_SKELETON"])
    cands += [
        os.path.join(video_out, "visual", "dedup", f"{V}_skeleton.json"),
        os.path.join(video_out, "visual", "graph_merge", f"{V}_skeleton.json"),
        os.path.join(video_out, "visual", "graph_merge", "skeleton.json"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return json.load(open(p))
    sys.exit(f"[seg] 骨架不存在（SCENE_SKELETON/dedup/graph_merge 均无），先跑骨架")


def load_audio_asr(video_out, V):
    """④ ASR 台词：audio/dialogue/<V>_dialogue.json → {scene_id: [speaker|text, ...]}
    2026-08-19 大名：ASR 骨架 = scene 级 asr（merge_speaker 落盘），scene_id = dedup 列表序，
    台词直接按 scene 取，不再按 shot_range 聚合。"""
    p = os.path.join(video_out, "audio", "dialogue", f"{V}_dialogue.json")
    if not os.path.isfile(p):
        return {}
    scenes = json.load(open(p)).get("scenes", [])
    n = sum(len(sc.get("asr") or []) for sc in scenes)
    print(f"[seg] audio/dialogue: {len(scenes)} scenes / {n} 条台词")
    return {sc["scene_id"]: sc.get("asr") or [] for sc in scenes}


def load_vlm_desc(video_out):
    """⑤ 上游 vlm 产出：vlm/<hash>_desc.json → {帧号: 描述}（帧名 <prefix>_f<num><后缀>：<描述>）"""
    cands = glob.glob(os.path.join(video_out, "vlm", "*_desc.json"))
    if not cands:
        return {}
    d = json.load(open(cands[0]))
    m = {}
    for seg in d.get("segments", []):
        for f in seg.get("frames", []):
            mm = re.match(r"^[0-9a-f]+_f(\d+)(?:.*?[：:]\s*(.*))?$", f)
            if mm:
                m[int(mm.group(1))] = (mm.group(2) or "").strip()
    print(f"[seg] vlm desc.json: {len(m)} 帧有描述")
    return m


def load_person_map(video_out, V):
    """⑥ 登场人物：face_head_fusion/<V>_person_timeline.json → {scene_idx: [pid, ...]}
    （scene_idx = dedup scene 列表序，face_recognition frame_to_scene 同约定）"""
    p = os.path.join(video_out, "visual", "face_head_fusion", f"{V}_person_timeline.json")
    if not os.path.isfile(p):
        return {}
    tl = json.load(open(p)).get("timeline", [])
    m = {}
    for t in tl:
        for iv in t.get("intervals", []):
            for si in range(iv["start_scene"], iv["end_scene"] + 1):
                m.setdefault(si, []).append(t["person_id"])
    print(f"[seg] person_timeline: {len(m)} scenes 有人物")
    return m


def build_transcript(scenes, use_asr=True, shots=None, vlm_desc=None, person_map=None):
    """⑦ 拼装：逐 scene 文本 + 参考格式嵌入骨架（大名 2026-08-18 定稿格式）。

    返回 (transcript 纯文本, embedded 参考格式骨架列表)。"""
    max_f = int(os.environ.get("TOKEN_SEG_MAX_F", "0"))                # 0=全量
    max_asr = int(os.environ.get("TOKEN_SEG_MAX_ASR_LINES", "0"))      # 0=全量
    embedded = []
    parts = []
    for i, sc in enumerate(scenes):
        sid = i                                                        # scene_id = dedup scene 列表序
        sr = sc.get("shot_range") or {}
        st, ed = int(sr.get("start", 0)), int(sr.get("end", 0))
        persons = person_map.get(i, []) if person_map else []
        persons_str = ",".join(f"P{x}" for x in persons) if persons else "无"

        # frames：scene 内帧号 → vlm desc，F+序号 与 intro.prompt「F18=该 scene 第18帧」一致
        fr_lines = []
        if vlm_desc:
            for fn in sc.get("frames", []):
                dsc = vlm_desc.get(int(fn))
                if dsc:
                    fr_lines.append(f"F{len(fr_lines)+1}|{dsc}")
        if max_f > 0:
            fr_lines = fr_lines[:max_f]

        # asr：ASR 骨架已按 scene 聚合（dialogue json scenes[].asr = speaker|text，
        # scene_id = 列表序对齐 sid，先后顺序即落盘序）
        asr_lines = []
        if use_asr and shots:
            for line in shots.get(sid) or []:
                txt = line.split("|", 1)[1] if "|" in line else line
                if txt.strip() and txt.strip() not in ("None", "无"):
                    asr_lines.append(line)
        if max_asr > 0:
            asr_lines = asr_lines[:max_asr]

        embedded.append({
            "scene_id": sid,
            "scene范围": [st, ed],
            "登场人物": [f"P{x}" for x in persons],
            "frames": fr_lines,
            "asr": asr_lines,
        })

        # prompt 文本段
        desc_str = ""
        if fr_lines:
            desc_str = "\n  画面:\n" + "\n".join("    " + ln for ln in fr_lines)
        asr_str = ""
        if asr_lines:
            lines = []
            for line in asr_lines:
                p2 = line.split("|", 1)
                lines.append(f"    {p2[0]}: {p2[1]}" if len(p2) == 2 else f"    {line}")
            asr_str = "\n  对白:\n" + "\n".join(lines)

        if desc_str or asr_str:
            body = f"Scene{sid} [persons: {persons_str}]" + desc_str + asr_str
        else:
            body = f"Scene{sid} [persons: {persons_str}]\n  (None)"
        parts.append(body)
    return "\n\n".join(parts), embedded


def parse_boundaries(response):
    """⑧ 输出解析：只解析格式，零过滤（同旧版拼装）"""
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


def main():
    V = sys.argv[1] if len(sys.argv) > 1 else "video1"
    _out_root = os.environ.get("OUT_ROOT")
    if _out_root:
        video_out = _out_root
    elif len(sys.argv) >= 3:
        video_out = sys.argv[2]
    else:
        video_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "output", V)
    use_asr = os.environ.get("TOKEN_SEG_USE_ASR", "1") != "0"

    sk = load_skeleton(video_out, V)
    scenes = sk["scenes"]
    print(f"[seg] 骨架: scenes={len(scenes)}")

    intro_txt = load_intro(len(scenes))
    prompt_txt = load_prompt(len(scenes))

    shots = load_audio_asr(video_out, V)
    vlm_desc = load_vlm_desc(video_out)
    person_map = load_person_map(video_out, V)
    transcript, embedded = build_transcript(scenes, use_asr, shots, vlm_desc, person_map)

    # ① 嵌入骨架落盘（参考格式，chapter 定稿产物）
    embed_dir = os.path.join(video_out, "chapter", "scene_embedded")
    os.makedirs(embed_dir, exist_ok=True)
    embed_path = os.path.join(embed_dir, f"{V}_skeleton.json")
    json.dump({"video": V, "n_scenes": len(scenes), "scenes": embedded},
              open(embed_path, "w"), ensure_ascii=False, indent=1)
    print(f"[seg] 嵌入骨架 -> {embed_path}")

    prompt_full = intro_txt + "\n\n" + prompt_txt + "\n\n" + transcript
    print(f"[seg] {len(scenes)} scenes, transcript chars: {len(transcript)}, "
          f"intro chars: {len(intro_txt)}, rules chars: {len(prompt_txt)}, use_asr={use_asr}")

    # ② API 推理（无本地模型装载）
    api_base, _ = load_api()
    print(f"[seg] api: {os.environ.get('LLM_API_MODEL', 'Qwen/Qwen3.5-122B-A10B')} @{api_base}")
    response = api_chat(prompt_full)
    print(f"[seg] raw: {response}")
    final = parse_boundaries(response)
    # 2026-08-12：过滤无效边界（0=首块起点无语义，>=n_scenes 越界）——EP02 模型输出过 [0,...]
    final = [b for b in final if 0 < b < len(scenes)]
    sizes = ([final[0] - 1] + [final[i] - final[i - 1] for i in range(1, len(final))]
             + [len(scenes) - final[-1] + 1]) if final else [len(scenes)]
    print(f"[seg] boundaries ({len(final)}): {final}")
    print(f"[seg] sizes: {sizes}")

    out_dir = os.path.join(video_out, "chapter", "chapter_boundaries")
    os.makedirs(out_dir, exist_ok=True)
    sfx = os.environ.get("TOKEN_SEG_OUT_SUFFIX", "")
    out_path = os.path.join(out_dir, f"chapter_boundaries_vlm_tokens{sfx}.json")
    json.dump({"n_scenes": len(scenes), "boundaries": final, "n_chapters": len(final) + 1,
               "method": "prompt_text_api", "pooling": None,
               "tokens_per_scene": [0] * len(scenes),
               "use_asr": use_asr,
               "api_model": os.environ.get("LLM_API_MODEL", "Qwen/Qwen3.5-122B-A10B"),
               "api_base": api_base},
              open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[seg] -> {out_path}")


if __name__ == "__main__":
    main()
