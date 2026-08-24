#!/usr/bin/env python3
"""audio/transcribe.py — Qwen3-ASR-1.7B via vllm 0.27.1 原生批量转录，标点切句。

用法: python3 audio/transcribe.py <vn>
依赖: output/<vn>/audio/wav/audio.wav + speaker/speakers.json（声纹必须先跑）
产物: output/<vn>/audio/transcribe/raw_segments.json（每句一 segment，含 speaker）

流程：
  1. pyannote说话人段（speakers.json）
  2. 相邻同 speaker 碎段合并（270 段里多数 <1s）后整批喂 vllm Qwen3-ASR
  3. 解析 language<asr_text> 输出（自带 52 语种 LID，替代 guess_lang）、按标点切句、删标点
  4. 段内多句按字符插值时间（段短，误差小）；首句用段起点
  5. speaker继承pyannote段

2026-08-20 大名定案：vllm 0.27.1 原生 Qwen3-ASR。
  - qwen-asr 包需 transformers==4.57.6，与 vllm 0.27.1（需 transformers>=5.5.3）同 env 冲突，
    弃用 transformers backend；vllm 0.27.1 内置 qwen3_asr 模型实现（config 已修 thinker_config 顺序），
    零环境改动。
  - 显存实测：加载后 ~5.9G、推理 ~6.0G；5 段共 53s 音频 1.3s（连续批量）。
"""
import json, os, re, shutil, sys, tempfile, time
import numpy as np
import soundfile as sf
import torch
from vllm import LLM, SamplingParams, TokensPrompt

SR = 16000
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


QWEN3_ASR_DIR = _locate_model(("QWEN3_ASR_MODEL_DIR",),
                              "{hf}/hub/Qwen/Qwen3-ASR-1.7B")
# 不含空格：英文句「Oh no. i mean guys」按空格切会碎成单词
PUNCT = r'[，,。！？!?…、；;：:（）()"《》<>【】\[\]]+'

# Qwen3-ASR 输出自带 language 标签（52 语种 LID，官方 2026 定稿）→ 替代 guess_lang 启发式
_LANG_MAP = {
    "chinese": "zh", "cantonese": "yue", "japanese": "ja", "korean": "ko",
    "english": "en", "arabic": "ar", "german": "de", "french": "fr",
    "spanish": "es", "portuguese": "pt", "russian": "ru", "italian": "it",
    "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "turkish": "tr", "hindi": "hi", "dutch": "nl", "swedish": "sv",
    "polish": "pl", "czech": "cs", "filipino": "fil", "persian": "fa",
    "greek": "el", "romanian": "ro", "hungarian": "hu", "ukrainian": "uk",
    "bengali": "bn", "nepali": "ne", "urdu": "ur",
}


def parse_qwen3asr(raw):
    """Qwen3-ASR 输出形如 'language Japanese<asr_text>はい。' → (lang, text)。
    lang 映射失败返回 None（调用方回退 guess_lang）。"""
    raw = (raw or "").strip()
    lang = None
    text = raw
    m = re.match(r"language\s+([A-Za-z]+)", raw)
    if m:
        lang = _LANG_MAP.get(m.group(1).lower())
        text = raw[m.end():]
    m2 = re.match(r"<asr_text>(.*)$", text, re.S)
    if m2:
        text = m2.group(1)
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return lang, text


def split_sentences(text):
    """按标点切句、删标点。"""
    return [p.strip() for p in re.split(PUNCT, text) if p.strip()]


def guess_lang(text):
    """fallback：输出无 language 标签时按字符集启发式判定（假名=ja 优先级最高）"""
    ja = sum(1 for c in text if '぀' <= c <= 'ヿ')
    ko = sum(1 for c in text if '가' <= c <= '힯')
    latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    han = sum(1 for c in text if '一' <= c <= '鿿')
    if ja:
        return "ja"
    if ko:
        return "ko"
    if han >= latin:
        return "zh"
    if latin:
        return "en"
    return "none"


# ── 2026-08-19 大名定稿：语言白名单 = CJK+EN（zh/ja/ko/en）──
# 视频只有中/日（+少量英/韩），Qwen3-ASR 的 52 语种 auto LID 在超短段上会把
# 日语「はい」误判成粤语（係 同音）、偶发越南语/葡语。白名单外的段一律视为误判，
# 用 language hint 自动重听（ja→zh→en→ko 取首个有效文本）。
LANG_WHITELIST = set(os.environ.get("ASR_LANG_WHITELIST", "zh,ja,ko,en").split(","))
HINT_ORDER = os.environ.get("ASR_HINT_ORDER", "ja,zh,en,ko").split(",")
_LANG_NAME = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean", "en": "English"}


def build_hint_prompt(lang):
    """language hint 强制 prompt：与 vllm get_generation_prompt 一致
    （user 音频占位 + assistant 前缀 language {Lang}<asr_text>）。"""
    full = _LANG_NAME.get(lang, lang)
    return (f"<|im_start|>user\n"
            f"<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\nlanguage {full}<asr_text>")


def retry_with_hint(model, wave, sp):
    """白名单外段：依次试 hint 语言，取首个非空文本。返回 (lang, text)。
    实测：hint 强制后模型直接续文本（无 language 前缀），超快（~0.1s/段）。"""
    tok = model.get_tokenizer()
    wav32 = np.asarray(wave, dtype=np.float32)
    for lang in HINT_ORDER:
        ids = tok.encode(build_hint_prompt(lang))
        tp = TokensPrompt(prompt_token_ids=ids, multi_modal_data={"audio": wav32})
        out = model.generate([tp], sampling_params=sp)[0]
        raw = out.outputs[0].text if out.outputs else ""
        l, t = parse_qwen3asr(raw)
        t = t.strip()
        if t:
            return (l or lang), t
    return None, ""


def merge_by_speaker(results, shots=None, fps=None):
    """2026-08-19 大名定稿：按时间顺序聚类（非全局）。同 speaker 且句间间隙
    < ASR_TURN_GAP_MS（默认 2000ms）且簇时长 <= ASR_TURN_MAX_MS（默认 10s）
    且**同 shot**（帧号落同一镜头 range，大名：不跨 shot 合并）→ 并入同一话轮；
    否则新话轮。shots/fps 来自 dedup 骨架，缺省时不做 shot 约束（退化）。"""
    gap_ms = int(os.environ.get("ASR_TURN_GAP_MS", "2000"))
    max_ms = int(os.environ.get("ASR_TURN_MAX_MS", "10000"))

    def shot_id(fn):
        if shots is None or fps is None:
            return None
        for sh in shots:
            r = sh["range"]
            if r["start"] <= fn <= r["end"]:
                return sh["id"]
        return None

    def same_shot(a, b):
        if shots is None or fps is None:
            return True
        fa = int(a["start_ms"] / 1000.0 * fps)
        fb = int(b["start_ms"] / 1000.0 * fps)
        return shot_id(fa) == shot_id(fb)

    out = []
    for s in results:
        if (out and out[-1]["speaker"] == s["speaker"]
                and s["start_ms"] - out[-1]["end_ms"] < gap_ms
                and (s["end_ms"] - out[-1]["start_ms"]) <= max_ms
                and same_shot(out[-1], s)):
            prev = out[-1]
            prev["text"] += s["text"]
            prev["end_ms"] = s["end_ms"]
        else:
            out.append(dict(s))
    return out


def main():
    """Qwen3-ASR 批量转录：读声纹段 → 整批喂 vllm → 切句插值 → 按说话人聚类。"""
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <vn> [<project> <vid_name>]"); sys.exit(1)

    vn = sys.argv[1]
    mode_b = (len(sys.argv) == 4)
    project_name = sys.argv[2] if mode_b else None
    vid_name = sys.argv[3] if mode_b else vn
    sfx = f"_{vid_name}"

    # 2026-08-14 修复：batch_pipeline 以 OUT_ROOT 调用，子脚本需与其布局一致（见 asr_pipeline.py）
    OUT_ROOT = os.environ.get("OUT_ROOT")
    if OUT_ROOT:
        audio_base = os.path.join(OUT_ROOT, "audio")
    elif mode_b:
        audio_base = os.path.join(PROJECT_DIR, "output", project_name, "audio")
    else:
        audio_base = os.path.join(PROJECT_DIR, "output", vn, "audio")
    wav_path   = os.path.join(audio_base, "wav", f"{vid_name}_audio.wav")
    spk_path   = os.path.join(audio_base, "speaker", f"{vid_name}_speakers.json")
    out_dir    = os.path.join(audio_base, "transcribe")
    os.makedirs(out_dir, exist_ok=True)

    # 读 dedup 骨架的 shots（帧范围）→ 合并不跨 shot（2026-08-19 大名定稿）
    base_dir = os.path.dirname(audio_base)
    skel_path = os.path.join(base_dir, "visual", "dedup", f"{vid_name}_skeleton.json")
    shots, fps = None, None
    if os.path.isfile(skel_path):
        skel = json.load(open(skel_path))
        shots = skel.get("shots")
        fps = skel.get("fps")
    else:
        print(f"[transcribe] 警告: 无 dedup 骨架 {skel_path}，合并不限制 shot", flush=True)

    spk_segs = json.load(open(spk_path))
    audio, sr = sf.read(wav_path)
    if sr != SR:
        import torchaudio.functional as AF
        audio = AF.resample(torch.tensor(audio).unsqueeze(0), sr, SR).squeeze().numpy()

    required_model_files = ("config.json", "model.safetensors.index.json")
    missing = [
        name
        for name in required_model_files
        if not os.path.isfile(os.path.join(QWEN3_ASR_DIR, name))
    ]
    if missing:
        print(
            f"[transcribe] Qwen3-ASR local model incomplete: "
            f"{QWEN3_ASR_DIR} missing {missing}"
        )
        sys.exit(1)

    # 音频块临时目录（vllm audio_url 走 file://，allowed_local_media_path 限此目录）
    tmpdir = tempfile.mkdtemp(prefix="qwen3asr_chunks_", dir="/tmp")
    try:
        print(f"[transcribe] loading Qwen3-ASR-1.7B (vllm 0.27.1 原生) from {QWEN3_ASR_DIR}...")
        # 显存实测（2026-08-20 定稿，8G 卡可跑）：util=0.50 + max_num_batched_tokens=2048
        # + seq=64 → 峰值 6799 MiB（省 43%，12288 上曾 11869）；核心是 max_num_batched_tokens
        # 决定 encoder cache budget（默认 8192，压 2048 后 profiling 需求降 4 倍，util 才能压到 0.50）。
        # enforce_eager 免 cudagraph/torch.compile 工作区（省内存）。全部 env 可调。
        model = LLM(
            model=QWEN3_ASR_DIR,
            gpu_memory_utilization=float(os.environ.get("ASR_GPU_UTIL", "0.80")),
            max_model_len=int(os.environ.get("ASR_MAX_MLEN", "2048")),
            max_num_seqs=int(os.environ.get("ASR_MAX_SEQS", "64")),
            max_num_batched_tokens=int(os.environ.get("ASR_MAX_BATCH_TOKENS", "4096")),
            enforce_eager=(os.environ.get("ASR_EAGER", "1") == "1"),
            allowed_local_media_path=tmpdir,
            disable_log_stats=True,
        )
        print(f"[transcribe] {len(spk_segs)} speaker segments")

        t0 = time.time()
        results = []
        sid = 0

        # 相邻同 speaker 碎段先合并（270 段里多数 <1s），再整批送 vllm 连续批量。
        # 输出算法（切句/插值/lang/speaker）全部不动。
        MERGE_MAX_MS = 6000
        merged = []
        for seg in spk_segs:
            if (merged and merged[-1]["speaker"] == seg["speaker"]
                    and (seg["end_ms"] - merged[-1]["start_ms"]) <= MERGE_MAX_MS):
                merged[-1]["end_ms"] = seg["end_ms"]
            else:
                merged.append(dict(seg))
        n_merge = len(spk_segs) - len(merged)
        print(f"[transcribe] {len(spk_segs)} segs → 合并同speaker碎段后 {len(merged)} 段（并 {n_merge}）", flush=True)

        # 收集全部有效段（<300ms 跳过），整批送 vllm
        jobs = []
        for seg in merged:
            ms0, ms1 = seg["start_ms"], seg["end_ms"]
            chunk = audio[int(ms0/1000*SR):int(ms1/1000*SR)]
            if len(chunk) < SR * 0.3:
                continue
            jobs.append((seg, chunk))
        n_jobs = len(jobs)
        print(f"[transcribe] {n_jobs} 有效段 → vllm Qwen3-ASR 批量推理...", flush=True)

        # 写临时 wav（vllm audio_url 需 file:// 路径）→ 整批 chat
        chunks = [c for _, c in jobs]
        paths = []
        for i, c in enumerate(chunks):
            p = os.path.join(tmpdir, f"c{i}.wav")
            sf.write(p, c, SR)
            paths.append(p)
        convos = [
            [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "file://" + p}}]}]
            for p in paths
        ]
        sp = SamplingParams(temperature=0.01, max_tokens=int(os.environ.get("ASR_MAX_TOKENS", "512")))
        res_list = model.chat(convos, sampling_params=sp)
        if len(res_list) != len(jobs):
            print(f"[transcribe] 警告: vllm 返回 {len(res_list)} != 输入 {len(jobs)}，截断对齐", flush=True)

        for i, ((seg, chunk), res) in enumerate(zip(jobs, res_list)):
            ms0, ms1 = seg["start_ms"], seg["end_ms"]
            raw_text = res.outputs[0].text if res and res.outputs else ""
            lang, body = parse_qwen3asr(raw_text)
            if lang is None:
                lang = guess_lang(body)

            # 语言白名单（CJK+EN）：auto LID 出白名单外（yue/pt/vi…）→ 自动 hint 重听
            if lang not in LANG_WHITELIST:
                lang2, body2 = retry_with_hint(model, chunk, sp)
                if body2:
                    lang, body = lang2, body2

            # 按标点切句，段内多句字符插值时间
            parts = split_sentences(body)
            if not parts:
                continue
            dur = ms1 - ms0
            total_chars = sum(len(p) for p in parts)
            cpos = 0
            for part in parts:
                plen = len(part)
                t0i = ms0 + int(dur * cpos / total_chars) if total_chars else ms0
                t1i = ms0 + int(dur * (cpos + plen) / total_chars) if total_chars else ms1
                cpos += plen
                results.append({
                    "id":       sid,
                    "start_ms": t0i,
                    "end_ms":   t1i,
                    "text":     part,
                    "lang":     lang,
                    "speaker":  seg["speaker"],
                })
                sid += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                print(f"[transcribe] {i+1}/{len(jobs)} 段处理 ({time.time()-t0:.0f}s)", flush=True)

        elapsed = time.time() - t0
        print(f"[transcribe] {len(results)} sentences from {len(spk_segs)} segs ({elapsed:.0f}s)")

        # 按时间顺序聚类（大名定稿）：同 speaker + 间隙 + 簇长 + 同 shot
        n_before = len(results)
        results = merge_by_speaker(results, shots=shots, fps=fps)
        print(f"[transcribe] 按说话人合并: {n_before} → {len(results)} 句")

        out_path = os.path.join(out_dir, f"{vid_name}_raw_segments.json")
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
