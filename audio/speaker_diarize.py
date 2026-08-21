#!/usr/bin/env python3
"""audio/speaker_diarize.py — pyannote speaker-diarization-3.1 全片说话人分割。

用法: python3 audio/speaker_diarize.py <vn>
依赖: output/<vn>/audio/wav/audio.wav（extract_audio 产物）
产物: output/<vn>/audio/speaker/speakers.json（[{id,start_ms,end_ms,speaker}]）
"""
import json, os, sys, time
from pyannote.audio import Pipeline

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


MODEL_PATH = _locate_model(("PYANNOTE_MODEL_DIR",),
                        "glob:{hf}/hub/models--pyannote--speaker-diarization-3.1/snapshots/*")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    """读 wav → pyannote 说话人分割 → 输出每说话人片段 {id,start_ms,end_ms,speaker}。"""
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <vn> [<project> <vid_name>]"); sys.exit(1)

    vn = sys.argv[1]
    mode_b = (len(sys.argv) == 4)
    project_name = sys.argv[2] if mode_b else None
    vid_name = sys.argv[3] if mode_b else vn
    sfx = f"_{vid_name}"

    # 2026-08-14 修复：batch_pipeline 以 OUT_ROOT 调用，asr_pipeline 已支持，
    # 子脚本缺失导致路径错位（wav 在 out_root/audio/wav/ 却读 output/<vn>/）→ 统一优先 OUT_ROOT
    OUT_ROOT = os.environ.get("OUT_ROOT")
    if OUT_ROOT:
        base_dir = OUT_ROOT
    elif mode_b:
        base_dir = os.path.join(PROJECT_DIR, "output", project_name)
    else:
        base_dir = os.path.join(PROJECT_DIR, "output", vn)

    wav_path = os.path.join(base_dir, "audio", "wav", f"{vid_name}_audio.wav")
    out_dir  = os.path.join(base_dir, "audio", "speaker")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[diarize] loading pyannote speaker-diarization-3.1 (local)...")
    # 2026-08-13 重装修复：
    #   - 4.0.5 默认 plda 指向 gated pyannote/speaker-diarization-community-1（联网 403）
    #     → config 注入本地占位 plda（AgglomerativeClustering 推理不碰 plda，加载仅需文件存在）
    #   - dict config 走本地分支，零联网
    import yaml
    plda_local = os.environ.get("PLDA_MODEL_DIR", "/models/hf/plda_local")
    with open(os.path.join(MODEL_PATH, "config.yaml")) as fp:
        cfg = yaml.safe_load(fp)
    cfg.setdefault("pipeline", {}).setdefault("params", {})["plda"] = plda_local
    pipeline = Pipeline.from_pretrained(cfg)
    import torch
    pipeline.to(torch.device("cuda"))

    print(f"[diarize] processing {wav_path} ...")
    t0 = time.time()
    # 2026-08-13 重装修复：torchcodec 0.15 缺系统 FFmpeg 5.x 库（libavutil.so.56），
    # pyannote 内置解码不可用 → 改 torchaudio 读入，按官方回退传 {'waveform':…} dict
    # 2026-08-14 修复：torch 2.13 升级后 torchaudio 2.11 强制走 torchcodec，
    # 系统无 FFmpeg 4-8 库（libavutil.so.56-60）→ RuntimeError。改 soundfile 读入
    # （libsndfile，零 ffmpeg/torchcodec 依赖），格式与 pyannote dict 一致 (1, n) float32
    import soundfile as sf
    wav, sr = sf.read(wav_path, dtype="float32")
    wav = torch.from_numpy(wav)[None]
    diarization = pipeline({"waveform": wav, "sample_rate": sr})
    elapsed = time.time() - t0
    print(f"[diarize] done ({elapsed:.0f}s)")

    # 输出：每个说话人片段 {id, start_ms, end_ms, speaker}
    # pyannote speaker label: "SPEAKER_00", "SPEAKER_01" → 转成 0, 1, 2...
    # 2026-08-13 重装修复：pyannote 4.0.5 推理返回 DiarizeOutput（含 .speaker_diarization
    # 的 Annotation）或 legacy Annotation，两者兼容
    annotation = getattr(diarization, "speaker_diarization", None)
    if annotation is None:
        annotation = diarization
    label_to_int = {}
    results = []
    for i, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        if speaker not in label_to_int:
            label_to_int[speaker] = len(label_to_int)
        results.append({
            "id":       i,
            "start_ms": int(turn.start * 1000),
            "end_ms":   int(turn.end   * 1000),
            "speaker":  label_to_int[speaker],
        })

    out_path = os.path.join(out_dir, f"{vid_name}_speakers.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(r["speaker"] for r in results)
    print(f"[diarize] {len(results)} segments, {len(label_to_int)} speakers: {dict(sorted(dist.items()))}")
    print(f"  -> {out_path}")

if __name__ == "__main__":
    main()
