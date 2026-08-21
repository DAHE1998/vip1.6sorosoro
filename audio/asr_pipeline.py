#!/usr/bin/env python3
"""audio/asr_pipeline.py — 音频管线编排器: 抽音轨→声纹→转录→场景对话。

用法: python3 audio/asr_pipeline.py <vn>
依赖: output/<vn>/visual/dedup/<vh>_skeleton.json（视觉流水线须先跑到 dedup，每个视频的终点）
产物: output/<vn>/audio/dialogue/<vh>_dialogue.json、<vh>_dialogue_text.txt

流程（顺序关键：transcribe 依赖 speakers.json，声纹必须先跑）：
  Step 1: extract_audio        → wav/audio.wav          (ffmpeg GPU 解码)
  Step 2: speaker_diarize.py   → speaker/speakers.json  (pyannote-3.1 说话人分割)
  Step 2.5: 字幕检测（控制分支）→ 有字幕: 跳过 ASR，merge_speaker 直读 srt
                              → 无字幕: Step 3 照旧
  Step 3: transcribe.py        → transcribe/raw_segments.json (Qwen3-ASR-1.7B 转录+标点切句)
  Step 4: merge_speaker.py     → dialogue/<vh>_dialogue.json (scene 级 asr：台词按 scene
                              聚合，对应 <哈希>_shot_id；2026-08-19 大名)
"""
import json, os, sys, subprocess as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srt_to_segments import find_srt

SR = 16000
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FFMPEG_BIN = os.environ.get("FFMPEG_BIN") or os.path.join(PROJECT_DIR, "ffmpeg-9.0", "ffmpeg")

def extract_audio(video_path, out_wav):
    """视频 → 16kHz mono WAV（项目根 ffmpeg-9.0 标准编译版，纯 CPU 抽音轨）"""
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    sp.run([
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ar", str(SR), "-ac", "1",
        "-c:a", "pcm_s16le", out_wav
    ], check=True)
    print(f"[pipeline] audio extracted: {out_wav}")

def main():
    """编排四步（抽音轨→声纹→转录/字幕→场景对话），支持单视频与 B 路线模式。"""
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <vn> [<project> <vid_name>]")
        sys.exit(1)

    vn = sys.argv[1]
    mode_b = (len(sys.argv) == 4)
    project_name = sys.argv[2] if mode_b else None
    vid_name = sys.argv[3] if mode_b else vn
    sfx = f"_{vid_name}"

    OUT_ROOT = os.environ.get("OUT_ROOT")
    if OUT_ROOT:
        base_dir = OUT_ROOT
    elif mode_b:
        base_dir = os.path.join(PROJECT_DIR, "output", project_name)
    else:
        base_dir = os.path.join(PROJECT_DIR, "output", vn)

    # 读 dedup 骨架（2026-08-19 大名：每个视频的终点就到 dedup；ASR 输出物按哈希
    # 命名对齐，video_hash 透传 dedup。graph_merge/onion 已废弃不进）
    skeleton_path = os.path.join(base_dir, "visual", "dedup", f"{vid_name}_skeleton.json")
    if os.path.isfile(skeleton_path):
        info = json.load(open(skeleton_path))
    else:
        print(f"[pipeline] 找不到 dedup 骨架: {skeleton_path}（先跑视觉流水线至 dedup）")
        sys.exit(1)

    video_path = info.get("video", "")
    if not video_path or not os.path.isfile(video_path):
        print(f"[pipeline] 视频不存在: {video_path}")
        sys.exit(1)

    # 输出路径
    audio_base = os.path.join(base_dir, "audio")
    wav_path   = os.path.join(audio_base, "wav", f"{vid_name}_audio.wav")
    spk_path   = os.path.join(audio_base, "speaker", f"{vid_name}_speakers.json")
    raw_path   = os.path.join(audio_base, "transcribe", f"{vid_name}_raw_segments.json")
    dia_path   = os.path.join(audio_base, "dialogue", f"{vid_name}_dialogue.json")

    here = os.path.dirname(__file__)

    def run_step(cmd):
        sp.run(cmd, check=True)

    # ── Step 1: 抽音轨 ──
    if not os.path.isfile(wav_path):
        extract_audio(video_path, wav_path)
    else:
        print(f"[pipeline] audio exists: {wav_path}")

    # ── Step 2: 声纹分割 (pyannote) ── 必须在转录之前

    # 不能跳过（merge_speaker 按声纹段时间对齐给 srt 句子填 speaker）
    if not os.path.isfile(spk_path):
        print("[pipeline] speaker diarization (pyannote)...")
        if mode_b:
            run_step([sys.executable, os.path.join(here, "speaker_diarize.py"), vid_name, project_name, vid_name])
        else:
            run_step([sys.executable, os.path.join(here, "speaker_diarize.py"), vn])
    else:
        print(f"[pipeline] speakers exists: {spk_path}")

    # ── Step 2.5: 字幕检测（有字幕跳过转录，merge_speaker 直读 srt）──
    # find_srt 遍历 input/ 目录树找字幕
    srt_path = find_srt(video_path, info.get("video_hash", ""))
    if srt_path:
        print(f"[pipeline] srt found: {srt_path} (skip ASR transcribe)")

    # ── Step 3: 转录 (Qwen3-ASR-1.7B，读 speakers.json) ──
    elif not os.path.isfile(raw_path):
        print("[pipeline] transcribing (Qwen3-ASR-1.7B)...")
        if mode_b:
            run_step([sys.executable, os.path.join(here, "transcribe.py"), vid_name, project_name, vid_name])
        else:
            run_step([sys.executable, os.path.join(here, "transcribe.py"), vn])
    else:
        print(f"[pipeline] transcript exists: {raw_path}")

    # ── Step 4: 场景对话 (句子落 shot) ──
    if not os.path.isfile(dia_path):
        print("[pipeline] scene dialogue...")
        if mode_b:
            run_step([sys.executable, os.path.join(here, "merge_speaker.py"), vid_name, project_name, vid_name])
        else:
            run_step([sys.executable, os.path.join(here, "merge_speaker.py"), vn])
    else:
        print(f"[pipeline] dialogue exists: {dia_path}")

    print(f"\n[pipeline] done -> {audio_base}/")
    print(f"  dialogue/{vid_name}_dialogue.json")
    print(f"  dialogue/dialogue_text.txt")

if __name__ == "__main__":
    main()
