#!/usr/bin/env python3
"""audio/srt_to_segments.py — 字幕映射模块（与 ASR 无关）。

用法: 模块导入（find_srt / parse_srt），无独立 CLI
依赖: —
产物: —（纯模块，不落盘）

职责: SRT 字幕的查找与解析，供 merge_speaker 嵌入 shot 时直读。
  find_srt(video_path, video_hash) → 三级查找字幕路径:
      1. 视频同名 .srt（<视频路径去扩展名>.srt）
      2. 项目 input/subtitles/<video_hash>.srt（哈希索引，防改名）
      3. 遍历 input/ 目录树找 .srt
         同名可能对不上 → 递归找，视频同名优先，无同名取第一个）
  parse_srt(path) → [{id, start_ms, end_ms, text, lang, speaker}]
      说话人默认 -1，由 merge_speaker 按声纹段时间对齐填充
"""
import os, re

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_srt(video_path, video_hash=""):
    """字幕查找:
    1.视频同名 .srt/.ass  2.input/subtitles/<video_hash>.srt  3.视频同目录内同名。找不到返回 None。"""
    base = os.path.splitext(video_path)[0]
    for ext in (".srt", ".ass"):
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    if video_hash:
        cand2 = os.path.join(PROJECT_DIR, "input", "subtitles", f"{video_hash}.srt")
        if os.path.isfile(cand2):
            return cand2
    # 同目录内找同名（WebDL 每集一目录，srt 与视频同目录但可能为 .ass）
    # 2026-08-13 重装修复：原实现 os.walk(input/) 无同名取第一个 → japan girl 错用 vivant EP02 字幕
    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    if os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.lower().endswith((".srt", ".ass")) and os.path.splitext(f)[0] == video_base:
                return os.path.join(video_dir, f)
    return None

_TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")

def parse_srt(path):
    """SRT → segments 列表 [{id, start_ms, end_ms, text, lang, speaker}]"""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    segs = []
    sid = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.isdigit() and i + 1 < n and "-->" in lines[i + 1]:
            m = _TIME_RE.match(lines[i + 1])
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                t0 = ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1
                t1 = ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2
                j = i + 2
                txt = []
                while j < n and lines[j].strip():
                    txt.append(lines[j].strip())
                    j += 1
                text = " ".join(txt)
                if text:
                    segs.append({
                        "id": sid, "start_ms": t0, "end_ms": t1,
                        "text": text, "lang": "none", "speaker": -1,
                    })
                    sid += 1
                i = j
                continue
        i += 1
    return segs
