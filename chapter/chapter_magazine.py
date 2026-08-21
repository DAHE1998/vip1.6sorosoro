#!/usr/bin/env python3
"""chapter/chapter_magazine.py — 杂志风划章编辑器：渲染 scene_asr.json（含帧）+ 拖拽编辑。

用法: python3 chapter/chapter_magazine.py <视频名>
     或 python3 chapter/chapter_magazine.py <视频名> <项目名> <视频输出目录>（B 路线）
依赖: output/chapter/<视频>/scene_asr/scene_asr.json + chapter_boundaries.json、
     output/frames/<视频>/f{帧号}.jpg、visual/dedup/*_skeleton.json、audio/dialogue/<V>_dialogue.json、
     剧情总结_api_qwen35_122b_v2.json、face_head_fusion person_timeline / sweep_records、
     onion_model skeleton、v01_visual_group exp_sweep union_detail_0.40.html
产物: output/chapter/<视频>/editor/scene_asr_editor.html（B 路线 <V>_scene_asr_editor.html）

说明: 功能——展示每个 scene 的帧缩略图 + ASR 台词；按 chapter 分组，横向滚动；支持拖拽
scene 卡片到其他 chapter 并自动重算边界；保存 JSON（POST /api/boundaries/<视频>，
serve_viz.py 写 chapter_boundaries.json，渲染时手工文件优先于 API 默认分块）；输入 JSON
弹窗编辑 boundaries 后应用并保存；划分依据独立按钮：按 chapter 划分（可编辑）/ 按 face id
划分（每个 face 一个 chapter，块内按视频时间线顺序，只读视图；无人脸 scene 归「未分配」章）。
"""
import json, os, sys, base64, re, io
import numpy as np
from PIL import Image

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
V = sys.argv[1] if len(sys.argv) > 1 else "japanese_street_girls"
mode_b = (len(sys.argv) >= 4)
project_name = sys.argv[2] if mode_b else None
video_out_dir = sys.argv[3] if mode_b and len(sys.argv) > 3 else None

def fnv6(name):
    """视频名 FNV-1a 32 位 → 6 位 hex（与 fullres_extract C 版同算法，文件名前缀用）"""
    h = 2166136261
    for b in name.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xffffffff
    return "%06x" % (h & 0xffffff)

def frame_prefix(frames_dir):
    """帧目录实际文件名前缀（2026-08-14：骨架重建后按 fnv6(视频全名) 命名，
    不能 fnv6(V) 目录名算——vivant≠9d5dab。取第一帧文件前缀最稳）"""
    if not os.path.isdir(frames_dir):
        return None
    for fn in sorted(os.listdir(frames_dir)):
        if fn.endswith(".jpg") and "_f" in fn:
            return fn.split("_")[0]
    return None

# B 路线产物 per-video 命名（内部哈希，对齐需求文档 §3.2；BND_FILE/BND_OUT env 仍可覆盖）
BND_NAME = os.environ.get("BND_FILE") or (f"chapter_boundaries_{V}.json" if (mode_b and video_out_dir) else "chapter_boundaries.json")
OUT_NAME = os.environ.get("BND_OUT") or (f"{V}_scene_asr_editor.html" if (mode_b and video_out_dir) else "scene_asr_editor.html")

if mode_b and video_out_dir:
    ASR_PATH = os.path.join(video_out_dir, "audio", "dialogue", f"{V}_dialogue.json")
    BND_PATH = os.path.join(video_out_dir, "chapter", "chapter_boundaries", BND_NAME)
    FRAMES = os.path.join(video_out_dir, "preproc", "frames")
    OUT = os.path.join(video_out_dir, "chapter", "editor", OUT_NAME)
else:
    ASR_PATH = f"{BASE}/output/{V}/audio/dialogue/{V}_dialogue.json"
    BND_PATH = f"{BASE}/output/{V}/chapter/chapter_boundaries/{BND_NAME}"
    FRAMES = f"{BASE}/output/{V}/preproc/frames"
    OUT = f"{BASE}/output/{V}/chapter/editor/{OUT_NAME}"

# 帧文件前缀（2026-08-14：骨架重建后按 fnv6(视频全名) 命名，不能 fnv6(V) 目录名算）
FP = frame_prefix(FRAMES) or fnv6(V)

PALETTE = [
    "#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
    "#3498db","#9b59b6","#e91e63","#00bcd4","#8bc34a",
    "#ff5722","#795548","#607d8b","#ff9800","#4caf50",
    "#2196f3","#673ab7","#f06292","#26c6da","d4e157","ef5350",
]
SPK_COLORS = {"A":"#e74c3c","B":"#3498db","C":"#2ecc71","D":"#f39c12",
              "E":"#9b59b6","F":"#1abc9c","G":"#e67e22","H":"#e91e63","I":"#00bcd4"}

# load：scene 骨架 = dedup（scene_id 按列表序，chapter 边界编号对齐）；ASR = audio/dialogue
# scene 级 asr（2026-08-19 大名：台词按 scene 聚合，先后顺序 = 落盘序，不显示时间）
import glob as _glob
if mode_b and video_out_dir:
    _dedup_glob = _glob.glob(os.path.join(video_out_dir, "visual", "dedup", "*_skeleton.json"))
else:
    _dedup_glob = _glob.glob(f"{BASE}/output/{V}/visual/dedup/*_skeleton.json")
DEDUP_PATH = _dedup_glob[0] if _dedup_glob else (os.path.join(video_out_dir, "visual", "dedup", V + "_skeleton.json") if (mode_b and video_out_dir) else f"{BASE}/output/{V}/visual/dedup/{V}_skeleton.json")
dedup = json.load(open(DEDUP_PATH))

with open(ASR_PATH) as f:
    asr_data = json.load(f)
# 2026-08-19 大名：ASR 骨架 = scene 级 asr（scene_id = dedup 列表序），不读 shot 级台词；
# 台词先后顺序 = 落盘序（scene 内按 shot+首字时间，scene 间按 scene_id）
audio_scenes = {}
for sc in asr_data.get("scenes", []):
    spk_txt = []
    for line in sc.get("asr") or []:
        spk, _, txt = line.partition("|")
        spk_txt.append((spk, txt))
    audio_scenes[sc["scene_id"]] = spk_txt

scenes = dedup["scenes"]
for n, sc in enumerate(scenes):
    sc["scene_id"] = n                       # dedup id 是字符串，编辑器统一按序号
scene_first_frame = {sc["scene_id"]: (sc.get("frames") or [None])[0] for sc in scenes}
# 帧 → scene_id（scene 骨架 frames 反向）→ scene 级 asr（同一 scene 的帧共用该 scene 台词）
frame_scene = {}
for sc in scenes:
    for fn in (sc.get("frames") or []):
        frame_scene[int(fn)] = sc["scene_id"]

if os.path.isfile(BND_PATH):
    bnd = json.load(open(BND_PATH))
else:
    bnd = {}
    print(f"[bound] 无 {BND_PATH}，仅用 API 段落默认边界（BND_FILE 可指 chapter_boundaries_vlm_tokens.json）")

def api_boundaries():
    """默认分块 = Qwen3.5-122B 剧情总结 v2 API 的段落起始 scene；
    手工 chapter_boundaries.json 只在 API 结果缺失时 fallback。
    返回 (boundaries, {chid: 标题}, {chid: 剧情总结})——标题=Chapter 名字、剧情=标题下 AI 总结行"""
    p = os.path.join(video_out_dir, "剧情总结_api_qwen35_122b_v2.json") if (mode_b and video_out_dir) else os.path.join(BASE, "output", V, "chapter", "剧情总结_api_qwen35_122b_v2.json")
    if not os.path.isfile(p):
        return [], {}, {}
    try:
        s = json.load(open(p))
        raw = s.get("summary_raw")
        if isinstance(raw, str):
            raw = json.loads(raw)
        paras = (raw or {}).get("段落", []) if isinstance(raw, dict) else (raw or [])
        bs, titles, plots = [], {}, {}
        for i, para in enumerate(paras):
            if i >= 1:  # 第一段起点=Scene0，不构成边界
                m = re.match(r"Scene(\d+)", str(para.get("范围", "")))
                if m:
                    bs.append(int(m.group(1)))
            titles[i] = str(para.get("标题", "")).strip()
            plots[i] = str(para.get("剧情", "")).strip()
        return sorted(set(bs)), titles, plots
    except Exception:
        return [], {}, {}

# 边界来源：手工 chapter_boundaries.json（页面「保存 JSON」）优先；
# 初始值已同步为 API 段落边界，未保存过即等价于「默认用 API 的结果」。
# 标题/剧情总结仍始终来自 API 段落。
boundaries, chapter_titles, chapter_plots = api_boundaries()
if bnd.get("boundaries"):
    boundaries = sorted(bnd["boundaries"])
    print("[bound] 使用手工 chapter_boundaries.json（页面保存的边界优先于 API 默认）:", boundaries)
else:
    print("[bound] 使用 API 段落默认边界:", boundaries)

def get_chapter_id(scene_id):
    return sum(1 for x in boundaries if x <= scene_id)

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ═══ 下拉视频列表（需求 §七：下拉选择视频，按视频名字排序，不是哈希）═══
def vi_video_list():
    """扫描可视范围全部 dedup 骨架 → [(hash, 显示名, 页面 URL), ...]。
    显示名 = skeleton.video basename 去扩展名（改文件名重跑自动跟变），fallback video_id，按显示名排序；
    URL 顶层 = relpath(BASE/output) 第一段（A=视频名目录，B=项目名目录）；顶层==显示名（A 型）
    → scene_asr_editor.html，否则（B 型）→ {hash}_scene_asr_editor.html"""
    if mode_b and video_out_dir:
        g = _glob.glob(os.path.join(video_out_dir, "visual", "dedup", "*_skeleton.json"))
    else:
        g = _glob.glob(f"{BASE}/output/*/visual/dedup/*_skeleton.json")
    out_rel = os.path.join(BASE, "output")
    items = []
    for p in sorted(g):
        h = os.path.basename(p).rsplit("_skeleton.json", 1)[0]
        try:
            sk = json.load(open(p))
        except Exception:
            continue
        disp = os.path.basename(sk.get("video") or "").rsplit(".", 1)[0] or sk.get("video_id") or h
        top = os.path.relpath(p, out_rel).replace(os.sep, "/").split("/")[0]
        fn = "scene_asr_editor.html" if top == disp else f"{h}_scene_asr_editor.html"
        items.append((h, disp, "/" + top + "/chapter/editor/" + fn))
    return items

cur_hash = os.path.basename(DEDUP_PATH).rsplit("_skeleton.json", 1)[0]
cur_disp = esc(V)  # 主标题显示视频名（内部用 V=hash，显示映射具体文件名）
vi_opts = []
for _h, _disp, _url in sorted(vi_video_list(), key=lambda x: x[1]):
    _sel = " selected" if _h == cur_hash else ""
    if _h == cur_hash:
        cur_disp = esc(_disp)
    vi_opts.append('<option value="' + esc(_url) + '"' + _sel + '>' + esc(_disp) + '</option>')
vi_select = ('<select class="btn" id="viVideoSel" title="切换视频" onchange="location.href=this.value">'
             + "".join(vi_opts) + '</select>') if len(vi_opts) > 1 else ""
# saveJSON 保存路径：B 路线两段（/api/boundaries/<项目名>/<hash>），A 路线保持单段（现状）
save_path = ("/api/boundaries/" + project_name + "/" + V) if (mode_b and video_out_dir) else ("/api/boundaries/" + V)

# 帧集 = graph_merge scene 的 frames 全部展开（每帧独立卡片，Scene 号重复使用）；
# ASR = 帧所在 shot 的 shot 级 dialogue；色条在顶部。
# dedup 帧集仍计算（face 模式 legend/h1 用），模式一卡片走 chapter_scenes。
# 2026-08-14：dedup 骨架按 fnv6 前缀匹配（同 select_cases.load_skeleton），不再用 {V}_skeleton.json
# （dedup 已在头部载入为 scenes 源，DEDUP_PATH 头部已定）

# face 模式用：dedup 全帧集（scenes.frames ∪ shots.key_frames）去重升序
all_frames = set()
for sc in dedup["scenes"]:
    all_frames.update(sc.get("frames") or [])
for sh in dedup.get("shots", []):
    all_frames.update(sh.get("key_frames") or [])
all_frames = sorted(all_frames)

# 模式一（chapter 视图）：graph_merge scene 按 chapter 边界分组（边界 scene 号 → chapter id）
chapter_scenes = {}
for sc in scenes:
    chid = get_chapter_id(sc["scene_id"])
    chapter_scenes.setdefault(chid, []).append(sc)

# scene → 人物链（scene-gap 圆点左端标 P{n} 用；洋葱 persons 已随 graph_merge 退场）
_person_tl = {}
_pt_path = os.path.join((video_out_dir if mode_b and video_out_dir else f"{BASE}/output/{V}"),
                        "visual", "face_head_fusion", f"{V}_person_timeline.json")
if os.path.isfile(_pt_path):
    for _t in json.load(open(_pt_path)).get("timeline", []):
        for _iv in _t.get("intervals", []):
            for _si in range(_iv["start_scene"], _iv["end_scene"] + 1):
                _person_tl.setdefault(_si, []).append(f"P{_t['person_id']}")
scene_persons = {sc["scene_id"]: _person_tl.get(sc["scene_id"], []) for sc in scenes}

# 人物链连续边（scene-gap 圆点虚线不插）：2026-08-19 大名定稿——
# 人物链 = 洋葱 proto_scenes（按人物链切点切分）；相邻 proto 共享人物 → 同一链延续。
# 链段内部连续 scene 不插虚线；fragment（无 persons）段照常虚线。
# 小脸过滤 = face_recognition dyn_thr（上游）、孤儿过滤 = onion MIN_PERSON_SCENES=3，链上已含。
_chain_edges = set()
_onion_path = os.path.join((video_out_dir if mode_b and video_out_dir else f"{BASE}/output/{V}"),
                           "visual", "onion_model", f"{V}_skeleton.json")
if os.path.isfile(_onion_path):
    _onion = json.load(open(_onion_path))
    _prev_end, _prev_pset = None, None
    for _ps in _onion.get("proto_scenes", []):
        _sr = _ps.get("scene_range") or {}
        _pset = set(_ps.get("persons") or [])
        if not _pset:
            _prev_end, _prev_pset = None, None
            continue
        for _si in range(int(_sr["start"]), int(_sr["end"])):
            _chain_edges.add((_si, _si + 1))                # proto 内部连续边
        if _prev_end is not None and _prev_pset & _pset:
            _chain_edges.add((_prev_end, int(_sr["start"])))  # 跨 proto 共享人物 → 边界边
        _prev_end, _prev_pset = int(_sr["end"]), _pset
    print(f"[chain] 洋葱 {len(_onion.get('proto_scenes', []))} proto → 人物链内连续边 {len(_chain_edges)} 条")

print(f"Building: {len(scenes)} scenes, {len(chapter_scenes)} chapters (dedup scene 卡 + audio/dialogue ASR, 无 VLM)")

# 视频信息块
vi_dur = ""
vi_fps = ""
try:
    _fps = float(asr_data.get("fps") or 0)
    _tf = int(asr_data.get("total_frames") or 0)
    if _fps > 0 and _tf > 0:
        _s = int(round(_tf / _fps))
        vi_dur = "%d:%02d" % (_s // 60, _s % 60)
    if _fps > 0:
        vi_fps = ("%.2f" % _fps).rstrip("0").rstrip(".")
except Exception:
    pass
video_info_html = (
    '<div class="video-info">'
    '<div class="vi-title">' + cur_disp + '</div>'
    + vi_select +
    '<span class="vi-item">时长 ' + vi_dur + '</span>'
    '<span class="vi-item">' + str(asr_data.get("width", "?")) + "×" + str(asr_data.get("height", "?")) + '</span>'
    '<span class="vi-item">' + str(asr_data.get("total_frames", 0)) + ' 帧</span>'
    '<span class="vi-item">' + vi_fps + ' fps</span>'
    '<span class="vi-item">' + str(len(scenes)) + ' scenes</span>'
    '<span class="vi-item" id="viChapters">' + str(len(chapter_scenes)) + ' chapters</span>'
    '</div>'
)

# frame loader（V16 hash 命名：{fnv6(V)}_f{fn}.jpg；fnv6 定义见文件头）

def load_b64(fn):
    """帧引用路径：改回服务器静态文件引用——serve_viz 直接 serve output/，
    src=/<视频名>/preproc/frames/{FP}_f{fn}.jpg 浏览器按需加载，HTML 只有几百 KB"""
    p = os.path.join(FRAMES, f"{FP}_f" + str(fn) + ".jpg")
    if not os.path.exists(p):
        return None
    return "/" + os.path.relpath(p, os.path.join(BASE, "output")).replace(os.sep, "/")

def frame_size(fn):
    """读 JPEG SOF 段拿实际帧尺寸（零依赖，不加载整图）。
    检测坐标系是 1620x1080（sweep/detail），实际帧图可能 1440x960——头像裁剪必须按实际尺寸换算"""
    p = os.path.join(FRAMES, f"{FP}_f" + str(fn) + ".jpg")
    if not os.path.exists(p):
        return None
    data = open(p, "rb").read()
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2):  # SOF0/1/2
            return (int.from_bytes(data[i + 7:i + 9], "big"),
                    int.from_bytes(data[i + 5:i + 7], "big"))  # (width, height)
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
        else:
            seg = int.from_bytes(data[i + 2:i + 4], "big")
            i += 2 + seg
    return None

# ===== 卡片渲染（boundary 模式）=====
def frame_card(fn, sid, color):
    """单帧卡片：顶部色条（左 Scene 号、右帧号）+ 静止帧图。
    模式一帧直嵌 base64，face 模式从 #chapters-boundary 复用；Scene 号重复使用（同 scene 每帧标同号）。"""
    url = load_b64(fn) if fn is not None else None
    thumb = (
        '<img class="shot-img" data-fn="' + str(fn) + '"' + ((' src="' + url + '"') if url else '') + '>'
    ) if fn is not None else '<div class="shot-img" style="background:#1a1a1a"></div>'
    return (
        # data-scene-id：JS 边界增删/重建按 scene 定位
        '<div class="card" draggable="true" data-frame="' + str(fn) + '" data-scene-id="' + str(sid) + '">'
        '<div class="card-bar" style="background:' + color + '">'
        '<span>Scene ' + str(sid) + '</span>'
        '<span class="card-bar-frame">f' + (str(fn) if fn is not None else "?") + '</span>'
        '</div><div class="shot-wrap">' + thumb + '</div></div>'
    )

def asr_note(fn):
    """帧卡片的 ASR 图注：帧 → scene 反向 → 显示该 scene 的台词行，不做兜底；
    台词只有先后顺序要讲究（2026-08-19 大名：不显示时间）"""
    asr_parts = []
    for spk, text in audio_scenes.get(frame_scene.get(fn)) or []:
        if text == "无":
            asr_parts.append(("", "(无)"))
            continue
        if spk:
            asr_parts.append((spk, text))
        else:
            asr_parts.append(("", text))

    lines_html = []
    for spk, text in asr_parts:
        if spk:
            colored = '<span style="color:%s;font-weight:600">%s</span>' % (SPK_COLORS.get(spk, "#ccc"), spk)
            lines_html.append('<div class="asr-line">' + colored + ' ' + text + '</div>')
        else:
            lines_html.append('<div class="asr-line" style="color:#444">' + text + '</div>')
    asr_html = "".join(lines_html) if asr_parts else '<div class="asr-line" style="color:#444">(无)</div>'
    return '<div class="asr-note">' + asr_html + '</div>'

# ===== 模式一：按 chapter 边界划分 =====
cards_html = []
prev_chid = -1

for chid in sorted(chapter_scenes):
    color = PALETTE[chid % len(PALETTE)]

    # chapter header
    if chid != prev_chid:
        if prev_chid != -1:
            cards_html.append('</div></div>')
        n_frames = sum(len(sc.get("frame_files") or sc.get("frames") or []) for sc in chapter_scenes[chid])
        bound_mark = ' <span style="color:#f1c40f;font-size:11px">&#9733;</span>' if chid > 0 else ''
        bnd_range = ('Scene' + str(boundaries[chid - 1]) + '~' + str(boundaries[chid] - 1)) if 0 < chid < len(boundaries) else ''
        ch_name = esc(chapter_titles.get(chid, "")) if chapter_titles.get(chid, "").strip() else 'CHAPTER ' + str(chid)
        ch_plot = esc(chapter_plots.get(chid, ""))
        cards_html.append(
            '<div class="ch" data-chapter-id="' + str(chid) + '">'
            '<div class="ch-hdr" style="background:' + color + '18">'
            '<div class="ch-hdr-top">'
            '<span style="color:' + color + ';font-weight:700;font-size:13px">' + ch_name + bound_mark + '</span>'
            '<span class="ch-size" style="color:#666;font-size:11px;margin-left:8px">' + str(n_frames) + ' 帧</span>'
            '<span class="ch-boundary" style="color:#888;font-size:10px;margin-left:8px">' + bnd_range + '</span>'
            '</div>'
            + (('<div class="ch-summary">' + ch_plot + '</div>') if ch_plot else '')
            + '</div><div class="ch-row" data-chapter-id="' + str(chid) + '">'
        )
        prev_chid = chid

    # 帧展开（保持 scene/帧时间序）→ 均分 4 列（杂志式：竖线+卡片+ASR 自由文本）
    frame_entries = []
    for sc in chapter_scenes[chid]:
        sid = sc["scene_id"]
        for fn in (sc.get("frame_files") or sc.get("frames") or []):
            frame_entries.append((fn, sid))
    # 均匀分摊 4 列：每列 base 个，余数依次分给前列（列间差最多 1，无突出列）
    cols = []
    n = len(frame_entries)
    base, extra = divmod(n, 4)
    start = 0
    for k in range(4):
        size = base + (1 if k < extra else 0)
        cols.append(frame_entries[start:start + size])
        start += size
    col_html = []
    for c in cols:
        inner = '<div class="vline" style="background:' + color + '"></div>'
        prev_sid = -1
        for fn, sid in c:
            # scene 切换（且不在人物链内）→ 插圆点虚线；同一 scene 的相邻帧不插
            # 有 P{n} 时圆点左端标 P{n}（P 与虚线剧中对齐）
            if prev_sid != -1 and sid != prev_sid and (prev_sid, sid) not in _chain_edges:
                ps = scene_persons.get(sid) or []
                p_tag = ('<span class="gap-p" style="color:' + color + '">'
                         + " ".join(ps) + '</span>') if ps else ''
                inner += ('<div class="scene-gap">' + p_tag
                          + '<span class="gap-dots" style="background-image:radial-gradient(circle,' + color + '55 1px,transparent 1px);background-size:6px 3px;background-repeat:repeat-x"></span>'
                          + '</div>')
            inner += frame_card(fn, sid, color)
            inner += asr_note(fn)
            prev_sid = sid
        col_html.append('<div class="mag-col">' + inner + '</div>')
    cards_html.append("".join(col_html))

cards_html.append('</div></div>')

# ===== 模式二：按 face id 划分 =====
# graph_merge 洋葱语义的 persons 次数失真（P20 44->16）；face id 视图直接以
# dedup skeleton（175 scenes）+ 匹配明细 union_detail 链归属划分，
# 卡片 = 帧在上 + ASR 在下，无 VLM（分个人脸进 vlm 没意思）
# 归属数据 = union_detail_0.40.html（thr=0.40 canonical 匹配链，逐帧精确到 SC n）。

# 进人物块（64/95/125/126/148 等 20 个假脸），「没人就没人」——只用真检测到脸的 scene
# 模式一渲染的帧号集合（face 图片复用来源：fillFaceImages 从 #chapters-boundary 取 base64）
gm_frame_set = set()
for sc in scenes:
    gm_frame_set.update(sc.get("frame_files") or sc.get("frames") or [])

# DEDUP_PATH 已在头部定义（chapter 视图同源），此处直接用

face_scenes = {}
scene_full_frames = {}  # scene_id -> 该 scene shot_range 内全部 shots 的 key_frames（259 全覆盖）
avatar_map = {}  # pid -> (scene_id, frame, bbox) 链内 DINO 最稳定帧（顶部人物栏头像用）
if os.path.isfile(DEDUP_PATH):
    dedup_all = json.load(open(DEDUP_PATH))
    dedup_scenes = dedup_all["scenes"]
    for n, sc in enumerate(dedup_scenes):
        sc["scene_id"] = n  # dedup 用 id 字符串，face 卡片标注统一按序号
    # 帧集展开：dedup scenes.frames 只留每 scene 的 key_frame（180 帧，被合并吞掉的
    # 帧缺失）。face 视图每 scene 的帧 = shot_range 内全部 shots 的 key_frames，
    # 259 帧全覆盖，与 preproc/frames 的帧文件一一对应
    dedup_shots = dedup_all.get("shots", [])
    for sc in dedup_scenes:
        r = sc["shot_range"]
        frames = []
        for s in range(r["start"], r["end"] + 1):
            if 0 <= s < len(dedup_shots):
                frames.extend(dedup_shots[s].get("key_frames") or [])
        scene_full_frames[sc["scene_id"]] = sorted(set(frames))
    # sweep_records：每 scene top-1 脸（真实像素 bbox，1440x960 原图坐标系；
    # skeleton 元数据 1620x1080 是假的勿用；face_emb = insightface 512 维嵌入）。
    # 同 scene 多条记录取 det_score 最高
    SWEEP_PATH = os.path.join(
        (video_out_dir if mode_b and video_out_dir else f"{BASE}/output/{V}"),
        "visual", "face_head_fusion", "sweep_records.json")
    # 每 scene 保留全部检出脸（按 det_score 降序）。只留 top-1 会吞掉同框的另一人，
    # 导致同框两链（如 P17/P18）的候选池相同 → 头像选到同一张脸（共脸 bug）
    sweep_best = {}
    if os.path.isfile(SWEEP_PATH):
        for r in json.load(open(SWEEP_PATH)).get("recs", []):
            if not (r.get("has_face") and r.get("face_bbox")):
                continue
            sweep_best.setdefault(r["scene_id"], []).append(
                (r.get("det_score", 0), r["frame"], r["face_bbox"], r.get("face_emb")))
        for k in sweep_best:
            sweep_best[k].sort(key=lambda x: -x[0])

    def pick_avatar(slist):
        """链内最好的一帧：候选 = 链内带脸 scene 的 sweep top-1 帧；
        最好 = face_emb 与链均值 cos 最大者（最典型一帧）；
        退化：嵌入不可用 → 链内第一个带脸帧；仍无 → None"""
        cands = []
        for sc in slist:
            for r in sweep_best.get(sc["scene_id"], []):
                if r[3]:
                    cands.append((sc["scene_id"], r[1], r[2], r[3]))
        if not cands:
            for sc in slist:
                faces = sweep_best.get(sc["scene_id"], [])
                if faces:
                    return (sc["scene_id"], faces[0][1], faces[0][2])
            return None
        if len(cands) > 1:
            vecs = np.array([c[3] for c in cands], dtype=np.float32)
            mean = vecs.mean(axis=0)
            mnorm = np.linalg.norm(mean)
            sims = [float(v @ mean / (np.linalg.norm(v) * mnorm)) for v in vecs]
            best = int(np.argmax(sims))
        else:
            best = 0
        c = cands[best]
        return (c[0], c[1], c[2])

    # face 归属 = union_detail 链（thr=0.40 canonical，逐帧 SC 精确归属）。
    # 只有真正检测到脸且匹配进该人物链的 scene 才进块；无脸 scene 一律未分配
    DETAIL_PATH = os.path.join(
        (video_out_dir if mode_b and video_out_dir else f"{BASE}/output/{V}"),
        "visual", "v01_visual_group", "exp_sweep", "union_detail_0.40.html")
    covered = set()
    if os.path.isfile(DETAIL_PATH):
        detail = open(DETAIL_PATH, encoding="utf-8").read()
        for chain in detail.split('<div class=chain>')[1:]:
            m = re.search(r'class=pid[^>]*>P(\d+)<', chain)
            if not m:
                continue
            pid = int(m.group(1))
            slist = []
            for n in sorted(set(int(x) for x in re.findall(r'SC(\d+) f\d+', chain))):
                if 0 <= n < len(dedup_scenes):
                    slist.append(dedup_scenes[n])
                    covered.add(n)
            if slist:
                face_scenes[pid] = slist
                avatar_map[pid] = pick_avatar(slist)
    unassigned_scenes = [dedup_scenes[n] for n in range(len(dedup_scenes)) if n not in covered]
else:
    # dedup skeleton 缺失 fallback：退回原 persons 分组逻辑
    for sc in scenes:
        for p in sc.get("persons", []):
            face_scenes.setdefault(p, []).append(sc)
    unassigned_scenes = [sc for sc in scenes if not sc.get("persons", [])]
# 块顺序 = 出现次数从多到少
face_order = sorted(face_scenes, key=lambda p: len(face_scenes[p]), reverse=True)

def face_chapter_html(title, slist, color):
    # 一帧一卡：每张卡片 = 一帧，卡头标该帧所属 Scene N（同一 scene 多帧 = 多卡，各标一次）。
    # 帧集 = scene_full_frames 展开表（shot_range 内全部 shots 的 key_frames，259 全覆盖，
    # 含被 dedup 合并吞掉的帧）；fallback 用 dedup scenes.frames
    # gm_frame_set 内的帧留 data-fn 由 JS 复用 base64，缺失帧直嵌，保证不漏图
    cards = []
    for sc in slist:
        sid = sc["scene_id"]
        for fn in scene_full_frames.get(sid) or sc.get("frames") or []:
            v = ('<img class="shot-img" data-fn="' + str(fn) + '"'
                 + ((' src="' + load_b64(fn) + '"') if fn not in gm_frame_set and load_b64(fn) else '') + '>')
            cards.append(
                '<div class="fcard">'
                '<div class="fcard-hdr">Scene ' + str(sid) + '</div>'
                '<div class="fcard-body">' + v + '</div>'
                '</div>'
            )
    return (
        '<div class="ch" data-face-id="' + title + '">'
        '<div class="ch-hdr" style="background:' + color + '18">'
        '<span style="color:' + color + ';font-weight:700;font-size:13px">' + title + '</span>'
        '<span class="ch-size" style="color:#666;font-size:11px;margin-left:8px">' + str(len(slist)) + ' scenes</span>'
        '</div><div class="ch-row">' + "".join(cards) + '</div></div>'
    )

def avatar_crop(fn, bbox):
    """最终头像四步：判长边短边 → 短边扩至长边（中心对齐，不拉伸）→ 放大 1.3 倍（中心锚定 1:1）
    → 越界平移截图 88x88。bbox = sweep_records 真实像素坐标（1440x960；skeleton 1620x1080 是假的勿用）"""
    p = os.path.join(FRAMES, f"{FP}_f" + str(fn) + ".jpg")
    if not os.path.isfile(p):
        return None
    img = Image.open(p)
    W, H = img.size
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bw, bh = bx2 - bx1, by2 - by1
    # 第一步：判长边短边；第二步：短边扩至长边（中心对齐）
    if bw >= bh:
        x1, x2 = bx1, bx2
        pad = (bw - bh) / 2.0
        y1, y2 = by1 - pad, by2 + pad
    else:
        y1, y2 = by1, by2
        pad = (bh - bw) / 2.0
        x1, x2 = bx1 - pad, bx2 + pad
    # 第三步：按比例放大 1.3 倍（中心锚定，保持 1:1）
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = (x2 - x1) * 1.3
    x1, y1 = cx - side / 2.0, cy - side / 2.0
    x2, y2 = x1 + side, y1 + side
    # 第四步：越界平移（保持 1:1，不拉伸）
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > W:
        x1 -= x2 - W
        x2 = W
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > H:
        y1 -= y2 - H
        y2 = H
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    crop = img.crop((x1, y1, x2, y2)).resize((88, 88), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def people_bar_html(avatar_map):
    """顶部人物栏：每人物一小卡 = 左头像 + 右名字（可改，localStorage 持久化）。
    头像 = avatar_crop 四步裁剪，bbox = avatar_map 里 DINO 最稳定帧的 sweep 真实像素坐标（1440x960）"""
    cards = []
    for pid in face_order:
        av = avatar_map.get(pid)
        if not av:
            continue
        sid, fn, bbox = av
        b64 = avatar_crop(fn, bbox)
        img = ('<img src="data:image/jpeg;base64,' + b64 + '">') if b64 else '<div class="pav-miss">?</div>'
        cards.append(
            '<div class="pcard" title="P' + str(pid) + '">'
            '<div class="pavatar">' + img + '</div>'
            '<input class="pname" data-v="' + V + '" data-pid="' + str(pid) + '" value="P' + str(pid) + '" maxlength="12" spellcheck="false">'
            '</div>'
        )
    return '<div class="people-bar">' + "".join(cards) + '</div>'

face_cards_html = [people_bar_html(avatar_map)]
for i, p in enumerate(face_order):
    face_cards_html.append(face_chapter_html(
        "P" + str(p), face_scenes[p], PALETTE[i % len(PALETTE)]
    ))
if unassigned_scenes:
    face_cards_html.append(face_chapter_html("未分配", unassigned_scenes, "#888"))

HTML = """<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>
<title>Scene Editor - """ + cur_disp + """</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;padding:16px}
h1{font-size:15px;margin-bottom:4px}
.sub{font-size:11px;color:#555;margin-bottom:8px}
.toolbar{display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
.btn{padding:6px 14px;border:1px solid #444;background:#1a1a2e;color:#ddd;border-radius:4px;cursor:pointer;font-size:12px}
.btn:hover{background:#2a2a4e}
select.btn{padding:6px 8px;appearance:auto}
.btn.active{background:none;border:1px solid #444;border-left:4px solid #3498db;color:#fff;font-weight:600;padding:8px 18px;font-size:13px}
.btn.danger{background:#e74c3c;border-color:#e74c3c;color:#fff}
.btn.danger:hover{background:#c0392b}
.info{font-size:11px;color:#888;margin-left:auto}
.ch{margin-bottom:14px;background:#0d0d14;border-radius:6px;overflow:hidden;transition:opacity 0.2s}
.ch-hdr{padding:5px 12px;display:flex;align-items:center;flex-wrap:wrap;gap:4px}
/* 仅 chapter 视图——face 模式保持单行不变 */
#chapters-boundary .ch-hdr{padding:10px 14px;display:block}
#chapters-boundary .ch-hdr-top{display:flex;align-items:center;flex-wrap:wrap;gap:4px}
#chapters-boundary .ch-summary{margin-top:6px;font-size:12px;line-height:1.6;color:#aaa}
.ch-id{font-weight:700;font-size:13px}
.ch-size{font-size:10px;color:#666}
.ch-actions{display:flex;gap:4px;margin-left:auto}
.ch-actions button{padding:2px 8px;font-size:10px;border:1px solid #444;background:#1a1a2e;color:#ddd;border-radius:3px;cursor:pointer}
.ch-actions button:hover{background:#2a2a4e}
.ch-actions button.del{color:#e74c3c;border-color:#e74c3c}
.ch-actions button.del:hover{background:#e74c3c;color:#fff}
.ch-row{display:flex;gap:8px;overflow-x:auto;padding:8px;scroll-snap-type:x proximity;min-height:200px;transition:background 0.2s}
.ch-row.drag-over{background:#1a2a3a !important;outline:2px dashed #3498db;outline-offset:-2px}
.ch-row::-webkit-scrollbar{height:10px}
.ch-row::-webkit-scrollbar-thumb{background:#444;border-radius:5px}
.ch-row::-webkit-scrollbar-track{background:#1a1a1a}
.card{flex:0 0 300px;scroll-snap-align:start;background:#161622;border:1px solid #333;border-radius:6px;padding:8px;cursor:grab;transition:transform 0.15s,box-shadow 0.15s,opacity 0.15s}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.5)}
/* 剧情 chapter 视图：开四列，每列=竖线+帧卡片+自由 ASR 文本。
   列内顺序（从左到右）：竖线 1.5单位空白 卡片 1.5单位空白 ASR文本 4单位空白 → 下一竖线（1单位=10px）；
   卡片=帧图，左上角 Scene 号、右上角 F 编码；ASR 不被卡片约束（像图注介绍卡片）；
   只作用于 #chapters-boundary，face 模式不受影响 */
/* 列数 = JS colCount()（视口宽/370，4K 自动 5~8 列）；auto-fit 让 mag-col 一行排满 */
#chapters-boundary .ch-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));column-gap:40px;overflow:visible;min-height:0;align-items:start}
#chapters-boundary .ch{margin-bottom:32px}
#chapters-boundary .mag-col{position:relative;display:grid;grid-template-columns:1px 15px 1fr 15px 1fr;row-gap:15px;align-items:start;min-height:48px}
/* 竖线 absolute 贯穿整列（grid-row:1/-1 对隐式行无效，高度会塌成 0） */
#chapters-boundary .mag-col .vline{position:absolute;left:0;top:0;bottom:0;width:1px;border-radius:1px}
#chapters-boundary .mag-col .card{grid-column:3;flex:none;width:auto;padding:0;background:#161622;border-radius:4px;overflow:hidden}
/* 卡片顶部色条：左 Scene 号、右帧号，帧图保持原始比例 */
#chapters-boundary .card-bar{display:flex;justify-content:space-between;align-items:center;padding:2px 8px;font-size:11px;font-weight:700;color:#111;line-height:1.5;white-space:nowrap}
#chapters-boundary .card-bar-frame{color:#ffffffb3;font-size:10px;font-weight:600}
#chapters-boundary .mag-col .asr-note{grid-column:5}
/* 上下 scene 分界圆点虚线：chapter 主题色半透明、圆点半径1.5px 中心距6px，
   radial-gradient 实现，居中于两卡片之间 */
#chapters-boundary .scene-gap{display:flex;align-items:center;gap:6px;grid-column:3/-1;height:14px}
#chapters-boundary .scene-gap .gap-p{font-size:10px;font-weight:600;white-space:nowrap;flex:none}
#chapters-boundary .scene-gap .gap-dots{flex:1;height:3px}
#chapters-boundary .shot-wrap{position:relative}
/* 帧图顶部直角 */
#chapters-boundary .shot-img{border-radius:0 0 3px 3px}
#chapters-boundary .asr-note{font-size:12px;line-height:1.6;color:#ccc}
#chapters-boundary .asr-note .asr-line{margin-bottom:2px;word-break:break-word}
/* 关键修复：.gap 分隔条（拖拽分章手柄）也占 grid cell，会破坏列布局；
   chapter 视图隐藏 gap（face 模式不受影响，仍保留拖拽手柄） */
#chapters-boundary .gap{display:none}
.card.dragging{opacity:0.4;transform:scale(0.95)}
.card-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.card-label{font-size:13px;font-weight:800}
.card-meta{font-size:9px;color:#555}
.card-body{display:flex;gap:8px}
.shots-col{flex:0 0 40%;display:flex;flex-direction:column;gap:4px}
.shot-wrap{position:relative}
.shot-img{width:100%;height:auto;border-radius:3px;display:block}
.shot-badge{position:absolute;top:2px;left:2px;background:#000a;color:#ccc;font-size:8px;padding:1px 4px;border-radius:2px}
.info-col{flex:1;display:flex;flex-direction:column;min-width:0}
/* face id 模式专用布局：一帧一卡（卡头标 Scene N），5 张一行多行，不改 boundary 模式 */
#chapters-face .ch-row{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;overflow:visible}
#chapters-face .fcard{background:#161622;border:1px solid #333;border-radius:6px;padding:6px}
#chapters-face .fcard-hdr{font-size:11px;font-weight:700;color:#f1c40f;margin-bottom:5px}
#chapters-face .shot-img{width:100%;height:auto;border-radius:3px;display:block}
/* 顶部人物栏：一小卡 = 头像 + 可改名*/
#chapters-face .people-bar{display:flex;gap:6px;flex-wrap:wrap;padding:0 0 12px;border-bottom:1px solid #333;margin-bottom:14px}
#chapters-face .pcard{display:flex;align-items:center;gap:6px;background:#161622;border:1px solid #333;border-radius:6px;padding:3px 8px 3px 3px}
#chapters-face .pavatar{width:44px;height:44px;overflow:hidden;border-radius:4px;flex-shrink:0;background:#000}
#chapters-face .pavatar img{width:100%;height:100%;object-fit:cover;display:block}
#chapters-face .pav-miss{width:44px;height:44px;display:flex;align-items:center;justify-content:center;color:#555;font-size:13px}
#chapters-face .pname{width:62px;background:transparent;border:none;outline:none;color:#eee;font-size:12px;font-weight:600}
#chapters-face .pname:focus{border-bottom:1px solid #8af}
.asr-box{font-size:11px;line-height:1.55;color:#ccc}
.asr-line{margin-bottom:3px;word-break:break-word}
.spk-tag{display:inline-block;color:#111;font-weight:700;font-size:9px;padding:0 5px;border-radius:3px;margin-right:3px}
.b-input{background:#0d0d14;border:1px solid #444;border-radius:4px;color:#ddd;padding:6px 10px;font-size:12px;font-family:monospace;width:300px}
.b-input:focus{outline:none;border-color:#3498db}
.toast{position:fixed;bottom:20px;right:20px;background:#2ecc71;color:#fff;padding:8px 16px;border-radius:4px;font-size:12px;opacity:0;transition:opacity 0.3s;z-index:2000}
.toast.show{opacity:1}
.help{font-size:10px;color:#666;margin-top:4px}
.gap{flex:0 0 14px;align-self:stretch;display:flex;align-items:center;justify-content:center;border-radius:4px;position:relative;cursor:pointer}
.gap-btn{width:16px;height:16px;line-height:16px;text-align:center;font-size:11px;font-weight:700;border-radius:50%;opacity:0;transition:opacity .15s;color:#fff;border:none;cursor:pointer;padding:0}
.gap:hover .gap-btn{opacity:1}
.gap-add .gap-btn{background:#2ecc71}
.gap-add:hover{background:rgba(46,204,113,.15)}
.gap-remove .gap-btn{background:#e74c3c}
.gap-remove:hover{background:rgba(231,76,60,.15)}
.gap-rm-btn{margin-left:auto;padding:2px 10px;font-size:10px;border:1px solid #e74c3c;background:transparent;color:#e74c3c;border-radius:3px;cursor:pointer;opacity:0;transition:opacity .15s}
.ch:hover .gap-rm-btn{opacity:1}
.gap-rm-btn:hover{background:#e74c3c;color:#fff}
/* 视频信息块：页面最顶部，其下才是模式切换按钮块 */
.video-info{display:flex;align-items:center;gap:10px;padding:12px 16px;background:#13131d;border:1px solid #2a2a3a;border-radius:8px;margin-bottom:14px;flex-wrap:wrap}
.vi-title{font-size:16px;font-weight:800;color:#eee;margin-right:6px}
.vi-item{font-size:12px;color:#aab;background:#1c1c2a;border:1px solid #2a2a3a;padding:3px 10px;border-radius:4px;white-space:nowrap}
/* chapter 概览条：C0 (41f) 等章标签放页面上方，点击滚动到对应块 */
.ch-overview{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.ch-ov{display:flex;align-items:center;gap:6px;padding:5px 12px;background:#161622;border:1px solid #2a2a3a;border-radius:6px;cursor:pointer;font-size:12px;user-select:none}
.ch-ov:hover{background:#1d1d2c}
.ch-ov-name{font-weight:800;color:var(--c,#3498db)}
.ch-ov-f{color:#999;font-size:11px}
</style>
</head><body>
""" + video_info_html + """
<div class="ch-overview" id="chOverview"></div>
<div class="toolbar">
<button class="btn active" id="modeBoundaryBtn" onclick="setMode('boundary')">按 chapter 划分</button>
<button class="btn" id="modeFaceBtn" onclick="setMode('face')">按 face id 划分</button>
<!-- + 右边保存 JSON（写服务器） -->
<input id="boundaryInput" class="b-input" title="chapter 边界 scene 号，逗号分隔（如 30, 51, 66, 86, 97）" spellcheck="false">
<button class="btn primary" id="saveBtn" onclick="saveJSON()">保存 JSON</button>
<button class="btn" id="resetBtn" onclick="resetAll()">Reset</button>
<span class="info" id="info"></span>
</div>
"""
HTML += '<div id="chapters-boundary">'
HTML += "".join(cards_html)
HTML += '</div>'
HTML += '<div id="chapters-face" style="display:none">'
HTML += "".join(face_cards_html)
HTML += '</div>'
HTML += """
<div class="toast" id="toast"></div>
<script>
// 圆点虚线末尾裁剪：
// mask 遮掉末尾不足 6px 的部分（tile 保持 6px 密集平铺不平摊）；只裁圆点，不裁左端 P{n}
function fitGaps() {
  document.querySelectorAll('#chapters-boundary .scene-gap .gap-dots').forEach(g => {
    const w = g.getBoundingClientRect().width;
    const n = Math.floor(w / 6);
    const mask = 'linear-gradient(90deg, black 0, black ' + (n * 6) + 'px, transparent ' + (n * 6) + 'px)';
    g.style.webkitMaskImage = mask;
    g.style.maskImage = mask;
  });
}
fitGaps();
window.addEventListener('resize', fitGaps);
// 概览条帧数 = dedup 全帧集（all_frames, 259）按块边界起始帧归属（与 Python frame_chapter_id 一致）
const ALL_FRAMES = """ + json.dumps(all_frames) + """;
const START_FRAME_OF = """ + json.dumps(scene_first_frame) + """;
// scene → 人物链（scene-gap 圆点左端 P{n} 重建用）
const SCENE_PERSONS = """ + json.dumps(scene_persons) + """;
// 人物链内相邻 scene 对（洋葱链段内部，链内不插圆点虚线）
const CHAIN_EDGES = """ + json.dumps(sorted([list(e) for e in _chain_edges])) + """;
const CHAIN_EDGES_SET = new Set(CHAIN_EDGES.map(e => e[0] + ',' + e[1]));
const N_TOTAL = """ + str(len(scenes)) + """;
const N_FACES = """ + str(len(face_order)) + """;
const N_UNASSIGNED = """ + str(len(unassigned_scenes)) + """;
let mode = 'boundary';  // 'boundary' | 'face'
let boundaries = """ + json.dumps(boundaries) + """;
let draggedSceneId = null;
updateOverview();

function setMode(m) {
  mode = m;
  document.getElementById('modeBoundaryBtn').classList.toggle('active', m === 'boundary');
  document.getElementById('modeFaceBtn').classList.toggle('active', m === 'face');
  const faceMode = (m === 'face');
  document.getElementById('chapters-boundary').style.display = faceMode ? 'none' : '';
  document.getElementById('chapters-face').style.display = faceMode ? '' : 'none';
  // face id 模式为只读：禁用边界输入框/保存/重置
  ['boundaryInput', 'saveBtn', 'resetBtn'].forEach(id => {
    const b = document.getElementById(id);
    b.disabled = faceMode;
    b.style.opacity = faceMode ? 0.35 : 1;
    b.style.cursor = faceMode ? 'not-allowed' : 'pointer';
  });
  const ovEl = document.getElementById('chOverview');
  if (ovEl) ovEl.style.display = faceMode ? 'none' : '';
  const info = document.getElementById('info');
  if (faceMode) {
    document.querySelectorAll('.gap').forEach(g => g.remove());
    info.textContent = 'face id 模式（只读）: ' + N_FACES + ' 个人脸章 · ' + N_UNASSIGNED + ' scenes 无人脸';
  } else {
    info.textContent = getChapters().length + ' chapters, ' + boundaries.length + ' boundaries';
    refreshGaps();
    refreshBoundaryButtons();
    updateChapterLabels();
  }
}

// face id 模式图片复用：从边界模式卡片取 base64，填充 data-fn 占位（避免文件翻倍）
// 选择器用 img[data-fn]：face 卡片 + 顶部人物栏头像共用
function fillFaceImages() {
  const srcs = {};
  document.querySelectorAll('#chapters-boundary .shot-wrap img').forEach(img => {
    if (img.dataset.fn && img.getAttribute('src')) srcs[img.dataset.fn] = img.getAttribute('src');
  });
  document.querySelectorAll('#chapters-face img[data-fn]').forEach(img => {
    if (img.dataset.fn && !img.getAttribute('src') && srcs[img.dataset.fn]) img.setAttribute('src', srcs[img.dataset.fn]);
  });
}

// 人物栏名字：可编辑，改完存 localStorage（key=视频+pid）+ 同步下方 face 章标题，刷新恢复
function applyFaceName(pid, name) {
  const hdr = document.querySelector('#chapters-face .ch[data-face-id="P' + pid + '"] .ch-hdr > span:first-child');
  if (hdr) hdr.textContent = name;
}
function initPersonNames() {
  document.querySelectorAll('#chapters-face .pname').forEach(inp => {
    const k = 'faceName_' + inp.dataset.v + '_' + inp.dataset.pid;
    const saved = localStorage.getItem(k);
    if (saved) { inp.value = saved; applyFaceName(inp.dataset.pid, saved); }
    inp.addEventListener('change', () => {
      localStorage.setItem(k, inp.value);
      applyFaceName(inp.dataset.pid, inp.value);
    });
  });
}

function getChapters() {
  const chapters = [];
  let prev = 0;
  for (const b of boundaries) {
    chapters.push({start: prev, end: b - 1});
    prev = b;
  }
  chapters.push({start: prev, end: N_TOTAL - 1});
  return chapters;
}

function recalcBoundaries() {
  // Remove empty chapters first（只作用于边界模式容器）
  const chapters = document.querySelectorAll('#chapters-boundary .ch');
  chapters.forEach(ch => {
    const cards = ch.querySelectorAll('.card');
    if (cards.length === 0) {
      ch.remove();
    }
  });

  // Recalculate boundaries from remaining chapters
  const remainingChapters = document.querySelectorAll('#chapters-boundary .ch');
  boundaries = [];
  for (let i = 1; i < remainingChapters.length; i++) {
    const firstCard = remainingChapters[i].querySelector('.card');
    if (firstCard) {
      const sid = parseInt(firstCard.getAttribute('data-scene-id'));
      if (!isNaN(sid)) {
        boundaries.push(sid);
      }
    }
  }
  updateChapterLabels();
}

function updateChapterLabels() {
  if (mode !== 'boundary') return;
  const chapters = getChapters();
  const chEls = document.querySelectorAll('#chapters-boundary .ch');
  chEls.forEach((el, i) => {
    if (i >= chapters.length) return;
    const ch = chapters[i];
    const idEl = el.querySelector('.ch-id');
    const sizeEl = el.querySelector('.ch-size');
    const boundEl = el.querySelector('.ch-boundary');
    if (idEl) idEl.textContent = 'CHAPTER ' + i;
    if (sizeEl) sizeEl.textContent = (ch.end - ch.start + 1) + ' scenes';
    if (boundEl) {
      if (i > 0) {
        boundEl.textContent = '(boundary: ' + ch.start + ')';
      } else {
        boundEl.textContent = '';
      }
    }
  });
  document.getElementById('info').textContent = chapters.length + ' chapters, ' + boundaries.length + ' boundaries';
  const viB = document.getElementById('viChapters');
  if (viB) viB.textContent = chapters.length + ' chapters';
  updateOverview();
  syncBoundaryInput();
}

// 每块帧数 = dedup 全帧集按当前边界起始帧归属（照 Python frame_chapter_id：bf <= fn 计入后块）
function chapterFrameCounts() {
  const counts = [];
  for (const fn of ALL_FRAMES) {
    let chid = 0;
    for (const b of boundaries) {
      const s = START_FRAME_OF[b];
      if (s == null) continue;
      if (fn >= s) chid++; else break;
    }
    counts[chid] = (counts[chid] || 0) + 1;
  }
  return counts;
}

// chapter 概览条：视频信息下 C{n} (Nf) 小卡，点击滚动到对应块；边界变化后重建
function updateOverview() {
  const ov = document.getElementById('chOverview');
  if (!ov || !document.getElementById('chapters-boundary')) return;
  const cnts = chapterFrameCounts();
  let h = '';
  document.querySelectorAll('#chapters-boundary .ch').forEach((ch, i) => {
    const colorEl = ch.querySelector('.ch-hdr span');
    const c = colorEl ? (colorEl.style.color || '#3498db') : '#3498db';
    h += '<div class="ch-ov" style="--c:' + c + '" onclick="scrollToChapter(' + i + ')" title="滚动到 C' + i + '">'
       + '<span class="ch-ov-name">C' + i + '</span>'
       + '<span class="ch-ov-f">(' + (cnts[i] || 0) + 'f)</span></div>';
  });
  ov.innerHTML = h;
}
function scrollToChapter(i) {
  const ch = document.querySelectorAll('#chapters-boundary .ch')[i];
  if (ch) ch.scrollIntoView({behavior: 'smooth', block: 'start'});
}

// 边界输入框 → 当前边界同步显示（拖拽/✕/Reset/保存后刷新）
function syncBoundaryInput() {
  const inp = document.getElementById('boundaryInput');
  if (inp && document.activeElement !== inp) inp.value = boundaries.join(', ');
}

// → POST 服务器
// （serve_viz.py 写 chapter_boundaries.json；渲染时该手工文件优先于 API 默认）
function saveJSON() {
  if (mode !== 'boundary') { showToast('face id 模式为只读，不能保存边界'); return; }
  const raw = document.getElementById('boundaryInput').value.trim();
  if (!raw) { showToast('边界为空'); return; }
  const parts = raw.split(/[,，\s]+/).map(Number);
  if (parts.some(x => !Number.isInteger(x) || x <= 0 || x >= N_TOTAL)) {
    showToast('边界需为 1~' + (N_TOTAL - 1) + ' 的整数，逗号分隔');
    return;
  }
  boundaries = [...new Set(parts)].sort((a, b) => a - b);
  rebuildChapters();
  syncBoundaryInput();
  const data = {n_scenes: N_TOTAL, boundaries: boundaries, n_chapters: boundaries.length + 1};
  fetch('""" + save_path + """', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(() => showToast('已保存到服务器: ' + boundaries.length + ' boundaries'))
    .catch(e => showToast('保存失败: ' + e.message));
}

// 按当前 boundaries 全量重建分章 DOM。
// 杂志列结构：.ch-row 下 4 个 .mag-col，每列 = vline + (scene-gap) + card + asr-note 单位。
// 重建 = 按 DOM 视觉序收集 chunk（card + 紧随的 asr-note），按 scene 范围归块后均分 4 列。
// ═══ 响应式列数═══
// 与 CSS 网格对齐（minmax(320px,1fr) + 40px gap = 每列 360px 起）：
// floor((innerWidth+40)/360) —— 13" MBA(1470)→4、MBP(1512~1728)→4、
// 外接 1920→5、2304→6、3840→8（cap 8）。修正：原 370 阈值让 1470/370=3.97 砍成 3 列
function colCount() {
  return Math.min(8, Math.max(2, Math.floor((window.innerWidth + 40) / 360)));
}

// 从块内按视觉序收集 chunk（card + 紧随的 asr-note）+ 块级样式（竖线同色；scene-gap 的 P 标签/圆点由数据重建）
function collectChunks(ch) {
  const chunks = [];
  let vlineStyle = '';
  ch.querySelectorAll('.mag-col').forEach(col => {
    if (!vlineStyle) {
      const v = col.querySelector('.vline');
      if (v) vlineStyle = v.getAttribute('style') || '';
    }
    col.querySelectorAll('.card').forEach(card => {
      // asr-note 紧随卡片（中间可能被 hover 的 .gap 插入，跳过它们）
      let note = card.nextElementSibling;
      while (note && note.classList && (note.classList.contains('gap') || note.classList.contains('scene-gap'))) note = note.nextElementSibling;
      chunks.push({
        sid: parseInt(card.getAttribute('data-scene-id')),
        card,
        note: (note && note.classList && note.classList.contains('asr-note')) ? note : null
      });
    });
  });
  return {chunks, vlineStyle};
}

// scene-gap 重建：圆点虚线 + （下方 scene 有人物链时）左端 P{n}，P 与虚线剧中对齐
function makeGap(sid, vlineStyle) {
  const m = (vlineStyle || '').match(/#[0-9a-fA-F]{6}/);
  const color = m ? m[0] : '#3498db';
  const gap = document.createElement('div');
  gap.className = 'scene-gap';
  const ps = SCENE_PERSONS[sid] || [];
  if (ps.length) {
    const tag = document.createElement('span');
    tag.className = 'gap-p';
    tag.style.color = color;
    tag.textContent = ps.join(' ');
    gap.appendChild(tag);
  }
  const dots = document.createElement('span');
  dots.className = 'gap-dots';
  dots.style.backgroundImage = 'radial-gradient(circle,' + color + '55 1px,transparent 1px)';
  dots.style.backgroundSize = '6px 3px';
  dots.style.backgroundRepeat = 'repeat-x';
  gap.appendChild(dots);
  return gap;
}

// chunks 均分 nCols 列 → mag-col（vline + scene-gap + card + asr-note）挂入 row
function buildCols(row, chunks, vlineStyle, nCols) {
  const n = chunks.length;
  const base = Math.floor(n / nCols), extra = n % nCols;
  let idx = 0;
  for (let k = 0; k < nCols; k++) {
    const size = base + (k < extra ? 1 : 0);
    const col = document.createElement('div');
    col.className = 'mag-col';
    col.innerHTML = '<div class="vline" style="' + (vlineStyle || 'background:#3498db') + '"></div>';
    let prevSid = -1;
    for (let i = 0; i < size; i++) {
      const ch = chunks[idx + i];
      if (!ch) continue;
      if (prevSid !== -1 && ch.sid !== prevSid) {
        const ga = Math.min(prevSid, ch.sid), gb = Math.max(prevSid, ch.sid);
        if (!CHAIN_EDGES_SET.has(ga + ',' + gb)) col.appendChild(makeGap(ch.sid, vlineStyle));
      }
      col.appendChild(ch.card);
      if (ch.note) col.appendChild(ch.note);
      prevSid = ch.sid;
    }
    idx += size;
    // 2026-08-14：空列也 append（帧少的块跳过空列 → 列数变少 → auto-fit 拉伸栏宽，
    // 与其他 chapter 列宽不一致）；空列仅竖线可见，靠 .mag-col min-height 不塌）
    row.appendChild(col);
  }
}

// 视口变化 → 全部块按新列数重排（不重建块头，face 模式不动）
function reflowCols() {
  if (mode !== 'boundary') return;
  const nCols = colCount();
  document.querySelectorAll('#chapters-boundary .ch').forEach(ch => {
    const row = ch.querySelector('.ch-row');
    if (!row) return;
    const {chunks, vlineStyle} = collectChunks(ch);
    row.querySelectorAll('.mag-col').forEach(c => c.remove());
    buildCols(row, chunks, vlineStyle, nCols);
  });
  refreshGaps();  // 行内 hover +/✕ 按钮随重排重建
  fitGaps();
}

let reflowTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(reflowTimer);
  reflowTimer = setTimeout(reflowCols, 250);
});

// 旧章的 AI 标题头按「起始 scene」归档复用；起始 scene 无对应旧块的用 NEW 头。
function rebuildChapters() {
  const container = document.getElementById('chapters-boundary');
  const oldChs = [...container.querySelectorAll(':scope > .ch')];

  // 按起始 scene 归档旧块头 / 竖线样式（重建后复用同色；scene-gap 的 P 标签由 SCENE_PERSONS 数据重建）
  const hdrByStart = {}, vlineStyleByStart = {};
  const chunks = [];  // {sid, card, note} 按当前视觉序
  oldChs.forEach(ch => {
    const {chunks: sub, vlineStyle} = collectChunks(ch);
    chunks.push(...sub);
    const first = ch.querySelector('.card');
    if (first) {
      const start = parseInt(first.getAttribute('data-scene-id'));
      const hdr = ch.querySelector('.ch-hdr');
      if (hdr) {
        const clone = hdr.cloneNode(true);
        const btn = clone.querySelector('.gap-rm-btn');
        if (btn) btn.remove();  // 克隆的按钮没有 onclick，refreshBoundaryButtons 会重建
        hdrByStart[start] = clone;
      }
      vlineStyleByStart[start] = vlineStyle;
    }
  });
  oldChs.forEach(ch => ch.remove());

  const ranges = [];
  let prev = 0;
  for (const b of boundaries) { ranges.push([prev, b - 1]); prev = b; }
  ranges.push([prev, N_TOTAL - 1]);

  const defaultHdr = '<div class="ch-hdr"><div class="ch-hdr-top"><span class="ch-id" style="font-weight:700;font-size:13px;color:#fff">NEW</span><span class="ch-size" style="color:#666;font-size:11px;margin-left:8px"></span><span class="ch-boundary" style="color:#888;font-size:10px;margin-left:8px"></span></div></div>';

  ranges.forEach((range, chi) => {
    const [start, end] = range;
    const inChapter = chunks.filter(ch => ch.sid >= start && ch.sid <= end);
    const ch = document.createElement('div');
    ch.className = 'ch';
    ch.setAttribute('data-chapter-id', String(chi));
    ch.innerHTML = (hdrByStart[start] ? hdrByStart[start].outerHTML : defaultHdr);
    const row = document.createElement('div');
    row.className = 'ch-row';
    row.setAttribute('data-chapter-id', String(chi));

    // 均分 N 列（响应式列数：视口宽度决定，
    buildCols(row, inChapter, vlineStyleByStart[start], colCount());
    ch.appendChild(row);
    container.appendChild(ch);
  });

  updateChapterLabels();
  refreshGaps();
  refreshBoundaryButtons();
  fitGaps();
}

function resetAll() {
  if (mode !== 'boundary') { showToast('face id 模式为只读，不能保存边界'); return; }
  if (confirm('Reset all boundaries to original?')) {
    boundaries = """ + json.dumps(boundaries) + """;
    location.reload();
  }
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

// Drag and drop
document.addEventListener('dragstart', function(e) {
  if (mode !== 'boundary') return;  // face id 模式只读
  const card = e.target.closest('.card');
  if (!card) return;
  draggedSceneId = card.getAttribute('data-scene-id');
  card.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', draggedSceneId);
});

document.addEventListener('dragend', function(e) {
  const card = e.target.closest('.card');
  if (card) card.classList.remove('dragging');
  draggedSceneId = null;
  // Remove all drag-over highlights
  document.querySelectorAll('.ch-row').forEach(row => row.classList.remove('drag-over'));
});

document.addEventListener('dragover', function(e) {
  const row = e.target.closest('.ch-row');
  if (!row) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  row.classList.add('drag-over');
});

document.addEventListener('dragleave', function(e) {
  const row = e.target.closest('.ch-row');
  if (!row) return;
  // Only remove if actually leaving the row
  const rect = row.getBoundingClientRect();
  if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
    row.classList.remove('drag-over');
  }
});

document.addEventListener('drop', function(e) {
  e.preventDefault();
  if (mode !== 'boundary') return;  // face id 模式只读
  const row = e.target.closest('.ch-row');
  if (!row || !draggedSceneId) return;
  row.classList.remove('drag-over');

  const draggedCard = document.querySelector('.card[data-scene-id="' + draggedSceneId + '"]');
  if (!draggedCard) return;

  // Find the closest card in the target row to determine insertion point
  const targetChapter = row.closest('.ch');
  const targetChapterId = parseInt(targetChapter.getAttribute('data-chapter-id'));

  const sourceChapter = draggedCard.closest('.ch');
  const sourceChapterId = parseInt(sourceChapter.getAttribute('data-chapter-id'));

  if (sourceChapterId === targetChapterId) return; // Same chapter, no move needed

  // Move the card to the target chapter
  const allCardsInTarget = Array.from(row.querySelectorAll('.card'));
  const targetSceneIds = allCardsInTarget.map(c => parseInt(c.getAttribute('data-scene-id')));

  // Find insertion index: place after the closest scene_id
  let insertBefore = null;
  for (const card of allCardsInTarget) {
    const sid = parseInt(card.getAttribute('data-scene-id'));
    if (sid > parseInt(draggedSceneId)) {
      insertBefore = card;
      break;
    }
  }

  if (insertBefore) {
    row.insertBefore(draggedCard, insertBefore);
  } else {
    row.appendChild(draggedCard);
  }

  // 重算边界 + 全量重建（杂志列结构；拖拽 = 移动 scene 的归属边界）
  recalcBoundaries();
  rebuildChapters();
  showToast('Scene ' + draggedSceneId + ' moved to Chapter ' + targetChapterId);
});

// Click on card to highlight it
document.addEventListener('click', function(e) {
  const card = e.target.closest('.card');
  if (!card) return;
  // Remove previous selection
  document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
});

// ===== 边界交互：hover 添加/移除 =====
function refreshGaps() {
  if (mode !== 'boundary') return;  // face id 模式只读
  document.querySelectorAll('.gap').forEach(g => g.remove());
  document.querySelectorAll('#chapters-boundary .ch-row').forEach(row => {
    const cards = Array.from(row.querySelectorAll('.card'));
    for (let i = 0; i < cards.length - 1; i++) {
      const a = parseInt(cards[i].getAttribute('data-scene-id'));
      const b = parseInt(cards[i + 1].getAttribute('data-scene-id'));
      const gap = document.createElement('div');
      if (boundaries.includes(b)) {
        gap.className = 'gap gap-remove';
        gap.title = '移除边界 (scene ' + b + ' 起)';
        gap.innerHTML = '<button class="gap-btn" type="button">&#10005;</button>';
        gap.onclick = function() { removeBoundaryAt(b); };
      } else {
        gap.className = 'gap gap-add';
        gap.title = '添加边界 (scene ' + b + ' 起)';
        gap.innerHTML = '<button class="gap-btn" type="button">+</button>';
        gap.onclick = function() { addBoundaryAt(b); };
      }
      cards[i].after(gap);
    }
  });
}

function refreshBoundaryButtons() {
  if (mode !== 'boundary') return;  // face id 模式只读
  document.querySelectorAll('#chapters-boundary .ch').forEach((ch, i) => {
    if (i === 0) return;
    const first = ch.querySelector('.card');
    if (!first) return;
    const sid = parseInt(first.getAttribute('data-scene-id'));
    let btn = ch.querySelector('.gap-rm-btn');
    if (!btn) {
      btn = document.createElement('button');
      btn.className = 'gap-rm-btn';
      btn.title = '移除边界 (scene ' + sid + ' 起)';
      btn.innerHTML = '&#10005; 移除边界';
      btn.onclick = function() { removeBoundaryAt(sid); };
      // 按钮进标题行（ch-hdr 现在是标题行+AI总结行两段式）
      (ch.querySelector('.ch-hdr-top') || ch.querySelector('.ch-hdr')).appendChild(btn);
    }
  });
}

function addBoundaryAt(sid) {
  if (boundaries.includes(sid)) return;
  if (sid <= 0 || sid >= N_TOTAL) return;
  boundaries.push(sid);
  boundaries.sort((x, y) => x - y);
  rebuildChapters();  // 杂志列结构下全量重建（2026-08-12）
  showToast('边界已添加: scene ' + sid + ' 起');
}

function removeBoundaryAt(sid) {
  const idx = boundaries.indexOf(sid);
  if (idx < 0) return;
  boundaries.splice(idx, 1);
  rebuildChapters();  // 杂志列结构下全量重建（2026-08-12）
  showToast('边界已移除: scene ' + sid);
}

refreshGaps();
refreshBoundaryButtons();
fillFaceImages();
initPersonNames();

// Initial state
updateChapterLabels();
syncBoundaryInput();
reflowCols();  // 首屏按当前视口列数重排（4K 外接屏自动加列）
</script>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("-> " + OUT + "  (" + str(os.path.getsize(OUT) // 1024) + "KB)")
