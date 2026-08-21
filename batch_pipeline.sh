#!/bin/bash
# batch_pipeline.sh — 全流水线（VIP1.6zenkiru；A/B 逻辑按内容自动判定，自动跳过已完成步骤）
# 用法: bash batch_pipeline.sh（调度脚本 run_batch.sh / remux_and_run.sh 调用；OUT_ROOT 可覆盖输出根）
# 依赖: input/*.mp4 或 input/<folder>/ 视频；amaterasu conda 环境 + 模型路径（env/HF_HOME 布局推导）
# 产物: output/<视频名或文件夹>/preproc/<模块>/<video_hash>_<产物> + visual/audio/vlm/... 下游产物
#
# 输入处理（无严格 Mode，按内容自动适配）:
#   input/*.mp4             → A 逻辑（单视频全流程）
#   input/<folder>/ 多视频  → B 逻辑（DINO/人脸全局合并）
#   input/<folder>/ 单视频  → A 逻辑（无全局合并）
#   混合（视频+文件夹共存）→ 先处理视频，再处理文件夹
# 输出协议:
#   A 逻辑 → output/<视频名>/；B 逻辑 → output/<文件夹名>/；OUT_ROOT 显式覆盖
#   preproc/ 下: cuts / skeleton / select_frames / events / features / frames / frames224
#   命名: <video_hash>_<模块>.json（内容哈希前缀，A/B 统一；scdet → cuts，不暴露滤镜名）
# 流程: scdet→skeleton→select_frames→抽帧→DINOv3→face_detect→dedup→face_recognition
#   →onion_model→body_detect→audio→vlm(select→fuse→submit)

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

conda_activate() {
  source "${CONDA_SH:?set CONDA_SH to your conda.sh path}"
  conda activate "${CONDA_ENV:-base}"
}

# ═══ 模型路径（统一环境变量管理，零硬编码，═══
# 每个模型同一模式：显式 <VAR> 优先 → 按 HF_HOME（默认 /models/hf）布局推导 → 找不到报错提示
# 布局推导：$HF_HOME/<org>/<id>/（手动下载）→ $HF_HOME/hub/models--<org>--<id>/snapshots/*（HF 标准）
#           $HF_HOME/hub/<org>/<id>/（hub 直放）→ $HF_HOME/../modelscope/<org>/<id>（modelscope）
HF_HOME="${HF_HOME:-/models/hf}"
model_locate() { # <变量名> <候选目录...>：显式 env 有效则用，否则取第一个存在的候选，全无则报错
  local var="$1"; shift
  local val="${!var}"
  if [ -n "$val" ] && [ -e "$val" ]; then return 0; fi
  local c
  for c in "$@"; do
    [ -n "$c" ] && [ -e "$c" ] && { export "$var=$c"; return 0; }
  done
  echo "[ERR] $var 未设置，HF_HOME（$HF_HOME）下也找不到对应模型，请 export $var=<模型目录>" >&2
  return 1
}
# DINOv3（visual/dino_cluster.py；camenduru 是 hub 直放格式，无 snapshots）
model_locate DINO_MODEL_DIR \
  "$HF_HOME/facebook/dinov3-vitl16-pretrain-lvd1689m" \
  "$(ls -d "$HF_HOME"/hub/models--*--dinov3-vitl16-pretrain-lvd1689m/snapshots/* 2>/dev/null | head -1)" \
  "$(ls -d "$HF_HOME"/hub/models--*--dinov3-vitl16-pretrain-lvd1689m 2>/dev/null | head -1)" || exit 1
# insightface buffalo_l（visual/face_detect.py / face_recognition.py；标准位 /models/hf/insightface/）
model_locate INSIGHTFACE_DIR \
  "$HF_HOME/insightface/models/buffalo_l" || exit 1
# Qwen3-ASR-1.7B（audio/transcribe.py：QWEN3_ASR_MODEL_DIR）
model_locate QWEN3_ASR_MODEL_DIR \
  "$HF_HOME/hub/Qwen/Qwen3-ASR-1.7B" || exit 1
# pyannote 3.1（audio/speaker_diarize.py）
model_locate PYANNOTE_MODEL_DIR \
  "$(ls -d "$HF_HOME"/hub/models--pyannote--speaker-diarization-3.1/snapshots/* 2>/dev/null | head -1)" || exit 1
# Qwen（vlm/submit_segments.py：Qwen3-VL-4B vllm；compress_asr 已停用 2026-08-09）
# 视觉引擎 Qwen3-VL-4B-Instruct 优先（submit 加载的就是它，vllm bnb4bit）
model_locate QWEN_MODEL_DIR \
  "$HF_HOME/hub/Qwen/Qwen3-VL-4B-Instruct" \
  "$HF_HOME/qwen35_all4bit" \
  "$HF_HOME/qwen35_split" \
  "$HF_HOME/qwen35_hybrid_4bit" \
  "$HF_HOME/hub/Qwen/Qwen3___5-4B" \
  "$HF_HOME/hub/Qwen/Qwen3.5-9B" || exit 1
# chapter 划章脚本兼容旧 MODEL_PATH 变量名
export MODEL_PATH="$QWEN_MODEL_DIR"

input_dir="$PWD/input"
mkdir -p "$input_dir"

# ── 扫描输入 ──

FOLDERS=()
while IFS= read -r -d '' d; do
  FOLDERS+=("$d")
done < <(find "$input_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

VIDEO_FILES=()
while IFS= read -r -d '' f; do
  VIDEO_FILES+=("$f")
done < <(find "$input_dir" -maxdepth 1 -type f \
  \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.m4v" \) \
  -print0 | sort -z)

if [ "${#VIDEO_FILES[@]}" -gt 0 ]; then
  echo "=== 直接视频 ${#VIDEO_FILES[@]} 个（A 逻辑）==="
fi
if [ "${#FOLDERS[@]}" -gt 0 ]; then
  echo "=== 文件夹 ${#FOLDERS[@]} 个（逐个按视频数判定）==="
fi

# ── 路径辅助 ──
# 输出文件夹：A 逻辑 input/ 根视频 → output/<视频名>/；
# B 逻辑 input/<文件夹>/ → output/<文件夹名>/；调度脚本可用 OUT_ROOT 显式覆盖（run_batch.sh 的 vivant）
get_out_root() {
  local mp4="$1"
  if [ -n "$OUT_ROOT" ]; then
    echo "$OUT_ROOT"
  elif [ "$(dirname "$mp4")" = "$input_dir" ]; then
    echo "$PWD/output/$(basename "$mp4" | sed 's/\.[^.]*$//')"
  else
    echo "$PWD/output/$(basename "$(dirname "$mp4")")"
  fi
}

# 视频内容指纹 video_hash（文件 SHA256 前 6 位 hex，2026-08-19 大名定稿：
# 判重/产物命名唯一依据 = 内容哈希，与文件名/路径无关；旧 fnv6 文件名哈希、
# 按 video 字段（路径名）查骨架取哈希均已废——直接对视频文件算，不使用名字）
content_hash() {
  sha256sum "$1" | awk '{print substr($1,1,6)}'
}

# ── 步骤 1-4: 逐视频（cuts/skeleton/select_frames/events/features/dino_input/抽帧）──

process_video_steps_1_4() {
  local mp4="$1"
  local project_label="$2"
  local out_root="$3"
  local vid_name
  vid_name=$(basename "$mp4")
  vid_name="${vid_name%.*}"

  echo ""
  echo "======= [$vid_name] $(date) ======="
  echo "  视频: $mp4"
  echo "  输出: $out_root/preproc/"

  # 1-4. duo_analyze（双线一体，一次解码全产出：cuts/skeleton/select_frames/events/features
  #       + 抽帧（短边 960，+ frames224；cubin 按 /proc/self/exe 定位，无需 cd）
  # 缓存判断（2026-08-19 大名：判重只用内容哈希）：先预检测输入视频内容哈希，
  # preproc 骨架 <hash>_skeleton.json 已存在 = 该视频预处理已完整产出 → 停止处理
  local vh
  vh=$(content_hash "$mp4")
  if [ -f "$out_root/preproc/skeleton/${vh}_skeleton.json" ]; then
    echo "[ 1-4/12] duo_analyze: cached ($vh)"
  else
    echo "[ 1-4/12] duo_analyze（双线一体，短边 960 抽帧）..."
    ./preproc/duo_analyze -o "$out_root" "$mp4" 2>&1 | tail -5
  fi
}

# ── 步骤 5: DINO（每视频 / 全局）──
# $1=out_root  $2=project_label  $3=vid_name（vid_name 为空 → 全局模式）

run_dino() {
  local out_root="$1" project_label="$2" vid_name="$3" mp4="$4"
  local vh
  vh=$(content_hash "$mp4")

  if [ -n "$vid_name" ]; then
    # 每视频 DINO（帧前缀 = video_hash 内容指纹；2026-08-19：只用内容哈希）
    local npz="$out_root/visual/dino/${vh}_key_frame_embeddings.npz"
    if [ -f "$npz" ]; then
      echo "[ 5/12] dino_cluster [$vh]: cached"
      return
    fi
    echo "[ 5/12] dino_cluster [$vh]..."
    conda_activate
    if [ -n "$project_label" ]; then
      OUT_ROOT="$out_root" python3 visual/dino_cluster.py "$vh" "$project_label" "$vh" 2>&1 | tail -5
    else
      OUT_ROOT="$out_root" python3 visual/dino_cluster.py "$vh" 2>&1 | tail -5
    fi
  else
    # 全局 DINO（B 逻辑）
    local dino_npz="$out_root/visual/dino/key_frame_embeddings.npz"
    local dino_meta="$out_root/visual/dino/model_meta.json"
    if [ -f "$dino_npz" ] && [ -f "$dino_meta" ] &&
       grep -q '"model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"' "$dino_meta"; then
      echo "[ 5/12] dino_cluster (全局): cached"
      return
    fi
    echo "[ 5/12] dino_cluster (全局)..."
    conda_activate
    OUT_ROOT="$out_root" python3 visual/dino_cluster.py "" "$project_label" 2>&1 | tail -5
  fi
}

# ── 步骤 6: face_detect（逐视频）──
# $1=out_root  $2=mp4  $3=project_label

run_face_detect() {
  local out_root="$1" mp4="$2" project_label="$3"
  local vhash
  vhash=$(content_hash "$mp4")
  local face_map="$out_root/visual/face_detect/${vhash}_face_map.json"
  # 依赖 DINO meta（哈希前缀协议）：全局 DINO → model_meta.json；每视频 DINO → <vhash>_model_meta.json
  local dino_meta="$out_root/visual/dino/model_meta.json"
  if [ ! -f "$dino_meta" ]; then
    dino_meta="$out_root/visual/dino/${vhash}_model_meta.json"
  fi
  if [ -f "$face_map" ] && [ "$face_map" -nt "$dino_meta" ]; then
    echo "[ 6/12] face_detect [$vhash]: cached"
    return
  fi
  echo "[ 6/12] face_detect [$vhash]..."
  conda_activate
  if [ -n "$project_label" ]; then
    OUT_ROOT="$out_root" python3 visual/face_detect.py "$vhash" "$project_label" "$vhash" 2>&1 | tail -5
  else
    OUT_ROOT="$out_root" python3 visual/face_detect.py "$vhash" 2>&1 | tail -5
  fi
}

# ── 步骤 6-12: A 逻辑全流程（face_detect/dedup/face_rec/onion/graph/audio/chapter）──
# $1=mp4  $2=project_label  $3=out_root

process_video_steps_6_12() {
  local mp4="$1"
  local project_label="$2"
  local out_root="$3"
  local vid_name
  vid_name=$(basename "$mp4"); vid_name="${vid_name%.*}"
  local vh
  vh=$(content_hash "$mp4")

  # 6. face_detect
  run_face_detect "$out_root" "$mp4" "$project_label"

  # 7. dedup
  local dedup_meta dedup_sk dino_meta
  dedup_meta="$out_root/visual/dedup/${vh}_model_meta.json"
  dedup_sk="$out_root/visual/dedup/${vh}_skeleton.json"
  dino_meta="$out_root/visual/dino/${vh}_model_meta.json"
  if [ -f "$dedup_sk" ] && [ -f "$dedup_meta" ] &&
     grep -q '"dino_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"' "$dedup_meta" &&
     grep -q '"face_map_used": true' "$dedup_meta" &&
     [ "$dedup_sk" -nt "$dino_meta" ]; then
    echo "[ 7/12] dedup: cached"
  else
    echo "[ 7/12] dedup..."
    conda_activate
    if [ -n "$project_label" ]; then
      OUT_ROOT="$out_root" python3 visual/dedup.py "$vh" "$project_label" "$vh" 2>&1 | tail -5
    else
      OUT_ROOT="$out_root" python3 visual/dedup.py "$vh" 2>&1 | tail -5
    fi
  fi

  # 7.5 global_cos（全局帧处理目录：挨个读目录下所有 dedup 骨架 → 全局聚簇 → gc_skeleton.json）
  local gc_sk
  gc_sk="$out_root/visual/global_cos/gc_skeleton.json"
  if [ -f "$gc_sk" ] && [ "$gc_sk" -nt "$dedup_sk" ]; then
    echo "[7.5/12] global_cos: cached"
  else
    echo "[7.5/12] global_cos（全局聚簇）..."
    conda_activate
    OUT_ROOT="$out_root" python3 visual/global_cos.py "$vid_name" 2>&1 | tail -5
  fi

  # 8. face_recognition（每视频）
  local pt_meta pt_path
  pt_meta="$out_root/visual/face_head_fusion/${vh}_person_timeline_meta.json"
  pt_path="$out_root/visual/face_head_fusion/${vh}_person_timeline.json"
  if [ -f "$pt_path" ] && [ -f "$pt_meta" ] &&
     grep -q '"dino_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"' "$pt_meta" &&
     [ "$pt_path" -nt "$dedup_sk" ]; then
    echo "[ 8/12] face_recognition: cached"
  else
    echo "[ 8/12] face_recognition..."
    conda_activate
    if [ -n "$project_label" ]; then
      OUT_ROOT="$out_root" python3 visual/face_recognition.py "$vh" --dino-filter --project "$project_label" --video "$vh" 2>&1 | tail -8
    else
      OUT_ROOT="$out_root" python3 visual/face_recognition.py "$vh" --dino-filter 2>&1 | tail -8
    fi
  fi

  # 9. onion_model
  local onion_sk
  onion_sk="$out_root/visual/onion_model/${vh}_skeleton.json"
  if [ -f "$onion_sk" ] && [ "$onion_sk" -nt "$pt_path" ] && [ "$onion_sk" -nt "$dedup_sk" ]; then
    echo "[ 9/12] onion_model: cached"
  else
    echo "[ 9/12] onion_model..."
    conda_activate
    if [ -n "$project_label" ]; then
      OUT_ROOT="$out_root" python3 visual/onion_model.py "$vh" "$project_label" "$vh" 2>&1 | tail -5
    else
      OUT_ROOT="$out_root" python3 visual/onion_model.py "$vh" 2>&1 | tail -5
    fi
  fi

  # 9.5 body_detect（人体检测；每项目全局一份 body_bbox.json，键对齐 gc 帧名 <hash>_f<fn>，
  #      fuse 读身体骨架做主体/路人分群；graph_merge 已废弃）
  local body_bbox body_dep_sk
  body_bbox="$out_root/visual/body_detect/body_bbox.json"
  body_dep_sk="$out_root/visual/dino/${vh}_skeleton.json"
  if [ -f "$body_bbox" ] && [ "$body_bbox" -nt "$gc_sk" ]; then
    echo "[9.5/12] body_detect: cached"
  else
    echo "[9.5/12] body_detect..."
    conda_activate
    OUT_ROOT="$out_root" python3 visual/body_detect.py "${project_label:-$vid_name}" 2>&1 | tail -5
  fi

  # 10. audio ASR pipeline
  local audio_dia
  audio_dia="$out_root/audio/dialogue/${vh}_dialogue.json"
  if [ -f "$audio_dia" ] && [ "$audio_dia" -nt "$dedup_sk" ]; then
    echo "[10/12] audio: cached"
  else
    echo "[10/12] audio..."
    conda_activate
    if [ -n "$project_label" ]; then
      OUT_ROOT="$out_root" python3 audio/asr_pipeline.py "$vh" "$project_label" "$vh" 2>&1 | tail -8
    else
      OUT_ROOT="$out_root" python3 audio/asr_pipeline.py "$vh" 2>&1 | tail -8
    fi
  fi

  # 11.5 vlm 三件套① select_segments（选帧；读 dedup/dino/face_map/body_bbox）。
  #      vlm 只读 ASR 判 has_asr（有无台词），台词不嵌进骨架、只读不写 ASR 的东西；
  #      缓存判断不依赖 audio（vlm 不依赖 ASR，graph_merge 已废弃不进，
  #      pipeline 到 vlm desc 为止——chapter 划章待修不串）
  local sel_skel
  sel_skel="$out_root/vlm/${vh}_skeleton.json"
  if [ -f "$sel_skel" ] && [ "$sel_skel" -nt "$body_dep_sk" ]; then
    echo "[11.5/12] vlm select: cached"
  else
    echo "[11.5/12] vlm select (选帧)..."
    conda_activate
    python3 vlm/select_segments.py "$vid_name" 2>&1 | tail -5
  fi

  # 11.6 vlm 三件套② fuse_segments（融合拼图；读 select 骨架 + frames + DINO，
  #      落 vlm/segments/ + vlm/<vhash>_fused.json）
  local fused_out
  fused_out="$out_root/vlm/${vh}_fused.json"
  if [ -f "$fused_out" ] && [ "$fused_out" -nt "$sel_skel" ]; then
    echo "[11.6/12] vlm fuse: cached"
  else
    echo "[11.6/12] vlm fuse (融合拼图)..."
    conda_activate
    python3 vlm/fuse_segments.py "$vid_name" 2>&1 | tail -5
  fi

  # 11.7 vlm 三件套③ submit_segments（送检 VLM desc；vllm continuous batching，
  #      落 vlm/<vhash>_desc.json；vllm env 跑，禁 amaterasu）——pipeline 终点
  local desc_out
  desc_out="$out_root/vlm/${vh}_desc.json"
  if [ -f "$desc_out" ] && [ "$desc_out" -nt "$fused_out" ]; then
    echo "[11.7/12] vlm desc: cached"
  else
    echo "[11.7/12] vlm desc (Qwen3-VL-4B vllm, amaterasu)..."
    conda_activate
    python3 vlm/submit_segments.py "$vid_name" 2>&1 | tail -6
  fi

  echo "=== [$vid_name] 完成 ==="
}

# ── 步骤 9-12: B 逻辑（全局 face_recognition 之后；9. onion_model 默认跑，A/B 同款）──
# $1=mp4  $2=project_label  $3=out_root  $4=person_offset

process_video_steps_9_12() {
  local mp4="$1"
  local project_label="$2"
  local out_root="$3"
  local person_offset="${4:-0}"   # 本集人物段偏移（前集 dedup scenes 累计，2026-08-12）
  local vid_name
  vid_name=$(basename "$mp4"); vid_name="${vid_name%.*}"
  local vh
  vh=$(content_hash "$mp4")

  local pt_dep="$out_root/visual/face_head_fusion/person_timeline.json"
  local dedup_sk="$out_root/visual/dedup/${vh}_skeleton.json"
  local body_dep_sk="$out_root/visual/dino/${vh}_skeleton.json"

  # 9. onion_model（B 路线默认也走洋葱：全局 person_timeline + 偏移映射本集，
  #    产物落盘展示；2026-08-16 大名拍板 A/B 都跑洋葱）
  local onion_sk
  onion_sk="$out_root/visual/onion_model/${vh}_skeleton.json"
  if [ -f "$onion_sk" ] && [ "$onion_sk" -nt "$pt_dep" ] && [ "$onion_sk" -nt "$dedup_sk" ]; then
    echo "[ 9/11] onion_model: cached"
  else
    echo "[ 9/11] onion_model..."
    conda_activate
    PERSON_SCENE_OFFSET="$person_offset" OUT_ROOT="$out_root" python3 visual/onion_model.py "$vh" "$project_label" "$vh" 2>&1 | tail -5
  fi

  # 10. audio ASR pipeline
  local audio_dia
  audio_dia="$out_root/audio/dialogue/${vh}_dialogue.json"
  if [ -f "$audio_dia" ] && [ "$audio_dia" -nt "$dedup_sk" ]; then
    echo "[10/11] audio: cached"
  else
    echo "[10/11] audio..."
    conda_activate
    OUT_ROOT="$out_root" python3 audio/asr_pipeline.py "$vh" "$project_label" "$vh" 2>&1 | tail -8
  fi

  # 10.5 vlm 三件套① select_segments（选帧；B 逻辑 video_dir=项目名 --ep=本集）。
  #      vlm 只读 ASR 判 has_asr（有无台词），台词不嵌进骨架、只读不写 ASR 的东西；
  #      缓存判断不依赖 audio（vlm 不依赖 ASR）
  local sel_skel
  sel_skel="$out_root/vlm/${vh}_skeleton.json"
  if [ -f "$sel_skel" ] && [ "$sel_skel" -nt "$body_dep_sk" ]; then
    echo "[10.5/11] vlm select: cached"
  else
    echo "[10.5/11] vlm select (选帧)..."
    conda_activate
    python3 vlm/select_segments.py "$project_label" --ep "$vh" 2>&1 | tail -5
  fi

  # 10.6 vlm 三件套② fuse_segments（融合拼图；读 select 骨架 + frames + DINO）
  local fused_out
  fused_out="$out_root/vlm/${vh}_fused.json"
  if [ -f "$fused_out" ] && [ "$fused_out" -nt "$sel_skel" ]; then
    echo "[10.6/11] vlm fuse: cached"
  else
    echo "[10.6/11] vlm fuse (融合拼图)..."
    conda_activate
    python3 vlm/fuse_segments.py "$project_label" --ep "$vh" 2>&1 | tail -5
  fi

  # 10.7 vlm 三件套③ submit_segments（送检 VLM desc；vllm continuous batching，
  #      落 vlm/<vhash>_desc.json；vllm env 跑，禁 amaterasu）——pipeline 终点
  local desc_out
  desc_out="$out_root/vlm/${vh}_desc.json"
  if [ -f "$desc_out" ] && [ "$desc_out" -nt "$fused_out" ]; then
    echo "[10.7/11] vlm desc: cached"
  else
    echo "[10.7/11] vlm desc (Qwen3-VL-4B vllm, amaterasu)..."
    conda_activate
    python3 vlm/submit_segments.py "$project_label" --ep "$vh" 2>&1 | tail -6
  fi

  echo "=== [$vid_name] 完成 ==="
}

# ── 步骤 8: face_recognition 全局（B 逻辑专属）──
# $1=out_root  $2=project_label

run_global_face_recognition() {
  local out_root="$1" project_label="$2"
  local pt_path="$out_root/visual/face_head_fusion/person_timeline.json"
  local pt_meta="$out_root/visual/face_head_fusion/person_timeline_meta.json"

  if [ -f "$pt_path" ] && [ -f "$pt_meta" ] &&
     grep -q '"n_tracks"' "$pt_meta"; then
    echo "[ 8/12] face_recognition (全局): cached"
    return
  fi

  echo "[ 8/12] face_recognition (全局)..."
  conda_activate
  OUT_ROOT="$out_root" python3 visual/face_recognition.py "" --dino-filter --project "$project_label" 2>&1 | tail -8
}

# ── 处理一个文件夹（按视频数判定 A/B 逻辑）──

process_folder() {
  local folder="$1"
  local folder_name
  folder_name=$(basename "$folder")

  # 输入素材允许套娃（一直翻文件夹直到找到视频）；输出目录保持标准平铺格式
  local FOLDER_VIDEOS=()
  while IFS= read -r -d '' f; do
    FOLDER_VIDEOS+=("$f")
  done < <(find "$folder" -type f \
    \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.m4v" \) \
    -print0 | sort -z)

  local n_vids=${#FOLDER_VIDEOS[@]}
  if [ "$n_vids" -eq 0 ]; then
    echo "[项目] $folder_name 里没有视频，跳过。"
    return
  fi

  # 输出文件夹：B 逻辑 output/<文件夹名>/（传文件夹本身，套娃深层视频取顶层文件夹名；OUT_ROOT 显式覆盖）
  local out_root
  out_root=$(get_out_root "$folder")

  echo ""
  echo "############################################################"
  echo "# 项目: $folder_name"
  echo "# 输出: $out_root/preproc/"
  echo "############################################################"

  if [ "$n_vids" -eq 1 ]; then
    echo "[项目] $folder_name 只有 1 个视频 → A 逻辑（无全局合并）"
    local mp4="${FOLDER_VIDEOS[0]}"
    local vid_name
    vid_name=$(basename "$mp4"); vid_name="${vid_name%.*}"

    # ── 步骤 1-4: 逐视频 ──
    process_video_steps_1_4 "$mp4" "$folder_name" "$out_root"

    # ── 步骤 5: DINO 每视频 ──
    run_dino "$out_root" "$folder_name" "$vid_name" "$mp4"

    # ── 步骤 6-12: 全流程 ──
    process_video_steps_6_12 "$mp4" "$folder_name" "$out_root"
  else
    echo "[项目] $folder_name 共 $n_vids 个视频 → B 逻辑（全局合并）"

    # ── 步骤 1-4: 逐视频 ──
    for mp4 in "${FOLDER_VIDEOS[@]}"; do
      process_video_steps_1_4 "$mp4" "$folder_name" "$out_root"
    done

    # ── 步骤 5: DINO 逐视频──
    for mp4 in "${FOLDER_VIDEOS[@]}"; do
      local bvid_d
      bvid_d=$(basename "$mp4"); bvid_d="${bvid_d%.*}"
      run_dino "$out_root" "" "$bvid_d" "$mp4"
    done

    # ── 步骤 6: face_detect 逐视频 ──
    for mp4 in "${FOLDER_VIDEOS[@]}"; do
      run_face_detect "$out_root" "$mp4" "$folder_name"
    done

    # ── 步骤 7: dedup 逐视频──
    for mp4 in "${FOLDER_VIDEOS[@]}"; do
      local vh
      vh=$(content_hash "$mp4")
      local ddm_meta ddm_sk dino_meta_d
      ddm_meta="$out_root/visual/dedup/${vh}_model_meta.json"
      ddm_sk="$out_root/visual/dedup/${vh}_skeleton.json"
      dino_meta_d="$out_root/visual/dino/${vh}_model_meta.json"
      if [ -f "$ddm_sk" ] && [ -f "$ddm_meta" ] &&
         grep -q '"dino_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"' "$ddm_meta" &&
         grep -q '"face_map_used": true' "$ddm_meta" &&
         [ "$ddm_sk" -nt "$dino_meta_d" ]; then
        echo "[ 7/12] dedup [$vh]: cached"
      else
        echo "[ 7/12] dedup [$vh]..."
        conda_activate
        OUT_ROOT="$out_root" python3 visual/dedup.py "$vh" 2>&1 | tail -5
      fi
    done

    # ── 步骤 7.5: global_cos 全局（挨个读目录下所有 dedup 骨架 → 全局聚簇 → gc_skeleton.json）──
    local gc_sk
    gc_sk="$out_root/visual/global_cos/gc_skeleton.json"
    if [ -f "$gc_sk" ]; then
      echo "[7.5/12] global_cos: cached"
    else
      echo "[7.5/12] global_cos（全局聚簇）..."
      conda_activate
      OUT_ROOT="$out_root" python3 visual/global_cos.py "$folder_name" 2>&1 | tail -5
    fi

    # ── 步骤 9.5: body_detect 全局（人体检测；每项目全局一份 body_bbox.json，键对齐 gc
    #      帧名 <hash>_f<fn>；fuse 读身体骨架做主体/路人分群；graph_merge 已废弃）──
    local body_bbox
    body_bbox="$out_root/visual/body_detect/body_bbox.json"
    if [ -f "$body_bbox" ] && [ "$body_bbox" -nt "$gc_sk" ]; then
      echo "[9.5/11] body_detect: cached"
    else
      echo "[9.5/11] body_detect（全局）..."
      conda_activate
      OUT_ROOT="$out_root" python3 visual/body_detect.py "$folder_name" 2>&1 | tail -5
    fi

    # ── 步骤 8: face_recognition 全局 ──
    run_global_face_recognition "$out_root" "$folder_name"

    # ── 步骤 9-12: 逐视频（人物段偏移 = 前集 dedup scenes 累计）──
    person_offset=0
    for mp4 in "${FOLDER_VIDEOS[@]}"; do
      process_video_steps_9_12 "$mp4" "$folder_name" "$out_root" "$person_offset"
      local vh; vh=$(content_hash "$mp4")
      person_offset=$((person_offset + $(python3 -c "import json;print(len(json.load(open('$out_root/visual/dedup/${vh}_skeleton.json'))['scenes']))")))
    done
  fi
}

# ── 分发 ──

# 1) 先处理 input/ 直接视频（A 逻辑；ONLY_VIDEO 指定时只跑该视频）
for mp4 in "${VIDEO_FILES[@]}"; do
  vid_name=$(basename "$mp4")
  vid_name="${vid_name%.*}"
  if [ -n "$ONLY_VIDEO" ] && [ "$ONLY_VIDEO" != "$vid_name" ]; then
    continue
  fi
  out_root=$(get_out_root "$mp4")
  process_video_steps_1_4 "$mp4" "" "$out_root"
  run_dino "$out_root" "" "$vid_name" "$mp4"
  process_video_steps_6_12 "$mp4" "" "$out_root"
done

# 2) 再处理文件夹（各自按视频数判定；ONLY_VIDEO 过滤）
for folder in "${FOLDERS[@]}"; do
  if [ -n "$ONLY_VIDEO" ] && [ "$ONLY_VIDEO" != "$(basename "$folder")" ]; then
    continue
  fi
  process_folder "$folder"
done

echo ""
echo "===== $(date) 全部完成 ====="
