#!/bin/bash
# run_batch.sh — B 路线试跑：vivant EP01+EP02（步骤 12 = 现役 ModelScope API 划章）
# 用法: bash run_batch.sh
# 依赖: input/vivant/EP0*/ 素材；API 配置优先环境变量，否则读 chapter/api_key.txt
#       （第 1 行 base_url，第 2 行 key；缺 key 直接退出）
# 产物: output/vivant/ 下全流水线产物（OUT_ROOT 可覆盖）
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export CONDA_SH=/home/dahe/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=amaterasu
export OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/output/vivant}"
API_KEY_FILE="$SCRIPT_DIR/chapter/api_key.txt"
if [ -f "$API_KEY_FILE" ]; then
  export LLM_API_BASE="${LLM_API_BASE:-$(sed -n '1p' "$API_KEY_FILE" | tr -d '\r\n ')}"
  export LLM_API_KEY="${LLM_API_KEY:-$(sed -n '2p' "$API_KEY_FILE" | tr -d '\r\n ')}"
fi
export LLM_API_BASE="${LLM_API_BASE:-https://api-inference.modelscope.cn/v1}"
if [ -z "$LLM_API_KEY" ]; then
  echo "[run] 缺 API key（chapter/api_key.txt 或环境变量 LLM_API_KEY）" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT"
exec bash batch_pipeline.sh
