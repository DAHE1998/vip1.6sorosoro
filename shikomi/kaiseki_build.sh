#!/bin/bash
# shikomi/kaiseki_build.sh — 编译 kaiseki（单路 singlepass 解析器；cubin 复用现役 CUDA_KaKu.cubin）
# 用法: bash kaiseki_build.sh（需在 shikomi/ 目录下执行）
#
# 全部路径相对项目根（自包含，零硬编码，2026-09-01 定稿）：
#   env/sorosoro/                    自带 conda 环境（python/torch/vllm + libcrypto）
#   env/sorosoro/.../nvidia/cu13/    NVIDIA pip CUDA 13 工具链（cuda.h / nvcc）
#   include/ffnvcodec/               nv-codec-headers（NVCUVID）
#   include/nvjpeg/                  nvjpeg.h（从 CUDA toolkit 收编，随项目走）
#   shikomi/ffmpeg_static/           自包含静态库 libavformat/avcodec/avutil（无外部依赖）
#   ffmpeg/include/                  FFmpeg 9.0 标准布局头（libav*/xx.h；与静态库同版本 9.0。
#                                    ffmpeg_static/include 缺内部头（codec_desc.h 等），勿用）
# 链接 -lcuda/-lz/-lm/-ldl 走系统默认路径（宿主只出 CPU/GPU/驱动，勿写死路径）
#
# 产物: shikomi/kaiseki（kaiseki.c + hako.c；hako 静态链 ffmpeg 自包含库）
set -e
cd "$(dirname "$0")"
PROJ_ROOT="$(cd .. && pwd)"
ENV_DIR="$PROJ_ROOT/env/sorosoro"
CU13="$ENV_DIR/lib/python3.11/site-packages/nvidia/cu13"

[ -d "$ENV_DIR" ] || { echo "[ERR] 缺自带环境 $ENV_DIR"; exit 1; }
[ -f "$PROJ_ROOT/shikomi/CUDA_KaKu.cubin" ] || echo "[WARN] 缺 CUDA_KaKu.cubin（kaiseki 运行时加载，需先 nvcc 编）"

gcc -O2 -o kaiseki kaiseki.c hako.c \
  -I "$ENV_DIR/include" \
  -I "$CU13/include" \
  -I "$PROJ_ROOT/include/ffnvcodec" \
  -I "$PROJ_ROOT/include/nvjpeg" \
  -I "$PROJ_ROOT/ffmpeg/include" \
  -lcuda -lpthread -lcrypto \
  -L "$ENV_DIR/lib" \
  -L ./ffmpeg_static/lib -lavformat -lavcodec -lavutil -lz -lm -ldl
echo "OK: kaiseki 编译完成"
