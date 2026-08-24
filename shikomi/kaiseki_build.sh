#!/bin/bash
# shikomi/kaiseki_build.sh — 编译 kaiseki（单路 singlepass 解析器；cubin 复用现役 CUDA_KaKu.cubin）
# 用法: bash kaiseki_build.sh（需在 shikomi/ 目录下执行）
# 依赖: sorosoro conda 环境；cuda/ffnvcodec 头；ffmpeg_static/ 下自包含静态库(libavformat/avcodec/avutil)
# 产物: shikomi/kaiseki（由 kaiseki.c + hako.c 编译；hako 静态链 ffmpeg 自包含库）
set -e
cd "$(dirname "$0")"
source /home/dahe/miniconda3/etc/profile.d/conda.sh
conda activate sorosoro
/usr/bin/gcc -O2 -o kaiseki kaiseki.c hako.c \
  -I $CONDA_PREFIX/include \
  -I $CONDA_PREFIX/targets/x86_64-linux/include \
  -I /usr/local/cuda/include \
  -I /home/dahe/videotools/vip/vip1.6zenkiru/include/ffnvcodec \
  -I ./ffmpeg_static/include \
  -lcuda -lpthread -lcrypto \
  -L $CONDA_PREFIX/lib -L /usr/lib/x86_64-linux-gnu \
  -L ./ffmpeg_static/lib -lavformat -lavcodec -lavutil -lz -lm -ldl
echo "OK: kaiseki 编译完成"
