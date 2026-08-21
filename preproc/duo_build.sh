#!/bin/bash
# preproc/duo_build.sh — 编译 duo_analyze（双线·线 A；cubin 复用现役 unified_kernels.cubin）
# 用法: bash duo_build.sh（需在 preproc/ 目录下执行）
# 依赖: amaterasu conda 环境；/usr/local/cuda/include + ffnvcodec 头 + libnvjpeg/libcrypto
# 产物: preproc/duo_analyze 可执行文件
set -e
cd "$(dirname "$0")"
source /home/dahe/miniconda3/etc/profile.d/conda.sh
conda activate amaterasu
/usr/bin/gcc -O2 -o duo_analyze duo_analyze.c mp4_mov.c \
  -I $CONDA_PREFIX/include \
  -I $CONDA_PREFIX/targets/x86_64-linux/include \
  -I /usr/local/cuda/include \
  -I /home/dahe/videotools/vip/vip1.6zenkiru/include/ffnvcodec \
  -lz -lm -ldl -lcuda -lpthread -lcrypto \
  -L $CONDA_PREFIX/lib -L /usr/lib/x86_64-linux-gnu
echo "OK: duo_analyze 编译完成"
