#!/bin/bash
# remux_and_run.sh — 无损转封装 mkv→mp4 + 重启 B 路线（素材/输出均在脚本所在目录下）
# 用法: bash remux_and_run.sh（逐个 mkv→mp4 转封装并改名 .off，再后台重启 run_batch.sh）
# 依赖: input/vivant/EP0*/ 下 .mkv 素材；ffmpeg（-c copy 无损转封装）
# 产物: 同名 .mp4（转封装）+ output/vivant/batch_run.log（重启日志）
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

for d in input/vivant/*EP0*/; do
  mkv=$(find "$d" -maxdepth 1 -name '*.mkv' | head -1)
  mp4="${mkv%.mkv}.mp4"
  if [ -f "$mp4" ]; then
    echo "[remux] 已有 $mp4，跳过"
  else
    echo "[remux] $mkv -> $mp4"
    ffmpeg -y -v error -i "$mkv" -map 0:v -map 0:a? -c copy "$mp4"
  fi
  mv "$mkv" "$mkv.off"
  echo "[remux] mkv 已改名 .off"
done

echo "═══ 最终素材清单（find 匹配）═══"
find input/vivant -type f \( -iname '*.mp4' -o -iname '*.mkv' \) | sort

# 重启
mkdir -p output/vivant
setsid nohup bash run_batch.sh </dev/null > output/vivant/batch_run.log 2>&1 &
echo "[run] 已重启 PID=$!"
sleep 5
head -12 output/vivant/batch_run.log
