#!/usr/bin/env bash
# 学生完成相应 TODO 后再运行。MAX_PROCS=2/4/8；不会用 GPU。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
MAX_PROCS=${MAX_PROCS:-2}
python -m pytest -q --exercises
for case in groups linear embedding loss model; do
  torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case "$case" --tp 2
 done
for axis in tp pp dp; do
  torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --"$axis" 2 --clip 0.25
 done
# PP 的单 microbatch 边界，不应依赖 M >= PP。
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case step --pp 2 --microbatches 1
# PP=4 时也测试 M<PP；使用 4/8 进程须确认主机内存足够。
if (( MAX_PROCS >= 4 )); then
  for pair in "--tp 2 --pp 2" "--tp 2 --dp 2" "--pp 2 --dp 2"; do
    read -r -a axes <<< "$pair"
    torchrun --standalone --nproc_per_node=4 -m tools.check_distributed --case step "${axes[@]}" --clip 0.25
  done
  torchrun --standalone --nproc_per_node=4 -m tools.check_distributed --case step --pp 4 --microbatches 1
fi
if (( MAX_PROCS >= 8 )); then
  torchrun --standalone --nproc_per_node=8 -m tools.check_distributed --case step --tp 2 --pp 2 --dp 2 --clip 0.25
fi
torchrun --standalone --nproc_per_node=2 -m tools.check_distributed --case checkpoint --tp 2
