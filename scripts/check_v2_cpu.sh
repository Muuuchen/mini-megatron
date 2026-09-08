#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Not an all-green starter command: unfinished cases deliberately fail their preflight.
# Run one selected case first: CASES=fsdp bash scripts/check_v2_cpu.sh
export OMP_NUM_THREADS=1
for case in ${CASES:-transport sp cp zero1 zero2 fsdp moe pipeline}; do
  torchrun --standalone --nproc_per_node=2 -m megatron_mini.lab --case "$case"
done
