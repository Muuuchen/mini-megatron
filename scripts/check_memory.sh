#!/usr/bin/env bash
set -euo pipefail
# All student cases fail-fast while TODO remains. CPU transfer checks are emulation.
DEVICE=${DEVICE:-cpu}
PROCS=${PROCS:-2}
for case in residency zero3 sac zero3_sac offload_transfer offload_optimizer offload_param offload_activation offload_checkpoint; do
  torchrun --standalone --nproc_per_node="$PROCS" -m megatron_mini.lab --case "$case" --device "$DEVICE"
done
# Full zero3_cpu_sac recipe is a separate T55/T71 integration acceptance, not this script.
