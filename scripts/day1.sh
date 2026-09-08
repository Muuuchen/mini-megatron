#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python -m megatron_mini.train --mode baseline --device cpu --steps 20 --output runs/baseline_cpu.jsonl
python -m megatron_mini.tasks
