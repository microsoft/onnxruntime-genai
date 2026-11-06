#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash run_benchmark.sh MODEL_DIRECTORY OUTPUT_JSON" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1
export ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1
"${PYTHON:-python}" "$SCRIPT_DIR/benchmark_qwen_mtp.py" \
  --model "$1" --mode mtp --warmup 1 --repetitions 5 \
  --overlay "$SCRIPT_DIR/width2_overlay.json" --output "$2"
