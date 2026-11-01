#!/usr/bin/env bash
# Run under run_monitored.py; this deliberately builds and tests sequentially.
set -euo pipefail
source "$(dirname "$0")/xavier/env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python"
export MLC_LIBRARY_PATH="$PWD/build"
export TVM_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MLC_GEMMA4_TEXT_ONLY=1 MLC_DFLASH_CUDA_GRAPH=1
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_VERIFY_THREE=1 MLC_JP5_Q4_CUBLAS=1
export MLC_XAVIER_Q4=1
mkdir -p logs dist/libs
variant=dp4a-attn
export MLC_XAVIER_ATTN=1
if [[ ${1:-} == --q4-only ]]; then
  variant=dp4a
  unset MLC_XAVIER_ATTN
elif [[ $# != 0 ]]; then
  echo "Usage: $0 [--q4-only]" >&2
  exit 2
fi
taskset -c 0,1 python tests/python/op/test_xavier_q4.py > logs/xavier-q4-test.log 2>&1
if [[ $variant == dp4a-attn ]]; then
  taskset -c 0,1 python tests/python/op/test_xavier_attention.py \
    --output logs/xavier-attention-test.json > logs/xavier-attention-test.log 2>&1
  taskset -c 0,1 python tests/python/op/test_xavier_cache.py > logs/xavier-cache-test.log 2>&1
fi
taskset -c 0,1 python -m mlc_llm compile dist/target/mlc-chat-config.json \
  --device cuda --opt O2 -o "dist/libs/target-dflash-${variant}-sm72.so" \
  > "logs/compile-target-${variant}.log" 2>&1
