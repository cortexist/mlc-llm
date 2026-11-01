#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/xavier/env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python" MLC_LIBRARY_PATH="$PWD/build" OPENBLAS_NUM_THREADS=1
for variant in baseline bounds width hybrid; do
  python -u scripts/run_monitored.py --workload-log "logs/prefill-${variant}.log" -- \
    timeout --kill-after=10s 300 python -u scripts/bench_xavier_dflash.py \
    --prefill-variant "$variant" --target-lib dist/libs/target-dflash-dp4a-attn-sm72.so \
    --draft-tokens 2 --cases chat,448,1768 --tokens 128 --warmups 1 --repeats 2 \
    --output "logs/prefill-${variant}.json"
done
