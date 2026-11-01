#!/bin/bash
# Full chain: tvm-ffi (+orcjit) -> tvm-tirx -> tvm -> mlc_llm. Each step logs to stdout.
set -euo pipefail
d="$(dirname "$0")"
for step in ${STEPS:-ffi tirx tvm mlc}; do
  echo "===== STEP $step ====="
  "$d/build-$step.sh"
done
echo "===== ALL DONE ====="
