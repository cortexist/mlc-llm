#!/usr/bin/env bash
# Requires the compiler stack and model inputs described in README.md.
set -euo pipefail
source "$(dirname "$0")/xavier/env.sh"
DFLASH_SRC=$(cd "$(dirname "$0")/.." && pwd)
cd "$DFLASH_SRC"
export PYTHONPATH="$DFLASH_SRC/python"
export MLC_LIBRARY_PATH="$DFLASH_SRC/build"
export MLC_GEMMA4_TEXT_ONLY=1
export MLC_DFLASH_CUDA_GRAPH=1
export TVM_NUM_THREADS=2
export OMP_NUM_THREADS=2
mkdir -p logs dist/libs
if [[ ! -f 3rdparty/tvm/CMakeLists.txt ]]; then
  echo "Initialize dependencies with: git submodule update --init --recursive" >&2
  exit 1
fi
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_CUBLAS=ON -DUSE_THRUST=OFF -DUSE_CUTLASS=OFF \
  -DUSE_NVTX=OFF -DUSE_LLVM=OFF -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_SHARED_LINKER_FLAGS=-pthread -DCMAKE_EXE_LINKER_FLAGS=-pthread \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  > logs/configure.log 2>&1
cmake --build build --target mlc_llm mlc_llm_module -j2 > logs/build.log 2>&1
python scripts/prepare_xavier_dflash.py > logs/prepare.log 2>&1
python -m mlc_llm convert_weight dist/draft-hf --quantization q0f16 \
  --model-type dflash --device cpu -o dist/draft > logs/convert-draft.log 2>&1
MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_VERIFY_THREE=1 MLC_JP5_Q4_CUBLAS=1 taskset -c 0,1 python -m mlc_llm compile \
  dist/target/mlc-chat-config.json --device cuda --opt O2 \
  -o dist/libs/target-dflash-graph-b3-sm72.so > logs/compile-target-graph-b3.log 2>&1
env -u MLC_JP5_Q4_CUBLAS taskset -c 0,1 python -m mlc_llm compile \
  dist/draft/mlc-chat-config.json --device cuda --opt O2 \
  -o dist/libs/draft-cached-tc-sm72.so > logs/compile-draft-tc.log 2>&1
python - <<'PY'
import json
from pathlib import Path
p = Path("dist/draft/mlc-chat-config.json")
c = json.loads(p.read_text())
c["model_config"]["cache_context_kv"] = False
p.with_name("mlc-chat-config-uncached.json").write_text(json.dumps(c, indent=2) + "\n")
PY
env -u MLC_JP5_Q4_CUBLAS taskset -c 0,1 python -m mlc_llm compile \
  dist/draft/mlc-chat-config-uncached.json --device cuda --opt O2 \
  -o dist/libs/draft-uncached-tc-sm72.so > logs/compile-draft-uncached-tc.log 2>&1
python tests/python/op/test_dflash_argmax.py --output logs/argmax.json
TVM_NUM_THREADS=2 python tests/python/model/test_dflash_context_cache.py
python scripts/check_draft_reference.py > logs/draft-reference.log
python scripts/bench_draft_core.py --variant-suffix=-tc --output logs/draft-core-tc-control.json
