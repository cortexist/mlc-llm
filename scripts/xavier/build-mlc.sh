#!/bin/bash
# Build + install mlc_llm (C++ engine + Python package) against the installed TVM/tirx.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$MLC_SRC"
date
uv pip install -v --no-build-isolation --no-deps . \
  -C cmake.define.USE_CUDA=ON \
  -C cmake.define.USE_CUBLAS=ON \
  -C cmake.define.USE_THRUST=OFF \
  -C cmake.define.USE_CUTLASS=OFF \
  -C cmake.define.USE_NVTX=OFF \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=72 \
  -C cmake.define.CMAKE_SHARED_LINKER_FLAGS=-pthread \
  -C cmake.define.CMAKE_EXE_LINKER_FLAGS=-pthread \
  -C cmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache \
  -C cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
cd /tmp && python -c "import mlc_llm, tvm; print(\"mlc_llm OK\", mlc_llm.__file__)"
date
