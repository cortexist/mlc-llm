#!/bin/bash
# Build + install apache-tvm-tirx (TVM low-level compiler: LLVM + CUDA codegen) from MLC's pinned submodule.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TIRX_SRC"
date
uv pip install -v --no-build-isolation --no-deps --reinstall-package apache-tvm-tirx . \
  -C cmake.define.TVM_TIRX_USE_LLVM="$LLVM_CONFIG --ignore-libllvm --link-static" \
  -C cmake.define.TVM_TIRX_USE_CUDA=/usr/local/cuda \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=72 \
  -C cmake.define.TVM_TIRX_USE_Z3=OFF \
  -C cmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache \
  -C cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
python -c "import tvm_tirx; print(\"tvm_tirx OK\", tvm_tirx.__file__)"
date
