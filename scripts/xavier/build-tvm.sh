#!/bin/bash
# Build + install apache-tvm (Relax compiler) from MLC's pinned submodule, against the installed tvm-tirx.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TVM_SRC"
date
uv pip install -v --no-build-isolation --no-deps --reinstall-package apache-tvm . \
  -C cmake.define.TVM_TIRX_USE_CUDA=ON \
  -C cmake.define.TVM_USE_CUBLAS=ON \
  -C cmake.define.TVM_USE_THRUST=OFF \
  -C cmake.define.TVM_USE_CUTLASS=OFF \
  -C cmake.define.TVM_USE_NVTX=OFF \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=72 \
  -C cmake.define.CMAKE_SHARED_LINKER_FLAGS=-pthread \
  -C cmake.define.CMAKE_EXE_LINKER_FLAGS=-pthread \
  -C cmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache \
  -C cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
python -c "import tvm, tvm_tirx; d = tvm_tirx.cuda(0); print(\"tvm OK\", tvm.__version__, d.exist, d.compute_version)"
date
