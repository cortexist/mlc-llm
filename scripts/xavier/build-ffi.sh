#!/bin/bash
# Build + install apache-tvm-ffi and its orcjit addon from the source pinned inside tvm-tirx,
# so ffi / tirx / tvm / mlc all share one FFI ABI (the PyPI wheel lags the pinned source).
set -euo pipefail
source "$(dirname "$0")/env.sh"
F="$TIRX_SRC/3rdparty/tvm-ffi"
date
uv pip install -q "cython>=3.2.8" setuptools-scm "scikit-build-core>=0.10.0,!=1.1.0"
cd "$F"
uv pip install -v --no-build-isolation --reinstall-package apache-tvm-ffi . \
  -C cmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache -C cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
# orcjit addon skipped: needs LLVM >= 23.1.1 (focal has <= 21); tirx imports it optionally (JIT only).
cd /tmp && python -c "import tvm_ffi; print(\"tvm_ffi OK\", tvm_ffi.__version__)"
date
