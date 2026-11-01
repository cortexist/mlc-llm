#!/usr/bin/env bash
# Source from an initialized JetPack 5 / CUDA 11.4 build environment.
export MLC_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VIRTUAL_ENV="${MLC_XAVIER_VENV:-$HOME/Workspaces/mlc-venv}"
export CUDA_HOME=/usr/local/cuda
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export TVM_SRC="$MLC_SRC/3rdparty/tvm"
export TIRX_SRC="$TVM_SRC/3rdparty/tvm-tirx"
export LLVM_CONFIG="${LLVM_CONFIG:-/usr/lib/llvm-20/bin/llvm-config}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export TVM_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$MLC_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export MLC_LIBRARY_PATH="$MLC_SRC/build"
