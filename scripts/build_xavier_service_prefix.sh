#!/usr/bin/env bash
# Stop all GPU inference before invoking. Service lifecycle belongs to the caller.
set -euo pipefail
source scripts/xavier/env.sh
patch_file="$MLC_SRC/patches/tvm-linear-sliding-prefix.patch"
if git -C "$TVM_SRC" apply --check "$patch_file" 2>/dev/null; then
    git -C "$TVM_SRC" apply "$patch_file"
else
    git -C "$TVM_SRC" apply --reverse --check "$patch_file"
fi
cmake --build "$TVM_SRC/build" --target tvm_runtime_extra -j2
cp "$TVM_SRC/build/lib/libtvm_runtime_extra.so" "$MLC_SRC/build/lib/"
# Python imports TVM before MLC and loads this package copy first.
tvm_package=$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("tvm").origin).parent)')
cp "$TVM_SRC/build/lib/libtvm_runtime_extra.so" "$tvm_package/lib/"
cmake --build "$MLC_SRC/build" -j2
python -c 'import tvm, tvm_ffi; assert tvm_ffi.get_global_func("vm.builtin.xavier_linear_fork_supported")()'
