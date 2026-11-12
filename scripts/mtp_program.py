"""Load a CUDA VM, sharing weights only between identical parameter schemas."""

import json
import os
from pathlib import Path

import tvm_ffi

import mlc_llm  # noqa: F401  Register runtime functions before loading compiled programs.


def release_file_cache(model):
    """Drop the weight files of `model` from the page cache once they are on the GPU.

    Xavier GPU memory is system RAM, and a large allocation fails rather than wait
    for the kernel to reclaim cached file pages.
    """
    for shard in Path(model).glob("params_shard_*.bin"):
        fd = os.open(shard, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


class Program:
    def __init__(self, model, library, share_params_from=None):
        self.module = tvm_ffi.load_module(str(library))
        self.vm = self.module["vm_load_executable"]()
        self.vm["vm_initialization"](2, 0, 1, 0)
        metadata = json.loads(self.vm["_metadata"]())
        self.context_length = metadata["context_window_size"]
        self.parameter_metadata = metadata["params"]
        if share_params_from is not None:
            if self.parameter_metadata != share_params_from.parameter_metadata:
                raise ValueError("Shared model parameter metadata must match exactly")
            self.params = share_params_from.params
            return
        get = tvm_ffi.get_global_func
        get("vm.builtin.tensor_cache.load")(str(model), 2, 0)
        self.params = get("vm.builtin.param_array_from_cache_by_name")(
            [p["name"] for p in metadata["params"]]
        )
        get("vm.builtin.tensor_cache.clear")()
        release_file_cache(model)

    def call(self, name, *args):
        return self.vm[name](*args, self.params)
