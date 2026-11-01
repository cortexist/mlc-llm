"""Compile and instantiate the actual cache callback wrapper before a model build."""

import os
from pathlib import Path

import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.dispatch_kv_cache_creation import DispatchKVCacheCreation
from mlc_llm.nn import PagedKVCache, RopeMode


class Probe(nn.Module):
    def create_paged_kv_cache(
        self,
        max_batch_size,
        max_total_seq_len,
        prefill_chunk_size,
        page_size,
        support_sliding_window,
    ):
        return PagedKVCache.create_generic(
            attn_kind=["mha", "mha_sliding"],
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            qk_head_dim=512,
            v_head_dim=512,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=10000,
            dtype="float16",
            layer_sliding_window_size=512,
        )


os.environ["MLC_XAVIER_ATTN"] = "1"
spec = {
    "create_paged_kv_cache": dict(
        max_batch_size=int,
        max_total_seq_len=int,
        prefill_chunk_size=int,
        page_size=int,
        support_sliding_window=int,
        **{"$": {"param_mode": "none", "effect_mode": "none"}},
    )
}
mod, _ = Probe().export_tvm(spec=spec)
target = Target(
    {"kind": "cuda", "arch": "sm_72", "max_shared_memory_per_block": 49152}, host="llvm"
)
mod = DispatchKVCacheCreation(target, False, {})(mod)
exe = relax.build(mod, target=target)
path = Path("dist/libs/xavier-cache-test.so")
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), tvm_tirx.cuda(0))
cache = vm["create_tir_paged_kv_cache"](*[tvm_ffi.Shape([v]) for v in [1, 2048, 512, 16, 1]])
print("Cache callback factory round trip:", type(cache).__name__, flush=True)
