"""Opt-in prefill experiments using unchanged model libraries and cache storage."""

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import tvm
import tvm_ffi
import tvm_tirx
from tvm.relax.frontend.nn.llm import _kernel_common, _prefill_kernels
from tvm_tirx.target import Target

ROOT = Path(__file__).resolve().parents[3]
TARGET = Target(
    {"kind": "cuda", "arch": "sm_72", "max_shared_memory_per_block": 49152}, host="llvm"
)


def build_tir(variant):
    """Clone the pinned factories, leaving the installed TVM sources untouched."""
    assert variant in ("bounds", "width")
    source = inspect.getsource(_prefill_kernels)
    tree = ast.parse(source)
    replacements = []
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in (
            "_attention_prefill",
            "_attention_prefill_ragged",
        ):
            continue
        code = "".join(lines[node.lineno - 1 : node.end_lineno])
        loop = "for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):"
        assert code.count(loop) == 1
        local = node.name == "_attention_prefill"
        low = (
            "T.if_then_else(causal > 0, T.max(0, kv_chunk_len[0] - "
            "T.max(sliding_window_size - LH_start // group_size - 1, 0)), 0)"
            if local
            else "0"
        )
        high = (
            "T.if_then_else(causal > 0, T.max(0, T.min(kv_chunk_len[0], kv_chunk_len[0] - "
            "(q_indptr[b_idx + 1] - q_indptr[b_idx]) + T.min((LH_start + tile_x - 1) // "
            "group_size + 1, q_indptr[b_idx + 1] - q_indptr[b_idx]))), kv_chunk_len[0])"
        )
        if local:
            low = f"({low} if sliding_window else 0)"
            high = f"(kv_chunk_len[0] if sliding_window else {high})"
        code = code.replace(
            loop, f"for iterator in T.serial(({low}) // tile_z, T.ceildiv({high}, tile_z)):"
        )
        if local and variant == "width":
            marker = "    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z ="
            assert code.count(marker) == 1
            code = code.replace(
                marker, "    physical_d = d\n    if sliding_window:\n        d = 256\n" + marker
            )
            # Only external tensors retain the padded storage stride. Shared memory and
            # arithmetic use d=256; the native wrapper clears the unused output half.
            code = code.replace("(total_len, h_q, d)", "(total_len, h_q, physical_d)")
            code = code.replace(
                "(max_num_pages, 2, h_kv, page_size, d)",
                "(max_num_pages, 2, h_kv, page_size, physical_d)",
            )
            code = code.replace(
                "[max_num_pages, 2, h_kv, page_size, d]",
                "[max_num_pages, 2, h_kv, page_size, physical_d]",
            )
        replacements.append((node.lineno - 1, node.end_lineno, code))
    assert len(replacements) == 2
    for start, stop, code in reversed(replacements):
        lines[start:stop] = [code + "\n"]
    folder = ROOT / "dist/prefill-experiments"
    folder.mkdir(parents=True, exist_ok=True)
    common_source = inspect.getsource(_kernel_common)
    # Exact zero is required when all keys of a row are masked: the original
    # finite sentinel gives exp2(-50000 - -50000)=1 in that case.
    common_source = common_source.replace("T.exp2(-5e4 - m_new[i])", "T.float32(0)", 1)
    common_path = folder / f"common_{variant}.py"
    common_path.write_text(common_source)
    common_name = f"tvm.relax.frontend.nn.llm._xavier_common_{variant}"
    common_spec = importlib.util.spec_from_file_location(common_name, common_path)
    common_module = importlib.util.module_from_spec(common_spec)
    sys.modules[common_name] = common_module
    common_spec.loader.exec_module(common_module)
    path = folder / f"factories_{variant}.py"
    path.write_text(
        "".join(lines).replace(
            "from ._kernel_common import", f"from ._xavier_common_{variant} import"
        )
    )
    name = f"tvm.relax.frontend.nn.llm._xavier_{variant}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    funcs = {"ragged": module._attention_prefill_ragged(2, 8, 512, 512, "float16", {}, TARGET)}
    for local in (0, 1):
        funcs[f"paged{local}"] = module._attention_prefill(
            2, 8, 512, "float16", bool(local), {}, TARGET, sliding_window_size=512
        )
    funcs = {name: fn.with_attr("global_symbol", name) for name, fn in funcs.items()}
    executable = tvm.compile(tvm_tirx.IRModule(funcs), target=TARGET)
    output = folder / f"{variant}.so"
    executable.export_library(str(output))
    return output


def install(variant, target_config=None):
    """Intercept cache construction once; all per-layer callbacks remain native."""
    names = ("bounds", "width", "wmma", "hybrid", "hybrid-width", "paged", "paged-prefill")
    assert variant in names
    native = tvm_ffi.load_module(str(ROOT / "dist/prefill-experiments/native.so"))
    # "paged": hybrid-width for the current chunk, tensor cores over the cached pages.
    # "paged-prefill": the same for chunks of more than four rows only, so that
    # decode and verification keep the arithmetic of hybrid-width.
    paged = variant if variant.startswith("paged") else None
    if paged:
        variant = "hybrid-width"
    hybrid = variant in ("hybrid", "hybrid-width")
    tir_variant = "width" if hybrid else variant
    module = (
        native
        if variant == "wmma"
        else tvm_ffi.load_module(str(ROOT / f"dist/prefill-experiments/{tir_variant}.so"))
    )
    key = "vm.builtin.paged_attention_kv_cache_create"
    original = tvm_ffi.get_global_func(key)
    receipt = {"variant": paged or variant, "target_caches": 0}
    if variant == "hybrid-width":
        path = Path(target_config or ROOT / "dist/target-16k/mlc-chat-config.json")
        config = json.loads(path.read_text())["model_config"]["text_config"]
        assert (
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            config["global_head_dim"],
        ) == (8, 2, 256, 512)
        layers = config["layer_types"]
        physical_layers = config["num_hidden_layers"] - config["num_kv_shared_layers"]
        assert len(layers) == config["num_hidden_layers"] and physical_layers > 0
        assert set(layers) <= {"sliding_attention", "full_attention"}
        layout = "".join("l" if x == "sliding_attention" else "g" for x in layers[:physical_layers])
        receipt["physical_layer_layout"] = layout
        for suffix in ("attention_with_fused_qkv", "attention_with_shared_kv", "self_attention"):
            name = "vm.builtin.attention_kv_cache_" + suffix
            fn = tvm_ffi.get_global_func(name)
            tvm_ffi.register_global_func(
                name, native["wrap_layer_width"](fn, layout), override=True
            )

    def create(*args):
        args = list(args)
        if tuple(map(int, args[2:6])) == (8, 2, 512, 512):
            assert len(args) in (28, 29) and int(args[0][5]) == 512 and int(args[8]) == 0
            if variant == "hybrid-width":
                assert tuple(map(int, args[1])) == (0, physical_layers)
                assert int(args[0][2]) <= config["sliding_window"] == 512
            before = list(args)
            for index, name in ((15, "ragged"), (16, "paged0"), (18, "paged1")):
                old = args[index]
                assert len(old) == 2 and str(old[0]) == "tirx"
                clear = int((variant == "width" or hybrid) and index == 18)
                candidate = native[name] if hybrid and index == 15 else module[name]
                if variant == "hybrid-width" and index == 15:
                    candidate = native["ragged_layer_width"]
                wrapped = native["wrap_prefill"](old[1], candidate, clear, 7 if index == 15 else 8)
                args[index] = tvm_ffi.convert([old[0], wrapped])
            # Global and local, prefill and decode: every call over the cached pages.
            for index in {"paged": (16, 17, 18, 19), "paged-prefill": (16, 18)}.get(paged, ()):
                old = args[index]
                assert len(old) == 2 and str(old[0]) == "tirx"
                wrapped = native["wrap_paged"](old[1], int(index >= 18), index % 2)
                if paged == "paged-prefill":
                    wrapped = native["wrap_prefill"](before[index][1], wrapped, 0, 8)
                args[index] = tvm_ffi.convert([old[0], wrapped])
            receipt["target_caches"] += 1
            print("Prefill experiment installed:", receipt, flush=True)
        return original(*args)

    tvm_ffi.register_global_func(key, create, override=True)
    return receipt
