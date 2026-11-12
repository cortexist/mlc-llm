#!/usr/bin/env bash
# Run from the repository root with the Xavier Python/compiler environment active.
set -euo pipefail
assistant_src="${1:?Pass the original Unified 12B assistant directory}"
target_weights="${2:?Pass the converted q4f16_1 Unified 12B target directory}"
assistant_out="${3:?Pass the output q0f16 assistant directory}"
target_out="dist/gemma4-12b-mtp-target"
export MLC_GEMMA4_TEXT_ONLY=1
export TVM_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
unset MLC_XAVIER_ATTN MLC_XAVIER_Q4 MLC_JP5_Q4_CUBLAS MLC_DFLASH_CUDA_GRAPH
mkdir -p "$assistant_out" "$target_out" dist/libs
if [[ ! -f "$assistant_out/tensor-cache.json" ]]; then
    taskset -c 0,1 python -m mlc_llm convert_weight "$assistant_src" \
        --quantization q0f16 --device cpu -o "$assistant_out"
fi
python -m mlc_llm gen_config "$assistant_src" \
    --quantization q0f16 --conv-template gemma4_instruction \
    --context-window-size 1024 --prefill-chunk-size 512 --max-batch-size 1 \
    -o "$assistant_out"
python - "$target_weights" "$target_out" <<'PY'
import json
import sys
from pathlib import Path

original, out = (Path(p).resolve() for p in sys.argv[1:])
config = json.loads((original / 'mlc-chat-config.json').read_text())
if config['model_type'] != 'gemma4_unified' or config['quantization'] != 'q4f16_1':
    raise ValueError('Expected a converted Unified 12B q4f16_1 target')
config['model_type'] = 'gemma4_unified_mtp_target'
config['context_window_size'] = config['model_config']['context_window_size'] = 1024
(out / 'mlc-chat-config.json').write_text(json.dumps(config, indent=2) + '\n')
for source in original.iterdir():
    if source.name != 'mlc-chat-config.json' and not (out / source.name).exists():
        (out / source.name).symlink_to(source)
PY
opt='flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0'
taskset -c 0,1 python -m mlc_llm compile "$assistant_out/mlc-chat-config.json" \
    --device cuda --opt "$opt" -o dist/libs/gemma4-12b-mtp-assistant-sm72.so
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_Q4_CUBLAS=1 MLC_XAVIER_Q4=1
taskset -c 0,1 python -m mlc_llm compile "$target_out/mlc-chat-config.json" \
    --device cuda --opt "$opt" -o dist/libs/gemma4-12b-mtp-target-sm72.so
# Token selection of the controller, on the GPU.
nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_72 scripts/xavier_mtp_runtime.cu \
    -o dist/libs/xavier-mtp-runtime.so
