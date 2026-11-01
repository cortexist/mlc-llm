#!/usr/bin/env bash
# From the repo root, with Nemotron serving stopped and its existing artifacts built.
set -euo pipefail
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1 CMAKE_BUILD_PARALLEL_LEVEL=2
export TVM_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
mkdir -p dist/libs
python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
assistant = Path.home() / 'Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant-q0f16-MLC'
for source, destination, context in [
    (root/'dist/target', root/'dist/target-16k', 16384),
    (root/'dist/draft', root/'dist/draft-16k', 16384),
    (root/'dist/gemma4-12b-mtp-target', root/'dist/gemma4-12b-mtp-target-4k', 4096),
    (assistant, root/'dist/gemma4-12b-mtp-assistant-4k', 4096),
]:
    destination.mkdir(exist_ok=True)
    config = json.loads((source/'mlc-chat-config.json').read_text())
    config['context_window_size'] = config['model_config']['context_window_size'] = context
    (destination/'mlc-chat-config.json').write_text(json.dumps(config, indent=2)+'\n')
    for item in source.iterdir():
        if item.name != 'mlc-chat-config.json' and not (destination/item.name).exists():
            (destination/item.name).symlink_to(item.resolve())
PY
opt='flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0'
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_VERIFY_THREE=1 MLC_JP5_Q4_CUBLAS=1
export MLC_XAVIER_Q4=1 MLC_XAVIER_ATTN=1 MLC_DFLASH_CUDA_GRAPH=1
taskset -c 0,1 python -m mlc_llm compile dist/target-16k/mlc-chat-config.json \
    --device cuda --opt "$opt" -o dist/libs/target-dflash-dp4a-attn-16k-sm72.so
unset MLC_XAVIER_Q4 MLC_JP5_Q4_CUBLAS MLC_XAVIER_ATTN
taskset -c 0,1 python -m mlc_llm compile dist/draft-16k/mlc-chat-config.json \
    --device cuda --opt "$opt" -o dist/libs/draft-cached-tc-16k-sm72.so
unset MLC_DFLASH_CUDA_GRAPH MLC_JP5_VERIFY_THREE
taskset -c 0,1 python -m mlc_llm compile dist/gemma4-12b-mtp-assistant-4k/mlc-chat-config.json \
    --device cuda --opt "$opt" -o dist/libs/gemma4-12b-mtp-assistant-4k-sm72.so
export MLC_XAVIER_Q4=1 MLC_JP5_Q4_CUBLAS=1
taskset -c 0,1 python -m mlc_llm compile dist/gemma4-12b-mtp-target-4k/mlc-chat-config.json \
    --device cuda --opt "$opt" -o dist/libs/gemma4-12b-mtp-target-4k-sm72.so
