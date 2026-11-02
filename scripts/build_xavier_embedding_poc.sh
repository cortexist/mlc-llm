#!/usr/bin/env bash
# Run from the repository root, with inference stopped.
set -euo pipefail
source scripts/xavier/env.sh
export CMAKE_BUILD_PARALLEL_LEVEL=2 TVM_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset MLC_XAVIER_ATTN MLC_XAVIER_Q4 MLC_JP5_Q4_CUBLAS MLC_DFLASH_CUDA_GRAPH MLC_JP5_VERIFY_THREE
mkdir -p dist/libs
python scripts/prepare_embeddinggemma2_text.py "$HOME/Models/embeddinggemma-2" \
    dist/embeddinggemma2-text-hf
python -m mlc_llm convert_weight dist/embeddinggemma2-text-hf \
    --model-type embedding_gemma2_poc --quantization q0f32 --device cpu \
    -o dist/embeddinggemma2-text-q0f32
python -m mlc_llm gen_config dist/embeddinggemma2-text-hf \
    --model-type embedding_gemma2_poc --quantization q0f32 --conv-template gemma4_instruction \
    --context-window-size 128 --prefill-chunk-size 128 --max-batch-size 1 \
    -o dist/embeddinggemma2-text-q0f32
taskset -c 0,1 python -m mlc_llm compile \
    dist/embeddinggemma2-text-q0f32/mlc-chat-config.json --device cuda \
    --opt 'flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0' \
    -o dist/libs/embeddinggemma2-text-f32-s128-sm72.so
