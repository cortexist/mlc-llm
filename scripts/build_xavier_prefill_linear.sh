#!/usr/bin/env bash
# From the initialized Xavier checkout; stop GPU serving before running this script.
set -euo pipefail
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1 CMAKE_BUILD_PARALLEL_LEVEL=2
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_VERIFY_THREE=1 MLC_JP5_Q4_CUBLAS=1
export MLC_XAVIER_Q4=1 MLC_XAVIER_ATTN=1 MLC_DFLASH_CUDA_GRAPH=1
export MLC_XAVIER_DEQUANT=1
opt='flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0'
taskset -c 0,1 python -m mlc_llm compile dist/target-16k/mlc-chat-config.json \
    --device cuda --opt "$opt" -o dist/libs/target-dflash-vector-dequant-16k-sm72.so
