#!/usr/bin/env bash
# Stop GPU serving before running; restart it after the build/checks.
set -euo pipefail
source scripts/xavier/env.sh
unset MLC_XAVIER_ATTN MLC_XAVIER_Q4 MLC_JP5_Q4_CUBLAS MLC_DFLASH_CUDA_GRAPH MLC_JP5_VERIFY_THREE
precision=${MLC_D1_PRECISION:-f32}
case "$precision" in f32|f16) ;; *) exit 2;; esac
context=${MLC_D1_CONTEXT:-256}
case "$context" in 256|512|1024|2048) ;; *) exit 2;; esac
mkdir -p dist/libs
if [[ ${MLC_D1_REUSE_WEIGHTS:-0} != 1 ]]; then
    python scripts/prepare_d1_text.py "$HOME/Models/d1-Omni-2B" dist/d1-text-hf
    python -m mlc_llm convert_weight dist/d1-text-hf --model-type d1_lfm3_text \
        --quantization q0${precision} --device cpu -o dist/d1-text-q0${precision}
fi
test -f "dist/d1-text-q0${precision}/tensor-cache.json"
config_dir=dist/d1-text-q0${precision}
if [[ $context != 256 ]]; then
    config_dir=dist/d1-text-config-${precision}-s${context}
fi
python -m mlc_llm gen_config dist/d1-text-hf --model-type d1_lfm3_text \
    --quantization q0${precision} --conv-template gemma4_instruction \
    --context-window-size "$context" --prefill-chunk-size "$context" --max-batch-size 1 \
    -o "$config_dir"
taskset -c 0,1 python -m mlc_llm compile "$config_dir/mlc-chat-config.json" \
    --device cuda --opt 'flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0' \
    -o dist/libs/d1-text-${precision}-s${context}-sm72.so
