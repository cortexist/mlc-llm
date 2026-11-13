#!/usr/bin/env bash
# From the repo root on Xavier, with the resident service stopped: what
# scripts/qwen35_service.py serves.
#
#   dist/qwen35-target, dist/qwen35-draft                       links to the converted weights
#   dist/libs/qwen35-target-sm72.so, dist/libs/qwen35-draft-sm72.so
#   dist/prefill-experiments/native.so                          tensor-core attention kernels
#
# TARGET and DRAFT are converted q4f16_1 packages of one Qwen3.5 family (same vocabulary).
# TEMPLATE is chat_template.jinja of the target's original checkpoint, which MLC conversion
# does not copy. A token of CONTEXT costs the 27B target 64 KiB of attention cache and the
# 2B draft 12 KiB.
set -euo pipefail
source scripts/xavier/env.sh
TARGET=${TARGET:-$HOME/Models/qwen3.8-27B-uncensored-q4f16_1-MLC}
DRAFT=${DRAFT:-$HOME/Models/qwen-3.8-2B-q4f16_1-MLC}
TEMPLATE=${TEMPLATE:-$HOME/Models/qwen3.8-27B-chat_template.jinja}
CONTEXT=${CONTEXT:-8192}
CHUNK=${CHUNK:-512}
export TVM_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
mkdir -p dist/libs dist/qwen35-target dist/qwen35-draft
for pair in "$TARGET:dist/qwen35-target" "$DRAFT:dist/qwen35-draft"; do
    for item in "${pair%%:*}"/*; do
        ln -sfn "$(realpath "$item")" "${pair##*:}/$(basename "$item")"
    done
done
install -m 644 "$TEMPLATE" dist/qwen35-target/chat_template.jinja
# Grouped small-batch projections, vectorized weight expansion for prefill, the gated delta
# kernel with verify_hold/commit_rows, and verification of up to three proposals.
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_Q4_CUBLAS=1 MLC_XAVIER_Q4=2 MLC_XAVIER_DEQUANT=1
export MLC_XAVIER_GDN=1 MLC_MTP_VERIFY_LEN=4
opt='flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=1'
overrides="context_window_size=$CONTEXT;prefill_chunk_size=$CHUNK;max_batch_size=1"
for name in target draft; do
    taskset -c 0,1 python -m mlc_llm compile "dist/qwen35-$name/mlc-chat-config.json" \
        --device cuda --opt "$opt" --overrides "$overrides" -o "dist/libs/qwen35-$name-sm72.so"
done
# The libraries hold TVM's attention kernels; the tensor-core ones are installed when
# the caches are created (mlc_llm/op/xavier_prefill.py, install_qwen35).
taskset -c 0,1 python scripts/build_xavier_prefill.py --native-only
