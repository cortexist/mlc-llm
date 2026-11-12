#!/usr/bin/env bash
# Run from the repository root with the resident GPU service stopped.
# Existing Nemotron q4f16_1 weights must be available at dist/target-16k.
set -euo pipefail
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1 MLC_E4B_MTP_TOPK=1
mtp_source="${1:-$HOME/Models/gemma-4-E4B-IT-assistant}"
mtp_weights="$HOME/Models/gemma-4-E4B-IT-assistant-q0f16-MLC"
if [[ ! -f "$mtp_weights/tensor-cache.json" ]]; then
 taskset -c 0,1 python -m mlc_llm convert_weight "$mtp_source" --quantization q0f16 --device cpu -o "$mtp_weights"
fi
python -m mlc_llm gen_config "$mtp_source" --quantization q0f16 --conv-template gemma4_instruction --context-window-size 1024 --prefill-chunk-size 512 --max-batch-size 1 -o "$mtp_weights"
python - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
assistant=Path.home()/'Models/gemma-4-E4B-IT-assistant-q0f16-MLC'
configs = [
 (root/'dist/target-16k',root/'dist/nemotron-e4b-mtp-1k','gemma4_e4b_mtp_target',1024),
 (root/'dist/target-16k',root/'dist/nemotron-e4b-mtp-16k','gemma4_e4b_mtp_target',16384),
]+[(assistant,root/'dist'/f'gemma4-e4b-mtp-assistant-{n}k','gemma4_assistant',n*1024)
   for n in [2,4,8,16]]
for source,dest,kind,context in configs:
 dest.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((source/'mlc-chat-config.json').read_text())
 cfg['model_type']=kind
 cfg['context_window_size']=cfg['model_config']['context_window_size']=context
 (dest/'mlc-chat-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 for item in source.iterdir():
  if item.name!='mlc-chat-config.json' and not (dest/item.name).exists():
   (dest/item.name).symlink_to(item.resolve())
PY
opt='flashinfer=0;cublas_gemm=1;faster_transformer=0;cudagraph=0;cutlass=0'
mkdir -p dist/libs
unset MLC_XAVIER_Q4 MLC_XAVIER_ATTN MLC_XAVIER_DEQUANT MLC_JP5_Q4_CUBLAS
taskset -c 0,1 python -m mlc_llm compile "$mtp_weights/mlc-chat-config.json" --device cuda --opt "$opt" -o dist/libs/gemma4-e4b-mtp-assistant-topk-1k-sm72.so
for size in 2k 4k 8k 16k; do
 taskset -c 0,1 python -m mlc_llm compile "dist/gemma4-e4b-mtp-assistant-$size/mlc-chat-config.json" --device cuda --opt "$opt" -o "dist/libs/gemma4-e4b-mtp-assistant-topk-$size-sm72.so"
done
# One 16K assistant that attends the valid context only, in place of the buckets
# (deploy/xavier/xavier-full.conf).
MLC_E4B_MTP_ATTN=1 taskset -c 0,1 python -m mlc_llm compile "dist/gemma4-e4b-mtp-assistant-16k/mlc-chat-config.json" --device cuda --opt "$opt" -o dist/libs/gemma4-e4b-mtp-assistant-native-16k-sm72.so
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_Q4_CUBLAS=1 MLC_XAVIER_Q4=1 MLC_XAVIER_ATTN=1 MLC_XAVIER_DEQUANT=1
for size in 1k 16k; do
 taskset -c 0,1 python -m mlc_llm compile "dist/nemotron-e4b-mtp-$size/mlc-chat-config.json" --device cuda --opt "$opt" -o "dist/libs/nemotron-e4b-mtp-target-$size-sm72.so"
done
# The 16K target of deploy/xavier/xavier-full.conf: 4-bit projections that apply the
# scales once per group of 32 weights, and a verification step of four rows (three
# proposals).
MLC_XAVIER_Q4=2 MLC_MTP_VERIFY_LEN=4 taskset -c 0,1 python -m mlc_llm compile "dist/nemotron-e4b-mtp-16k/mlc-chat-config.json" --device cuda --opt "$opt" -o dist/libs/nemotron-e4b-mtp-target-grouped-16k-sm72.so
nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_72 scripts/xavier_mtp_runtime.cu -o dist/libs/xavier-mtp-runtime.so
# Attention callbacks: hybrid-width and the tensor-core kernel over the cached pages.
taskset -c 0,1 python scripts/build_xavier_prefill.py
