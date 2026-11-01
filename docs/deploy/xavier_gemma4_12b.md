# Gemma4 Unified 12B on Xavier

The additional `gemma4_unified` model entry supports text generation from
`~/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic` on cerebellum. Its source
contains 11,907,350,320 text parameters in BF16; the small audio and vision
components are excluded. MLC converts these values to `q4f16_1`. The source's
`Q4_0-unquantized` name describes its QAT lineage, not its on-disk tensor dtype.

This dense architecture differs from the E-series model used by Nemotron:
48 layers, hidden size 3840, 16 query heads, eight local KV heads and one global
KV head, no per-layer embeddings or layer-to-layer KV sharing, and global
layers that derive both K and V from the same projection. The port preserves
separate K normalization/RoPE and unscaled V normalization.

The current uniform cache broadcasts the global layer's single KV head into
eight cache slots. It also pads local head dimensions from 256 to 512, as the
E-series implementation does. Both choices preserve the attention calculation
but cost memory and attention work. The Nemotron-specific attention/prefill
callbacks are not enabled for this model. Small-batch Q8/DP4A projections and
cuBLAS prefill projections are enabled in the measured library.

## Plain-decoding results

November 1, 2026, MAXN/fixed Xavier clocks, 4K context, chunks of 512, one
sequence, prefix caching disabled, greedy generation and 128 forced output
tokens. Each case uses one warmup and two measured runs; the table reports
means. Output text matches between repeats in all three cases. A separate
natural-stop arithmetic check returns `391` for 17 × 23.

| Prompt tokens | Prefill tok/s | Decode tok/s | First token | Complete request |
|---|---:|---:|---:|---:|
| 161 | 90.82 | 11.72 | 1.777 s | 12.616 s |
| 448 | 109.65 | 10.59 | 4.091 s | 16.082 s |
| 1,768 | 50.44 | 8.73 | 35.063 s | 49.607 s |

Converted tensors occupy 6,698,991,200 bytes across 159 shards. The runtime
estimates 11,793.8 MiB including the 4K KV cache and temporary buffers. FP16
rounding and Q8 activations change arithmetic; these functional and throughput
checks do not establish broad accuracy equivalence. This first port has not
received the Nemotron-specific attention optimizations.

## Reproduce

Use the Xavier compiler/runtime build described in the main README. Stop the
Nemotron service before GPU work, and restore it afterward. Compilation and
inference must run sequentially.

```bash
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1
model_src="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic"
model_out="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic-q4f16_1-MLC"
mkdir -p "$model_out" dist/libs logs
python -m mlc_llm convert_weight "$model_src" \
  --quantization q4f16_1 --device cuda -o "$model_out"
python -m mlc_llm gen_config "$model_src" \
  --quantization q4f16_1 --conv-template gemma4_instruction \
  --context-window-size 4096 --prefill-chunk-size 512 --max-batch-size 1 \
  -o "$model_out"
export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_Q4_CUBLAS=1 MLC_XAVIER_Q4=1
unset MLC_XAVIER_ATTN
taskset -c 0,1 python -m mlc_llm compile "$model_out/mlc-chat-config.json" \
  --device cuda --opt O2 -o dist/libs/gemma4-12b-heretic-q4f16_1-sm72.so
python scripts/bench_xavier_dflash.py \
  --target "$model_out" --target-lib dist/libs/gemma4-12b-heretic-q4f16_1-sm72.so \
  --draft-tokens 0 --context-size 4096 --cases chat,448,1768 \
  --tokens 128 --warmups 1 --repeats 2 --output logs/gemma4-12b-heretic.json
```

`tests/python/model/test_gemma4_unified.py` checks the new local/global
projection, normalization and RoPE paths against the installed Transformers
implementation on CPU, including positions 0, 17 and 1025. It also checks the
original E-series configuration. All source text tensors are accounted for by
the loader; no checkpoint tensors are synthesized.

## Native MTP assistant

The local `gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant` checkpoint is
`Gemma4UnifiedAssistantForCausalLM`: four layers, hidden size 1024 and backbone
hidden size 3840. It consumes the target's shared KV states and embeddings.
This requires a separate Gemma4 MTP execution path; it cannot replace the
diffusion drafter in the current DFlash implementation. No assistant/MTP speed
is claimed by the plain-decoding benchmark above.
