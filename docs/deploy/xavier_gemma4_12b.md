# Gemma4 Unified 12B on Xavier

The official QAT assistant now has a separate
[native MTP prototype and short measured trial](xavier_gemma4_mtp.md), using
`Gemma4UnifiedAssistantForCausalLM`. It accepts proposals with this target;
the DFlash experiment below remains a separate, unsuccessful pairing.

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

## DFlash experiment

The renamed local file `~/Models/gemma-4-12B-IT-DFlash-assistant/` matches
[z-lab/gemma4-12B-it-DFlash](https://huggingface.co/z-lab/gemma4-12B-it-DFlash)
at revision `541d351de6dc9cd698fc348612bd5cdf03c636d0`. Its weight SHA-256 is
`8b32507dcff3ca4c1bbbb6c1cca7d17749bbf394480024e505228b6ed40afa02`.
It has five Qwen-style layers and projects six target-layer outputs
`[1, 10, 19, 27, 36, 45]` from width 23040 to 3840. The weights convert from
BF16 to FP16 without further quantization.

The upstream draft code reads raw embedding-table values. MLC's Gemma4 embed
entry point includes the model's embedding multiplier, so the draft now accepts
explicit `target_embedding_scale` and `input_embedding_scale` configuration.
This experiment uses `61.96875` (FP16 sqrt(3840)) and `1.0`, respectively, in
the draft's `model_config`. Both default to 1, preserving existing Nemotron
behavior. Scaling is corrected before the first unnormalized residual.
The CPU regression compares an entire draft forward pass with raw inputs
against scaled target inputs and measures a maximum absolute error of
`4.77e-7`. Existing cached/uncached context checks also pass.

The correction is necessary for that input convention, but **does not resolve
this model pairing's acceptance problem**:

| Configuration | Prompt tokens | Decode tok/s | Accepted / proposed draft tokens |
|---|---:|---:|---:|
| Plain | 161 | 11.72 | — |
| DFlash, original embedding path | 161 | 6.79 | 0 / 508 |
| DFlash, original embedding path | 448 | 5.26 | 0 / 508 |
| DFlash, corrected raw embedding path | 161 | 6.84 | 0 / 508 |

Each DFlash row averages two warmed, forced-128-token requests with two draft
proposals. Their output text matches plain decoding exactly because target
verification rejects all proposals. Both natural arithmetic checks return
`391`. A separate 12-token diagnostic has finite context, hidden states and
logits, but repeatedly proposes token 258882 (the image-ending token), which
the target rejects. This pairing is not qualified as an accelerator. Separating
checkpoint/target compatibility from remaining implementation differences needs
a reference comparison of draft inputs and outputs; the trace alone cannot
establish the cause.

Device power losses interrupted the long initial run and the corrected sweep;
no long-context corrected rate is reported. Power-supply investigation remains
deferred. The completed plain benchmark is unaffected, and Nemotron remains
the serving model.

Experimental artifacts on cerebellum:

- `dist/gemma4-12b-dflash-target`: target config with the six feature layers,
  linking to the converted 12B weights under `~/Models`.
- `dist/gemma4-12b-dflash`: converted FP16 draft and its chat configuration.
- `dist/libs/gemma4-12b-dflash-q4f16_1-sm72.so`: target with feature exports.
- `dist/libs/gemma4-12b-draft-rawembed-sm72.so`: corrected draft library.

The benchmark runner accepts `--target`, `--draft`, `--target-lib` and
`--draft-lib` for these artifacts. Detailed local logs are retained under
`~/Workspaces/cerebellum-llm-bench/mlc-jp5/logs/gemma4-12b-heretic-20261101/`
on nowhere; the numerical summary is versioned in
[`docs/benchmarks/xavier/gemma4-12b-heretic-summary.json`](../benchmarks/xavier/gemma4-12b-heretic-summary.json).

## Native MTP assistant

The local `gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant` checkpoint is
`Gemma4UnifiedAssistantForCausalLM`: four layers, hidden size 1024 and backbone
hidden size 3840. It consumes the target's shared KV states and embeddings.
This requires a separate Gemma4 MTP execution path; it cannot replace the
diffusion drafter in the current DFlash implementation. No assistant/MTP speed
is claimed by the plain-decoding benchmark above.
