# Current Xavier model comparisons — November 2, 2026

The README's two current tables compare Nemotron 3.5 Bolt E4B and Gemma4
Unified 12B Heretic on the Jetson AGX Xavier 32GB. Each target is tested in
MLC and llama.cpp with plain decoding and two speculative proposals.
Nemotron uses DFlash; Unified 12B uses Google's original native MTP assistant.

## Protocol

- MAXN, existing fixed device clocks, CUDA 11.4 / sm_72; two CPU threads.
- Exactly 161, 448 and 910 input token IDs, including the chat template.
  Use the source tokenizer's deployed chat template with thinking disabled.
  Truncate/pad only the user body token stream to reach the exact length.
  Each MLC/llama model pair receives identical IDs. The runner checks GGUF
  tokenizer agreement on the original template before the adjusted inputs.
- One sequence, FP16 KV cache, 512-token prefill chunks/microbatches, prefix
  reuse disabled; 16K context for Nemotron and 4K for Unified 12B.
- Greedy generation, 128 forced output tokens, including continuation past EOS.
  One full warmup and two measured requests per case and mode. Report means.
- The prefill columns use plain-mode requests. Engine-reported decode rates
  exclude prefill; the native MLC 12B controller measures
  `(output_tokens - 1) / (end - first_token)`.
- Compilation/conversion and GPU measurements run sequentially. The resident
  service is stopped for the measurements and restored afterward.

The MLC Nemotron library includes vectorized Q4 expansion and `hybrid-width`
attention. Experimental dense GEMM retuning is disabled. The 12B MLC path is
its current direct-VM native-MTP controller; both its plain control and MTP
mode include CPU copies of shared state. These costs are part of the deployed
prototype's performance. The earlier MLCEngine plain benchmark uses a different
controller and is retained separately in the historical record.

Both engines use the same source target checkpoint within each table:
NVIDIA's BF16 Nemotron, or the local BF16 Unified 12B Heretic checkpoint.
MLC uses q4f16_1; llama.cpp uses Q4_K_M. Both assistants are F16. The 12B
llama target is freshly converted from the Heretic source, with the original
QAT native MTP assistant converted separately. The other local 12B distill
GGUF is not used. These are practical engine comparisons with different
quantization/arithmetic, rather than accuracy-equivalence measurements.

Raw requests, generated continuations, native MTP acceptance counters and
server logs are kept in `mlc-jp5/logs/current-matrix-20261102/` in the private
benchmark workspace. The public summary retains individual measurements,
output hashes, protocol and artifact hashes.

## Output checks

Some 128-token greedy speculative continuations differ from the corresponding
plain run in both engines. The public summary records these checks and output
hashes. The MLC 12B 448-token case matches; its 161/910-token cases differ.
Earlier 24-token prototype checks remain historical short checks. These
throughput measurements do not establish general speculative equivalence or
broad model quality; no cause for the differences is asserted here.

The [public summary](../benchmarks/xavier/current-matrix-summary.json) contains
all 48 measured requests, the table means and the exact prompt hashes. Short
startup pilots are excluded. llama.cpp is revision `ddf48f85c`; the MLC model
libraries and attention adapter are identified by SHA-256 in the summary.

## Prepare the matching 12B GGUF files

From an initialized Xavier checkout with llama.cpp beside it:

```sh
source scripts/xavier/env.sh
model_src="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic"
assistant_src="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant"
gguf_out="$HOME/Models/gemma4-12b-heretic-gguf"
mkdir -p "$gguf_out"
python ../llama.cpp/convert_hf_to_gguf.py "$model_src" \
  --outtype f16 --outfile "$gguf_out/target-f16.gguf"
../llama.cpp/build/bin/llama-quantize "$gguf_out/target-f16.gguf" \
  "$gguf_out/target-Q4_K_M.gguf" Q4_K_M 2
python ../llama.cpp/convert_hf_to_gguf.py "$assistant_src" \
  --outtype f16 --outfile "$gguf_out/assistant-f16.gguf"
```

Stop serving before these preparation steps. The benchmark script manages
stopping/restoring the service during the subsequent measurements:

```sh
bench_log="logs/current-matrix-$(date +%Y%m%d-%H%M%S)"
MLC_CURRENT_BENCH_LOG="$bench_log" bash scripts/run_xavier_current.sh
python scripts/summarize_xavier_current.py "$bench_log" \
  --output docs/benchmarks/xavier/current-matrix-summary.json
```

`MLC_CURRENT_BENCH_LOG` changes the log destination. The short pilot runs are
excluded from the tables. The 910-token case crosses the 512-token chunk
boundary; no full-context or sustained-load qualification is implied.
