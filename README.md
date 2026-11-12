![MLC-LLM for Jetson AGX Xavier](docs/_static/img/945-82972-0040-000.png)

This fork runs **NVIDIA Nemotron 3.5 Bolt E4B** and **Gemma4 Unified 12B** on
**Jetson AGX Xavier 32 GB**, using JetPack 5 / CUDA 11.4. It adds speculative
decoding and kernels suited to Xavier's Volta GPU. With every Xavier option
enabled (November 11), Nemotron with **Gemma E4B MTP decodes at 38.94 tokens/s**
on short prompts and **35.3 tokens/s at 9,500 tokens of context**, and
prefills **808 tokens/s** at 910 tokens. Gemma4 Unified 12B with its native
MTP assistant decodes at **26–37 tokens/s**.

Based on upstream MLC-LLM `44fd06d0`. DFlash builds on
[Lev Stambler's implementation](https://github.com/Lev-Stambler/mlc-llm--DFlash/tree/b6c7bf7d269fd4127fd1a48c478b01d9dc8ff42b).
The original project overview is preserved in [README.upstream.md](README.upstream.md).

## Nemotron with all Xavier options — November 11, 2026

Three additions to the Gemma E4B MTP path, enabled together by
[`deploy/xavier/xavier-full.conf`](deploy/xavier/xavier-full.conf):

- **Tensor-core attention over the cached pages.** A Volta WMMA kernel reads
  the paged FP16 cache in place, for prefill chunks and for decode and
  verification steps.
- **One assistant for every context length.** The assistant attends only the
  valid part of its shared K/V state, so a draft step costs 1.3–1.9 ms at any
  context instead of 2 ms (1K) to 33 ms (16K) across five libraries.
- **K/V-only prefill.** The last 18 of Nemotron's 42 layers store no K/V, and
  only the last prompt token is predicted from, so every other prompt token
  runs the first 24 layers only.

Same protocols as the tables below, now with the **16K libraries** of the
service; the November 7 configuration was rerun alongside. Decode: 60 / 68 / 90
token prompts, 128 forced outputs, one warmup and two measured repetitions,
nothing cached. Prefill: the 161 / 448 / 910-token fixture, prompt tokens over
time to first token.

| Decode tok/s | Prose (60) | Code (68) | Scheduling (90) | Pooled |
|---|---:|---:|---:|---:|
| MTP controller plain, November 7 profile | 24.52 | 24.30 | 23.93 | 24.25 |
| MTP controller plain, all options | 25.36 | 25.19 | 25.32 | 25.29 |
| Gemma E4B MTP, 2 drafts, November 7 profile | 33.25 | 43.31 | 36.60 | 37.27 |
| **Gemma E4B MTP, 2 drafts, all options** | **34.79** | **44.88** | **38.44** | **38.94** |

| Prefill tok/s | 161 | 448 | 910 |
|---|---:|---:|---:|
| November 7 profile | 329.46 | 522.26 | 423.54 |
| **All options** | **504.00** | **807.04** | **807.50** |

The gains grow with context. One run each, 160 forced outputs:

| Prompt tokens | 300 | 2,500 | 5,000 | 9,500 |
|---|---:|---:|---:|---:|
| Decode tok/s, November 7 profile | 39.6 | 28.3 | 20.4 | 14.2 |
| **Decode tok/s, all options** | **45.5** | **39.3** | **39.3** | **35.3** |
| First token, new prompt (s), November 7 profile | 0.66 | 8.21 | 23.56 | 68.52 |
| **First token, new prompt (s), all options** | **0.43** | **3.19** | **6.93** | **14.65** |
| First token, 200 tokens added to a cached prompt (s), November 7 profile | 0.65 | 1.22 | 1.82 | 2.90 |
| **First token, 200 tokens added (s), all options** | **0.34** | **0.41** | **0.48** | **0.61** |

K/V-only prefill is exact: K/V states and logits are bit-identical with and
without it. The new attention kernel is not bit-identical to the November 7
kernels. Decode activations are quantized to int8, so small attention
differences flip near-tie greedy choices: 748 of 768 teacher-forced
predictions match the November 7 profile (the two previously accepted
attention variants agree on 755), and one of the three recorded 128-token
outputs is reproduced exactly; the others diverge after 88 and 124 tokens.
No task-quality benchmark was run.

[Build, enable and roll back](docs/deploy/xavier_nemotron_mtp.md#all-xavier-options--november-11)
· [Per-run measurements, equivalence checks and hashes](docs/benchmarks/xavier/nemotron-mtp-full-summary.json).

## Gemma4 12B with all Xavier options — November 11, 2026

The attention kernels above now take the 12B layout (16 query heads, 8 stored
K/V heads, a 1,024-token local window), and its native MTP path has additions
of its own. Enabled together by
[`deploy/xavier/gemma4-12b-full.conf`](deploy/xavier/gemma4-12b-full.conf):

- **Tensor-core attention over the cached pages** and **vectorized expansion
  of the 4-bit weights** for prefill: a 512-token chunk behind 3,072 cached
  tokens takes 3.3 s instead of 16.8 s.
- **Projections with grouped arithmetic.** The small-batch 4-bit kernel applies
  its scales once per 32 weights instead of once per 8. A verification step of
  four rows then costs what one row costs: 87 against 83 ms at 512 tokens of
  context (109 against 89 ms before).
- **A GPU-resident controller and a 4-bit assistant** that attends only the
  valid part of its shared K/V state: a draft step takes 4 ms instead of 33 ms.
- **Three proposals per round**, which the cheap verification rows now pay for.

4K context, 128 forced outputs, one warmup and two measured repetitions,
nothing cached. The baseline is the November 2 configuration rerun with
current `main`; both use the same Heretic target and original QAT assistant.

| 161 / 448 / 910-token fixture | 161 | 448 | 910 |
|---|---:|---:|---:|
| Prefill tok/s, baseline | 90.91 | 111.03 | 76.05 |
| **Prefill tok/s, all options** | **138.34** | **202.63** | **189.90** |
| Plain decode tok/s, baseline | 11.50 | 10.40 | 9.00 |
| Plain decode tok/s, all options | 12.67 | 12.10 | 11.52 |
| MTP decode tok/s, 2 proposals, baseline | 13.26 | 12.33 | 9.46 |
| MTP decode tok/s, 2 proposals, all options | 27.26 | 29.67 | 26.78 |
| **MTP decode tok/s, 3 proposals, all options** | **34.37** | **36.70** | **33.73** |

| Short prompts, decode tok/s | Prose (64) | Code (72) | Scheduling (94) |
|---|---:|---:|---:|
| Plain, baseline | 11.86 | 11.87 | 11.71 |
| Plain, all options | 12.99 | 12.99 | 12.93 |
| MTP, 2 proposals, baseline | 11.78 | 14.18 | 11.85 |
| MTP, 2 proposals, all options | 23.16 | 28.26 | 23.47 |
| **MTP, 3 proposals, all options** | **25.72** | **35.07** | **26.11** |

The fixture's forced continuations run past the end of a short answer and are
easier to predict than the fresh text of the short prompts. Neither the new
attention nor the grouped projections are bit-identical to what they replace:
with tokens forced to the baseline's outputs, 761 of 765 predictions match the
baseline's (the other four were near-ties, 0.01–0.16 logits apart), and 6 of
the 12 recorded outputs with zero or two proposals are reproduced exactly; the
others diverge after 4 to 114 tokens.
No task-quality benchmark was run. The 12B stays an optional second model of
the service.

[Options, per-step gains and build](docs/deploy/xavier_gemma4_mtp.md#all-xavier-options--november-11-2026)
· [Per-run measurements, equivalence checks and hashes](docs/benchmarks/xavier/gemma4-12b-mtp-full-summary.json).

## Nemotron with Gemma E4B MTP — November 7, 2026

All rates below are **decode tokens/second** on AGX Xavier 32 GB, MAXN and
fixed clocks. Three prompts contain exactly **60 / 68 / 90 tokens**; each has
128 forced outputs, one warmup and two measured repetitions. Rates pool
post-first-token outputs over decode time. MLC uses the same Nemotron
`q4f16_1` target weights and `q0f16` assistants, one sequence, 512-token
prefill chunks, a **1K runtime context**, and no prefix reuse.

| Configuration | Prose (60) | Code (68) | Scheduling (90) | Pooled |
|---|---:|---:|---:|---:|
| Native MLCEngine plain | 24.90 | 24.47 | 24.15 | 24.50 |
| Native MLCEngine DFlash, 2 drafts | 30.04 | 32.75 | 30.28 | 30.98 |
| MTP controller plain control | 24.60 | 24.44 | 24.10 | 24.38 |
| Gemma E4B MTP, 1 draft | 30.33 | 35.36 | 31.43 | 32.24 |
| **Gemma E4B MTP, 2 drafts** | **33.20** | **43.07** | **36.87** | **37.28** |

Two-draft MTP is **20.4% faster than DFlash** in this short-context comparison.
The first MTP implementation measured 17.31 / 14.80 tokens/s with one / two
drafts. Parallelizing the assistant's centroid selection reduced its step
from **37.01 to 1.88 ms**; keeping KV, hidden states and embeddings on GPU
removed repeated host transfers. Two-draft throughput improved **2.52×**.

The optimized assistant's captured logits, hidden state and candidate IDs
match the original exactly. All measured optimized output tokens and
acceptance counts also match their original same-width MTP runs. Plain and
speculative modes can still produce different greedy continuations; these
measurements do not establish broad answer quality. The MTP controller uses
direct TVM VMs and Python scheduling, while DFlash uses native MLCEngine.
These short fixtures are separate from the older 161 / 448 / 910-token matrix.

[Implementation and serving instructions](docs/deploy/xavier_nemotron_mtp.md)
· [Per-run measurements and validation](docs/benchmarks/xavier/nemotron-gemma-mtp-summary.json).

The selected service profile uses **Nemotron + Gemma E4B MTP with two drafts**,
a 16K target context, a pinned server prefix and two cached conversations.
Assistant buckets share weights across 1K/2K/4K/8K/16K contexts. Streaming,
tool round trips, validated JSON and the resident embedding/d1 workers remain
available through the shared GPU queue. The table above measures the separate
1K benchmark configuration, not full-capacity serving performance.

## Earlier benchmarks — November 2, 2026

All rates are **tokens/second**; unqualified columns are this MLC fork.
Each cell averages **two measured runs after one warmup**, with **128 forced
output tokens**. Prefill comes from plain mode. Each engine pair receives
identical token IDs with thinking/prefix reuse disabled. Targets use the same
source checkpoint: MLC q4f16_1 versus llama.cpp Q4_K_M, F16 assistants/FP16 KV.

### Nemotron 3.5 Bolt E4B

16K context, 512-token chunks/microbatches, one sequence. The fork uses
`hybrid-width` attention and vectorized Q4 expansion. Both speculative
columns use **DFlash with two proposals** (including `llama decode MTP`).

| Prompt tokens | Prefill | llama prefill | Decode plain | llama decode plain | Decode MTP/DFlash | llama decode MTP |
|---|---:|---:|---:|---:|---:|---:|
| 161 | 329.40 | 216.42 | 22.77 | 18.74 | 30.46 | 22.71 |
| 448 | 517.58 | 352.96 | 19.19 | 18.60 | 29.61 | 23.04 |
| 910 | 423.65 | 357.78 | 18.20 | 18.49 | 30.21 | 20.12 |

### Gemma4 Unified 12B Heretic

4K context, 512-token chunks/microbatches, one sequence. Both engines use the
**same Heretic source weights**, freshly converted to GGUF for llama.cpp.
Both speculative columns use the **original QAT native MTP assistant, with
two proposals**. MLC's current direct-VM controller includes CPU shared-state
copies in its prefill, plain control and MTP timings.

| Prompt tokens | Prefill | llama prefill | Decode plain | llama decode plain | Decode MTP/DFlash | llama decode MTP |
|---|---:|---:|---:|---:|---:|---:|
| 161 | 89.74 | 87.10 | 11.34 | 9.88 | 13.23 | 14.00 |
| 448 | 110.46 | 138.30 | 10.25 | 9.74 | 12.37 | 12.90 |
| 910 | 75.78 | 139.05 | 8.92 | 9.62 | 9.44 | 12.78 |

**Qualification:** Some 128-token greedy plain/speculative continuations
differ in both engines; output comparisons and hashes are recorded. Forced
continuation includes tokens past EOS. These are throughput measurements,
with no claim of identical generation or broad model-quality validation.
The 910-token case is the longest input in this refresh.

[Protocol, matching-checkpoint conversion and reproduction](docs/deploy/xavier_current_benchmarks.md)
· [Per-run measurements and artifact hashes](docs/benchmarks/xavier/current-matrix-summary.json).
The earlier [attention-width](docs/deploy/xavier_prefill_followup.md) and
[weight-expansion](docs/deploy/xavier_prefill_linear.md) experiments retain
their separate 16-output-token controls and optimization results.

<details>
<summary>Historical fork versus baseline MLC and llama.cpp — October 31–November 1 (161 / 448 / 1,768 tokens)</summary>

### Historical benchmarks: this fork versus baseline MLC

All results below run **Nemotron 3.5 Bolt E4B** on the same Xavier. Rates are
in tokens/second. Baseline uses upstream MLC with the CUDA 11.4 compatibility
fixes needed to run it, q4f16_1 weights and plain decoding. The optimized fork
uses the same target quantization, an F16 DFlash draft with two proposals,
Q8 activation kernels, and `--prefill-variant hybrid`.

| Prompt tokens | Baseline prefill | Fork prefill | Baseline decode | Fork decode | Fork decode MTP |
|---|---:|---:|---:|---:|---:|
| 161 | 108.28 | **240.79** | 11.90 | 23.24 | **31.68** |
| 448 | 115.47 | **395.16** | 10.80 | 19.63 | **34.16** |
| 1,768 | 82.66 | **278.67** | 10.06 | 17.18 | **31.10** |

**Fork decode** is the November 1 native-matrix, non-speculative control,
measured before the prefill changes (see the
[experiment record](docs/deploy/xavier_dflash.md)). **Fork decode MTP** uses
two-token DFlash speculative decoding with the final `hybrid` prefill configuration.

This compares the complete optimized configuration with the initial engine;
it includes speculative decoding and changed arithmetic. Baseline measurements
were recorded on October 31; fork measurements on November 1, 2026. The power
connection changed between sessions, and CPU compilation overlapped the initial
baseline. These are historical end-to-end engine comparisons. For an isolated
prefill comparison, the following November 1 sweep keeps the weights, compiled
model library, DFlash settings and machine configuration identical:

| Prefill implementation | 161 tokens | 448 tokens | 1,768 tokens | Long first token |
|---|---:|---:|---:|---:|
| Fork before prefill changes | 218.91 | 271.75 | 143.94 | 12.292 s |
| Skip fully masked tiles | 235.07 | 345.13 | 201.56 | 8.781 s |
| + Actual-width local heads (`width`) | 233.16 | 340.26 | 250.50 | 7.068 s |
| + Volta WMMA for current-chunk attention (`hybrid`) | **240.79** | **395.16** | **278.67** | **6.354 s** |

The combined prefill changes improve the long prompt **1.94×** and cut its
complete 128-token request from **16.354 to 10.438 seconds**. Decode stays
approximately constant. The `width` variant preserves all six measured baseline
output strings; `hybrid` changes the long forced continuation.

### Historical benchmarks: this fork versus llama.cpp

Both engines run **Nemotron 3.5 Bolt E4B with two DFlash proposals**, the same
prompt text and a forced 128-token output. MLC uses q4f16_1 plus Q8 activations;
llama.cpp uses Q4_K_M and the F16 draft, with CUDA offload and Flash Attention.
The tokenizer/template counts differ slightly.

| Prompt tokens MLC / llama | Fork prefill | llama prefill | Fork decode | llama decode | Fork request | llama request |
|---|---:|---:|---:|---:|---:|---:|
| 161 / 168 | **240.79** | 189.90 | **31.68** | 26.80 | **4.681 s** | 5.635 s |
| 448 / 454 | **395.16** | 316.46 | **34.16** | 25.09 | **4.856 s** | 6.511 s |
| 1,768 / 1,774 | 278.67 | **341.88** | **31.10** | 24.37 | 10.438 s | 10.427 s |

The fork is faster on short and medium complete requests. Llama prefills the
long input faster; the complete long request is effectively tied. The llama
control was recorded earlier on November 1 at revision `ddf48f85`, before the
final MLC prefill sweep. Different quantizations and arithmetic mean this is a
practical serving comparison, with no claim of equal model accuracy.

**Method:** AGX Xavier 32 GB, sm_72, MAXN, pinned clocks (GPU 1,377 MHz, CPU
2,265.6 MHz, EMC 2,133 MHz), one sequence, context 8,192, MLC chunks 512,
prefix caching disabled, greedy generation, one warmup and two measured runs
per case. Tables report means; profiler timings are excluded. Forced output
past EOS measures throughput rather than natural answer quality. Raw summaries
are in [docs/benchmarks/xavier](docs/benchmarks/xavier); detailed validation and
history are in [the experiment record](docs/deploy/xavier_dflash.md).

</details>

## Additional model: Gemma4 Unified 12B

An experimental port of the official `Gemma4UnifiedAssistantForCausalLM` also
supports native MTP trials with this target. See the
[MTP implementation, validation and build instructions](docs/deploy/xavier_gemma4_mtp.md);
its current speed is in the [November 11 section](#gemma4-12b-with-all-xavier-options--november-11-2026) above.
It currently uses a greedy direct-VM controller with a 4K deployed context and native tool calls.
The optional HTTP adapter keeps it resident alongside Nemotron and routes
requests by model ID through one shared GPU queue.

The fork also supports text generation from
`gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic`, converted from BF16 to MLC
`q4f16_1` (6.70 GB of tensors). These are **plain-decoding** results on the
same Xavier, with a 4K context, chunks of 512, one warmup and two measured
128-token runs per prompt:

| Prompt tokens | Prefill tok/s | Decode tok/s | Complete request |
|---|---:|---:|---:|
| 161 | 90.82 | 11.72 | 12.616 s |
| 448 | 109.65 | 10.59 | 16.082 s |
| 1,768 | 50.44 | 8.73 | 49.607 s |

This first port uses the fast projection kernels and generic attention
(the tensor-core attention took the 12B head layout on November 11). See [the 12B experiment](docs/deploy/xavier_gemma4_12b.md) for
conversion/build instructions, validation and native-MTP compatibility, and
[the saved summary](docs/benchmarks/xavier/gemma4-12b-heretic-summary.json)
for the measured means.

The experimental z-lab DFlash pairing did not accelerate this checkpoint:
**6.84 versus 11.72 tok/s** on the short prompt, with **zero accepted draft
tokens**. The 12B experiment records the input-scaling correction and remaining
compatibility investigation; these plain rates do not include speculative decoding.

## Improvements in this fork

- **CUDA 11.4 support:** pinned TVM/TIRx compatibility patches, nvcc instead of
  the newer NVRTC bindings, and a fix for exporting shared module graphs.
- **Gemma4 support:** E4B/Nemotron with two KV heads, plus a text-only Unified
  12B port with mixed local/global KV head counts and global K=V projections.
- **Native Unified MTP prototype:** official QAT assistant conversion, shared-KV
  draft steps, greedy target verification, and an optional resident HTTP adapter;
  with a 4-bit assistant and three proposals it reaches 26–37 tok/s on the 12B.
- **Nemotron + Gemma E4B MTP:** trained centroid-filtered vocabulary head,
  parallel top-32 selection and GPU-resident shared state; two drafts reach
  37.28 tok/s in the short-context comparison above.
- **Resident model routing:** Nemotron shares one authenticated endpoint and
  FIFO GPU queue with either 12B + MTP or EmbeddingGemma 2 text embeddings;
  see the [resident switching PoC](docs/deploy/xavier_embedding_switch_poc.md).
- **Faster projections:** one-token GEMV specialization; vectorized Q4-to-FP16 expansion +
  cuBLAS for large prefill; native DP4A/Q8-activation kernels for 1–4 rows, with
  grouped arithmetic that makes four rows cost about what one does.
- **DFlash execution:** target feature extraction, GPU greedy selection, cached
  draft K/V, bounded context attention, FP32 residual fixes and CUDA graphs.
- **Faster attention:** native 256/512-wide verification; prefill skips masked
  tiles, computes local paged heads at width 256, and optionally uses Volta
  Tensor Cores for current-chunk attention.
- **Reproducible experiments:** fixed prompts, component tests, monitored runners
  and separate prefill variants.
- **Xavier HTTPS service:** authenticated OpenAI-compatible chat and native
  Nemotron/Gemma4 tool round-trips for Pi; see the
  [deployment runbook](deploy/xavier/README.md).

## Build on Jetson AGX Xavier

### 1. System and Python environment

Tested with **JetPack 5.1.8 / L4T R35.7.0, Ubuntu 20.04, CUDA 11.4,
gcc 9.4, Python 3.12, LLVM 20.1.8 and CMake 4.4.4**. Install LLVM 20 development
packages using the [LLVM apt repository](https://apt.llvm.org/), plus a recent
Rust toolchain for the tokenizer build. Keep `/usr/local/cuda` pointed at CUDA
11.4. The system Python 3.8 is too old for this source tree.

```bash
sudo apt-get install build-essential git ccache pkg-config libssl-dev bc llvm-20-dev
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
mkdir -p ~/Workspaces
cd ~/Workspaces
git clone --recursive https://github.com/cortexist/mlc-llm.git
cd mlc-llm
uv venv --python 3.12 --seed ~/Workspaces/mlc-venv
source scripts/xavier/env.sh
uv pip install 'cmake==4.4.4' ninja 'scikit-build-core>=0.11,!=1.1.0' \
  setuptools-scm 'cython>=3.2.8' numpy ml_dtypes packaging psutil \
  datasets fastapi openai pandas prompt_toolkit requests safetensors \
  sentencepiece shortuuid tiktoken tqdm transformers uvicorn
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps apache-tvm-ffi-orcjit==0.1.3
```

CPU PyTorch is sufficient for the conversion/validation helpers. The tested
packages were torch `2.15.0+cpu`, transformers `5.22.1` and numpy `2.5.4`.
FlashInfer starts at SM75; Xavier is SM72, so this build uses its own attention
paths. The optional source-built orcjit addon requires a newer LLVM; the tested
setup used the prebuilt 0.1.3 package above.

### 2. Build the pinned compiler stack and runtime

The recursive checkout pins TVM `629cc65820`, TIRx `198a04573e` and TVM-FFI
`6628c464e3`. Apply the included dependency patch **before building**:

```bash
source scripts/xavier/env.sh
git -C "$TIRX_SRC" apply --check "$MLC_SRC/scripts/xavier/patches/tirx-cuda114.patch"
git -C "$TIRX_SRC" apply "$MLC_SRC/scripts/xavier/patches/tirx-cuda114.patch"
bash scripts/xavier/build-all.sh
```

Apply the patch once per clean dependency checkout. It includes the CUDA launch
compatibility definitions, nvcc default and export-order fix. Build order is
FFI → TIRx → TVM → MLC. The scripts enable CUDA and cuBLAS, disable
Thrust/CUTLASS/NVTX, target architecture 72, and link pthread explicitly for
Ubuntu 20.04. Compilation defaults to two jobs. Installing all four from the
same pinned source avoids FFI ABI mismatches.

### 3. Prepare Nemotron and the DFlash draft

Place the BF16 Hugging Face checkpoint
[`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16)
in `~/Models/nemotron-3.5-Bolt-E4B-BF16`, and its matching DFlash F16 GGUF at
`~/Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf`.
The tested draft uses block size 8, mask token 4 and target layers
`[5, 11, 17, 23, 35]` (zero-based). Keep a llama.cpp checkout at
`~/Workspaces/llama.cpp` for `gguf-py` and the comparison server.

```bash
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1
mkdir -p dist/nemotron-q4 dist/libs logs
python -m mlc_llm convert_weight ~/Models/nemotron-3.5-Bolt-E4B-BF16 \
  --quantization q4f16_1 --device cuda -o dist/nemotron-q4
python -m mlc_llm gen_config ~/Models/nemotron-3.5-Bolt-E4B-BF16 \
  --quantization q4f16_1 --conv-template gemma4_instruction \
  --context-window-size 8192 --prefill-chunk-size 512 --max-batch-size 1 \
  -o dist/nemotron-q4
python scripts/prepare_xavier_dflash.py --target dist/nemotron-q4 \
  --draft ~/Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf
python -m mlc_llm convert_weight dist/draft-hf --quantization q0f16 \
  --model-type dflash --device cpu -o dist/draft
export MLC_DFLASH_CUDA_GRAPH=1 MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_VERIFY_THREE=1
export MLC_JP5_Q4_CUBLAS=1
# The draft's Tensor Core compilation path uses its model-specific override.
env -u MLC_JP5_Q4_CUBLAS python -m mlc_llm compile dist/draft/mlc-chat-config.json \
  --device cuda --opt O2 -o dist/libs/draft-cached-tc-sm72.so
bash scripts/build_xavier_kernels.sh
python scripts/build_xavier_prefill.py
```

`prepare_xavier_dflash.py` exports the F16/F32 draft tensors without
requantization and reuses the converted target via local symlinks. It also
accepts `--gguf-python` for a different `gguf-py` location. Keep the original
converted target directory available. `MLC_GEMMA4_TEXT_ONLY=1` omits audio entry
points that do not compile on this target. Only text generation was qualified.

### 4. Validate and reproduce the benchmark

```bash
source scripts/xavier/env.sh
sudo nvpmodel -m 0
sudo jetson_clocks
python tests/python/op/test_xavier_prefill.py --output logs/prefill-components.json
bash scripts/run_xavier_prefill_experiments.sh
# Individual optimized run:
python scripts/bench_xavier_dflash.py --prefill-variant hybrid \
  --target-lib dist/libs/target-dflash-dp4a-attn-sm72.so --draft-tokens 2 \
  --output logs/nemotron-hybrid.json
# llama.cpp control; uses ~/Workspaces/llama.cpp/build/bin/llama-server:
python scripts/bench_xavier_llama.py --output logs/llama-control.json
```

`width` is the simpler prefill choice; `hybrid` additionally uses FP16
probabilities in WMMA attention. Both preserve the existing small-query
verification path. **108 component checks** passed, including independent
FP32 attention comparisons and three-query fallback checks. Elevated Compute
Sanitizer memcheck completed with **zero errors**. Natural-stop tests returned
`391`, the requested three summary bullets plus two risks, and `Done`.

The tested configuration is greedy, one sequence, two draft proposals, prefix
caching disabled and an 8K context. Q8 activations and WMMA change arithmetic;
broad accuracy/perplexity equivalence has not been established. Board shutdowns
also occurred during the investigation; these measurements do not establish
long-term hardware stability. Benchmark runners use temperature/memory stop
limits. Keep compilation and GPU inference sequential.

## HTTPS serving on Xavier

The optional service exposes an OpenAI-compatible API on the LAN or a
Headscale/Tailscale network. Use your own hostname, such as
`https://xavier.example.com/v1`. It supports resident Nemotron with either
the 12B/MTP adapter or the EmbeddingGemma 2 text worker. It uses a
certificate signed by a local CA, bearer authentication and streaming responses.
Requests share one GPU queue; the 12B adapter supports 4K text/tool chat, and
the embedding PoC accepts up to 128 text tokens.
See the
[deployment and client instructions](deploy/xavier/README.md) for CA trust,
credentials, service management and certificate renewal.

The optional [protected-prefix service profile](docs/deploy/xavier_service_prefix.md)
adds DFlash prefix reuse, safe independent KV copies, corrected LRU eviction,
and a pinned shared policy with two recent conversations. A short uncached
old/current/old check found no detectable raw-throughput regression; see the
[measurements and limits](docs/benchmarks/xavier/raw-regression-summary.json).

The [d1 decision worker](docs/deploy/xavier_d1_decisions.md) adds bounded text
yes/no and single-choice decisions from LiquidAI/d1-Omni-2B alongside resident
Nemotron and EmbeddingGemma 2. Its LFM3 port includes set attention and gated
convolutions. Initial 256-token FP16 forward time is 277 ms versus 995 ms
in FP32. A separately compiled 2K variant and JEV-style `/v1/systemone`
endpoint support structured mail decisions; audio and vision remain unsupported. The
[Pi specialist extension](docs/deploy/xavier_pi_specialists.md) lets Nemotron
call d1 decisions and EmbeddingGemma semantic ranking, with direct slash commands
for both workers.

The [runbook search/RAG trial](docs/deploy/xavier_runbook_rag.md) exercises
EmbeddingGemma 2 retrieval followed by cited Nemotron answers over 251 public
documentation passages. It includes a reusable client, a keyword-search control,
frozen questions and complete results, including incorrect answers.

## Upstream and attribution

[MLC-LLM](https://github.com/mlc-ai/mlc-llm) and its contributors provide the
compiler and serving engine. DFlash was ported from Lev Stambler's fork; its
history and source attribution are retained. This repository retains the
upstream [Apache-2.0 license](LICENSE). See [README.upstream.md](README.upstream.md)
for the broader upstream project, documentation and contribution links.
