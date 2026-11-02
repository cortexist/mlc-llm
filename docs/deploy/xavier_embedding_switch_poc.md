# Nemotron ↔ EmbeddingGemma 2 resident switching PoC

Cerebellum now exposes two complementary resident workers at the existing
`https://cerebellum.lan/v1` endpoint:

| API model | Task | Current limit |
|---|---|---|
| `nemotron` (also `nemotron:e4b`) | Existing DFlash chat/tools | 16K context |
| `embeddinggemma-2` | FP32 text embeddings | 128 tokens including BOS/EOS |

The optional `embedding-poc.conf` systemd drop-in disables the resident 12B
worker and enables EmbeddingGemma 2. It preserves the previous dual-chat
configuration and all checkpoints/libraries. Nemotron remains Pi's default.
Embedding models are accessed through `/v1/embeddings`, not Pi's chat picker.

## What was implemented

This is the actual local `~/Models/embeddinggemma-2` text checkpoint: 271,002,648
parameters, extracted without modifying the original multimodal checkpoint.
MLC converts them to FP32 (`q0f32`, 1,084,010,592 tensor bytes), with no additional
quantization. Xavier has no native BF16; the
[model card](https://huggingface.co/google/embeddinggemma-2) explicitly warns
against FP16. Vision/audio and d1-Omni are outside this first PoC.

The implementation includes bidirectional grouped-query attention, separate
local/global head layouts, Q/K/V normalization, RoPE, projection-only per-layer
inputs, layer scalars and the trained 512→768 projection. An entire small encoder
matches Transformers on CPU, including masked padding and a small sliding
window, with maximum errors below `2e-7`. FP16 conversion is explicitly rejected.

The GPU worker uses one fixed, padded 128-token bucket and processes batch items
sequentially. It adds the checkpoint's BOS and EOS explicitly: MLC's tokenizer
encodes content without Hugging Face post-processing. Valid tokens are
mean-pooled and L2-normalized. Longer inputs fail with HTTP 400; nothing is
silently truncated. The native checkpoint's advertised 8K context is not
implemented by this bounded PoC.

Four real-checkpoint vectors were compared against Transformers 5.22.1 CPU
FP32. Token counts match (64 total); maximum vector error was `2.51e-7`, with
all cosine similarities above `0.999999999997`. Query/relevant-document cosine
was 0.899829, versus 0.543209 for the unrelated document. These are functional
checks, not a retrieval-quality evaluation.

## Switching measurements

November 1, 2026; Xavier AGX 32GB, JetPack 5.1.8, CUDA 11.4, fixed MAXN clocks,
cool fan profile. One persistent HTTPS connection from nowhere, matched inputs,
two warmups per model, 16 samples per model/mode. Repeated-model blocks and
alternating pairs run in both orders. Prefix caching is disabled. The Nemotron
prompt asks for 17×23, with a 16-token output limit; every response is `391`.
The embedding input is the same short password-reset search query on every call.

| Median latency | Repeated same model | Alternating models | Difference |
|---|---:|---:|---:|
| Nemotron complete HTTPS response | 530.19 ms | 525.95 ms | −4.24 ms |
| Embedding complete HTTPS response | 55.40 ms | 56.12 ms | +0.71 ms |
| Embedding compiled forward + output copy | 42.13 ms | 41.40 ms | −0.73 ms |

The embedding forward timer synchronizes before the VM call and stops after
copying the per-token output to CPU. It excludes tokenization/input copies,
pooling, JSON serialization and networking; it is not a CUDA-event kernel-only
timer. The worker median including preprocessing/pooling was 43.91 / 43.17 ms.
The HTTPS embedding P10–P90 ranges were 54.20–59.11 ms repeated and
55.48–57.87 ms alternating. Both workers remained resident in the same PID
(1942), and the embedding instance ID stayed unchanged across all calls.

This experiment found no material alternating-request penalty. The small signed
differences overlap ordinary request variation; they are **not an isolated CUDA
context-switch measurement**, a p99 SLA, or evidence of zero hardware overhead.
It does demonstrate that switching needs no checkpoint reload or per-request
weight transfer. Full raw timings and verification are saved in
[the result summary](../benchmarks/xavier/embedding-switch-poc-summary.json).

The existing queue still serializes whole requests. In a concurrent check, a
short embedding waited 1,619 ms for the active Nemotron generation, then took
41.60 ms for its forward/copy. Token/block preemption and priority scheduling
have not been implemented. Resident weights remove loading delays; they do not
remove contention for GPU time.

## API

Use the existing CA trust and bearer key. No TLS/DNS/port change is needed.

```bash
curl https://cerebellum.lan/v1/embeddings \
  -H "Authorization: Bearer $CEREBELLUM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":"task: search result | query: how do I reset my password?"}'
```

Inputs are one string or 1–16 strings. Task prefixes are supplied by the caller;
the service does not choose a retrieval task silently. Documents can use
`title: none | text: ...`. The endpoint supports `dimensions` 128/256/512/768,
renormalizes after truncation, and returns floats or `encoding_format: base64`.
Headers report worker/forward time, queue wait and the resident embedding
instance ID. Input/normalization checks, base64 output and overlong rejection
were exercised over HTTPS.

## Build, reproduce and restore

With the Xavier toolchain from the main deployment instructions:

```bash
sudo systemctl stop mlc-nemotron
bash scripts/build_xavier_embedding_poc.sh
# Copy the new model registry/module and service helpers to the deployed checkout
# when building somewhere other than the active source tree.
sudo install -m 644 deploy/cerebellum/embedding-poc.conf \
  /etc/systemd/system/mlc-nemotron.service.d/embedding-poc.conf
sudo systemctl daemon-reload
sudo systemctl start mlc-nemotron
```

The build produces `dist/embeddinggemma2-text-q0f32` and
`dist/libs/embeddinggemma2-text-f32-s128-sm72.so`. It needs no new C++ runtime
build. Keep compilation and inference separate. The new serving script needs
`embeddinggemma2_service.py` and `bench_xavier_mtp.py` beside it. The embedding
VM is initialized and used on its own single thread; cancellation drains GPU
work before releasing the same queue used by chat/tool requests.

```bash
python scripts/bench_cerebellum_switch.py --repeats 8 --output /tmp/switch.json
python tests/python/model/test_embeddinggemma2.py
python tests/python/serve/test_xavier_model_queue.py
```

For reference vectors, run `scripts/check_embeddinggemma2_reference.py` against
the original checkpoint on CPU. Local experimental records are under
`nowhere:~/Workspaces/cerebellum-llm-bench/mlc-jp5/logs/embedding-switch-poc-20261101`
and the corresponding remote `mlc-dflash-work/logs` directory.

To restore Nemotron + 12B, move the optional `embedding-poc.conf` drop-in out of
`/etc/systemd/system/mlc-nemotron.service.d`, reload systemd and restart the unit.
The retained `dual-model.conf` enables 12B again. The power/stability issue
remains deferred; this PoC uses only short requests and is not an uptime test.
