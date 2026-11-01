// Small-query Gemma4 cross-attention in the existing 512-wide cache.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

__device__ float block_reduce(float v, float* shared, bool maximum) {
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  for (int d = 16; d; d /= 2) {
    float other = __shfl_xor_sync(0xffffffff, v, d);
    v = maximum ? fmaxf(v, other) : v + other;
  }
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < 8 ? shared[lane] : (maximum ? -CUDART_INF_F : 0.f);
    for (int d = 16; d; d /= 2) {
      float other = __shfl_xor_sync(0xffffffff, v, d);
      v = maximum ? fmaxf(v, other) : v + other;
    }
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  v = shared[0];
  __syncthreads();
  return v;
}

template <int D, bool Local>
__global__ void paged_attention(const half* q, const int* qptr, const half* pages, const int* pptr,
                                const int* pvalues, const int* length, half* output, float* lse,
                                int batch, int hq, int hkv, int page_size, int causal,
                                float sm_scale) {
  int qi = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
  int lane = t & 31, warp = t / 32;
  int seq = 0;
  while (seq + 1 < batch && qi >= qptr[seq + 1]) ++seq;
  int row = qi - qptr[seq], kh = head / (hq / hkv);
  int first_page = pptr[seq], num_pages = pptr[seq + 1] - first_page;
  int offset = Local ? length[batch + seq] : 0, sink = Local ? length[2 * batch + seq] : 0;
  int count = num_pages ? ((num_pages - 1) * page_size + length[seq] - offset + sink) : 0;
  int begin = Local && causal ? max(0, count - max(512 - row - 1, 0)) : 0;
  if (!Local && causal) count = min(count, count - (qptr[seq + 1] - qptr[seq]) + row + 1);
  float qreg[D / 32];
#pragma unroll
  for (int i = 0; i < D / 32; ++i)
    qreg[i] = __half2float(q[(qi * hq + head) * 512 + lane + 32 * i]);
  __shared__ float scores[512];
  __shared__ float reduction[8];
  float m = -50000.f, denom = 1.f, acc = 0.f, acc_upper = 0.f;
  for (int base = begin; base < count; base += 512) {
    int tile = min(512, count - base);
    for (int key = warp; key < tile; key += 8) {
      int logical = base + key;
      int slot = logical < sink ? logical : logical - sink + offset;
      int page = pvalues[first_page + slot / page_size], pos = slot % page_size;
      const half* k = pages + ((((int64_t)page * 2) * hkv + kh) * page_size + pos) * 512;
      float dot = 0;
#pragma unroll
      for (int i = 0; i < D / 32; ++i) dot += qreg[i] * __half2float(k[lane + 32 * i]);
      for (int d = 16; d; d /= 2) dot += __shfl_down_sync(0xffffffff, dot, d);
      if (lane == 0) scores[key] = dot * sm_scale * 1.4426950408889634f;
    }
    __syncthreads();
    float vmax = -CUDART_INF_F;
    for (int key = t; key < tile; key += 256) vmax = fmaxf(vmax, scores[key]);
    float next_m = fmaxf(m, block_reduce(vmax, reduction, true));
    float sum = 0;
    for (int key = t; key < tile; key += 256) {
      float v = exp2f(scores[key] - next_m);
      scores[key] = v;
      sum += v;
    }
    float next_d = block_reduce(sum, reduction, false);
    float factor = exp2f(m - next_m);
    acc *= factor;
    acc_upper *= factor;
    for (int key = 0; key < tile; ++key) {
      int logical = base + key;
      int slot = logical < sink ? logical : logical - sink + offset;
      int page = pvalues[first_page + slot / page_size], pos = slot % page_size;
      const half* v = pages + ((((int64_t)page * 2 + 1) * hkv + kh) * page_size + pos) * 512;
      acc += scores[key] * __half2float(v[t]);
      if constexpr (D == 512) acc_upper += scores[key] * __half2float(v[t + 256]);
    }
    denom = denom * factor + next_d;
    m = next_m;
    __syncthreads();
  }
  output[(qi * hq + head) * 512 + t] = __float2half(acc / denom);
  output[(qi * hq + head) * 512 + t + 256] = __float2half(acc_upper / denom);
  if (t == 0) lse[qi * hq + head] = m + log2f(denom);
}

extern "C" int xavier_attention_run(const void* q, const void* qi, const void* pages,
                                    const void* pi, const void* pv, const void* li, void* output,
                                    void* lse, int queries, int batch, int hq, int hkv,
                                    int page_size, int causal, int local, float scale,
                                    void* stream) {
#define LAUNCH(D, L)                                                                      \
  paged_attention<D, L><<<dim3(queries, hq), 256, 0, (cudaStream_t)stream>>>(             \
      (const half*)q, (const int*)qi, (const half*)pages, (const int*)pi, (const int*)pv, \
      (const int*)li, (half*)output, (float*)lse, batch, hq, hkv, page_size, causal, scale)
  if (local) {
    LAUNCH(256, true);
  } else {
    LAUNCH(512, false);
  }
#undef LAUNCH
  return int(cudaGetLastError());
}

#ifdef MLC_XAVIER_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace {
void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }
template <bool Local>
void attention_ffi(DLTensor* q, DLTensor* qi, DLTensor* pages, DLTensor* pi, DLTensor* pv,
                   DLTensor* li, DLTensor* kpos, DLTensor* qpos, DLTensor* o, DLTensor* lse,
                   int64_t causal, int64_t rotary, double rope_scale, double theta, double scale) {
  if (q->device.device_type != kDLCUDA || q->dtype.bits != 16 || q->shape[2] != 512 ||
      pages->shape[4] != 512 || li->ndim != (Local ? 2 : 1) || (Local && li->shape[0] != 3) ||
      rotary != 0)
    throw std::runtime_error(
        "Xavier local attention requires FP16 padded Gemma4 heads and pre-applied RoPE");
  int result = xavier_attention_run(data(q), data(qi), data(pages), data(pi), data(pv), data(li),
                                    data(o), data(lse), q->shape[0], qi->shape[0] - 1, q->shape[1],
                                    pages->shape[2], pages->shape[3], causal, Local, scale,
                                    TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  if (result) throw std::runtime_error(cudaGetErrorString(cudaError_t(result)));
}

tvm::ffi::Function wrap_attention(tvm::ffi::Function fallback, int64_t local) {
  const char* off = std::getenv("MLC_XAVIER_ATTN_OFF");
  bool enabled = !(off && std::strcmp(off, "1") == 0);
  tvm::ffi::Function native = local ? tvm::ffi::Function::FromTyped(attention_ffi<true>)
                                    : tvm::ffi::Function::FromTyped(attention_ffi<false>);
  return tvm::ffi::Function::FromPacked(
      [fallback, native, enabled](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        DLTensor* q = args[0].cast<DLTensor*>();
        // Keep the existing tiled kernel for long prefill batches.
        if (enabled && q->shape[0] <= 4)
          native.CallPacked(args, result);
        else
          fallback.CallPacked(args, result);
      });
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_local_paged_attention, attention_ffi<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_global_paged_attention, attention_ffi<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_wrap_attention, wrap_attention);
#endif
