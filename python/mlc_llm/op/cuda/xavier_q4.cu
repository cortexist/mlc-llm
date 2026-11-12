// Experimental sm_72 q4f16_1 GEMV. Weights stay in the existing NK layout.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void quant_q8(const half* x, uint32_t* q, float2* aux, int k, int b, bool tiled) {
  int lane = threadIdx.x & 31;
  int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (group >= b * (k / 32)) return;
  float v = __half2float(x[group * 32 + lane]);
  float maximum = fabsf(v);
  for (int d = 16; d; d /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
  float scale = maximum > 0 ? maximum / 127.f : 1.f;
  int a = __float2int_rn(v / scale);
  int sum = a;
  for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
  uint32_t even = 0, odd = 0;
  int sum8 = 0;
  int base = lane & ~7;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int e = __shfl_sync(0xffffffff, a, base + 2 * i);
    int o = __shfl_sync(0xffffffff, a, base + 2 * i + 1);
    even |= (e & 255u) << (8 * i);
    odd |= (o & 255u) << (8 * i);
    sum8 += e + o;
  }
  if ((lane & 7) == 0) {
    if (tiled)
      reinterpret_cast<uint4*>(q)[group * 4 + lane / 8] =
          make_uint4(even, odd, __float_as_uint(scale), uint32_t(sum8));
    else {
      q[group * 8 + (lane / 8) * 2] = even;
      q[group * 8 + (lane / 8) * 2 + 1] = odd;
    }
  }
  if (lane == 0) aux[group] = make_float2(scale, float(sum));
}

template <int B, int R, typename Out>
__global__ void mat_q4_tiled(const uint32_t* w, const half* scales, const uint4* q, Out* y, int n,
                             int k) {
  int lane = threadIdx.x & 31;
  int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) / 32) * R;
  float accum[B][R] = {};
  int chunks = k / 8;
  for (int c = lane; c < chunks; c += 32) {
    uint32_t weight[R];
    float scale[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      weight[r] = row0 + r < n ? w[(row0 + r) * chunks + c] : 0;
      scale[r] = row0 + r < n ? __half2float(scales[(row0 + r) * (k / 32) + c / 4]) : 0;
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      uint4 a = q[b * chunks + c];
      float as = __uint_as_float(a.z);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        int dot = __dp4a(int(weight[r] & 0x0f0f0f0f), int(a.x), -7 * int(a.w));
        dot = __dp4a(int((weight[r] >> 4) & 0x0f0f0f0f), int(a.y), dot);
        accum[b][r] += scale[r] * as * float(dot);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      for (int d = 16; d; d /= 2) accum[b][r] += __shfl_down_sync(0xffffffff, accum[b][r], d);
      if (lane == 0 && row0 + r < n) y[b * n + row0 + r] = Out(accum[b][r]);
    }
  }
}

// As mat_q4_tiled, with lanes that own groups of 32 weights instead of chunks of 8.
// The integer dot product of a group is exact, so its two scales are applied once
// instead of four times: about half of the arithmetic of a row.
template <int B, int R, typename Out>
__global__ void mat_q4_grouped(const uint4* w, const half* scales, const uint4* q,
                               const float2* aux, Out* y, int n, int groups) {
  constexpr uint32_t Low = 0x0f0f0f0f;
  int lane = threadIdx.x & 31;
  int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) / 32) * R;
  float accum[B][R] = {};
  for (int g = lane; g < groups; g += 32) {
    uint4 even[R], odd[R];
    float scale[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      bool valid = row0 + r < n;
      uint4 v = valid ? w[(row0 + r) * groups + g] : make_uint4(0, 0, 0, 0);
      even[r] = make_uint4(v.x & Low, v.y & Low, v.z & Low, v.w & Low);
      odd[r] = make_uint4((v.x >> 4) & Low, (v.y >> 4) & Low, (v.z >> 4) & Low, (v.w >> 4) & Low);
      scale[r] = valid ? __half2float(scales[(row0 + r) * groups + g]) : 0;
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      // Activations of even and odd positions alternate, a pair per chunk.
      uint4 a = q[(b * groups + g) * 2], c = q[(b * groups + g) * 2 + 1];
      float2 s = aux[b * groups + g];
      int offset = -7 * int(s.y);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        int dot = __dp4a(int(even[r].x), int(a.x), offset);
        dot = __dp4a(int(odd[r].x), int(a.y), dot);
        dot = __dp4a(int(even[r].y), int(a.z), dot);
        dot = __dp4a(int(odd[r].y), int(a.w), dot);
        dot = __dp4a(int(even[r].z), int(c.x), dot);
        dot = __dp4a(int(odd[r].z), int(c.y), dot);
        dot = __dp4a(int(even[r].w), int(c.z), dot);
        dot = __dp4a(int(odd[r].w), int(c.w), dot);
        accum[b][r] += scale[r] * s.x * float(dot);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      for (int d = 16; d; d /= 2) accum[b][r] += __shfl_down_sync(0xffffffff, accum[b][r], d);
      if (lane == 0 && row0 + r < n) y[b * n + row0 + r] = Out(accum[b][r]);
    }
  }
}

template <int B, bool Q8, typename Out>
__global__ void mat_q4(const uint32_t* w, const half* scales, const half* x, const uint32_t* q,
                       const float2* aux, Out* y, int n, int k) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (row >= n) return;
  float accum[B] = {};
  int groups = k / 32;
  for (int g = lane; g < groups; g += 32) {
    uint4 weight = reinterpret_cast<const uint4*>(w)[row * groups + g];
    uint32_t words[4] = {weight.x, weight.y, weight.z, weight.w};
    half hs = scales[row * groups + g];
    if constexpr (Q8) {
      float ws = __half2float(hs);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t v = words[j];
          dot = __dp4a(int(v & 0x0f0f0f0f), int(q[(b * groups + g) * 8 + 2 * j]), dot);
          dot = __dp4a(int((v >> 4) & 0x0f0f0f0f), int(q[(b * groups + g) * 8 + 2 * j + 1]), dot);
        }
        float2 a = aux[b * groups + g];
        accum[b] += ws * a.x * (float(dot) - 7.f * a.y);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          // Preserve the checkpoint's FP16 dequantization rounding.
          float v = __half2float(__hmul(__int2half_rn(int((words[j] >> (4 * i)) & 15) - 7), hs));
#pragma unroll
          for (int b = 0; b < B; ++b) accum[b] += v * __half2float(x[b * k + g * 32 + j * 8 + i]);
        }
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) {
    for (int d = 16; d; d /= 2) accum[b] += __shfl_down_sync(0xffffffff, accum[b], d);
    if (lane == 0) y[b * n + row] = Out(accum[b]);
  }
}

template <int B, typename Out>
void dispatch_mode(const uint32_t* w, const half* s, const half* x, uint32_t* q, float2* aux,
                   Out* y, int n, int k, int mode, cudaStream_t stream) {
  dim3 block(128), grid((n + 3) / 4);
  if (mode >= 6) {
#define GROUPED(R)                                                                      \
  mat_q4_grouped<B, R><<<(n + 4 * R - 1) / (4 * R), 128, 0, stream>>>(                  \
      (const uint4*)w, s, (const uint4*)q, aux, y, n, k / 32)
    switch (mode) {
      case 6:
        GROUPED(1);
        break;
      case 7:
        GROUPED(2);
        break;
      case 8:
        GROUPED(4);
        break;
      case 9:
        GROUPED(8);
        break;
    }
#undef GROUPED
  } else if (mode >= 2) {
#define TILE(R) \
  mat_q4_tiled<B, R><<<(n + 4 * R - 1) / (4 * R), 128, 0, stream>>>(w, s, (const uint4*)q, y, n, k)
    switch (mode) {
      case 2:
        TILE(1);
        break;
      case 3:
        TILE(2);
        break;
      case 4:
        TILE(4);
        break;
      case 5:
        TILE(8);
        break;
    }
#undef TILE
  } else if (mode)
    mat_q4<B, true><<<grid, block, 0, stream>>>(w, s, x, q, aux, y, n, k);
  else
    mat_q4<B, false><<<grid, block, 0, stream>>>(w, s, x, q, aux, y, n, k);
}

extern "C" int xavier_q4_run(const void* w, const void* s, const void* x, void* y, void* q,
                             void* aux, int n, int k, int b, int mode, int out_f32,
                             void* stream_ptr) {
  if (k % 32 || b < 1 || b > 4) return int(cudaErrorInvalidValue);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  if (mode)
    quant_q8<<<(b * (k / 32) + 3) / 4, 128, 0, stream>>>((const half*)x, (uint32_t*)q, (float2*)aux,
                                                         k, b, mode >= 2 && mode < 6);
#define CALL(B, T)                                                                      \
  dispatch_mode<B, T>((const uint32_t*)w, (const half*)s, (const half*)x, (uint32_t*)q, \
                      (float2*)aux, (T*)y, n, k, mode, stream)
#define CASE(B)       \
  case B:             \
    if (out_f32) {    \
      CALL(B, float); \
    } else {          \
      CALL(B, half);  \
    }                 \
    break
  switch (b) {
    CASE(1);
    CASE(2);
    CASE(3);
    CASE(4);
  }
#undef CASE
#undef CALL
  return int(cudaGetLastError());
}

#ifdef MLC_XAVIER_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <stdexcept>

namespace {
void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }
// The kernel of each row count, 1..4.
// One row favors one output per warp; several rows amortize across eight outputs.
constexpr int Tiled[] = {2, 5, 5, 5};
// Grouped arithmetic; tiles as measured on the Unified 12B.
constexpr int Grouped[] = {7, 7, 9, 8};

template <const int* Modes>
void q4_ffi(DLTensor* x, DLTensor* w, DLTensor* s, DLTensor* y, DLTensor* q, DLTensor* aux) {
  int k = x->shape[x->ndim - 1], n = w->shape[0], b = 1;
  for (int i = 0; i < x->ndim - 1; ++i) b *= x->shape[i];
  if (x->device.device_type != kDLCUDA || x->dtype.code != kDLFloat || x->dtype.bits != 16 ||
      w->dtype.code != kDLUInt || w->dtype.bits != 32 || k % 32 || w->shape[1] != k / 8 ||
      s->shape[1] != k / 32 || b < 1 || b > 4 || (y->dtype.bits != 16 && y->dtype.bits != 32))
    throw std::runtime_error("Xavier q4 requires CUDA q4f16_1, contiguous tensors and 1..4 rows");
  void* stream = TVMFFIEnvGetStream(kDLCUDA, x->device.device_id);
  int result = xavier_q4_run(data(w), data(s), data(x), data(y), data(q), data(aux), n, k, b,
                             Modes[b - 1], y->dtype.bits == 32, stream);
  if (result) throw std::runtime_error(cudaGetErrorString(cudaError_t(result)));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_q4, q4_ffi<Tiled>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_q4_grouped, q4_ffi<Grouped>);
#endif
