// Assistant attention over the shared K/V state, one query position.
// Only the valid keys are read: the whole context for a global layer, the last
// window + 1 for a local one. The state layout [1, capacity, K/V heads, D] is read
// in place. Query heads go in pairs that share a K/V head: one pair per K/V head
// in the E4B, eight pairs on the single global head of the Unified 12B.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <stdexcept>

namespace {
constexpr int G = 2, Tile = 64;  // query heads per block, keys per block

__device__ __forceinline__ int first_key(int n, int window) {
  return window ? max(0, n - window - 1) : 0;
}

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

// A warp scores one key for a pair of query heads; a lane owns D/32 dimensions.
// Block row p holds query heads 2p and 2p+1, which read K/V head p / pairs.
template <int D>
__global__ void score(const half* q, const half* keys, const int* length, float* scores,
                      int capacity, int window, int kv, int pairs) {
  constexpr int N = D / 32;
  int n = min(length[0], capacity), begin = first_key(n, window);
  int p = blockIdx.y, h = p / pairs, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float qa[N], qb[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    qa[i] = __half2float(q[(p * G) * D + lane * N + i]);
    qb[i] = __half2float(q[(p * G + 1) * D + lane * N + i]);
  }
  int first = begin + blockIdx.x * Tile, last = min(n, first + Tile);
  for (int key = first + warp; key < last; key += 8) {
    const half* k = keys + (int64_t(key) * kv + h) * D + lane * N;
    float a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float v = __half2float(k[i]);
      a += qa[i] * v;
      b += qb[i] * v;
    }
    for (int d = 16; d; d /= 2) {
      a += __shfl_xor_sync(0xffffffff, a, d);
      b += __shfl_xor_sync(0xffffffff, b, d);
    }
    if (lane == 0) {
      scores[(p * G) * capacity + key] = a;
      scores[(p * G + 1) * capacity + key] = b;
    }
  }
}

// Softmax of one query head. As in the reference, scores and weights round to FP16.
__global__ void normalize(const float* scores, const int* length, half* weights, int capacity,
                          int window) {
  int n = min(length[0], capacity), begin = first_key(n, window);
  const float* s = scores + blockIdx.x * capacity;
  half* w = weights + blockIdx.x * capacity;
  __shared__ float shared[8];
  float top = -CUDART_INF_F;
  for (int key = begin + threadIdx.x; key < n; key += 256)
    top = fmaxf(top, __half2float(__float2half_rn(s[key])));
  top = block_reduce(top, shared, true);
  float sum = 0;
  for (int key = begin + threadIdx.x; key < n; key += 256)
    sum += expf(__half2float(__float2half_rn(s[key])) - top);
  sum = block_reduce(sum, shared, false);
  for (int key = begin + threadIdx.x; key < n; key += 256)
    w[key] = __float2half_rn(expf(__half2float(__float2half_rn(s[key])) - top) / sum);
}

// A block sums one tile of values for a pair of query heads; a thread owns D/128 dimensions.
template <int D>
__global__ void accumulate(const half* weights, const half* values, const int* length,
                           float* partial, int capacity, int chunks, int window, int kv,
                           int pairs) {
  constexpr int N = D / 128;
  int n = min(length[0], capacity), begin = first_key(n, window);
  int p = blockIdx.y, h = p / pairs, t = threadIdx.x;
  int first = begin + blockIdx.x * Tile, last = min(n, first + Tile);
  if (first >= last) return;
  __shared__ float w[G][Tile];
  if (t < Tile) {
#pragma unroll
    for (int g = 0; g < G; ++g)
      w[g][t] = first + t < last ? __half2float(weights[(p * G + g) * capacity + first + t]) : 0.f;
  }
  __syncthreads();
  float acc[G][N] = {};
  for (int key = first; key < last; ++key) {
    const half* v = values + (int64_t(key) * kv + h) * D + t * N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float x = __half2float(v[i]);
#pragma unroll
      for (int g = 0; g < G; ++g) acc[g][i] += w[g][key - first] * x;
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int i = 0; i < N; ++i)
      partial[(int64_t(p * G + g) * chunks + blockIdx.x) * D + t * N + i] = acc[g][i];
}

// Tiles are added in key order, so the result does not depend on scheduling.
template <int D>
__global__ void combine(const float* partial, const int* length, half* output, int capacity,
                        int chunks, int window) {
  int n = min(length[0], capacity), begin = first_key(n, window);
  int used = (n - begin + Tile - 1) / Tile;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0;
    for (int c = 0; c < used; ++c) sum += partial[(int64_t(blockIdx.x) * chunks + c) * D + d];
    output[blockIdx.x * D + d] = __float2half_rn(sum);
  }
}

template <typename T>
T* data(DLTensor* t) {
  return reinterpret_cast<T*>(static_cast<char*>(t->data) + t->byte_offset);
}
bool is(DLTensor* t, int code, int bits, std::initializer_list<int64_t> shape) {
  if (t->device.device_type != kDLCUDA || t->dtype.code != code || t->dtype.bits != bits ||
      t->ndim != int(shape.size()) || !tvm::ffi::IsContiguous(*t))
    return false;
  int i = 0;
  for (int64_t extent : shape)
    if (t->shape[i++] != extent) return false;
  return true;
}

template <int Window>
void attention(DLTensor* q, DLTensor* keys, DLTensor* values, DLTensor* length, DLTensor* output,
               DLTensor* scores, DLTensor* weights, DLTensor* partial) {
  bool shaped = q->ndim == 4 && keys->ndim == 4;
  int64_t kv = shaped ? q->shape[1] : 0, group = shaped ? q->shape[2] : 0;
  int64_t dim = shaped ? q->shape[3] : 0, capacity = shaped ? keys->shape[1] : 0;
  int64_t heads = kv * group, chunks = ((Window ? Window + 1 : capacity) + Tile - 1) / Tile;
  if ((dim != 256 && dim != 512) || capacity < 1 || kv < 1 || group < G || group % G ||
      !is(q, kDLFloat, 16, {1, kv, group, dim}) ||
      !is(keys, kDLFloat, 16, {1, capacity, kv, dim}) ||
      !is(values, kDLFloat, 16, {1, capacity, kv, dim}) || !is(length, kDLInt, 32, {1}) ||
      !is(output, kDLFloat, 16, {1, kv, group, dim}) ||
      !is(scores, kDLFloat, 32, {heads, capacity}) ||
      !is(weights, kDLFloat, 16, {heads, capacity}) ||
      !is(partial, kDLFloat, 32, {heads, chunks, dim}))
    throw std::runtime_error("Assistant attention requires FP16 [1,H,G,D] queries with even G "
                             "and [1,capacity,H,D] shared K/V on CUDA");
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  int pairs = group / G;
  dim3 tiles(chunks, heads / G);
#define LAUNCH(D)                                                                            \
  score<D><<<tiles, 256, 0, stream>>>(data<half>(q), data<half>(keys), data<int>(length),   \
                                      data<float>(scores), capacity, Window, kv, pairs);    \
  normalize<<<heads, 256, 0, stream>>>(data<float>(scores), data<int>(length),              \
                                       data<half>(weights), capacity, Window);              \
  accumulate<D><<<tiles, 128, 0, stream>>>(data<half>(weights), data<half>(values),         \
                                           data<int>(length), data<float>(partial),         \
                                           capacity, chunks, Window, kv, pairs);            \
  combine<D><<<heads, 256, 0, stream>>>(data<float>(partial), data<int>(length),            \
                                        data<half>(output), capacity, chunks, Window)
  if (dim == 256) {
    LAUNCH(256);
  } else {
    LAUNCH(512);
  }
#undef LAUNCH
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma4_mtp_attention_local, attention<512>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma4_mtp_attention_local_1024, attention<1024>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma4_mtp_attention_global, attention<0>);
