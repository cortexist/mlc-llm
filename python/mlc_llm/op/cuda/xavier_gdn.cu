// The gated delta rule of the Qwen3.5 linear-attention layers on sm_72.
// One block per value head. A thread keeps sixteen rows of one state column in registers
// for the whole chunk, so a token costs two barriers instead of four passes over memory.
// Products and updates are those of the TIR kernel; the 128 terms of a dot product are
// added as eight partial sums of sixteen instead of in one sequence.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {
constexpr int Dim = 128;                       // key and value head width: state rows, columns
constexpr int Rows = 16;                       // state rows a thread owns
constexpr int Chunks = Dim / Rows;             // threads per column
constexpr float Scale = 0.08838834764831845f;  // 1 / sqrt(Dim)

__global__ void gated_delta(const half* q, const half* k, const half* v, const float* gate,
                            const float* beta, const float* state_in, float* out, float* state_out,
                            int steps, int key_heads, int value_heads) {
  __shared__ float keys[2][Dim], queries[2][Dim], decayed[Chunks][Dim], updated[Chunks][Dim];
  const int column = threadIdx.x % Dim, chunk = threadIdx.x / Dim;
  const int batch = blockIdx.x / value_heads, head = blockIdx.x % value_heads;
  const int key_head = head / (value_heads / key_heads);
  const int64_t first = int64_t(batch) * steps;
  const int64_t cell = (int64_t(batch) * value_heads + head) * Dim * Dim;
  float s[Rows], key[Rows];
#pragma unroll
  for (int i = 0; i < Rows; ++i) s[i] = state_in[cell + (chunk * Rows + i) * Dim + column];
  if (threadIdx.x < Dim) {
    int64_t at = (first * key_heads + key_head) * Dim + threadIdx.x;
    keys[0][threadIdx.x] = __half2float(k[at]);
    queries[0][threadIdx.x] = __half2float(q[at]);
  }
  __syncthreads();
  for (int t = 0; t < steps; ++t) {
    const int slot = t & 1;
    const int64_t row = (first + t) * value_heads + head;
    const float g = gate[row], b = beta[row];
    float dot = 0;
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      s[i] = s[i] * g;
      key[i] = keys[slot][chunk * Rows + i];
      dot = dot + s[i] * key[i];
    }
    decayed[chunk][column] = dot;
    __syncthreads();
    float predicted = 0;
#pragma unroll
    for (int j = 0; j < Chunks; ++j) predicted += decayed[j][column];
    const float error = __half2float(v[row * Dim + column]) - predicted;
    dot = 0;
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      s[i] = s[i] + key[i] * b * error;
      dot = dot + s[i] * queries[slot][chunk * Rows + i];
    }
    updated[chunk][column] = dot;
    // The next token's key and query go to the buffer nobody reads before the barrier.
    if (threadIdx.x < Dim && t + 1 < steps) {
      int64_t at = ((first + t + 1) * key_heads + key_head) * Dim + threadIdx.x;
      keys[slot ^ 1][threadIdx.x] = __half2float(k[at]);
      queries[slot ^ 1][threadIdx.x] = __half2float(q[at]);
    }
    __syncthreads();
    if (chunk == 0) {
      float value = 0;
#pragma unroll
      for (int j = 0; j < Chunks; ++j) value += updated[j][column];
      out[row * Dim + column] = value * Scale;
    }
  }
#pragma unroll
  for (int i = 0; i < Rows; ++i) state_out[cell + (chunk * Rows + i) * Dim + column] = s[i];
}

void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }

bool cuda_float(DLTensor* t, int bits, int ndim) {
  return t->device.device_type == kDLCUDA && t->dtype.code == kDLFloat && t->dtype.bits == bits &&
         t->dtype.lanes == 1 && t->ndim == ndim && tvm::ffi::IsContiguous(*t);
}

// q, k: (batch, steps, key heads, 128) FP16. v: (batch, steps, value heads, 128) FP16.
// gate, beta: (batch, steps, value heads) FP32. state: (batch, value heads, 128, 128) FP32.
void gdn(DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* gate, DLTensor* beta, DLTensor* state,
         DLTensor* out, DLTensor* state_out) {
  bool ok = cuda_float(q, 16, 4) && cuda_float(k, 16, 4) && cuda_float(v, 16, 4) &&
            cuda_float(gate, 32, 3) && cuda_float(beta, 32, 3) && cuda_float(state, 32, 4) &&
            cuda_float(out, 32, 4) && cuda_float(state_out, 32, 4);
  int64_t batch = 0, steps = 0, key_heads = 0, value_heads = 0;
  if (ok) {
    batch = v->shape[0], steps = v->shape[1], value_heads = v->shape[2], key_heads = q->shape[2];
    ok = batch > 0 && steps > 0 && key_heads > 0 && value_heads % key_heads == 0;
    for (DLTensor* t : {q, k, v, out})
      ok = ok && t->shape[0] == batch && t->shape[1] == steps && t->shape[3] == Dim;
    ok = ok && k->shape[2] == key_heads && out->shape[2] == value_heads;
    for (DLTensor* t : {gate, beta})
      ok = ok && t->shape[0] == batch && t->shape[1] == steps && t->shape[2] == value_heads;
    for (DLTensor* t : {state, state_out})
      ok = ok && t->shape[0] == batch && t->shape[1] == value_heads && t->shape[2] == Dim &&
           t->shape[3] == Dim;
    ok = ok && batch * value_heads <= INT32_MAX && steps <= INT32_MAX;
  }
  if (!ok)
    throw std::runtime_error(
        "Xavier gated delta requires contiguous CUDA tensors: FP16 q/k/v with 128-wide heads, "
        "FP32 gate/beta/state");
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, v->device.device_id));
  gated_delta<<<batch * value_heads, Chunks * Dim, 0, stream>>>(
      static_cast<const half*>(data(q)), static_cast<const half*>(data(k)),
      static_cast<const half*>(data(v)), static_cast<const float*>(data(gate)),
      static_cast<const float*>(data(beta)), static_cast<const float*>(data(state)),
      static_cast<float*>(data(out)), static_cast<float*>(data(state_out)), int(steps),
      int(key_heads), int(value_heads));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_gdn, gdn);
