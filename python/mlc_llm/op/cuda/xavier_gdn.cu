// The gated delta rule of the Qwen3.5 linear-attention layers on sm_72.
// One block per value head. A thread keeps sixteen rows of one state column in registers
// for the whole chunk, so a token costs two barriers instead of four passes over memory.
// Products and updates are those of the TIR kernel; the 128 terms of a dot product are
// added as eight partial sums of sixteen instead of in one sequence.
//
// xavier_gdn runs a chunk and returns its outputs and the new state. xavier_gdn_commit
// repeats the state updates of the first rows of a chunk from the values the model held
// back (speculative verification: the state advances only by the accepted rows).
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

// The tokens of one sequence as the layer computes them.
struct Chunk {
  const half *q, *k, *v;     // (steps, key heads, 128), same, (steps, value heads, 128)
  const float *gate, *beta;  // (steps, value heads)
  int key_heads, value_heads;
  __device__ float query(int t, int head, int i) const {
    return __half2float(q[(int64_t(t) * key_heads + head) * Dim + i]);
  }
  __device__ float key(int t, int head, int i) const {
    return __half2float(k[(int64_t(t) * key_heads + head) * Dim + i]);
  }
  __device__ float value(int t, int head, int i) const {
    return __half2float(v[(int64_t(t) * value_heads + head) * Dim + i]);
  }
  __device__ float decay(int t, int head) const { return gate[int64_t(t) * value_heads + head]; }
  __device__ float rate(int t, int head) const { return beta[int64_t(t) * value_heads + head]; }
};

// The same values held back in FP32, one row per token:
// convolution input | k | v | gate | beta.
struct Held {
  const float* row;
  int64_t width;
  int k, v, gate, beta;  // offsets in a row
  __device__ float query(int, int, int) const { return 0; }
  __device__ float key(int t, int head, int i) const { return row[t * width + k + head * Dim + i]; }
  __device__ float value(int t, int head, int i) const {
    return row[t * width + v + head * Dim + i];
  }
  __device__ float decay(int t, int head) const { return row[t * width + gate + head]; }
  __device__ float rate(int t, int head) const { return row[t * width + beta + head]; }
};

// `out` may be null: only the state is wanted.
template <typename Input>
__global__ void gated_delta(Input in, const float* state_in, float* out, float* state_out,
                            int steps, int key_heads, int value_heads) {
  __shared__ float keys[2][Dim], queries[2][Dim], decayed[Chunks][Dim], updated[Chunks][Dim];
  const int column = threadIdx.x % Dim, chunk = threadIdx.x / Dim;
  const int head = blockIdx.x, key_head = head / (value_heads / key_heads);
  const int64_t cell = int64_t(head) * Dim * Dim;
  float s[Rows], key[Rows];
#pragma unroll
  for (int i = 0; i < Rows; ++i) s[i] = state_in[cell + (chunk * Rows + i) * Dim + column];
  if (threadIdx.x < Dim) {
    keys[0][threadIdx.x] = in.key(0, key_head, threadIdx.x);
    queries[0][threadIdx.x] = in.query(0, key_head, threadIdx.x);
  }
  __syncthreads();
  for (int t = 0; t < steps; ++t) {
    const int slot = t & 1;
    const float g = in.decay(t, head), b = in.rate(t, head);
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
    const float error = in.value(t, head, column) - predicted;
    dot = 0;
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      s[i] = s[i] + key[i] * b * error;
      dot = dot + s[i] * queries[slot][chunk * Rows + i];
    }
    updated[chunk][column] = dot;
    // The next token's key and query go to the buffer nobody reads before the barrier.
    if (threadIdx.x < Dim && t + 1 < steps) {
      keys[slot ^ 1][threadIdx.x] = in.key(t + 1, key_head, threadIdx.x);
      queries[slot ^ 1][threadIdx.x] = in.query(t + 1, key_head, threadIdx.x);
    }
    __syncthreads();
    if (out && chunk == 0) {
      float value = 0;
#pragma unroll
      for (int j = 0; j < Chunks; ++j) value += updated[j][column];
      out[(int64_t(t) * value_heads + head) * Dim + column] = value * Scale;
    }
  }
#pragma unroll
  for (int i = 0; i < Rows; ++i) state_out[cell + (chunk * Rows + i) * Dim + column] = s[i];
}

// The convolution keeps its last inputs: those before the chunk, then the committed rows.
__global__ void conv_commit(const float* held, int64_t width, const half* before, half* after,
                            int rows, int taps, int channels) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= taps * channels) return;
  int at = rows + index / channels, channel = index % channels;
  after[index] = at < taps ? before[at * channels + channel]
                           : __float2half(held[(at - taps) * width + channel]);
}

void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }

bool cuda_float(DLTensor* t, int bits, int ndim) {
  return t->device.device_type == kDLCUDA && t->dtype.code == kDLFloat && t->dtype.bits == bits &&
         t->dtype.lanes == 1 && t->ndim == ndim && tvm::ffi::IsContiguous(*t);
}

cudaStream_t stream_of(DLTensor* t) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, t->device.device_id));
}

void check_launch() {
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
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
    ok = ok && value_heads <= INT32_MAX && steps <= INT32_MAX;
  }
  if (!ok)
    throw std::runtime_error(
        "Xavier gated delta requires contiguous CUDA tensors: FP16 q/k/v with 128-wide heads, "
        "FP32 gate/beta/state");
  for (int64_t b = 0; b < batch; ++b) {
    Chunk in{static_cast<const half*>(data(q)) + b * steps * key_heads * Dim,
             static_cast<const half*>(data(k)) + b * steps * key_heads * Dim,
             static_cast<const half*>(data(v)) + b * steps * value_heads * Dim,
             static_cast<const float*>(data(gate)) + b * steps * value_heads,
             static_cast<const float*>(data(beta)) + b * steps * value_heads,
             int(key_heads),
             int(value_heads)};
    gated_delta<Chunk><<<value_heads, Chunks * Dim, 0, stream_of(v)>>>(
        in, static_cast<const float*>(data(state)) + b * value_heads * Dim * Dim,
        static_cast<float*>(data(out)) + b * steps * value_heads * Dim,
        static_cast<float*>(data(state_out)) + b * value_heads * Dim * Dim, int(steps),
        int(key_heads), int(value_heads));
    check_launch();
  }
}

// held: (layers, steps, width) FP32, one sequence. state: (1, value heads, 128, 128) FP32.
// conv: (1, taps, channels) FP16 with channels = (2 * key heads + value heads) * 128.
void commit(DLTensor* held, int64_t layer, int64_t rows, DLTensor* state, DLTensor* conv,
            DLTensor* state_out, DLTensor* conv_out) {
  bool ok = cuda_float(held, 32, 3) && cuda_float(state, 32, 4) && cuda_float(state_out, 32, 4) &&
            cuda_float(conv, 16, 3) && cuda_float(conv_out, 16, 3);
  int64_t value_heads = 0, key_heads = 0, channels = 0, taps = 0;
  if (ok) {
    value_heads = state->shape[1], taps = conv->shape[1], channels = conv->shape[2];
    key_heads = (channels - value_heads * Dim) / (2 * Dim);
    for (DLTensor* t : {state, state_out})
      ok = ok && t->shape[0] == 1 && t->shape[1] == value_heads && t->shape[2] == Dim &&
           t->shape[3] == Dim;
    for (DLTensor* t : {conv, conv_out})
      ok = ok && t->shape[0] == 1 && t->shape[1] == taps && t->shape[2] == channels;
    ok = ok && key_heads > 0 && value_heads % key_heads == 0 &&
         channels == (2 * key_heads + value_heads) * Dim &&
         held->shape[2] == channels + (key_heads + value_heads) * Dim + 2 * value_heads &&
         layer >= 0 && layer < held->shape[0] && rows >= 1 && rows <= held->shape[1] &&
         taps * channels <= INT32_MAX;
  }
  if (!ok)
    throw std::runtime_error(
        "Xavier gated delta commit requires the held rows of one sequence, a layer and a row "
        "count inside them, an FP32 state and an FP16 convolution state");
  const int64_t width = held->shape[2];
  const float* row = static_cast<const float*>(data(held)) + layer * held->shape[1] * width;
  const int k = int(channels), v = k + int(key_heads) * Dim, gate = v + int(value_heads) * Dim;
  Held in{row, width, k, v, gate, gate + int(value_heads)};
  gated_delta<Held><<<value_heads, Chunks * Dim, 0, stream_of(held)>>>(
      in, static_cast<const float*>(data(state)), nullptr, static_cast<float*>(data(state_out)),
      int(rows), int(key_heads), int(value_heads));
  check_launch();
  conv_commit<<<(taps * channels + 255) / 256, 256, 0, stream_of(held)>>>(
      row, width, static_cast<const half*>(data(conv)), static_cast<half*>(data(conv_out)),
      int(rows), int(taps), int(channels));
  check_launch();
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_gdn, gdn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_gdn_commit, commit);
