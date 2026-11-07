// Stable top-32 selection for the E4B assistant's 2048 centroid scores.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <stdexcept>

namespace {
__device__ bool better(float a, int ai, float b, int bi) {
  return a > b || (a == b && ai < bi);
}

__global__ void select_centroids(const half* scores, int* output) {
  // Each thread holds eight scores; each warp and then the block reduce their
  // best remaining candidate. Lower IDs win ties, matching gating_topk.
  float values[8];
  const int t = threadIdx.x, lane = t & 31, warp = t / 32;
#pragma unroll
  for (int i = 0; i < 8; ++i) values[i] = __half2float(scores[t + 256 * i]);
  __shared__ float warp_values[8];
  __shared__ int warp_ids[8];
  __shared__ int selected;
  for (int rank = 0; rank < 32; ++rank) {
    float best = -CUDART_INF_F;
    int id = 2147483647;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int candidate = t + 256 * i;
      if (better(values[i], candidate, best, id)) {
        best = values[i];
        id = candidate;
      }
    }
    for (int offset = 16; offset; offset /= 2) {
      float other = __shfl_down_sync(0xffffffff, best, offset);
      int other_id = __shfl_down_sync(0xffffffff, id, offset);
      if (lane + offset < 32 && better(other, other_id, best, id)) {
        best = other;
        id = other_id;
      }
    }
    if (lane == 0) { warp_values[warp] = best; warp_ids[warp] = id; }
    __syncthreads();
    if (warp == 0) {
      best = lane < 8 ? warp_values[lane] : -CUDART_INF_F;
      id = lane < 8 ? warp_ids[lane] : 2147483647;
      for (int offset = 16; offset; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, best, offset);
        int other_id = __shfl_down_sync(0xffffffff, id, offset);
        if (lane + offset < 32 && better(other, other_id, best, id)) {
          best = other;
          id = other_id;
        }
      }
      if (lane == 0) { selected = id; output[rank] = id; }
    }
    __syncthreads();
    // All model scores are finite. Retiring a winner cannot hide a valid score.
#pragma unroll
    for (int i = 0; i < 8; ++i) if (t + 256 * i == selected) values[i] = -CUDART_INF_F;
    __syncthreads();
  }
}

void topk(DLTensor* scores, DLTensor* output) {
  if (scores->device.device_type != kDLCUDA || scores->dtype.code != kDLFloat ||
      scores->dtype.bits != 16 || scores->ndim != 2 || scores->shape[0] != 1 ||
      scores->shape[1] != 2048 || output->dtype.code != kDLInt ||
      output->dtype.bits != 32 || output->ndim != 2 || output->shape[0] != 1 ||
      output->shape[1] != 32)
    throw std::runtime_error("E4B centroid selection requires FP16 [1,2048] -> int32 [1,32]");
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, scores->device.device_id));
  select_centroids<<<1, 256, 0, stream>>>(
      reinterpret_cast<const half*>(static_cast<char*>(scores->data) + scores->byte_offset),
      reinterpret_cast<int*>(static_cast<char*>(output->data) + output->byte_offset));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma4_mtp_topk, topk);
