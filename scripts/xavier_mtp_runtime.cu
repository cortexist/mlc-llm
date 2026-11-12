// GPU token selection for the explicit-state greedy MTP benchmark controller.
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
__device__ void reduce(float& value, int& index) {
  __shared__ float values[8];
  __shared__ int indices[8];
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  for (int delta = 16; delta; delta /= 2) {
    float other = __shfl_down_sync(0xffffffff, value, delta);
    int other_index = __shfl_down_sync(0xffffffff, index, delta);
    if (lane + delta < 32 && (other > value || (other == value && other_index < index))) {
      value = other; index = other_index;
    }
  }
  if (lane == 0) { values[warp] = value; indices[warp] = index; }
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? values[lane] : -CUDART_INF_F;
    index = lane < 8 ? indices[lane] : 2147483647;
    for (int delta = 16; delta; delta /= 2) {
      float other = __shfl_down_sync(0xffffffff, value, delta);
      int other_index = __shfl_down_sync(0xffffffff, index, delta);
      if (lane + delta < 32 && (other > value || (other == value && other_index < index))) {
        value = other; index = other_index;
      }
    }
  }
}

__global__ void partial_argmax(const float* logits, float* values, int* indices, int width) {
  int row = blockIdx.y, part = blockIdx.x;
  float value = -CUDART_INF_F;
  int index = 2147483647;
  for (int i = part * 1024 + threadIdx.x; i < min(width, (part+1)*1024); i += 256) {
    float x = logits[row*width+i];
    if (x > value || (x == value && i < index)) { value = x; index = i; }
  }
  reduce(value,index);
  if (threadIdx.x == 0) { values[row*gridDim.x+part] = value; indices[row*gridDim.x+part] = index; }
}

__global__ void finish_argmax(const float* values, const int* indices, const int* ids,
                             int* result, int parts) {
  int row = blockIdx.x, t = threadIdx.x;
  float value = t < parts ? values[row*parts+t] : -CUDART_INF_F;
  int index = t < parts ? indices[row*parts+t] : 2147483647;
  reduce(value,index);
  if (t == 0) result[row] = ids ? ids[index] : index;
}
}

extern "C" int mtp_argmax(const float* logits, float* values, int* indices,
                          const int* ids, int* result, int rows, int width, void* stream) {
  int parts = (width+1023)/1024;
  if (rows < 1 || rows > 4 || parts > 256 || width < 1) return int(cudaErrorInvalidValue);
  partial_argmax<<<dim3(parts,rows),256,0,(cudaStream_t)stream>>>(logits,values,indices,width);
  finish_argmax<<<rows,256,0,(cudaStream_t)stream>>>(values,indices,ids,result,parts);
  return int(cudaGetLastError());
}
