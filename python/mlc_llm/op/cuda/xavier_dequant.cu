// q4f16_1 NK expansion: one packed word -> one aligned 16-byte output store.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <cstdint>
#include <stdexcept>

namespace {
__global__ void expand_q4(const uint32_t* weights, const half* scales, uint4* output,
                         int64_t words) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= words) return;
  uint32_t w = weights[i];
  half scale = scales[i / 4];
  half2 s = __halves2half2(scale, scale);
  uint32_t result[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    half2 v = __halves2half2(__int2half_rn(int((w >> (8*j)) & 15) - 7),
                           __int2half_rn(int((w >> (8*j+4)) & 15) - 7));
    half2 r = __hmul2(v, s);
    result[j] = uint32_t(__half_as_ushort(r.x)) | (uint32_t(__half_as_ushort(r.y)) << 16);
  }
  output[i] = make_uint4(result[0], result[1], result[2], result[3]);
}
void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }
void dequant(DLTensor* w, DLTensor* s, DLTensor* out) {
  if (w->ndim != 2 || s->ndim != 2 || out->ndim != 2 ||
      w->device.device_type != kDLCUDA || s->device.device_type != kDLCUDA ||
      out->device.device_type != kDLCUDA || w->device.device_id != s->device.device_id ||
      w->device.device_id != out->device.device_id || w->dtype.code != kDLUInt ||
      w->dtype.bits != 32 || s->dtype.code != kDLFloat || s->dtype.bits != 16 ||
      out->dtype.code != kDLFloat || out->dtype.bits != 16 || w->shape[1] % 4 ||
      w->dtype.lanes != 1 || s->dtype.lanes != 1 || out->dtype.lanes != 1 ||
      w->shape[0] <= 0 || w->shape[1] <= 0 ||
      s->shape[0] != w->shape[0] || s->shape[1] * 4 != w->shape[1] ||
      out->shape[0] != w->shape[0] || out->shape[1] != w->shape[1] * 8 ||
      !tvm::ffi::IsContiguous(*w) || !tvm::ffi::IsContiguous(*s) ||
      !tvm::ffi::IsContiguous(*out) || uintptr_t(data(out)) % 16)
    throw std::runtime_error("Xavier dequant requires contiguous CUDA q4f16_1 NK matrices");
  int64_t words = w->shape[0] * w->shape[1];
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, w->device.device_id));
  expand_q4<<<(words + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint32_t*>(data(w)), static_cast<const half*>(data(s)),
      static_cast<uint4*>(data(out)), words);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xavier_dequant, dequant);
