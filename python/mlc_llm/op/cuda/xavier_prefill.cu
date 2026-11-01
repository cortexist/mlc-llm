// Prefill-only callback adapter. Model/cache layout and <=4-row calls are retained.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <stdexcept>

#include "xavier_prefill_kernel.cuh"
namespace {
void* tensor_data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }
tvm::ffi::Function wrap_prefill(tvm::ffi::Function small, tvm::ffi::Function large, int64_t clear,
                                int64_t output_index) {
  return tvm::ffi::Function::FromPacked(
      [small, large, clear, output_index](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        DLTensor* q = args[0].cast<DLTensor*>();
        if (q->shape[0] <= 4) {
          small.CallPacked(args, result);
          return;
        }
        if (clear) {
          DLTensor* out = args[output_index].cast<DLTensor*>();
          auto status = cudaMemsetAsync(
              tensor_data(out), 0, out->shape[0] * out->shape[1] * out->shape[2] * 2,
              static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id)));
          if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        large.CallPacked(args, result);
      });
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrap_prefill, wrap_prefill);

namespace {
template <bool Paged, bool Local>
void launch_prefill(DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki, DLTensor* pv,
                    DLTensor* length, DLTensor* output, DLTensor* lse, int64_t causal,
                    int64_t rotary, double scale) {
  if (q->device.device_type != kDLCUDA || q->dtype.code != kDLFloat || q->dtype.bits != 16 ||
      q->shape[2] != 512 || q->shape[1] != 8 || rotary != 0 ||
      (Paged ? k->shape[2] != 2 || k->shape[4] != 512 : k->shape[1] != 2 || k->shape[2] != 512))
    throw std::runtime_error("WMMA prefill requires the padded FP16 Gemma4 E4B cache");
  int batch = qi->shape[0] - 1;
  dim3 grid((q->shape[0] + 15) / 16 + batch - 1, q->shape[1]);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  xavier_prefill::attention<Local ? 256 : 512, Paged, Local><<<grid, 128, 0, stream>>>(
      static_cast<half*>(tensor_data(q)), static_cast<int*>(tensor_data(qi)),
      static_cast<half*>(tensor_data(k)), v ? static_cast<half*>(tensor_data(v)) : nullptr,
      static_cast<int*>(tensor_data(ki)), pv ? static_cast<int*>(tensor_data(pv)) : nullptr,
      length ? static_cast<int*>(tensor_data(length)) : nullptr,
      static_cast<half*>(tensor_data(output)), static_cast<float*>(tensor_data(lse)), batch, 8, 2,
      Paged ? k->shape[3] : 16, causal, scale);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <bool Local>
void paged_prefill(DLTensor* q, DLTensor* qi, DLTensor* pages, DLTensor* pi, DLTensor* pv,
                   DLTensor* length, DLTensor* kpos, DLTensor* qpos, DLTensor* output,
                   DLTensor* lse, int64_t causal, int64_t rotary, double rope_scale, double theta,
                   double scale) {
  launch_prefill<true, Local>(q, qi, pages, nullptr, pi, pv, length, output, lse, causal, rotary,
                              scale);
}
void ragged_prefill(DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki,
                    DLTensor* qpos, DLTensor* kpos, DLTensor* output, DLTensor* lse, int64_t causal,
                    int64_t rotary, double rope_scale, double theta, double scale) {
  launch_prefill<false, false>(q, qi, k, v, ki, nullptr, nullptr, output, lse, causal, rotary,
                               scale);
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged0, paged_prefill<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged1, paged_prefill<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged, ragged_prefill);
