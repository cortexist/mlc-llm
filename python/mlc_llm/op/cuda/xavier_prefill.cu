// Attention callback adapters. The model and the cache layout are retained.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "xavier_paged_kernel.cuh"
#include "xavier_prefill_kernel.cuh"
namespace {
// The cache callback omits layer identity. Carry it through the synchronous
// host-side attention entrypoint; GPU launches remain asynchronous.
thread_local int ragged_width = 512;
struct WidthScope {
  int previous;
  explicit WidthScope(int width) : previous(ragged_width) { ragged_width = width; }
  ~WidthScope() { ragged_width = previous; }
};
tvm::ffi::Function wrap_layer_width(tvm::ffi::Function original, std::string layout) {
  for (char c : layout)
    if (c != 'l' && c != 'g') throw std::runtime_error("Invalid attention layer layout");
  return tvm::ffi::Function::FromPacked(
      [original, layout](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        int64_t layer = args[1].cast<int64_t>();
        WidthScope scope(layer >= 0 && layer < static_cast<int64_t>(layout.size()) &&
                                 layout[layer] == 'l' ? 256 : 512);
        original.CallPacked(args, result);
      });
}
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrap_layer_width, wrap_layer_width);

namespace {
template <bool Paged, bool Local, int D = (Local ? 256 : 512)>
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
  xavier_prefill::attention<D, Paged, Local><<<grid, 128, 0, stream>>>(
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
void ragged_layer_width(DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki,
                        DLTensor* qpos, DLTensor* kpos, DLTensor* output, DLTensor* lse,
                        int64_t causal, int64_t rotary, double rope_scale, double theta,
                        double scale) {
  if (ragged_width == 256)
    launch_prefill<false, false, 256>(q, qi, k, v, ki, nullptr, nullptr, output, lse,
                                     causal, rotary, scale);
  else
    ragged_prefill(q, qi, k, v, ki, qpos, kpos, output, lse, causal, rotary,
                  rope_scale, theta, scale);
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged0, paged_prefill<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged1, paged_prefill<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged, ragged_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_layer_width, ragged_layer_width);

namespace {
// Scratch of the paged kernel: a 512-row chunk in groups of eight rows, or 32 key segments.
constexpr int MaxBlocks = 128;
float* paged_scratch() {
  static float* memory = nullptr;
  if (!memory && cudaMalloc(&memory, sizeof(float) * MaxBlocks * 32 * (512 + 2)) != cudaSuccess)
    throw std::runtime_error("Cannot allocate the paged attention scratch");
  return memory;
}
template <int T, bool Local>
void launch_paged(const xavier_paged::Job& job, int groups, half* output, float* lse,
                  cudaStream_t stream) {
  dim3 grid(groups * job.segments, 2);
  xavier_paged::attend<T, Local><<<grid, 256, 0, stream>>>(job);
  xavier_paged::merge<T, Local><<<dim3(job.rows, 8), 256, 0, stream>>>(job, output, lse, grid.x);
}
// False when the call is not one sequence of the padded FP16 E4B cache.
bool paged(bool local, DLTensor* q, DLTensor* pages, DLTensor* page_ptr, DLTensor* page_ids,
           DLTensor* length, DLTensor* output, DLTensor* lse, int64_t causal, int64_t rotary,
           double scale) {
  int rows = q->shape[0], count = page_ids->shape[0];
  if (q->device.device_type != kDLCUDA || q->dtype.code != kDLFloat || q->dtype.bits != 16 ||
      q->shape[1] != 8 || q->shape[2] != 512 || pages->shape[2] != 2 || pages->shape[3] != 16 ||
      pages->shape[4] != 512 || page_ptr->shape[0] != 2 || length->ndim != (local ? 2 : 1) ||
      rotary != 0 || rows < 1 || rows > 512 || count < 1)
    return false;
  int tiles = rows > 4 ? 2 : 1, groups = (rows + 4 * tiles - 1) / (4 * tiles);
  // Few rows leave the GPU idle unless the keys are split as well.
  int segments = std::max(1, std::min(32 / groups, (count + 15) / 16));
  int segment_pages = ((count + segments - 1) / segments + 7) / 8 * 8;
  segments = (count + segment_pages - 1) / segment_pages;
  float* scratch = paged_scratch();
  xavier_paged::Job job{static_cast<half*>(tensor_data(q)),
                        static_cast<half*>(tensor_data(pages)),
                        static_cast<int*>(tensor_data(page_ids)),
                        static_cast<int*>(tensor_data(length)),
                        scratch,
                        scratch + MaxBlocks * 32 * 512,
                        rows,
                        count,
                        segment_pages,
                        segments,
                        static_cast<int>(causal),
                        static_cast<float>(scale)};
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  auto* o = static_cast<half*>(tensor_data(output));
  auto* l = static_cast<float*>(tensor_data(lse));
  if (tiles == 1 && local) launch_paged<1, true>(job, groups, o, l, stream);
  if (tiles == 1 && !local) launch_paged<1, false>(job, groups, o, l, stream);
  if (tiles == 2 && local) launch_paged<2, true>(job, groups, o, l, stream);
  if (tiles == 2 && !local) launch_paged<2, false>(job, groups, o, l, stream);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  return true;
}
// A decode call has no query index pointer and no causal flag.
tvm::ffi::Function wrap_paged(tvm::ffi::Function fallback, int64_t local, int64_t decode) {
  return tvm::ffi::Function::FromPacked(
      [fallback, local, decode](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        int at = decode ? 1 : 2;
        auto tensor = [&](int i) { return args[i].cast<DLTensor*>(); };
        if (!paged(local, tensor(0), tensor(at), tensor(at + 1), tensor(at + 2), tensor(at + 3),
                   tensor(at + 6), tensor(at + 7), decode ? 0 : args[10].cast<int64_t>(),
                   args[decode ? 9 : 11].cast<int64_t>(), args[decode ? 12 : 14].cast<double>()))
          fallback.CallPacked(args, result);
      });
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrap_paged, wrap_paged);
