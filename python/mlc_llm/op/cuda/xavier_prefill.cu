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
// host-side attention entrypoint; GPU launches remain asynchronous. The layer
// means something to the callback of its own cache only (ragged_by_layer), so
// models with different layouts can share a process.
thread_local int64_t current_layer = -1;
struct LayerScope {
  int64_t previous;
  explicit LayerScope(int64_t layer) : previous(current_layer) { current_layer = layer; }
  ~LayerScope() { current_layer = previous; }
};
tvm::ffi::Function wrap_layer(tvm::ffi::Function original) {
  return tvm::ffi::Function::FromPacked(
      [original](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        LayerScope scope(args[1].cast<int64_t>());
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrap_layer, wrap_layer);

namespace {
// S: the stored width of a head; D: the part of it in use.
template <bool Paged, bool Local, int D = (Local ? 256 : 512), int S = 512>
void launch_prefill(DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki, DLTensor* pv,
                    DLTensor* length, DLTensor* output, DLTensor* lse, int64_t causal,
                    int64_t rotary, double scale) {
  int heads = q->shape[1], kv_heads = k->shape[Paged ? 2 : 1];
  // The layouts it is checked with. Padded to 512: the E4B, or the current chunk of
  // the Unified 12B. 256 wide: the current chunk of the Qwen3.8 27B or 2B. Their
  // cached pages take xavier_paged.
  bool e4b = heads == 8 && kv_heads == 2, unified = heads == 16 && kv_heads == 8;
  bool qwen = (heads == 24 && kv_heads == 4) || (heads == 8 && kv_heads == 2);
  bool known = S == 512 ? e4b || (!Paged && unified) : !Paged && qwen;
  if (q->device.device_type != kDLCUDA || q->dtype.code != kDLFloat || q->dtype.bits != 16 ||
      q->shape[2] != S || !known || rotary != 0 || k->shape[Paged ? 4 : 2] != S)
    throw std::runtime_error("WMMA prefill is not built for this FP16 cache layout");
  int batch = qi->shape[0] - 1;
  dim3 grid((q->shape[0] + 15) / 16 + batch - 1, q->shape[1]);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  xavier_prefill::attention<D, Paged, Local, S><<<grid, 128, 0, stream>>>(
      static_cast<half*>(tensor_data(q)), static_cast<int*>(tensor_data(qi)),
      static_cast<half*>(tensor_data(k)), v ? static_cast<half*>(tensor_data(v)) : nullptr,
      static_cast<int*>(tensor_data(ki)), pv ? static_cast<int*>(tensor_data(pv)) : nullptr,
      length ? static_cast<int*>(tensor_data(length)) : nullptr,
      static_cast<half*>(tensor_data(output)), static_cast<float*>(tensor_data(lse)), batch, heads,
      kv_heads, Paged ? k->shape[3] : 16, causal, scale);
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
// The current chunk of a cache whose heads are S wide.
template <int S>
void ragged_prefill(DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki,
                    DLTensor* qpos, DLTensor* kpos, DLTensor* output, DLTensor* lse, int64_t causal,
                    int64_t rotary, double rope_scale, double theta, double scale) {
  launch_prefill<false, false, S, S>(q, qi, k, v, ki, nullptr, nullptr, output, lse, causal,
                                     rotary, scale);
}
// The current chunk of a cache whose physical layers are local ('l': 256 of the
// 512 stored dimensions are used) or global ('g').
tvm::ffi::Function ragged_by_layer(std::string layout) {
  for (char c : layout)
    if (c != 'l' && c != 'g') throw std::runtime_error("Invalid attention layer layout");
  return tvm::ffi::Function::FromTyped(
      [layout](DLTensor* q, DLTensor* qi, DLTensor* k, DLTensor* v, DLTensor* ki, DLTensor* qpos,
               DLTensor* kpos, DLTensor* output, DLTensor* lse, int64_t causal, int64_t rotary,
               double rope_scale, double theta, double scale) {
        int64_t layer = current_layer;
        if (layer >= 0 && layer < static_cast<int64_t>(layout.size()) && layout[layer] == 'l')
          launch_prefill<false, false, 256>(q, qi, k, v, ki, nullptr, nullptr, output, lse, causal,
                                           rotary, scale);
        else
          ragged_prefill<512>(q, qi, k, v, ki, qpos, kpos, output, lse, causal, rotary,
                              rope_scale, theta, scale);
      });
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged0, paged_prefill<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged1, paged_prefill<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged, ragged_prefill<512>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged256, ragged_prefill<256>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_by_layer, ragged_by_layer);

namespace {
// Scratch of the paged kernel: a 512-row chunk in groups of eight rows, or 32 key segments.
constexpr int MaxBlocks = 128;
float* paged_scratch() {
  static float* memory = nullptr;
  if (!memory && cudaMalloc(&memory, sizeof(float) * MaxBlocks * 32 * (512 + 2)) != cudaSuccess)
    throw std::runtime_error("Cannot allocate the paged attention scratch");
  return memory;
}
template <int T, typename L>
void launch_paged(const xavier_paged::Job& job, int groups, cudaStream_t stream) {
  dim3 grid(groups * job.segments, L::Heads / L::Group);
  xavier_paged::attend<T, L><<<grid, 256, 0, stream>>>(job);
  if (job.segments > 1)
    xavier_paged::merge<T, L><<<dim3(job.rows, L::Heads), 256, 0, stream>>>(job, grid.x);
}
using Paged = bool(DLTensor*, DLTensor*, DLTensor*, DLTensor*, DLTensor*, DLTensor*, DLTensor*,
                   int64_t, int64_t, double);
// False when the call is not one sequence of the padded FP16 cache of layer kind L.
template <typename L>
bool paged(DLTensor* q, DLTensor* pages, DLTensor* page_ptr, DLTensor* page_ids, DLTensor* length,
           DLTensor* output, DLTensor* lse, int64_t causal, int64_t rotary, double scale) {
  int rows = q->shape[0], count = page_ids->shape[0];
  if (q->device.device_type != kDLCUDA || q->dtype.code != kDLFloat || q->dtype.bits != 16 ||
      q->shape[1] != L::Heads || q->shape[2] != L::Stride || pages->shape[2] != L::KV ||
      pages->shape[3] != 16 || pages->shape[4] != L::Stride || page_ptr->shape[0] != 2 ||
      length->ndim != (L::Window ? 2 : 1) || rotary != 0 || rows < 1 || rows > 512 || count < 1)
    return false;
  // A block holds 16 queries per tile, which share K and V: the rows that fit.
  int tiles = rows > 4 ? 2 : 1, block_rows = 16 * tiles / L::Group;
  int groups = (rows + block_rows - 1) / block_rows;
  // Few rows leave the GPU idle unless the keys are split as well.
  int segments = std::max(1, std::min(64 * L::Group / L::Heads / groups, (count + 15) / 16));
  int segment_pages = ((count + segments - 1) / segments + 7) / 8 * 8;
  segments = (count + segment_pages - 1) / segment_pages;
  float* scratch = paged_scratch();
  xavier_paged::Job job{static_cast<half*>(tensor_data(q)),
                        static_cast<half*>(tensor_data(pages)),
                        static_cast<int*>(tensor_data(page_ids)),
                        static_cast<int*>(tensor_data(length)),
                        static_cast<half*>(tensor_data(output)),
                        static_cast<float*>(tensor_data(lse)),
                        scratch,
                        scratch + MaxBlocks * 32 * 512,
                        rows,
                        count,
                        segment_pages,
                        segments,
                        static_cast<int>(causal),
                        static_cast<float>(scale)};
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q->device.device_id));
  if (tiles == 1)
    launch_paged<1, L>(job, groups, stream);
  else
    launch_paged<2, L>(job, groups, stream);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  return true;
}
// The launcher of the layer kind with these numbers, among those the kernel is built for.
template <typename... L>
Paged* find_paged(int64_t heads, int64_t kv_heads, int64_t group, int64_t window,
                  int64_t stride) {
  Paged* found = nullptr;
  auto offer = [&](Paged* launcher, int h, int k, int g, int w, int s) {
    if (heads == h && kv_heads == k && group == g && window == w && stride == s)
      found = launcher;
  };
  (offer(paged<L>, L::Heads, L::KV, L::Group, L::Window, L::Stride), ...);
  return found;
}
// The numbers are those of xavier_paged::Layer. A decode call has no query index
// pointer and no causal flag.
tvm::ffi::Function wrap_paged(tvm::ffi::Function fallback, int64_t decode, int64_t heads,
                              int64_t kv_heads, int64_t group, int64_t window, int64_t stride) {
  using namespace xavier_paged;
  Paged* run = find_paged<E4BLocal, E4BGlobal, UnifiedLocal, UnifiedGlobal, Qwen27B, Qwen2B>(
      heads, kv_heads, group, window, stride);
  if (!run) throw std::runtime_error("The paged attention kernel is not built for this layer");
  return tvm::ffi::Function::FromPacked(
      [fallback, run, decode](tvm::ffi::PackedArgs args, tvm::ffi::Any* result) {
        int at = decode ? 1 : 2;
        auto tensor = [&](int i) { return args[i].cast<DLTensor*>(); };
        if (!run(tensor(0), tensor(at), tensor(at + 1), tensor(at + 2), tensor(at + 3),
                 tensor(at + 6), tensor(at + 7), decode ? 0 : args[10].cast<int64_t>(),
                 args[decode ? 9 : 11].cast<int64_t>(), args[decode ? 12 : 14].cast<double>()))
          fallback.CallPacked(args, result);
      });
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wrap_paged, wrap_paged);
