// Diagnostic only: FP16 inputs/output, FP32 compute type. Report split-K reduction.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace {
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulHeuristicResult_t candidates[32];
  int count = 0;
  ~Plan() {
    if (op) cublasLtMatmulDescDestroy(op);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
  }
};
struct Context {
  cublasLtHandle_t lt = nullptr;
  cublasHandle_t legacy = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  void* workspace = nullptr;
  size_t bytes = 33554432;
  std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;
  Context() {
    try {
      check(cublasLtCreate(&lt));
      check(cublasCreate(&legacy));
      check(cublasLtMatmulPreferenceCreate(&preference));
      if (cudaMalloc(&workspace, bytes) != cudaSuccess)
        throw std::runtime_error("GEMM workspace allocation failed");
      check(cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    } catch (...) {
      release();
      throw;
    }
  }
  void release() {
    plans.clear();
    if (workspace) cudaFree(workspace);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (legacy) cublasDestroy(legacy);
    if (lt) cublasLtDestroy(lt);
  }
  ~Context() { release(); }
  Plan& get(int m, int n, int k) {
    auto key = std::make_tuple(m, n, k);
    auto found = plans.find(key);
    if (found != plans.end()) return *found->second;
    auto ptr = std::make_unique<Plan>();
    auto& p = *ptr;
    check(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &trans, sizeof(trans)));
    check(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, k, n, k));
    check(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, k, m, k));
    check(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16F, n, m, n));
    check(cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.a, p.b, p.c, p.c,
                                       preference, 32, p.candidates, &p.count));
    if (!p.count) throw std::runtime_error("No GEMM candidates");
    plans.emplace(key, std::move(ptr));
    return p;
  }
};
Context& get_context() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device != 0)
    throw std::runtime_error("This diagnostic requires CUDA device zero");
  static thread_local Context context;
  return context;
}
int gemm_config(int m, int n, int k, int index, int attribute) {
  auto& p = get_context().get(m, n, k);
  if (index < 0 || index >= p.count) throw std::runtime_error("Invalid GEMM candidate");
  int value = 0;
  size_t written = 0;
  check(cublasLtMatmulAlgoConfigGetAttribute(
      &p.candidates[index].algo, static_cast<cublasLtMatmulAlgoConfigAttributes_t>(attribute),
      &value, sizeof(value), &written));
  return value;
}
void* data(DLTensor* t) { return static_cast<char*>(t->data) + t->byte_offset; }
bool matrix(DLTensor* t) {
  return t->ndim == 2 && t->shape[0] > 0 && t->shape[1] > 0 &&
         t->device.device_type == kDLCUDA && t->device.device_id == 0 &&
         t->dtype.code == kDLFloat && t->dtype.bits == 16 && t->dtype.lanes == 1 &&
         tvm::ffi::IsContiguous(*t);
}
int gemm_probe(DLTensor* x, DLTensor* w, DLTensor* y, int index) {
  if (!matrix(x) || !matrix(w) || !matrix(y) || w->shape[1] != x->shape[1] ||
      y->shape[0] != x->shape[0] || y->shape[1] != w->shape[0])
    throw std::runtime_error("Contiguous CUDA FP16 X[M,K], W[N,K], Y[M,N] required");
  int k = x->shape[1], n = w->shape[0], m = x->shape[0];
  auto& context = get_context();
  auto& p = context.get(m, n, k);
  if (index < 0) return p.count;
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, 0));
  float alpha = 1, beta = 0;
  if (index == 32) {
    check(cublasSetStream(context.legacy, stream));
    check(cublasGemmEx(context.legacy, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                      &alpha, data(w), CUDA_R_16F, k, data(x), CUDA_R_16F, k,
                      &beta, data(y), CUDA_R_16F, n,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    if (index >= p.count) throw std::runtime_error("Invalid GEMM candidate");
    check(cublasLtMatmul(context.lt, p.op, &alpha, data(w), p.a, data(x), p.b,
                         &beta, data(y), p.c, data(y), p.c, &p.candidates[index].algo,
                         context.workspace, context.bytes, stream));
  }
  return p.count;
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_probe, gemm_probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_config, gemm_config);
