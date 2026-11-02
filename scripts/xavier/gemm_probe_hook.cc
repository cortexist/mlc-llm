// Opt-in experiment: select measured CUDA 11.4/sm_72 cuBLASLt configurations.
// Used by the isolated benchmark via LD_PRELOAD; not part of the default runtime.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
std::atomic<unsigned> seen{0};
template <typename T>
bool layout(cublasLtMatrixLayout_t d, cublasLtMatrixLayoutAttribute_t a, T& v) {
  size_t written = 0;
  return cublasLtMatrixLayoutGetAttribute(d, a, &v, sizeof(v), &written) == CUBLAS_STATUS_SUCCESS;
}
template <typename T>
bool operation(cublasLtMatmulDesc_t d, cublasLtMatmulDescAttributes_t a, T& v) {
  size_t written = 0;
  return cublasLtMatmulDescGetAttribute(d, a, &v, sizeof(v), &written) == CUBLAS_STATUS_SUCCESS;
}
int config(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t a) {
  int value = -1;
  size_t written = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&algo, a, &value, sizeof(value), &written) !=
      CUBLAS_STATUS_SUCCESS) return -1;
  return value;
}
}
extern "C" cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    cublasLtMatmulPreference_t preference, int requested,
    cublasLtMatmulHeuristicResult_t* results, int* returned) {
  static auto original = reinterpret_cast<decltype(&cublasLtMatmulAlgoGetHeuristic)>(
      dlsym(RTLD_NEXT, "cublasLtMatmulAlgoGetHeuristic"));
  if (!original) return CUBLAS_STATUS_INTERNAL_ERROR;
  auto status = original(handle, op, a, b, c, d, preference, requested, results, returned);
  const char* enabled = std::getenv("MLC_XAVIER_GEMM_PROBE");
  if (!enabled || std::strcmp(enabled, "1") || status != CUBLAS_STATUS_SUCCESS ||
      requested != 1 || *returned != 1) return status;
  static const bool supported = [] {
    cudaDeviceProp p{};
    return CUDART_VERSION == 11040 && cudaGetDeviceProperties(&p, 0) == cudaSuccess &&
           p.major == 7 && p.minor == 2;
  }();
  int device = -1;
  if (!supported || cudaGetDevice(&device) != cudaSuccess || device != 0) return status;
  uint64_t k = 0, n = 0, m = 0;
  int ba = 0, bb = 0, bc = 0;
  cudaDataType_t ta, tb, tc;
  cublasOperation_t opa, opb;
  cublasLtEpilogue_t ep;
  cublasComputeType_t compute;
  if (!layout(a, CUBLASLT_MATRIX_LAYOUT_ROWS, k) ||
      !layout(a, CUBLASLT_MATRIX_LAYOUT_COLS, n) ||
      !layout(b, CUBLASLT_MATRIX_LAYOUT_COLS, m) ||
      !layout(a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ba) ||
      !layout(b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, bb) ||
      !layout(c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, bc) || ba != 1 || bb != 1 || bc != 1 ||
      !layout(a, CUBLASLT_MATRIX_LAYOUT_TYPE, ta) || ta != CUDA_R_16F ||
      !layout(b, CUBLASLT_MATRIX_LAYOUT_TYPE, tb) || tb != CUDA_R_16F ||
      !layout(c, CUBLASLT_MATRIX_LAYOUT_TYPE, tc) || tc != CUDA_R_16F ||
      !operation(op, CUBLASLT_MATMUL_DESC_TRANSA, opa) || opa != CUBLAS_OP_T ||
      !operation(op, CUBLASLT_MATMUL_DESC_TRANSB, opb) || opb != CUBLAS_OP_N ||
      !operation(op, CUBLASLT_MATMUL_DESC_EPILOGUE, ep) || ep != CUBLASLT_EPILOGUE_DEFAULT ||
      !operation(op, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute) || compute != CUBLAS_COMPUTE_32F)
    return status;
  const bool query = n == 2048 && k == 2560 && (m == 161 || m == 448);
  const bool down = n == 2560 && k == 10240 && (m == 448 || m == 512);
  if (!query && !down) return status;
  // For the down projection, preserve the baseline split count and reduction type.
  if (down && (config(results[0].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) != 3 ||
               config(results[0].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) != 1)) return status;
  cublasLtMatmulHeuristicResult_t choices[8];
  int count = 0;
  if (original(handle, op, a, b, c, d, preference, 8, choices, &count) != CUBLAS_STATUS_SUCCESS)
    return status;
  for (int i = 0; i < count; ++i) {
    const auto& algo = choices[i].algo;
    if (choices[i].state == CUBLAS_STATUS_SUCCESS &&
        config(algo, CUBLASLT_ALGO_CONFIG_ID) == (query ? 21 : 3) &&
        config(algo, CUBLASLT_ALGO_CONFIG_TILE_ID) == (query ? 20 : 24) &&
        config(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) == (query ? 1 : 3) &&
        config(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == (query ? 0 : 1) &&
        config(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) == (query ? 0 : 1) &&
        config(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID) == 0 &&
        config(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) == 0) {
      results[0] = choices[i];
      unsigned bit = 1u << (query ? (m == 161 ? 0 : 1) : (m == 448 ? 2 : 3));
      if (!(seen.fetch_or(bit) & bit))
        std::fprintf(stderr, "Xavier GEMM probe selected M=%llu N=%llu K=%llu candidate=%d\n",
                     (unsigned long long)m, (unsigned long long)n, (unsigned long long)k, i);
      break;
    }
  }
  return status;
}
