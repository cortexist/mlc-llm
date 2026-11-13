// Volta WMMA prefill: 16 queries x 16 keys; four warps split QK's reduction.
// S is the stored width of a Q/K/V head and D the part of it in use: of a padded
// Gemma cache (S = 512), a local layer uses 256.
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

namespace xavier_prefill {
using namespace nvcuda;
template <int D>
struct Shared {
  __align__(32) half q[16 * D];
  __align__(32) half kv[16 * D];
  __align__(32) float score[4 * 256];
  __align__(32) half probability[256];
  __align__(32) float product[4 * 256];
  float maximum[16], denominator[16], factor[16];
};

template <int D, bool Paged, bool Local, int S = 512>
__global__ void attention(const half* q, const int* qptr, const half* keys, const half* values,
                          const int* kptr, const int* page_values, const int* length, half* output,
                          float* lse, int batch, int hq, int hkv, int page_size, int causal,
                          float scale) {
  __shared__ Shared<D> s;
  int t = threadIdx.x, warp = t / 32, lane = t % 32, head = blockIdx.y;
  int tile = blockIdx.x, seq = 0;
  while (seq < batch) {
    int nt = (qptr[seq + 1] - qptr[seq] + 15) / 16;
    if (tile < nt) break;
    tile -= nt;
    ++seq;
  }
  if (seq == batch) return;
  int qbase = qptr[seq] + tile * 16, rows = min(16, qptr[seq + 1] - qbase);
  int rowbase = tile * 16, qcount = qptr[seq + 1] - qptr[seq], kh = head / (hq / hkv);
  int first = kptr[seq], count = kptr[seq + 1] - first;
  int offset = 0, sink = 0;
  if constexpr (Paged) {
    if constexpr (Local) {
      offset = length[batch + seq];
      sink = length[2 * batch + seq];
    }
    count = count ? (count - 1) * page_size + length[seq] - offset + sink : 0;
  }
  int begin = 0, end = count;
  if (causal) {
    if constexpr (Local)
      begin = max(0, count - max(512 - rowbase - 1, 0));
    else
      end = max(0, min(count, count - qcount + rowbase + rows));
  }
  begin = begin / 16 * 16;
  for (int i = t; i < 16 * D; i += 128) {
    int r = i / D, d = i % D;
    s.q[i] = r < rows ? q[((int64_t)(qbase + r) * hq + head) * S + d] : __float2half(0.f);
  }
  if (t < 16) {
    s.maximum[t] = -50000.f;
    s.denominator[t] = 1.f;
  }
  constexpr int Chunks = D / 64;
  float acc[Chunks][8] = {};
  __syncthreads();
  for (int base = begin; base < end; base += 16) {
    for (int i = t; i < 16 * D; i += 128) {
      int r = i / D, d = i % D, pos = base + r;
      half value = __float2half(0.f);
      if (pos < count) {
        if constexpr (Paged) {
          int slot = pos < sink ? pos : pos - sink + offset;
          int page = page_values[first + slot / page_size];
          value = keys[((((int64_t)page * 2) * hkv + kh) * page_size + slot % page_size) * S + d];
        } else
          value = keys[((int64_t)(first + pos) * hkv + kh) * S + d];
      }
      s.kv[i] = value;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
    wmma::fill_fragment(score, 0.f);
    for (int d = warp * 16; d < D; d += 64) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      wmma::load_matrix_sync(a, s.q + d, D);
      wmma::load_matrix_sync(b, s.kv + d, D);
      wmma::mma_sync(score, a, b, score);
    }
    wmma::store_matrix_sync(s.score + warp * 256, score, 16, wmma::mem_row_major);
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      int row = warp * 4 + j, pos = base + lane;
      bool valid = lane < 16 && row < rows && pos < count;
      if (causal) {
        if constexpr (Local)
          valid = valid && pos >= max(0, count - max(512 - rowbase - row - 1, 0));
        else
          valid = valid && pos < count - qcount + rowbase + row + 1;
      }
      float value = -CUDART_INF_F;
      if (valid) {
        int i = row * 16 + lane;
        value = ((s.score[i] + s.score[256 + i]) + s.score[512 + i]) + s.score[768 + i];
        value *= scale * 1.4426950408889634f;
      }
      float top = value;
      for (int d = 8; d; d /= 2) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, d));
      top = fmaxf(top, s.maximum[row]);
      float prob = valid ? exp2f(value - top) : 0.f;
      float sum = prob;
      for (int d = 8; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
      if (lane < 16) s.probability[row * 16 + lane] = __float2half_rn(prob);
      if (lane == 0) {
        float factor = exp2f(s.maximum[row] - top);
        s.factor[row] = factor;
        s.denominator[row] = s.denominator[row] * factor + sum;
        s.maximum[row] = top;
      }
    }
    __syncthreads();
    // Reuse the key tile for V after QK has finished.
    for (int i = t; i < 16 * D; i += 128) {
      int r = i / D, d = i % D, pos = base + r;
      half value = __float2half(0.f);
      if (pos < count) {
        if constexpr (Paged) {
          int slot = pos < sink ? pos : pos - sink + offset;
          int page = page_values[first + slot / page_size];
          value =
              keys[((((int64_t)page * 2 + 1) * hkv + kh) * page_size + slot % page_size) * S + d];
        } else
          value = values[((int64_t)(first + pos) * hkv + kh) * S + d];
      }
      s.kv[i] = value;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> probability;
    wmma::load_matrix_sync(probability, s.probability, 16);
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      int col = (warp * Chunks + chunk) * 16;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> value;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> product;
      wmma::load_matrix_sync(value, s.kv + col, D);
      wmma::fill_fragment(product, 0.f);
      wmma::mma_sync(product, probability, value, product);
      wmma::store_matrix_sync(s.product + warp * 256, product, 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int i = lane + j * 32, row = i / 16;
        acc[chunk][j] = acc[chunk][j] * s.factor[row] + s.product[warp * 256 + i];
      }
      __syncwarp();
    }
    __syncthreads();
  }
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int i = lane + j * 32, row = i / 16, col = (warp * Chunks + chunk) * 16 + i % 16;
      if (row < rows)
        output[((int64_t)(qbase + row) * hq + head) * S + col] =
            __float2half_rn(acc[chunk][j] / s.denominator[row]);
    }
  }
  if constexpr (D < S) {
    for (int i = t; i < rows * (S - D); i += 128)
      output[((int64_t)(qbase + i / (S - D)) * hq + head) * S + D + i % (S - D)] =
          __float2half(0.f);
  }
  if (t < rows) lse[(qbase + t) * hq + head] = s.maximum[t] + log2f(s.denominator[t]);
}
}  // namespace xavier_prefill
