// Attention over the cached pages on Volta tensor cores, for any number of
// query rows of one sequence: a decode step, a verification or a prefill chunk.
//
// A block takes the queries of up to 4*T rows for one K/V head, whose four query
// heads share K and V, and one segment of the keys. It reads the pages in place,
// 16 keys per tile: K twice (row maxima, then probabilities) and V once. Nothing
// is rescaled along the way; `merge` joins the segments by their maxima.
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

namespace xavier_paged {
using namespace nvcuda;
constexpr int Stride = 512;  // stored width of every head
constexpr int Keys = 16;     // keys per page and per tile
constexpr int Warps = 8;     // each scores one page of a step
using Tile = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
constexpr int E = Tile::num_elements;

struct Job {
  const half* q;        // [rows, 8, 512]
  const half* pages;    // [page, K or V, 2, 16, 512]
  const int* page_ids;  // pages of the sequence, oldest first
  const int* length;    // keys in the last page; local: then window offset and sink
  float* partial;       // [block, query, D] unnormalized outputs
  float* stat;          // [block, maximum or denominator, query]
  int rows, pages_count, segment_pages, segments, causal;
  float scale;
};

__device__ __forceinline__ const half* page(const Job& job, int index, int value, int head) {
  return job.pages + (((int64_t)job.page_ids[index] * 2 + value) * 2 + head) * Keys * Stride;
}

// Logical keys [begin, end) of the `count` cached ones that a row attends.
__device__ __forceinline__ void bounds(const Job& job, bool local, int row, int count, int& begin,
                                       int& end) {
  begin = end = 0;
  if (row >= job.rows) return;
  end = count;
  if (!job.causal) return;
  if (local)
    begin = max(0, count - max(512 - row - 1, 0));
  else
    end = max(0, min(count, count - job.rows + row + 1));
}

// Fold a 16x16 tile held by one warp to one value per query.
__device__ __forceinline__ void fold(const Tile& tile, float* scratch, float* out, bool maximum) {
  wmma::store_matrix_sync(scratch, tile, 16, wmma::mem_row_major);
  __syncwarp();
  int lane = threadIdx.x % 32;
  if (lane < 16) {
    float v = scratch[lane * 16];
    for (int key = 1; key < 16; ++key) {
      float other = scratch[lane * 16 + key];
      v = maximum ? fmaxf(v, other) : v + other;
    }
    out[lane] = v;
  }
  __syncwarp();
}

template <int T, bool Local>
__global__ void attend(Job job) {
  constexpr int D = Local ? 256 : 512, Q = 16 * T, F = D * T / (Warps * 16);
  __shared__ __align__(32) half queries[Q * D];           // later the V of a partial last page
  __shared__ __align__(32) half weight[Warps * 2 * 256];  // [page of the step][tile][query][key]
  __shared__ float reduced[Warps][Q];
  __shared__ float top[Q];
  int t = threadIdx.x, warp = t / 32, lane = t % 32, head = blockIdx.y;
  int group = blockIdx.x / job.segments, segment = blockIdx.x % job.segments;
  // A warp's two weight tiles double as its scratch outside the key loop.
  float* scratch = reinterpret_cast<float*>(weight + warp * 512);

  int last = job.length[0], offset = Local ? job.length[1] : 0, sink = Local ? job.length[2] : 0;
  int count = (job.pages_count - 1) * Keys + last - offset + sink;
  int first = segment * job.segment_pages, end = min(first + job.segment_pages, job.pages_count);
  float scale = job.scale * 1.4426950408889634f;

  for (int i = t; i < Q * D; i += 256) {
    int query = i / D, row = group * 4 * T + query / 4;
    queries[i] = row < job.rows
                     ? job.q[((int64_t)row * 8 + head * 4 + query % 4) * Stride + i % D]
                     : __float2half(0.f);
  }
  // Which query and key each element of a tile holds, read from a tile of indices.
  for (int i = lane; i < 256; i += 32) scratch[i] = i;
  __syncthreads();
  Tile order;
  wmma::load_matrix_sync(order, scratch, 16, wmma::mem_row_major);
  int at[E], lo[T][E], hi[T][E];
#pragma unroll
  for (int i = 0; i < E; ++i) {
    at[i] = int(order.x[i]);
#pragma unroll
    for (int tile = 0; tile < T; ++tile) {
      int begin, stop;
      bounds(job, Local, group * 4 * T + (tile * 16 + at[i] / 16) / 4, count, begin, stop);
      lo[tile][i] = max(begin, sink) - sink + offset;
      hi[tile][i] = max(stop, sink) - sink + offset;
    }
  }
  // Stored position `slot` is attended by the query of element i: in its window, or a sink key.
  auto seen = [&](int tile, int i, int slot) {
    if (slot >= lo[tile][i] && slot < hi[tile][i]) return true;
    if (slot >= sink) return false;
    int begin, stop;
    bounds(job, Local, group * 4 * T + (tile * 16 + at[i] / 16) / 4, count, begin, stop);
    return slot >= begin && slot < stop;
  };
  auto score = [&](Tile* s, const half* keys) {
#pragma unroll
    for (int tile = 0; tile < T; ++tile) wmma::fill_fragment(s[tile], 0.f);
    for (int d = 0; d < D; d += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      wmma::load_matrix_sync(b, keys + d, Stride);
#pragma unroll
      for (int tile = 0; tile < T; ++tile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::load_matrix_sync(a, queries + tile * 16 * D + d, D);
        wmma::mma_sync(s[tile], a, b, s[tile]);
      }
    }
  };

  // Pass 1: the maximum score of every query over this segment.
  Tile best[T], s[T];
#pragma unroll
  for (int tile = 0; tile < T; ++tile) wmma::fill_fragment(best[tile], -CUDART_INF_F);
  for (int index = first + warp; index < end; index += Warps) {
    score(s, page(job, index, 0, head));
#pragma unroll
    for (int tile = 0; tile < T; ++tile)
#pragma unroll
      for (int i = 0; i < E; ++i)
        if (seen(tile, i, index * Keys + at[i] % 16))
          best[tile].x[i] = fmaxf(best[tile].x[i], s[tile].x[i] * scale);
  }
#pragma unroll
  for (int tile = 0; tile < T; ++tile) fold(best[tile], scratch, reduced[warp] + tile * 16, true);
  __syncthreads();
  if (t < Q) {
    float v = reduced[0][t];
    for (int w = 1; w < Warps; ++w) v = fmaxf(v, reduced[w][t]);
    top[t] = v;
  }
  __syncthreads();
#pragma unroll
  for (int tile = 0; tile < T; ++tile) {
    for (int i = lane; i < 256; i += 32) scratch[i] = top[tile * 16 + i / 16];
    __syncwarp();
    wmma::load_matrix_sync(best[tile], scratch, 16, wmma::mem_row_major);
    __syncwarp();
  }

  // Pass 2: probabilities against those maxima, then values. A warp accumulates
  // F tiles of output dimensions for one tile of queries.
  int mine = warp * T / Warps, dim = warp % (Warps / T) * F * 16;
  int partial_page = last < Keys ? job.pages_count - 1 : -1;
  Tile sum[T], out[F];
#pragma unroll
  for (int tile = 0; tile < T; ++tile) wmma::fill_fragment(sum[tile], 0.f);
#pragma unroll
  for (int f = 0; f < F; ++f) wmma::fill_fragment(out[f], 0.f);
  auto values = [&](int w, const half* v, int stride) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::load_matrix_sync(a, weight + (w * 2 + mine) * 256, 16);
#pragma unroll
    for (int f = 0; f < F; ++f) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, v + dim + f * 16, stride);
      wmma::mma_sync(out[f], a, b, out[f]);
    }
  };
  for (int base = first; base < end; base += Warps) {
    if (base + warp < end) {
      score(s, page(job, base + warp, 0, head));
#pragma unroll
      for (int tile = 0; tile < T; ++tile)
#pragma unroll
        for (int i = 0; i < E; ++i) {
          float p = seen(tile, i, (base + warp) * Keys + at[i] % 16)
                        ? exp2f(s[tile].x[i] * scale - best[tile].x[i])
                        : 0.f;
          sum[tile].x[i] += p;
          weight[(warp * 2 + tile) * 256 + at[i]] = __float2half_rn(p);
        }
    }
    __syncthreads();
    for (int w = 0; w < Warps && base + w < end; ++w)
      if (base + w != partial_page) values(w, page(job, base + w, 1, head), Stride);
    __syncthreads();
  }
  if (partial_page >= first && partial_page < end) {
    // Unused keys of the last page may hold anything, and 0 * NaN is NaN:
    // stage its V, cleared, where the queries were.
    const half* v = page(job, partial_page, 1, head);
    for (int i = t; i < Keys * D; i += 256)
      queries[i] = i / D < last ? v[i / D * Stride + i % D] : __float2half(0.f);
    __syncthreads();
    values((partial_page - first) % Warps, queries, D);
    __syncthreads();
  }

#pragma unroll
  for (int tile = 0; tile < T; ++tile) fold(sum[tile], scratch, reduced[warp] + tile * 16, false);
  __syncthreads();
  int block = blockIdx.y * gridDim.x + blockIdx.x;
  if (t < Q) {
    float v = 0;
    for (int w = 0; w < Warps; ++w) v += reduced[w][t];
    job.stat[(block * 2) * Q + t] = top[t];
    job.stat[(block * 2 + 1) * Q + t] = v;
  }
  float* partial = job.partial + ((int64_t)block * Q + mine * 16) * D + dim;
#pragma unroll
  for (int f = 0; f < F; ++f)
    wmma::store_matrix_sync(partial + f * 16, out[f], D, wmma::mem_row_major);
}

// One block per query: join its segments and normalize. A query that attends
// no key gets zero output and the empty log-sum-exp, as the TIR kernels give.
template <int T, bool Local>
__global__ void merge(Job job, half* output, float* lse, int blocks) {
  constexpr int D = Local ? 256 : 512, Q = 16 * T;
  int row = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
  int query = row % (4 * T) * 4 + head % 4;
  int block = head / 4 * blocks + row / (4 * T) * job.segments;
  float top = -CUDART_INF_F, sum = 0, low = 0, high = 0;
  for (int s = 0; s < job.segments; ++s) top = fmaxf(top, job.stat[(block + s) * 2 * Q + query]);
  for (int s = 0; s < job.segments; ++s) {
    float m = job.stat[(block + s) * 2 * Q + query];
    if (m == -CUDART_INF_F) continue;
    float weight = exp2f(m - top);
    const float* partial = job.partial + ((int64_t)(block + s) * Q + query) * D;
    sum += weight * job.stat[((block + s) * 2 + 1) * Q + query];
    low += weight * partial[t];
    if constexpr (D == 512) high += weight * partial[t + 256];
  }
  half* o = output + ((int64_t)row * 8 + head) * Stride;
  o[t] = __float2half_rn(sum > 0 ? low / sum : 0.f);
  o[t + 256] = __float2half_rn(sum > 0 ? high / sum : 0.f);
  if (t == 0) lse[row * 8 + head] = sum > 0 ? top + log2f(sum) : -50000.f;
}
}  // namespace xavier_paged
