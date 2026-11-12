// Attention over the cached pages on Volta tensor cores, for any number of
// query rows of one sequence: a decode step, a verification or a prefill chunk.
//
// A block takes 16*T queries that share K and V: the Group query heads of one
// K/V head, for 16*T/Group rows, and one segment of the keys. It reads the pages
// in place, 16 keys per tile: K twice (row maxima, then probabilities) and V
// once. Nothing is rescaled along the way. A block that has all the keys writes
// the result; otherwise `merge` joins the segments by their maxima.
//
// Probabilities enter the value product as FP16. A step that is predicted from
// (T = 1: decode or verification) keeps two FP16 terms of each, 22 bits, which
// puts their rounding far below that of the FP16 result; a prefill chunk
// (T = 2) keeps one.
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

// A kind of layer, fixed at compile time. Heads: query heads. KV: stored K/V
// heads. Group: the query heads of a block; more than Heads / KV when the stored
// heads repeat one, which then serves every query head. Window: the keys a local
// layer attends, the query's own included; 0 for a global layer.
template <int H, int K, int G, int W>
struct Layer {
  static constexpr int Heads = H, KV = K, Group = G, Window = W;
};
using E4BLocal = Layer<8, 2, 4, 512>;
using E4BGlobal = Layer<8, 2, 4, 0>;
using UnifiedLocal = Layer<16, 8, 2, 1024>;
using UnifiedGlobal = Layer<16, 8, 16, 0>;  // one global K/V head, stored eight times

struct Job {
  const half* q;        // [rows, Heads, 512]
  const half* pages;    // [page, K or V, KV, 16, 512]
  const int* page_ids;  // pages of the sequence, oldest first
  const int* length;    // keys in the last page; local: then window offset and sink
  half* output;         // [rows, Heads, 512]
  float* lse;           // [rows, Heads] base-2 log-sum-exp
  float* partial;       // [block, query, D] unnormalized outputs of a segment
  float* stat;          // [block, maximum or denominator, query]
  int rows, pages_count, segment_pages, segments, causal;
  float scale;
};

template <typename L>
__device__ __forceinline__ const half* page(const Job& job, int index, int value, int head) {
  return job.pages +
         (((int64_t)job.page_ids[index] * 2 + value) * L::KV + head) * Keys * Stride;
}

// Logical keys [begin, end) of the `count` cached ones that a row attends.
__device__ __forceinline__ void bounds(const Job& job, int window, int row, int count, int& begin,
                                       int& end) {
  begin = end = 0;
  if (row >= job.rows) return;
  end = count;
  if (!job.causal) return;
  if (window)
    begin = max(0, count - max(window - row - 1, 0));
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

template <int T, typename L>
__global__ void attend(Job job) {
  constexpr bool Local = L::Window > 0;
  constexpr int D = Local ? 256 : 512, Q = 16 * T, F = D * T / (Warps * 16), Terms = 3 - T;
  constexpr int G = L::Group;
  __shared__ __align__(32) half queries[Q * D];           // later the V of a partial last page
  // [page of the step][tile, or term of the one tile][query][key]
  __shared__ __align__(32) half weight[Warps * 2 * 256];
  __shared__ float reduced[Warps][Q];
  __shared__ float top[Q], total[Q];
  int t = threadIdx.x, warp = t / 32, lane = t % 32, head = blockIdx.y;
  int first_row = blockIdx.x / job.segments * (Q / G), segment = blockIdx.x % job.segments;
  // A warp's two weight tiles double as its scratch outside the key loop.
  float* scratch = reinterpret_cast<float*>(weight + warp * 512);

  int last = job.length[0], offset = Local ? job.length[1] : 0, sink = Local ? job.length[2] : 0;
  int count = (job.pages_count - 1) * Keys + last - offset + sink;
  int first = segment * job.segment_pages, end = min(first + job.segment_pages, job.pages_count);
  float scale = job.scale * 1.4426950408889634f;

  for (int i = t; i < Q * D; i += 256) {
    int query = i / D, row = first_row + query / G;
    queries[i] =
        row < job.rows
            ? job.q[((int64_t)row * L::Heads + head * G + query % G) * Stride + i % D]
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
      bounds(job, L::Window, first_row + (tile * 16 + at[i] / 16) / G, count, begin, stop);
      lo[tile][i] = max(begin, sink) - sink + offset;
      hi[tile][i] = max(stop, sink) - sink + offset;
    }
  }
  // Stored position `slot` is attended by the query of element i: in its window, or a sink key.
  auto seen = [&](int tile, int i, int slot) {
    if (slot >= lo[tile][i] && slot < hi[tile][i]) return true;
    if (slot >= sink) return false;
    int begin, stop;
    bounds(job, L::Window, first_row + (tile * 16 + at[i] / 16) / G, count, begin, stop);
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
    score(s, page<L>(job, index, 0, head));
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
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[Terms];
#pragma unroll
    for (int term = 0; term < Terms; ++term)
      wmma::load_matrix_sync(a[term], weight + (w * 2 + mine + term) * 256, 16);
#pragma unroll
    for (int f = 0; f < F; ++f) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, v + dim + f * 16, stride);
#pragma unroll
      for (int term = 0; term < Terms; ++term) wmma::mma_sync(out[f], a[term], b, out[f]);
    }
  };
  for (int base = first; base < end; base += Warps) {
    if (base + warp < end) {
      score(s, page<L>(job, base + warp, 0, head));
#pragma unroll
      for (int tile = 0; tile < T; ++tile)
#pragma unroll
        for (int i = 0; i < E; ++i) {
          float p = seen(tile, i, (base + warp) * Keys + at[i] % 16)
                        ? exp2f(s[tile].x[i] * scale - best[tile].x[i])
                        : 0.f;
          sum[tile].x[i] += p;
          half leading = __float2half_rn(p);
          weight[(warp * 2 + tile) * 256 + at[i]] = leading;
          if constexpr (Terms == 2)
            weight[(warp * 2 + 1) * 256 + at[i]] = __float2half_rn(p - __half2float(leading));
        }
    }
    __syncthreads();
    for (int w = 0; w < Warps && base + w < end; ++w)
      if (base + w != partial_page) values(w, page<L>(job, base + w, 1, head), Stride);
    __syncthreads();
  }
  if (partial_page >= first && partial_page < end) {
    // Unused keys of the last page may hold anything, and 0 * NaN is NaN:
    // stage its V, cleared, where the queries were.
    const half* v = page<L>(job, partial_page, 1, head);
    for (int i = t; i < Keys * D; i += 256)
      queries[i] = i / D < last ? v[i / D * Stride + i % D] : __float2half(0.f);
    __syncthreads();
    values((partial_page - first) % Warps, queries, D);
    __syncthreads();
  }

#pragma unroll
  for (int tile = 0; tile < T; ++tile) fold(sum[tile], scratch, reduced[warp] + tile * 16, false);
  __syncthreads();
  if (t < Q) {
    float v = 0;
    for (int w = 0; w < Warps; ++w) v += reduced[w][t];
    total[t] = v;
  }
  __syncthreads();
  if (job.segments > 1) {
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    if (t < Q) {
      job.stat[(block * 2) * Q + t] = top[t];
      job.stat[(block * 2 + 1) * Q + t] = total[t];
    }
    float* partial = job.partial + ((int64_t)block * Q + mine * 16) * D + dim;
#pragma unroll
    for (int f = 0; f < F; ++f)
      wmma::store_matrix_sync(partial + f * 16, out[f], D, wmma::mem_row_major);
    return;
  }
  // Query `query` of this block, or null for the padding of the last group.
  auto result = [&](int query) -> half* {
    int row = first_row + query / G;
    return row < job.rows
               ? job.output + ((int64_t)row * L::Heads + head * G + query % G) * Stride
               : nullptr;
  };
#pragma unroll
  for (int f = 0; f < F; ++f)
#pragma unroll
    for (int i = 0; i < E; ++i) {
      int query = mine * 16 + at[i] / 16;
      if (half* o = result(query))
        o[dim + f * 16 + at[i] % 16] =
            __float2half_rn(total[query] > 0 ? out[f].x[i] / total[query] : 0.f);
    }
  if constexpr (Local) {
    for (int i = t; i < Q * 256; i += 256)
      if (half* o = result(i / 256)) o[256 + i % 256] = __float2half(0.f);
  }
  if (t < Q && result(t))
    job.lse[(first_row + t / G) * L::Heads + head * G + t % G] =
        total[t] > 0 ? top[t] + log2f(total[t]) : -50000.f;
}

// One block per query: join its segments and normalize. A query that attends
// no key gets zero output and the empty log-sum-exp, as the TIR kernels give.
template <int T, typename L>
__global__ void merge(Job job, int blocks) {
  constexpr int D = L::Window ? 256 : 512, Q = 16 * T, G = L::Group;
  int row = blockIdx.x, head = blockIdx.y, t = threadIdx.x;
  int query = row % (Q / G) * G + head % G;
  int block = head / G * blocks + row / (Q / G) * job.segments;
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
  half* o = job.output + ((int64_t)row * L::Heads + head) * Stride;
  o[t] = __float2half_rn(sum > 0 ? low / sum : 0.f);
  o[t + 256] = __float2half_rn(sum > 0 ? high / sum : 0.f);
  if (t == 0) job.lse[row * L::Heads + head] = sum > 0 ? top + log2f(sum) : -50000.f;
}
}  // namespace xavier_paged
