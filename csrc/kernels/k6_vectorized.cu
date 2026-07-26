#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K6 - float4 vectorized loads with a transposed A tile.
//
//  Same geometry as K5 (kBM x kBN results per block, kBK values of k per
//  tile, kTM x kTN results per thread). Three changes, all about how many
//  instructions the same bytes cost:
//
//  1. The A tile is stored transposed, As[k][row] instead of As[row][k].
//     In K5 the kTM values of A a thread needs for one k sat kBK floats
//     apart in shared memory; transposed they are contiguous.
//  2. The inner loop reads both operands as float4: 16 scalar shared
//     loads per k become 4 vector loads, for the same bytes.
//  3. Global loads are vectorized too: A is read four consecutive k at a
//     time (its rows are contiguous) and B four consecutive columns at a
//     time. The transposing store into shared memory costs four scalar
//     stores, but it happens once per tile while the inner loop runs kBK
//     times.
//
//  float4 access needs 16-byte alignment, so the vector path is only
//  taken when K (for A) and N (for B) are multiples of 4 and the group of
//  four stays inside the matrix; otherwise the loads fall back to element
//  at a time. Shared-memory accesses are always aligned because the tile
//  dimensions are fixed.
// =====================================================================

namespace {
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 16;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kThreadRows = kBM / kTM;
constexpr int kThreadCols = kBN / kTN;
constexpr int kThreads = kThreadRows * kThreadCols;
// float4 chunks in each tile, and how many each thread moves.
constexpr int kAChunks = (kBM * kBK) / 4;
constexpr int kBChunks = (kBK * kBN) / 4;
constexpr int kAPerThread = kAChunks / kThreads;
constexpr int kBPerThread = kBChunks / kThreads;
constexpr int kAChunksPerRow = kBK / 4;
constexpr int kBChunksPerRow = kBN / 4;
}  // namespace

static_assert(kTM % 4 == 0 && kTN % 4 == 0, "inner loads read float4");
static_assert(kAChunks % kThreads == 0 && kBChunks % kThreads == 0,
              "tile loading must divide evenly among the threads");

__global__ void k6_vectorized_kernel(int M, int N, int K, float alpha,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B, float beta,
                                     float* __restrict__ C) {
  const int tc = static_cast<int>(threadIdx.x);
  const int tr = static_cast<int>(threadIdx.y);
  const int tid = tr * kThreadCols + tc;
  const int m0 = static_cast<int>(blockIdx.y) * kBM;
  const int n0 = static_cast<int>(blockIdx.x) * kBN;
  const int rm = m0 + tr * kTM;  // first result row of this thread
  const int rn = n0 + tc * kTN;  // first result column
  const bool any_ok = rm < M && rn < N;

  __shared__ alignas(16) float As[kBK * kBM];  // transposed: As[k][row]
  __shared__ alignas(16) float Bs[kBK * kBN];

  const bool vec_a = (K % 4 == 0);
  const bool vec_b = (N % 4 == 0);

  float res[kTM * kTN] = {};
  for (int k0 = 0; k0 < K; k0 += kBK) {
    for (int s = 0; s < kAPerThread; ++s) {
      const int chunk = tid * kAPerThread + s;
      const int row = chunk / kAChunksPerRow;
      const int kc = (chunk % kAChunksPerRow) * 4;
      const int m = m0 + row;
      float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (m < M) {
        if (vec_a && k0 + kc + 3 < K) {
          const float4 t =
              *reinterpret_cast<const float4*>(&A[m * K + k0 + kc]);
          v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
        } else {
          for (int e = 0; e < 4; ++e) {
            const int k = k0 + kc + e;
            v[e] = (k < K) ? A[m * K + k] : 0.0f;
          }
        }
      }
      // Transposing store: four scalar stores, once per tile.
      for (int e = 0; e < 4; ++e) As[(kc + e) * kBM + row] = v[e];
    }

    for (int s = 0; s < kBPerThread; ++s) {
      const int chunk = tid * kBPerThread + s;
      const int kr = chunk / kBChunksPerRow;
      const int nc = (chunk % kBChunksPerRow) * 4;
      const int k = k0 + kr;
      const int n = n0 + nc;
      float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (k < K) {
        if (vec_b && n + 3 < N) {
          const float4 t = *reinterpret_cast<const float4*>(&B[k * N + n]);
          v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
        } else {
          for (int e = 0; e < 4; ++e) {
            const int nn = n + e;
            v[e] = (nn < N) ? B[k * N + nn] : 0.0f;
          }
        }
      }
      *reinterpret_cast<float4*>(&Bs[kr * kBN + nc]) =
          make_float4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();

    if (any_ok) {
      for (int tk = 0; tk < kBK; ++tk) {
        const float4 a0 =
            *reinterpret_cast<const float4*>(&As[tk * kBM + tr * kTM]);
        const float4 a1 =
            *reinterpret_cast<const float4*>(&As[tk * kBM + tr * kTM + 4]);
        const float4 b0 =
            *reinterpret_cast<const float4*>(&Bs[tk * kBN + tc * kTN]);
        const float4 b1 =
            *reinterpret_cast<const float4*>(&Bs[tk * kBN + tc * kTN + 4]);
        const float ra[kTM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float rb[kTN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        for (int i = 0; i < kTM; ++i)
          for (int j = 0; j < kTN; ++j) res[i * kTN + j] += ra[i] * rb[j];
      }
    }
    __syncthreads();
  }

  if (!any_ok) return;
  for (int i = 0; i < kTM && rm + i < M; ++i) {
    const int m = rm + i;
    for (int j = 0; j < kTN && rn + j < N; ++j) {
      const int n = rn + j;
      C[m * N + n] = alpha * res[i * kTN + j] + beta * C[m * N + n];
    }
  }
}

void sgemm_k6_vectorized(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  dim3 block(kThreadCols, kThreadRows);
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  k6_vectorized_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k6_vectorized");
}
