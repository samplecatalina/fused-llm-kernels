#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K5 - 2D thread tiling: each thread computes a kTM x kTN block of C.
//
//  For every k, a thread reads the kTM values of A it needs into one set
//  of registers and the kTN values of B into another, and accumulates
//  their outer product into kTM x kTN result registers. The inner
//  multiply-add never touches memory; shared memory is read kTM + kTN
//  times per k for kTM x kTN results.
//
//  Tile geometry: kBM x kBN results per block, kBK values of k per tile.
//  A 2D split divides the threads per block by kTN as well, and a 32x32
//  tile would leave 16 threads per block - less than one warp. With
//  128x128 results per block there are 16x16 = 256 threads.
//
//  Loading covers each shared cell exactly once:
//    As[row][tk]  (kBM x kBK) is loaded by thread (row / kTM, tk)
//    Bs[tk][col]  (kBK x kBN) is loaded by thread (tk, col / kTN)
//  which requires the thread grid to be kBK threads on each side.
// =====================================================================

namespace {
constexpr int kBM = 128;  // result rows per block
constexpr int kBN = 128;  // result columns per block
constexpr int kBK = 16;   // values of k per tile
constexpr int kTM = 8;    // result rows per thread
constexpr int kTN = 8;    // result columns per thread
constexpr int kThreadRows = kBM / kTM;
constexpr int kThreadCols = kBN / kTN;
}  // namespace

// The loading scheme above silently breaks if the tile parameters change
// without it; fail the build instead.
static_assert(kBM % kTM == 0 && kBN % kTN == 0, "thread tiles must divide the block");
static_assert(kThreadCols == kBK, "As loading needs one thread column per k in a tile");
static_assert(kThreadRows == kBK, "Bs loading needs one thread row per k in a tile");

__global__ void k5_tiling2d_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
  const int tc = static_cast<int>(threadIdx.x);
  const int tr = static_cast<int>(threadIdx.y);
  const int m0 = static_cast<int>(blockIdx.y) * kBM + tr * kTM;
  const int n0 = static_cast<int>(blockIdx.x) * kBN + tc * kTN;
  // Some result of this thread lies inside the matrix.
  const bool any_ok = m0 < M && n0 < N;

  __shared__ float As[kBM * kBK];
  __shared__ float Bs[kBK * kBN];

  float res[kTM * kTN] = {};
  float regA[kTM];
  float regB[kTN];
  for (int k0 = 0; k0 < K; k0 += kBK) {
    for (int i = 0; i < kTM; ++i) {
      const int m = m0 + i;
      As[(tr * kTM + i) * kBK + tc] =
          (m < M && k0 + tc < K) ? A[m * K + k0 + tc] : 0.0f;
    }
    for (int j = 0; j < kTN; ++j) {
      const int n = n0 + j;
      Bs[tr * kBN + tc * kTN + j] =
          (k0 + tr < K && n < N) ? B[(k0 + tr) * N + n] : 0.0f;
    }
    __syncthreads();

    if (any_ok) {
      for (int tk = 0; tk < kBK; ++tk) {
        for (int i = 0; i < kTM; ++i) regA[i] = As[(tr * kTM + i) * kBK + tk];
        for (int j = 0; j < kTN; ++j) regB[j] = Bs[tk * kBN + tc * kTN + j];
        for (int i = 0; i < kTM; ++i)
          for (int j = 0; j < kTN; ++j) res[i * kTN + j] += regA[i] * regB[j];
      }
    }
    __syncthreads();
  }

  if (!any_ok) return;
  for (int i = 0; i < kTM && m0 + i < M; ++i) {
    const int m = m0 + i;
    for (int j = 0; j < kTN && n0 + j < N; ++j) {
      const int n = n0 + j;
      C[m * N + n] = alpha * res[i * kTN + j] + beta * C[m * N + n];
    }
  }
}

void sgemm_k5_tiling2d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  dim3 block(kThreadCols, kThreadRows);
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  k5_tiling2d_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k5_tiling2d");
}
