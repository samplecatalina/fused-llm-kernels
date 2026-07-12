#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K4 - 1D thread tiling: each thread computes kTM results of C.
//
//  Same 32x32x32 shared-memory tiles as K3. Instead of one thread per
//  element, a thread owns a run of kTM consecutive rows in one column of
//  C. For every k it reads Bs[tk][tn] into a register once and reuses it
//  for all kTM results, so a block runs 32 x (32 / kTM) threads instead
//  of 32 x 32, each doing kTM times the work.
//
//  Loading still covers each shared cell exactly once: thread (tr, tn)
//  owns rows tr*kTM .. tr*kTM+kTM-1 of the tile, and loads As at those
//  rows in column tn and Bs at those k-rows in column tn.
// =====================================================================

namespace {
constexpr int kTile = 32;                // rows, columns and k-extent of a tile
constexpr int kTM = 8;                   // results per thread, along the rows
constexpr int kThreadRows = kTile / kTM; // thread rows per block
}

__global__ void k4_tiling1d_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
  const int tn = static_cast<int>(threadIdx.x);
  const int tr = static_cast<int>(threadIdx.y);
  const int n = static_cast<int>(blockIdx.x) * kTile + tn;
  const int row0 = tr * kTM;  // first tile row owned by this thread
  const int m0 = static_cast<int>(blockIdx.y) * kTile + row0;
  const bool col_ok = n < N;

  __shared__ float As[kTile * kTile];
  __shared__ float Bs[kTile * kTile];

  float res[kTM] = {};
  for (int k0 = 0; k0 < K; k0 += kTile) {
    for (int r = 0; r < kTM; ++r) {
      const int m = m0 + r;
      const int tk = row0 + r;  // the k-rows of Bs this thread loads
      As[(row0 + r) * kTile + tn] =
          (m < M && k0 + tn < K) ? A[m * K + k0 + tn] : 0.0f;
      Bs[tk * kTile + tn] = (k0 + tk < K && col_ok) ? B[(k0 + tk) * N + n] : 0.0f;
    }
    __syncthreads();

    if (col_ok) {
      for (int tk = 0; tk < kTile; ++tk) {
        const float b = Bs[tk * kTile + tn];
        for (int r = 0; r < kTM; ++r) res[r] += As[(row0 + r) * kTile + tk] * b;
      }
    }
    __syncthreads();
  }

  if (!col_ok) return;
  for (int r = 0; r < kTM; ++r) {
    const int m = m0 + r;
    if (m < M) C[m * N + n] = alpha * res[r] + beta * C[m * N + n];
  }
}

void sgemm_k4_tiling1d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  dim3 block(kTile, kThreadRows);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  k4_tiling1d_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k4_tiling1d");
}
