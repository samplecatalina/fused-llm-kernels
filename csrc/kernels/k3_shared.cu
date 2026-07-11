#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K3 - shared-memory tiling.
//
//  Keeps the coalesced mapping of K2 (threadIdx.x -> column n,
//  threadIdx.y -> row m) and stops reading global memory in the inner
//  loop. For every tile of kTile values of k, the block copies
//      As[tm][tk] = A[m][k0 + tk]     (kTile x kTile)
//      Bs[tk][tn] = B[k0 + tk][n]     (kTile x kTile)
//  into shared memory, and the inner products for that tile only touch
//  the shared copies. Global memory is read in batches of kTile, not once
//  per k.
//
//  Each thread (tm, tn) fills exactly one cell of each shared array:
//  As[tm][tn] and Bs[tm][tn]. That covers both arrays only because the
//  tile is square (rows of A, columns of B and the k-extent are all
//  kTile), which is why this rung fixes all three to the same value.
//
//  Threads outside the matrix still load and still synchronize: the cell
//  a thread loads belongs to a row or column that in-range threads of the
//  same block read. They only skip the final write.
// =====================================================================

namespace {
constexpr int kTile = 32;  // rows, columns and k-extent of a tile
}

__global__ void k3_shared_kernel(int M, int N, int K, float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B, float beta,
                                 float* __restrict__ C) {
  const int tn = static_cast<int>(threadIdx.x);
  const int tm = static_cast<int>(threadIdx.y);
  const int n = static_cast<int>(blockIdx.x) * kTile + tn;
  const int m = static_cast<int>(blockIdx.y) * kTile + tm;
  const bool row_ok = m < M;
  const bool col_ok = n < N;

  __shared__ float As[kTile * kTile];
  __shared__ float Bs[kTile * kTile];

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    // Out-of-range cells are zero, so they add nothing to any product.
    As[tm * kTile + tn] = (row_ok && k0 + tn < K) ? A[m * K + k0 + tn] : 0.0f;
    Bs[tm * kTile + tn] =
        (k0 + tm < K && col_ok) ? B[(k0 + tm) * N + n] : 0.0f;
    // Wait until the whole block has loaded the tile.
    __syncthreads();

    if (row_ok && col_ok)
      for (int tk = 0; tk < kTile; ++tk)
        acc += As[tm * kTile + tk] * Bs[tk * kTile + tn];
    // Wait until every thread is done reading before the next tile
    // overwrites the shared arrays.
    __syncthreads();
  }

  if (row_ok && col_ok) C[m * N + n] = alpha * acc + beta * C[m * N + n];
}

void sgemm_k3_shared(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  dim3 block(kTile, kTile);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  k3_shared_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k3_shared");
}
