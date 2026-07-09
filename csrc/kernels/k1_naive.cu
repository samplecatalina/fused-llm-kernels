#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K1 - naive SGEMM: one thread computes one element of C.
//
//  Spec: C[m][n] = alpha * sum_k A[m][k] * B[k][n] + beta * C[m][n]
//        All operands are row-major: A is MxK, B is KxN, C is MxN.
//          A[m][k] -> linear index m * K + k
//          B[k][n] -> linear index k * N + n
//          C[m][n] -> linear index m * N + n
//
//  No shared memory, no blocking. This rung is the floor of the ladder
//  and the reference profile every later rung is read against.
//
//  Thread mapping: threadIdx.x selects the row m, threadIdx.y the column
//  n. Threads that are adjacent in x therefore land on adjacent rows of
//  C: they read different rows of A and write different rows of C, which
//  are K and N elements apart in global memory. This mapping is
//  deliberately the uncoalesced one; swapping it is the next rung.
// =====================================================================

namespace {
constexpr int kBlock = 32;  // 32 x 32 = 1024 threads, the per-block limit
}

__global__ void k1_naive_kernel(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  const int m = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  // The grid is rounded up, so the last row and column of blocks contain
  // threads that fall outside the matrix.
  if (m >= M || n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
  C[m * N + n] = alpha * acc + beta * C[m * N + n];
}

void sgemm_k1_naive(int M, int N, int K, float alpha, const float* A,
                    const float* B, float beta, float* C) {
  dim3 block(kBlock, kBlock);
  dim3 grid((M + kBlock - 1) / kBlock, (N + kBlock - 1) / kBlock);
  k1_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k1_naive");
}
