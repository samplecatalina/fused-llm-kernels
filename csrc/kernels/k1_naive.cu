#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K1 - naive SGEMM. Not implemented yet; this is a stub that compiles
//  but produces wrong results (make test fails on it by design).
//
//  Spec: C[m][n] = alpha * sum_k A[m][k] * B[k][n] + beta * C[m][n]
//        All operands are row-major: A is MxK, B is KxN, C is MxN.
//          A[m][k] -> linear index m * K + k
//          B[k][n] -> linear index k * N + n
//          C[m][n] -> linear index m * N + n
//
//  Scope of this rung: one thread computes one element of C. No shared
//  memory, no blocking. It exists to establish the floor of the ladder
//  and to produce the reference ncu report every later rung is read
//  against.
// =====================================================================

__global__ void k1_naive_kernel(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  // TODO: derive the (m, n) this thread owns
  // TODO: bounds guard
  // TODO: accumulate along k
  // TODO: C[...] = alpha * acc + beta * C[...]
  (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
}

void sgemm_k1_naive(int M, int N, int K, float alpha, const float* A,
                    const float* B, float beta, float* C) {
  // TODO: pick the block shape and derive the grid from M/N (round up)
  dim3 block(1, 1);
  dim3 grid(1, 1);
  k1_naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k1_naive");
}
