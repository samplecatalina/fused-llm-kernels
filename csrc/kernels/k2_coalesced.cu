#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K2 - coalesced global access.
//
//  Same computation as K1, same launch shape. The only change is which
//  thread index selects which matrix index: threadIdx.x now selects the
//  column n, threadIdx.y the row m.
//
//  Threads adjacent in x are scheduled together in one warp. Under the K1
//  mapping they touched K and N elements apart in global memory; now, for
//  every k, the threads of a warp read the same element of A, adjacent
//  elements of one row of B, and write adjacent elements of one row of C.
//  Their accesses can be coalesced instead of being issued one by one.
// =====================================================================

namespace {
constexpr int kBlock = 32;  // 32 x 32 = 1024 threads, the per-block limit
}

__global__ void k2_coalesced_kernel(int M, int N, int K, float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B, float beta,
                                    float* __restrict__ C) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int m = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (m >= M || n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
  C[m * N + n] = alpha * acc + beta * C[m * N + n];
}

void sgemm_k2_coalesced(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
  dim3 block(kBlock, kBlock);
  // x spans the columns and y the rows, so the grid dimensions swap too.
  dim3 grid((N + kBlock - 1) / kBlock, (M + kBlock - 1) / kBlock);
  k2_coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k2_coalesced");
}
