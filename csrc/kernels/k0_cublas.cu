#include "../common/cuda_utils.h"
#include "kernels.h"

// cuBLAS is column-major; our data is row-major. Instead of transposing,
// use the identity:
//     a row-major C(MxN) is byte-for-byte a column-major C^T(NxM)
// and C^T = (A*B)^T = B^T * A^T.
// So passing (B, A) as column-major with dimensions (N, M, K) makes cuBLAS
// write exactly the row-major C we want.
// No copies, no transposes: the baseline has to see the same memory layout as
// the kernels it is compared against, otherwise the comparison is not fair.
void sgemm_k0_cublas(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  static cublasHandle_t handle = nullptr;
  if (!handle) CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                           N, A, K, &beta, C, N));
}
