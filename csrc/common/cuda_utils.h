#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__, __LINE__,     \
                   #call, cudaGetErrorString(err_));                          \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[cuBLAS] %s:%d %s -> status %d\n", __FILE__,      \
                   __LINE__, #call, static_cast<int>(st_));                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Call after every launch: catches bad launch configurations (illegal block
// shape, shared memory over the limit) as well as execution errors.
inline void check_launch(const char* where) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[launch] %s -> %s\n", where, cudaGetErrorString(e));
    std::exit(EXIT_FAILURE);
  }
}
