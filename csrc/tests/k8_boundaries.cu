#include <algorithm>
#include <cstdio>

#include "../common/matrix.h"
#include "../common/verify.h"
#include "../kernels/kernels.h"

// Nonzero initial C exercises beta, which the zero-initialized benchmark
// reference cannot check. A double-precision host reference is independent
// of the tiled implementation and covers the empty reduction as well.
int main() {
  const int shapes[][3] = {{1, 1, 1},     {3, 5, 7},      {63, 65, 15},
                           {65, 63, 16},  {127, 67, 17},  {129, 65, 31},
                           {129, 68, 32}, {131, 132, 33}, {257, 129, 65},
                           {9, 11, 0}};
  const float scales[][2] = {{1.0f, 0.0f}, {-0.75f, 0.5f}, {0.0f, -1.0f}};
  const SgemmFn kernels[] = {sgemm_k8_doublebuffer, sgemm_k8_c1, sgemm_k8_c2,
                             sgemm_k8_c3,           sgemm_k8_c4, sgemm_k8_c5};
  int checks = 0;
  for (const auto& shape : shapes) {
    const int M = shape[0], N = shape[1], K = shape[2];
    auto A = make_host_matrix(M, K, 11);
    auto B = make_host_matrix(K, N, 23);
    auto C = make_host_matrix(M, N, 37);
    float* dA = device_alloc(std::max<size_t>(A.size(), 1));
    float* dB = device_alloc(std::max<size_t>(B.size(), 1));
    float* dC = device_alloc(C.size());
    if (K) {
      host_to_device(dA, A);
      host_to_device(dB, B);
    }
    for (const auto& scale : scales) {
      std::vector<float> ref(C.size());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double sum = 0.0;
          for (int k = 0; k < K; ++k)
            sum += static_cast<double>(A[m * K + k]) * B[k * N + n];
          ref[m * N + n] = scale[0] * sum + scale[1] * C[m * N + n];
        }
      for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
        host_to_device(dC, C);
        kernels[i](M, N, K, scale[0], dA, dB, scale[1], dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto vr = verify(ref, device_to_host(dC, C.size()));
        if (!vr.passed(1e-3, 1e-5)) {
          std::fprintf(stderr,
                       "FAIL variant=%zu shape=%dx%dx%d alpha=%g beta=%g "
                       "max=%g fro=%g\n",
                       i, M, N, K, scale[0], scale[1], vr.max_rel, vr.fro_rel);
          return 1;
        }
        ++checks;
      }
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }
  std::printf("PASS: %d K8 boundary/epilogue checks\n", checks);
}
