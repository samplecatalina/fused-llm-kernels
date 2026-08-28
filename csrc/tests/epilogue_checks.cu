#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../common/matrix.h"
#include "../common/verify.h"
#include "../kernels/kernels.h"

// Epilogue kernels against a double-precision host reference. Covers row
// tails (N not a multiple of 4), partial tiles, K = 1, and alpha large
// enough to drive SiLU deep into both tails. D starts nonzero: neither path
// may read it.
namespace {
double silu(double x) {
  const double e = std::exp(-std::fabs(x));
  return x * (x >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e));
}
}  // namespace

int main() {
  const int shapes[][3] = {{1, 1, 1},      {3, 5, 7},      {63, 65, 15},
                           {65, 63, 16},   {127, 67, 17},  {129, 65, 31},
                           {129, 68, 32},  {131, 132, 33}, {257, 129, 65},
                           {300, 1027, 3}, {5, 2048, 40}};
  const float alphas[] = {1.0f, -0.75f, 40.0f};
  int checks = 0;
  for (const auto& shape : shapes) {
    const int M = shape[0], N = shape[1], K = shape[2];
    auto A = make_host_matrix(M, K, 11);
    auto B = make_host_matrix(K, N, 23);
    auto bias = make_host_matrix(1, N, 31);
    auto D0 = make_host_matrix(M, N, 37);
    float* dA = device_alloc(A.size());
    float* dB = device_alloc(B.size());
    float* dBias = device_alloc(bias.size());
    float* dD = device_alloc(D0.size());
    host_to_device(dA, A);
    host_to_device(dB, B);
    host_to_device(dBias, bias);
    for (float alpha : alphas) {
      std::vector<float> ref(D0.size());
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double sum = 0.0;
          for (int k = 0; k < K; ++k)
            sum += static_cast<double>(A[m * K + k]) * B[k * N + n];
          ref[m * N + n] = static_cast<float>(silu(alpha * sum + bias[n]));
        }
      for (int i = 0; i < kNumEpilogues; ++i) {
        host_to_device(dD, D0);
        kEpilogues[i].fn(M, N, K, alpha, dA, dB, dBias, dD);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto vr = verify(ref, device_to_host(dD, D0.size()));
        if (!vr.passed(1e-3, 1e-5)) {
          std::fprintf(
              stderr, "FAIL %s shape=%dx%dx%d alpha=%g max=%g fro=%g\n",
              kEpilogues[i].name, M, N, K, alpha, vr.max_rel, vr.fro_rel);
          return 1;
        }
        ++checks;
      }
    }
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dBias));
    CUDA_CHECK(cudaFree(dD));
  }
  std::printf("PASS: %d epilogue checks\n", checks);
}
