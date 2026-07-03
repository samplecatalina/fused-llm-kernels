#pragma once
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_utils.h"

// Row-major host matrix. The fixed seed keeps inputs identical across runs.
inline std::vector<float> make_host_matrix(int rows, int cols, unsigned seed) {
  std::vector<float> v(static_cast<size_t>(rows) * cols);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(gen);
  return v;
}

inline float* device_alloc(size_t n_elems) {
  float* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n_elems * sizeof(float)));
  return p;
}

inline void host_to_device(float* d, const std::vector<float>& h) {
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

inline std::vector<float> device_to_host(const float* d, size_t n_elems) {
  std::vector<float> h(n_elems);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return h;
}
