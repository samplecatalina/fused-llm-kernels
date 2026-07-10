#pragma once

// One signature for every rung: row-major, C = alpha * A(MxK) * B(KxN) + beta * C(MxN).
// Every kernel must accept arbitrary M/N/K, including sizes that are not a
// multiple of the tile shape.
using SgemmFn = void (*)(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C);

struct KernelEntry {
  const char* name;
  const char* desc;
  SgemmFn fn;
};

void sgemm_k0_cublas(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C);
void sgemm_k1_naive(int M, int N, int K, float alpha, const float* A,
                    const float* B, float beta, float* C);
void sgemm_k2_coalesced(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C);

extern const KernelEntry kKernels[];
extern const int kNumKernels;
const KernelEntry* find_kernel(const char* name);
