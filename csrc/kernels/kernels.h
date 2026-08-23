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
void sgemm_k3_shared(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C);
void sgemm_k4_tiling1d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C);
void sgemm_k5_tiling2d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C);
void sgemm_k6_vectorized(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C);

// K7 plus the configurations used by the parameter search.
void sgemm_k7_warptile(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c1(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c2(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c3(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c4(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c5(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c6(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c7(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);
void sgemm_k7_c8(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C);


// K8 double buffering and its bounded parameter search.
void sgemm_k8_doublebuffer(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);
void sgemm_k8_c1(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);
void sgemm_k8_c2(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);
void sgemm_k8_c3(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);
void sgemm_k8_c4(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);
void sgemm_k8_c5(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C);

extern const KernelEntry kKernels[];
extern const int kNumKernels;
const KernelEntry* find_kernel(const char* name);
