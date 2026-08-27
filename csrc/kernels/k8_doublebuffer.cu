#include "../common/cuda_utils.h"
#include "kernels.h"

// K8: double buffering. Two shared tiles alternate: while the FMAs for tile
// t read one buffer, the global reads for tile t+1 fill the other, so a
// single barrier per tile both publishes the next tile and retires the
// readers of the current one (K7 needs two). No warp may overwrite a buffer
// still being read by another warp.
//
// Two ways to stage the next tile are kept:
//   - direct (default): global loads are stored straight into the other
//     buffer. Register usage matches K7 at the same geometry.
//   - register prefetch (k8c1..k8c5): loads land in registers first and are
//     stored after the FMAs. This orders the stores after the compute but
//     holds extra register state and copies every value twice.
// Neither form guarantees that the global loads and FMAs overlap in time;
// C++ order only exposes independent work to the scheduler.

namespace {

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
struct Cfg {
  static constexpr int kWarpsM = BM / WM;
  static constexpr int kWarpsN = BN / WN;
  static constexpr int kWarps = kWarpsM * kWarpsN;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kLaneRows = WM / TM;  // threads along rows in a warp
  static constexpr int kLaneCols = WN / TN;  // threads along columns
  static constexpr int kAChunks = (BM * BK) / 4;
  static constexpr int kBChunks = (BK * BN) / 4;
  static constexpr int kAPerThread = kAChunks / kThreads;
  static constexpr int kBPerThread = kBChunks / kThreads;
  static constexpr int kAChunksPerRow = BK / 4;
  static constexpr int kBChunksPerRow = BN / 4;
  static constexpr int kSharedBytes = 2 * (BM * BK + BK * BN) * 4;
};

}  // namespace
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
__device__ __forceinline__ void prefetch(int M, int N, int K, const float* A,
                                         const float* B, int m0, int n0, int k0,
                                         float4* next_a, float4* next_b) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  const int tid = static_cast<int>(threadIdx.x);
  const bool vec_a = K % 4 == 0;
  const bool vec_b = N % 4 == 0;
  for (int s = 0; s < C_::kAPerThread; ++s) {
    const int chunk = tid * C_::kAPerThread + s;
    const int row = chunk / C_::kAChunksPerRow;
    const int kc = (chunk % C_::kAChunksPerRow) * 4;
    const int m = m0 + row;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (m < M) {
      if (vec_a && k0 + kc + 3 < K) {
        const float4 t = *reinterpret_cast<const float4*>(&A[m * K + k0 + kc]);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
      } else {
        for (int e = 0; e < 4; ++e) {
          const int k = k0 + kc + e;
          v[e] = (k < K) ? A[m * K + k] : 0.0f;
        }
      }
    }
    next_a[s] = make_float4(v[0], v[1], v[2], v[3]);
  }

  for (int s = 0; s < C_::kBPerThread; ++s) {
    const int chunk = tid * C_::kBPerThread + s;
    const int kr = chunk / C_::kBChunksPerRow;
    const int nc = (chunk % C_::kBChunksPerRow) * 4;
    const int k = k0 + kr;
    const int n = n0 + nc;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (k < K) {
      if (vec_b && n + 3 < N) {
        const float4 t = *reinterpret_cast<const float4*>(&B[k * N + n]);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
      } else {
        for (int e = 0; e < 4; ++e) {
          const int nn = n + e;
          v[e] = (nn < N) ? B[k * N + nn] : 0.0f;
        }
      }
    }
    next_b[s] = make_float4(v[0], v[1], v[2], v[3]);
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
__device__ __forceinline__ void stage(const float4* next_a,
                                      const float4* next_b, float* As,
                                      float* Bs) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int s = 0; s < C_::kAPerThread; ++s) {
    const int chunk = tid * C_::kAPerThread + s;
    const int row = chunk / C_::kAChunksPerRow;
    const int kc = (chunk % C_::kAChunksPerRow) * 4;
    const float4 v = next_a[s];
    As[(kc + 0) * BM + row] = v.x;
    As[(kc + 1) * BM + row] = v.y;
    As[(kc + 2) * BM + row] = v.z;
    As[(kc + 3) * BM + row] = v.w;
  }
#pragma unroll
  for (int s = 0; s < C_::kBPerThread; ++s) {
    const int chunk = tid * C_::kBPerThread + s;
    *reinterpret_cast<float4*>(&Bs[chunk * 4]) = next_b[s];
  }
}
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN,
          bool MaxShared>
__global__ void k8_doublebuffer_kernel(int M, int N, int K, float alpha,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B, float beta,
                                       float* __restrict__ C) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  static_assert(BM % WM == 0 && BN % WN == 0,
                "warp tiles must divide the block tile");
  static_assert(WM % TM == 0 && WN % TN == 0,
                "thread tiles must divide the warp tile");
  static_assert(C_::kLaneRows * C_::kLaneCols == 32,
                "the 32 threads of a warp must cover the warp tile exactly");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "inner loads read float4");
  static_assert(
      C_::kAChunks % C_::kThreads == 0 && C_::kBChunks % C_::kThreads == 0,
      "tile loading must divide evenly among the threads");
  static_assert(C_::kSharedBytes <= 48 * 1024,
                "shared memory per block exceeds the default limit");
  static_assert(C_::kThreads <= 1024,
                "threads per block exceeds the hardware limit");

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int wm =
      (warp / C_::kWarpsN) * WM;  // this warp's row offset in the tile
  const int wn = (warp % C_::kWarpsN) * WN;  // and its column offset
  const int lr =
      (lane / C_::kLaneCols) * TM;  // lane offset inside the warp tile
  const int lc = (lane % C_::kLaneCols) * TN;

  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int rm = m0 + wm + lr;  // first result row of this thread
  const int rn = n0 + wn + lc;  // first result column
  const bool any_ok = rm < M && rn < N;

  __shared__ alignas(16) float As[2][BK * BM];  // transposed: As[k][row]
  __shared__ alignas(16) float Bs[2][BK * BN];

  float res[TM * TN] = {};
  float4 next_a[C_::kAPerThread];
  float4 next_b[C_::kBPerThread];
  if (K > 0) {
    prefetch<BM, BN, BK, TM, TN, WM, WN>(M, N, K, A, B, m0, n0, 0, next_a,
                                         next_b);
    stage<BM, BN, BK, TM, TN, WM, WN>(next_a, next_b, As[0], Bs[0]);
    __syncthreads();
  }
  int current = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next)
      prefetch<BM, BN, BK, TM, TN, WM, WN>(M, N, K, A, B, m0, n0, k0 + BK,
                                           next_a, next_b);
    if (any_ok) {
      for (int tk = 0; tk < BK; ++tk) {
        float ra[TM];
        float rb[TN];
        for (int e = 0; e < TM; e += 4) {
          const float4 t = *reinterpret_cast<const float4*>(
              &As[current][tk * BM + wm + lr + e]);
          ra[e] = t.x;
          ra[e + 1] = t.y;
          ra[e + 2] = t.z;
          ra[e + 3] = t.w;
        }
        for (int e = 0; e < TN; e += 4) {
          const float4 t = *reinterpret_cast<const float4*>(
              &Bs[current][tk * BN + wn + lc + e]);
          rb[e] = t.x;
          rb[e + 1] = t.y;
          rb[e + 2] = t.z;
          rb[e + 3] = t.w;
        }
        for (int i = 0; i < TM; ++i)
          for (int j = 0; j < TN; ++j) res[i * TN + j] += ra[i] * rb[j];
      }
    }

    if (has_next) {
      stage<BM, BN, BK, TM, TN, WM, WN>(next_a, next_b, As[current ^ 1],
                                        Bs[current ^ 1]);
      // Publish the next tile and retire all readers of the current one.
      __syncthreads();
      current ^= 1;
    }
  }

  if (!any_ok) return;
  for (int i = 0; i < TM && rm + i < M; ++i) {
    const int m = rm + i;
    for (int j = 0; j < TN && rn + j < N; ++j) {
      const int n = rn + j;
      C[m * N + n] = alpha * res[i * TN + j] + beta * C[m * N + n];
    }
  }
}

// Direct variant: global loads for the next tile are stored straight into
// the other shared buffer, so no loaded value is held in registers across
// the FMAs. A single load call site (the loop starts one tile early) keeps
// the register count equal to K7's at the same geometry.
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
__device__ __forceinline__ void load_direct(int M, int N, int K, const float* A,
                                            const float* B, int m0, int n0,
                                            int k0, float* As, float* Bs) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  const int tid = static_cast<int>(threadIdx.x);
  const bool vec_a = K % 4 == 0;
  const bool vec_b = N % 4 == 0;
  for (int s = 0; s < C_::kAPerThread; ++s) {
    const int chunk = tid * C_::kAPerThread + s;
    const int row = chunk / C_::kAChunksPerRow;
    const int kc = (chunk % C_::kAChunksPerRow) * 4;
    const int m = m0 + row;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (m < M) {
      if (vec_a && k0 + kc + 3 < K) {
        const float4 t = *reinterpret_cast<const float4*>(&A[m * K + k0 + kc]);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
      } else {
        for (int e = 0; e < 4; ++e) {
          const int k = k0 + kc + e;
          v[e] = (k < K) ? A[m * K + k] : 0.0f;
        }
      }
    }
    for (int e = 0; e < 4; ++e) As[(kc + e) * BM + row] = v[e];
  }

  for (int s = 0; s < C_::kBPerThread; ++s) {
    const int chunk = tid * C_::kBPerThread + s;
    const int kr = chunk / C_::kBChunksPerRow;
    const int nc = (chunk % C_::kBChunksPerRow) * 4;
    const int k = k0 + kr;
    const int n = n0 + nc;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (k < K) {
      if (vec_b && n + 3 < N) {
        const float4 t = *reinterpret_cast<const float4*>(&B[k * N + n]);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
      } else {
        for (int e = 0; e < 4; ++e) {
          const int nn = n + e;
          v[e] = (nn < N) ? B[k * N + nn] : 0.0f;
        }
      }
    }
    *reinterpret_cast<float4*>(&Bs[chunk * 4]) =
        make_float4(v[0], v[1], v[2], v[3]);
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
__global__ void k8_direct_kernel(int M, int N, int K, float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B, float beta,
                                 float* __restrict__ C) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  static_assert(C_::kLaneRows * C_::kLaneCols == 32,
                "the 32 threads of a warp must cover the warp tile exactly");
  static_assert(
      C_::kAChunks % C_::kThreads == 0 && C_::kBChunks % C_::kThreads == 0,
      "tile loading must divide evenly among the threads");
  static_assert(C_::kSharedBytes <= 48 * 1024,
                "shared memory per block exceeds the default limit");

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int wm = (warp / C_::kWarpsN) * WM;
  const int wn = (warp % C_::kWarpsN) * WN;
  const int lr = (lane / C_::kLaneCols) * TM;
  const int lc = (lane % C_::kLaneCols) * TN;

  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int rm = m0 + wm + lr;
  const int rn = n0 + wn + lc;
  const bool any_ok = rm < M && rn < N;

  __shared__ alignas(16) float As[2][BK * BM];  // transposed: As[k][row]
  __shared__ alignas(16) float Bs[2][BK * BN];

  float res[TM * TN] = {};
  // Iteration k0 loads tile k0+BK into buffer ((k0+BK)/BK)&1 and computes
  // tile k0 from the other buffer. The barrier publishes the new tile and
  // retires every reader of the old one before it is overwritten.
  for (int k0 = -BK; k0 < K; k0 += BK) {
    const int nk = k0 + BK;
    if (nk < K)
      load_direct<BM, BN, BK, TM, TN, WM, WN>(
          M, N, K, A, B, m0, n0, nk, As[(nk / BK) & 1], Bs[(nk / BK) & 1]);
    if (k0 >= 0 && any_ok) {
      const int cur = (k0 / BK) & 1;
      for (int tk = 0; tk < BK; ++tk) {
        float ra[TM];
        float rb[TN];
        for (int e = 0; e < TM; e += 4) {
          const float4 t =
              *reinterpret_cast<const float4*>(&As[cur][tk * BM + wm + lr + e]);
          ra[e] = t.x;
          ra[e + 1] = t.y;
          ra[e + 2] = t.z;
          ra[e + 3] = t.w;
        }
        for (int e = 0; e < TN; e += 4) {
          const float4 t =
              *reinterpret_cast<const float4*>(&Bs[cur][tk * BN + wn + lc + e]);
          rb[e] = t.x;
          rb[e + 1] = t.y;
          rb[e + 2] = t.z;
          rb[e + 3] = t.w;
        }
        for (int i = 0; i < TM; ++i)
          for (int j = 0; j < TN; ++j) res[i * TN + j] += ra[i] * rb[j];
      }
    }
    if (nk < K) __syncthreads();
  }

  if (!any_ok) return;
  for (int i = 0; i < TM && rm + i < M; ++i) {
    const int m = rm + i;
    for (int j = 0; j < TN && rn + j < N; ++j) {
      const int n = rn + j;
      C[m * N + n] = alpha * res[i * TN + j] + beta * C[m * N + n];
    }
  }
}

namespace {
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN,
          bool MaxShared = false>
void launch(int M, int N, int K, float alpha, const float* A, const float* B,
            float beta, float* C, const char* where) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  if constexpr (MaxShared) {
    // A distinct template instance keeps this hint separate from the control.
    static const bool configured = [] {
      CUDA_CHECK(cudaFuncSetAttribute(
          k8_doublebuffer_kernel<BM, BN, BK, TM, TN, WM, WN, MaxShared>,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          cudaSharedmemCarveoutMaxShared));
      return true;
    }();
    (void)configured;
  }
  dim3 block(C_::kThreads);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  k8_doublebuffer_kernel<BM, BN, BK, TM, TN, WM, WN, MaxShared>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch(where);
}
}  // namespace

// Default: direct stores at the K7 geometry. It keeps K7's register count
// (so four blocks still fit on an SM) and avoids the extra register copy of
// the prefetch variants below, which are kept as the searched alternatives.
void sgemm_k8_doublebuffer(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  using C_ = Cfg<128, 64, 16, 8, 4, 32, 32>;
  dim3 block(C_::kThreads);
  dim3 grid((N + 64 - 1) / 64, (M + 128 - 1) / 128);
  k8_direct_kernel<128, 64, 16, 8, 4, 32, 32>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch("k8");
}
void sgemm_k8_c1(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 64, 16, 8, 4, 32, 32>(M, N, K, alpha, A, B, beta, C, "k8c1");
}
void sgemm_k8_c2(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<64, 64, 16, 4, 8, 32, 32>(M, N, K, alpha, A, B, beta, C, "k8c2");
}
void sgemm_k8_c3(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 64, 16, 8, 4, 32, 32, true>(M, N, K, alpha, A, B, beta, C,
                                          "k8c3");
}
void sgemm_k8_c4(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 128, 16, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k8c4");
}
void sgemm_k8_c5(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 64, 32, 8, 4, 32, 32>(M, N, K, alpha, A, B, beta, C, "k8c5");
}
