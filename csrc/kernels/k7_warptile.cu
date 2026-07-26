#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  K7 - warp tiling: a second blocking level between the block tile and
//  the per-thread register block.
//
//  In K6 the 32 threads of a warp were spread over TM rows and the full
//  width of the block tile, so for one k they touched TM + BN distinct
//  shared-memory locations. Here each warp owns a WM x WN sub-tile, and
//  its 32 threads cover it exactly: (WM/TM) x (WN/TN) == 32. For one k a
//  warp now touches WM + WN locations instead, a smaller and denser
//  footprint.
//
//  Loading is unchanged from K6 (transposed A tile, float4 loads with a
//  scalar fallback when K or N is not a multiple of 4).
//
//  The kernel is a template over (BM, BN, BK, TM, TN, WM, WN) so the
//  parameter search can instantiate several configurations from one
//  source; every configuration goes through the same correctness check.
// =====================================================================

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
  static constexpr int kSharedBytes = (BM * BK + BK * BN) * 4;
};

}  // namespace

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
__global__ void k7_warptile_kernel(int M, int N, int K, float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B, float beta,
                                   float* __restrict__ C) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  static_assert(BM % WM == 0 && BN % WN == 0, "warp tiles must divide the block tile");
  static_assert(WM % TM == 0 && WN % TN == 0, "thread tiles must divide the warp tile");
  static_assert(C_::kLaneRows * C_::kLaneCols == 32,
                "the 32 threads of a warp must cover the warp tile exactly");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "inner loads read float4");
  static_assert(C_::kAChunks % C_::kThreads == 0 && C_::kBChunks % C_::kThreads == 0,
                "tile loading must divide evenly among the threads");
  static_assert(C_::kSharedBytes <= 48 * 1024, "shared memory per block exceeds the default limit");
  static_assert(C_::kThreads <= 1024, "threads per block exceeds the hardware limit");

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int wm = (warp / C_::kWarpsN) * WM;  // this warp's row offset in the tile
  const int wn = (warp % C_::kWarpsN) * WN;  // and its column offset
  const int lr = (lane / C_::kLaneCols) * TM;  // lane offset inside the warp tile
  const int lc = (lane % C_::kLaneCols) * TN;

  const int m0 = static_cast<int>(blockIdx.y) * BM;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int rm = m0 + wm + lr;  // first result row of this thread
  const int rn = n0 + wn + lc;  // first result column
  const bool any_ok = rm < M && rn < N;

  __shared__ alignas(16) float As[BK * BM];  // transposed: As[k][row]
  __shared__ alignas(16) float Bs[BK * BN];

  const bool vec_a = (K % 4 == 0);
  const bool vec_b = (N % 4 == 0);

  float res[TM * TN] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int s = 0; s < C_::kAPerThread; ++s) {
      const int chunk = tid * C_::kAPerThread + s;
      const int row = chunk / C_::kAChunksPerRow;
      const int kc = (chunk % C_::kAChunksPerRow) * 4;
      const int m = m0 + row;
      float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (m < M) {
        if (vec_a && k0 + kc + 3 < K) {
          const float4 t = *reinterpret_cast<const float4*>(&A[m * K + k0 + kc]);
          v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
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
          v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
        } else {
          for (int e = 0; e < 4; ++e) {
            const int nn = n + e;
            v[e] = (nn < N) ? B[k * N + nn] : 0.0f;
          }
        }
      }
      *reinterpret_cast<float4*>(&Bs[kr * BN + nc]) =
          make_float4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();

    if (any_ok) {
      for (int tk = 0; tk < BK; ++tk) {
        float ra[TM];
        float rb[TN];
        for (int e = 0; e < TM; e += 4) {
          const float4 t =
              *reinterpret_cast<const float4*>(&As[tk * BM + wm + lr + e]);
          ra[e] = t.x; ra[e + 1] = t.y; ra[e + 2] = t.z; ra[e + 3] = t.w;
        }
        for (int e = 0; e < TN; e += 4) {
          const float4 t =
              *reinterpret_cast<const float4*>(&Bs[tk * BN + wn + lc + e]);
          rb[e] = t.x; rb[e + 1] = t.y; rb[e + 2] = t.z; rb[e + 3] = t.w;
        }
        for (int i = 0; i < TM; ++i)
          for (int j = 0; j < TN; ++j) res[i * TN + j] += ra[i] * rb[j];
      }
    }
    __syncthreads();
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

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
void launch(int M, int N, int K, float alpha, const float* A, const float* B,
            float beta, float* C, const char* where) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  dim3 block(C_::kThreads);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  k7_warptile_kernel<BM, BN, BK, TM, TN, WM, WN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  check_launch(where);
}

}  // namespace

// Default configuration: the one the parameter search selected -
// 128x64 results per block, 8x4 per thread, 32x32 warp tiles. A narrower
// block tile needs fewer registers per thread, which lets more blocks sit
// on an SM; the search shows that matters more here than the extra reuse
// a wider tile would buy.
void sgemm_k7_warptile(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  launch<128, 64, 16, 8, 4, 32, 32>(M, N, K, alpha, A, B, beta, C, "k7");
}

// Search configurations. c1 matches K6's tile and thread count exactly, so
// the difference between K6 and c1 is the warp-level blocking alone.
void sgemm_k7_c1(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 128, 16, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k7c1");
}
void sgemm_k7_c2(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 128, 16, 8, 8, 64, 32>(M, N, K, alpha, A, B, beta, C, "k7c2");
}
void sgemm_k7_c3(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 64, 16, 8, 4, 32, 32>(M, N, K, alpha, A, B, beta, C, "k7c3");
}
void sgemm_k7_c4(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<64, 64, 16, 4, 8, 32, 32>(M, N, K, alpha, A, B, beta, C, "k7c4");
}
void sgemm_k7_c5(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 128, 8, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k7c5");
}
void sgemm_k7_c6(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 128, 32, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k7c6");
}
void sgemm_k7_c7(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<256, 128, 16, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k7c7");
}
void sgemm_k7_c8(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  launch<128, 256, 16, 8, 8, 32, 64>(M, N, K, alpha, A, B, beta, C, "k7c8");
}
