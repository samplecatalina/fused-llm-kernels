#include <cmath>
#include <cstddef>

#include "../common/cuda_utils.h"
#include "kernels.h"

// =====================================================================
//  E track - fused epilogue: D = SiLU(alpha * A*B + bias), bias broadcast
//  over columns, SiLU(x) = x * sigmoid(x).
//
//  e1 (fused) runs K7's tiled GEMM and applies bias and SiLU in the
//  register block, right before the result is stored in D.
//
//  e0 (unfused) runs the same GEMM, stores alpha * A*B in an intermediate
//  M x N buffer, and then launches a separate element-wise kernel that
//  reads that buffer, adds the bias, applies SiLU and stores D. The extra
//  M x N store, the extra M x N load and the extra launch are what fusion
//  removes.
//
//  e0 deliberately does not call K7: K7 evaluates alpha*res + beta*C and so
//  loads C even when beta is 0, which would charge the unfused path for a
//  load the fused path does not have. Both paths share one template here and
//  differ only in the write-back.
// =====================================================================

namespace {

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN>
struct Cfg {
  static constexpr int kWarpsM = BM / WM;
  static constexpr int kWarpsN = BN / WN;
  static constexpr int kWarps = kWarpsM * kWarpsN;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kLaneRows = WM / TM;
  static constexpr int kLaneCols = WN / TN;
  static constexpr int kAChunks = (BM * BK) / 4;
  static constexpr int kBChunks = (BK * BN) / 4;
  static constexpr int kAPerThread = kAChunks / kThreads;
  static constexpr int kBPerThread = kBChunks / kThreads;
  static constexpr int kAChunksPerRow = BK / 4;
  static constexpr int kBChunksPerRow = BN / 4;
  static constexpr int kSharedBytes = (BM * BK + BK * BN) * 4;
};

// Element-wise kernel: one thread per float4 of a row.
constexpr int kPassThreads = 1024;

}  // namespace

// Numerically stable: exp(-|x|) never overflows.
__host__ __device__ __forceinline__ float silu(float x) {
  const float e = std::exp(-std::fabs(x));
  const float s = (x >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
  return x * s;
}

// Shared by both paths; the name contains "e0" and "e1" so that
// `make profile K=e0` and `K=e1` both select it.
template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, bool Act>
__global__ void e0e1_gemm_kernel(int M, int N, int K, float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 const float* __restrict__ bias,
                                 float* __restrict__ D) {
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
  const int wm = (warp / C_::kWarpsN) * WM;
  const int wn = (warp % C_::kWarpsN) * WN;
  const int lr = (lane / C_::kLaneCols) * TM;
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

  // Tile loading, barriers and accumulation are K7's, unchanged.
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
          const float4 t =
              *reinterpret_cast<const float4*>(&A[m * K + k0 + kc]);
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
          ra[e] = t.x;
          ra[e + 1] = t.y;
          ra[e + 2] = t.z;
          ra[e + 3] = t.w;
        }
        for (int e = 0; e < TN; e += 4) {
          const float4 t =
              *reinterpret_cast<const float4*>(&Bs[tk * BN + wn + lc + e]);
          rb[e] = t.x;
          rb[e + 1] = t.y;
          rb[e + 2] = t.z;
          rb[e + 3] = t.w;
        }
        for (int i = 0; i < TM; ++i)
          for (int j = 0; j < TN; ++j) res[i * TN + j] += ra[i] * rb[j];
      }
    }
    __syncthreads();
  }

  if (!any_ok) return;
  if constexpr (Act) {
    // bias is indexed by the global column rn + j, not by the tile offset.
    float b[TN];
    for (int e = 0; e < TN; e += 4) {
      if (rn + e + 3 < N) {
        const float4 t = *reinterpret_cast<const float4*>(&bias[rn + e]);
        b[e] = t.x;
        b[e + 1] = t.y;
        b[e + 2] = t.z;
        b[e + 3] = t.w;
      } else {
        for (int j = e; j < e + 4; ++j)
          b[j] = (rn + j < N) ? bias[rn + j] : 0.0f;
      }
    }
    for (int i = 0; i < TM && rm + i < M; ++i) {
      const int m = rm + i;
      for (int j = 0; j < TN && rn + j < N; ++j)
        D[m * N + rn + j] = silu(alpha * res[i * TN + j] + b[j]);
    }
  } else {
    // No beta term: D is not loaded.
    for (int i = 0; i < TM && rm + i < M; ++i) {
      const int m = rm + i;
      for (int j = 0; j < TN && rn + j < N; ++j)
        D[m * N + rn + j] = alpha * res[i * TN + j];
    }
  }
}

// D[m][n] = SiLU(C[m][n] + bias[n]). All threads of a block work on the
// same row, four elements each, so access stays contiguous. Each thread
// handles columns n0..n0+3 only; the scalar path covers row tails and rows
// whose start is not 4-aligned.
__global__ void e0_bias_silu_kernel(int M, int N, const float* __restrict__ C,
                                    const float* __restrict__ bias,
                                    float* __restrict__ D) {
  const int m = static_cast<int>(blockIdx.y);
  const int chunk = static_cast<int>(blockIdx.x) * kPassThreads +
                    static_cast<int>(threadIdx.x);
  const int n0 = 4 * chunk;
  if (m >= M || n0 >= N) return;
  const size_t row = static_cast<size_t>(m) * N;
  // float4 global access needs an offset that is a multiple of 4: row
  // starts are aligned only when N is.
  if (N % 4 == 0 && n0 + 3 < N) {
    const float4 c = *reinterpret_cast<const float4*>(&C[row + n0]);
    const float4 b = *reinterpret_cast<const float4*>(&bias[n0]);
    *reinterpret_cast<float4*>(&D[row + n0]) = make_float4(
        silu(c.x + b.x), silu(c.y + b.y), silu(c.z + b.z), silu(c.w + b.w));
  } else {
    const int n1 = (n0 + 4 < N) ? n0 + 4 : N;
    for (int n = n0; n < n1; ++n) D[row + n] = silu(C[row + n] + bias[n]);
  }
}

namespace {

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, bool Act>
void launch_gemm(int M, int N, int K, float alpha, const float* A,
                 const float* B, const float* bias, float* D,
                 const char* where) {
  using C_ = Cfg<BM, BN, BK, TM, TN, WM, WN>;
  dim3 block(C_::kThreads);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  e0e1_gemm_kernel<BM, BN, BK, TM, TN, WM, WN, Act>
      <<<grid, block>>>(M, N, K, alpha, A, B, bias, D);
  check_launch(where);
}

// The intermediate buffer is allocated once per size, before any timed
// iteration (the warmup calls this first), and reused afterwards.
float* scratch(size_t n) {
  static float* buf = nullptr;
  static size_t cap = 0;
  if (n > cap) {
    if (buf) CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaMalloc(&buf, n * sizeof(float)));
    cap = n;
  }
  return buf;
}

}  // namespace

void epilogue_e1_fused(int M, int N, int K, float alpha, const float* A,
                       const float* B, const float* bias, float* D) {
  launch_gemm<128, 64, 16, 8, 4, 32, 32, true>(M, N, K, alpha, A, B, bias, D,
                                               "e1");
}

void epilogue_e0_unfused(int M, int N, int K, float alpha, const float* A,
                         const float* B, const float* bias, float* D) {
  float* C = scratch(static_cast<size_t>(M) * N);
  launch_gemm<128, 64, 16, 8, 4, 32, 32, false>(M, N, K, alpha, A, B, nullptr,
                                                C, "e0 gemm");
  const int chunks = (N + 3) / 4;
  dim3 block(kPassThreads);
  dim3 grid((chunks + kPassThreads - 1) / kPassThreads, M);
  e0_bias_silu_kernel<<<grid, block>>>(M, N, C, bias, D);
  check_launch("e0 pass");
}
