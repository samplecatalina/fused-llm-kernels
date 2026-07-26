#include <cstring>

#include "kernels.h"

// Adding a rung = declare it in kernels.h + add one row to this table.
const KernelEntry kKernels[] = {
    {"k0", "cuBLAS SGEMM (baseline)", sgemm_k0_cublas},
    {"k1", "naive: one thread per C element", sgemm_k1_naive},
    {"k2", "coalesced access: threadIdx.x selects the column", sgemm_k2_coalesced},
    {"k3", "shared-memory tiling, 32x32x32 tiles", sgemm_k3_shared},
    {"k4", "1D thread tiling, 8 results per thread", sgemm_k4_tiling1d},
    {"k5", "2D thread tiling, 8x8 register block, 128x128x16 tiles", sgemm_k5_tiling2d},
    {"k6", "float4 vectorized loads, transposed A tile", sgemm_k6_vectorized},
    {"k7", "warp tiling, 128x64x16 tiles, 8x4 thread tile (search winner)", sgemm_k7_warptile},
    {"k7c1", "warp tiling, 128x128x16 tiles, 32x64 warp tile (K6 geometry)", sgemm_k7_c1},
    {"k7c2", "warp tiling, 64x32 warp tile", sgemm_k7_c2},
    {"k7c3", "warp tiling, 128x64x16 tiles, 8x4 thread tile", sgemm_k7_c3},
    {"k7c4", "warp tiling, 64x64x16 tiles, 4x8 thread tile", sgemm_k7_c4},
    {"k7c5", "warp tiling, BK=8", sgemm_k7_c5},
    {"k7c6", "warp tiling, BK=32", sgemm_k7_c6},
    {"k7c7", "warp tiling, 256x128x16 tiles, 512 threads", sgemm_k7_c7},
    {"k7c8", "warp tiling, 128x256x16 tiles, 512 threads", sgemm_k7_c8},
};
const int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

const KernelEntry* find_kernel(const char* name) {
  for (int i = 0; i < kNumKernels; ++i)
    if (std::strcmp(kKernels[i].name, name) == 0) return &kKernels[i];
  return nullptr;
}
