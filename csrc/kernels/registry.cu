#include <cstring>

#include "kernels.h"

// Adding a rung = declare it in kernels.h + add one row to this table.
const KernelEntry kKernels[] = {
    {"k0", "cuBLAS SGEMM (baseline)", sgemm_k0_cublas},
    {"k1", "naive: one thread per C element", sgemm_k1_naive},
    {"k2", "coalesced access: threadIdx.x selects the column", sgemm_k2_coalesced},
    {"k3", "shared-memory tiling, 32x32x32 tiles", sgemm_k3_shared},
};
const int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

const KernelEntry* find_kernel(const char* name) {
  for (int i = 0; i < kNumKernels; ++i)
    if (std::strcmp(kKernels[i].name, name) == 0) return &kKernels[i];
  return nullptr;
}
