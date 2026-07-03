#!/usr/bin/env bash
# Environment acceptance check. The point is to fail early on the one thing that
# can invalidate the whole project: hardware counters not being readable under
# WSL2, which would leave every optimization claim without evidence.
# Usage (inside WSL2): bash scripts/env_check.sh
set -u
PASS=0; FAIL=0
ok(){ echo "  [OK]   $*"; PASS=$((PASS+1)); }
bad(){ echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "=============================================="
echo " fused-llm-kernels - environment check"
echo " $(date)  |  $(uname -sr)"
echo "=============================================="

echo
echo "[1/5] GPU and driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader | sed 's/^/  /'
  echo "  Current state (a laptop GPU's power budget floats; these numbers are a"
  echo "  precondition for the benchmarks being trustworthy):"
  nvidia-smi --query-gpu=clocks.current.sm,power.limit,power.draw,temperature.gpu --format=csv | sed 's/^/    /'
  ok "nvidia-smi works (the GPU is visible from WSL2)"
else
  bad "nvidia-smi unavailable - no GPU visible from WSL2, nothing else can work"
fi

echo
echo "[2/5] CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
  for p in /usr/local/cuda/bin /usr/local/cuda-12*/bin; do
    [ -x "$p/nvcc" ] && export PATH="$p:$PATH" && break
  done
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2 | sed 's/^/  /'
  ok "nvcc found: $(command -v nvcc)"
else
  bad "nvcc not on PATH (note: CUDA installed on the Windows side does not put a toolkit inside WSL2)"
fi

echo
echo "[3/5] Compile + run + link against cuBLAS"
TMP=$(mktemp -d)
cat > "$TMP/smoke.cu" <<'CU'
#include <cstdio>
#include <cublas_v2.h>
__global__ void axpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
    int n = 1 << 20;
    float *x, *y;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cudaMemset(x, 0, n * sizeof(float));
    cudaMemset(y, 0, n * sizeof(float));
    axpy<<<(n + 255) / 256, 256>>>(n, 2.0f, x, y);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL_FAIL %s\n", cudaGetErrorString(e)); return 1; }
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS_FAIL\n"); return 1; }
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("SMOKE_OK sm_%d%d  SMs=%d  sharedPerBlock=%zuKB  L2=%dKB\n",
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024, prop.l2CacheSize / 1024);
    cublasDestroy(h);
    return 0;
}
CU
if command -v nvcc >/dev/null 2>&1 && nvcc -O2 -arch=sm_89 "$TMP/smoke.cu" -o "$TMP/smoke" -lcublas 2>"$TMP/nvcc.log"; then
  ok "nvcc compiles (-arch=sm_89, linking cuBLAS)"
  OUT=$("$TMP/smoke" 2>&1)
  echo "  $OUT"
  case "$OUT" in SMOKE_OK*) ok "kernel executed and cuBLAS handle created" ;; *) bad "run failed: $OUT" ;; esac
else
  bad "compilation failed, nvcc output:"; sed 's/^/    /' "$TMP/nvcc.log" 2>/dev/null | head -20
fi

echo
echo "[4/5] Nsight Compute hardware counters"
if command -v ncu >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/ncu ]; then
  NCU=$(command -v ncu || echo /usr/local/cuda/bin/ncu)
  "$NCU" --version 2>&1 | head -2 | sed 's/^/  /'
  if [ -x "$TMP/smoke" ]; then
    NOUT=$("$NCU" --set basic --print-summary per-kernel "$TMP/smoke" 2>&1)
    if echo "$NOUT" | grep -q "ERR_NVGPUCTRPERM\|insufficient permissions\|not permitted"; then
      bad "counters blocked by permissions (ERR_NVGPUCTRPERM)"
      echo "    Fix this on the Windows host, not inside WSL2:"
      echo "      NVIDIA Control Panel -> Desktop menu -> Enable Developer Settings"
      echo "      -> Developer -> Manage GPU Performance Counters"
      echo "      -> Allow access to the GPU performance counters to all users"
      echo "    Then run 'wsl --shutdown' in PowerShell, reopen the terminal and rerun this script."
      echo "    If it still fails, profile the same binary with Nsight Compute on the Windows side."
    elif echo "$NOUT" | grep -qi "duration\|sm frequency\|Section:"; then
      ok "ncu collected counters - profiling can stay inside WSL2"
    else
      bad "unexpected ncu output, first 15 lines:"; echo "$NOUT" | head -15 | sed 's/^/    /'
    fi
  fi
else
  bad "ncu not installed (the Nsight Compute component of the CUDA toolkit may have been skipped)"
fi

echo
echo "[5/5] Python / Triton (only needed for the Triton kernels)"
python3 -c 'import torch;print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")' 2>/dev/null && ok "PyTorch available" || echo "  [SKIP] PyTorch not installed - not required for the CUDA ladder"
python3 -c 'import triton;print(f"  triton {triton.__version__}")' 2>/dev/null && ok "Triton available" || echo "  [SKIP] Triton not installed - same as above"

rm -rf "$TMP"
echo
echo "=============================================="
echo " $PASS passed, $FAIL failed"
echo "=============================================="
