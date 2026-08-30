"""Check that the PyTorch / Triton environment can run on this GPU.

Four checks, each one a precondition for the Triton benchmarks:
  1. torch sees the GPU through CUDA, and reports its capability;
  2. a cuBLAS matmul on the GPU matches the same product on the CPU;
  3. a minimal Triton kernel compiles for this architecture and is correct;
  4. torch.compile produces a working kernel (the second baseline).

Usage: .venv/bin/python scripts/check_torch.py
"""
import platform
import sys

import torch
import triton
import triton.language as tl

failures = 0


def report(ok, what):
    global failures
    print(f"  [{'OK' if ok else 'FAIL'}]   {what}")
    failures += 0 if ok else 1


print(f"python {platform.python_version()}  torch {torch.__version__}  "
      f"triton {triton.__version__}  torch CUDA {torch.version.cuda}")

report(torch.cuda.is_available(), "torch.cuda.is_available()")
if not torch.cuda.is_available():
    sys.exit(1)
dev = torch.device("cuda")
cap = torch.cuda.get_device_capability()
print(f"  device: {torch.cuda.get_device_name()}  capability sm_{cap[0]}{cap[1]}  "
      f"arch list: {torch.cuda.get_arch_list()}")
report(f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
       or any(a.startswith(f"sm_{cap[0]}") for a in torch.cuda.get_arch_list()),
       "this capability is covered by the wheel's compiled architectures")

g = torch.Generator().manual_seed(0)
a = torch.rand(257, 129, generator=g)
b = torch.rand(129, 63, generator=g)
ref = a.double() @ b.double()
got = (a.to(dev) @ b.to(dev)).cpu().double()
rel = ((got - ref).norm() / ref.norm()).item()
report(rel < 1e-5, f"GPU matmul matches CPU (relative Frobenius error {rel:.2e})")


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + tl.load(y + offs, mask=mask),
             mask=mask)


n = 100_003  # not a multiple of the block size: exercises the mask
x = torch.rand(n, device=dev)
y = torch.rand(n, device=dev)
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
report(torch.equal(out, x + y), "Triton kernel compiles and matches eager")


def silu_bias(t, bias):
    return torch.nn.functional.silu(t + bias)


t = torch.randn(33, 517, device=dev)
bias = torch.randn(517, device=dev)
compiled = torch.compile(silu_bias)
diff = (compiled(t, bias) - silu_bias(t, bias)).abs().max().item()
report(diff < 1e-6, f"torch.compile output matches eager (max abs diff {diff:.1e})")

print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
sys.exit(1 if failures else 0)
