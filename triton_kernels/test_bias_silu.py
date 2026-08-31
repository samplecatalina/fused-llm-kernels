"""Checks for bias_silu beyond the benchmark shapes.

Tails of SiLU (inputs scaled far into both saturated regions), row and
column tails, an output tensor passed in, empty inputs, and the argument
checks of the wrapper. The reference is computed on the host in double
precision.

Usage: python -m triton_kernels.test_bias_silu
"""
import sys

import torch

from . import bias_silu as bs
from . import verify as vf

failures = 0
checks = 0


def check(name, cond):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {name}")


def compare(name, fn, x, b):
    got = fn(x, b)
    torch.cuda.synchronize()
    max_rel, fro_rel = vf.verify(vf.reference_bias_silu(x, b), got)
    check(f"{name} max_rel={max_rel:.2e} fro_rel={fro_rel:.2e}",
          vf.passed(max_rel, fro_rel))


def raises(name, fn, exc):
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name + " (did not raise)", False)


gen = torch.Generator().manual_seed(7)
shapes = [(1, 1), (1, 3), (5, 1), (17, 33), (129, 1000), (3, 65537), (1025, 7)]
scales = [1.0, 60.0]  # 60 drives exp(-t) far past float32 range on both sides
for rows, cols in shapes:
    for scale in scales:
        x = ((torch.rand(rows, cols, generator=gen) * 2 - 1) * scale).cuda()
        b = ((torch.rand(cols, generator=gen) * 2 - 1) * scale).cuda()
        for w in (2, 4, 8, 16):
            compare(f"{rows}x{cols} scale={scale} num_warps={w}",
                    lambda x, b, w=w: bs.bias_silu(x, b, num_warps=w), x, b)

# Output tensor passed in, prefilled: every element must be overwritten.
x = torch.randn(33, 517, device="cuda")
b = torch.randn(517, device="cuda")
out = torch.full_like(x, 123.0)
compare("out= prefilled", lambda x, b: bs.bias_silu(x, b, out=out), x, b)

# Views of a larger tensor. Rows that are contiguous inside are accepted
# (stride(0) and the storage offset are honoured); rows with a column step
# are rejected.
big = torch.randn(40, 700, device="cuda")
rows_only = big[::3]
check("row stride honoured: stride(0)=2100", rows_only.stride(0) == 2100)
compare("every third row", bs.bias_silu, rows_only, torch.randn(700, device="cuda"))
window = big[::3, 100:613]
compare("row and column window (offset 100)", bs.bias_silu, window,
        torch.randn(513, device="cuda"))
raises("column step rejected",
       lambda: bs.bias_silu(big[:, ::2], torch.randn(350, device="cuda")), ValueError)
raises("transposed rows rejected",
       lambda: bs.bias_silu(big.t(), torch.randn(40, device="cuda")), ValueError)

# Empty inputs.
check("zero rows", bs.bias_silu(torch.empty(0, 9, device="cuda"),
                                torch.empty(9, device="cuda")).shape == (0, 9))

# Argument checks.
raises("bias length mismatch",
       lambda: bs.bias_silu(torch.randn(2, 3, device="cuda"),
                            torch.randn(4, device="cuda")), ValueError)
raises("float64 rejected",
       lambda: bs.bias_silu(torch.randn(2, 3, device="cuda", dtype=torch.float64),
                            torch.randn(3, device="cuda", dtype=torch.float64)), TypeError)
raises("host tensors rejected",
       lambda: bs.bias_silu(torch.randn(2, 3), torch.randn(3)), ValueError)
raises("hidden above the block limit",
       lambda: bs.bias_silu(torch.randn(1, bs.MAX_COLS + 1, device="cuda"),
                            torch.randn(bs.MAX_COLS + 1, device="cuda")), ValueError)

print(f"{'PASS' if not failures else 'FAIL'}: {checks - failures}/{checks} bias_silu checks")
sys.exit(1 if failures else 0)
