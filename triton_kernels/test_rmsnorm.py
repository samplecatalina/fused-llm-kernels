"""Checks for rmsnorm beyond the benchmark shapes.

Row and column tails, all-zero rows, values whose squares underflow or
overflow float32, an output tensor passed in, views, empty inputs and the
argument checks of the wrapper. The reference is computed on the host in
double precision; where float32 itself overflows, the Triton kernel is
compared with eager PyTorch instead.

Usage: python -m triton_kernels.test_rmsnorm
"""
import sys

import torch

from . import rmsnorm as rn
from . import verify as vf

failures = 0
checks = 0


def check(name, cond):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {name}")


def compare(name, fn, x, w):
    got = fn(x, w)
    torch.cuda.synchronize()
    max_rel, fro_rel = vf.verify(rn.reference(x, w), got)
    check(f"{name} max_rel={max_rel:.2e} fro_rel={fro_rel:.2e}",
          vf.passed(max_rel, fro_rel))


def raises(name, fn, exc):
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name + " (did not raise)", False)


gen = torch.Generator().manual_seed(11)
shapes = [(1, 1), (1, 3), (5, 1), (17, 33), (129, 1000), (3, 65537), (1025, 7)]
for rows, cols in shapes:
    for scale in (1.0, 1e3, 1e-3):
        x = ((torch.rand(rows, cols, generator=gen) * 2 - 1) * scale).cuda()
        w = (torch.rand(cols, generator=gen) * 2 - 1).cuda()
        for nw in (2, 4, 8, 16):
            compare(f"{rows}x{cols} scale={scale} num_warps={nw}",
                    lambda x, w, nw=nw: rn.rmsnorm(x, w, num_warps=nw), x, w)

# All-zero rows and rows whose squares underflow to 0 in float32: the result
# is x * w / sqrt(eps), i.e. 0 and a small multiple of x respectively.
x = torch.randn(6, 300, device="cuda")
x[1] = 0.0
x[3] = 1e-24
w = torch.randn(300, device="cuda")
compare("zero and underflowing rows", rn.rmsnorm, x, w)

# Squares that overflow float32: both eager and Triton get r = 0 there.
x = torch.randn(4, 50, device="cuda")
x[2, 7] = 3e38
w = torch.randn(50, device="cuda")
got = rn.rmsnorm(x, w)
ref = rn.eager(x, w)
torch.cuda.synchronize()
check("overflowing row matches eager (zeros)",
      torch.equal(got[2] == 0, ref[2] == 0) and bool((got[2] == 0).all()))
check("other rows unaffected by the overflowing one",
      vf.passed(*vf.verify(rn.reference(x[[0, 1, 3]], w), got[[0, 1, 3]])))

# Output tensor passed in, prefilled.
x = torch.randn(33, 517, device="cuda")
w = torch.randn(517, device="cuda")
out = torch.full_like(x, 123.0)
compare("out= prefilled", lambda x, w: rn.rmsnorm(x, w, out=out), x, w)

# Views: row stride and storage offset are honoured; column steps rejected.
big = torch.randn(40, 700, device="cuda")
compare("every third row", rn.rmsnorm, big[::3], torch.randn(700, device="cuda"))
compare("row and column window", rn.rmsnorm, big[::3, 100:613],
        torch.randn(513, device="cuda"))
raises("column step rejected",
       lambda: rn.rmsnorm(big[:, ::2], torch.randn(350, device="cuda")), ValueError)
raises("transposed rows rejected",
       lambda: rn.rmsnorm(big.t(), torch.randn(40, device="cuda")), ValueError)

# The three reference implementations agree with the host reference.
x = torch.randn(64, 1000, device="cuda")
w = torch.randn(1000, device="cuda")
compare("eager", rn.eager, x, w)
mod = rn.native_module(w)
compare("torch.nn.RMSNorm", lambda x, w: mod(x), x, w)
check("native module weight does not require grad", not mod.weight.requires_grad)

check("zero rows", rn.rmsnorm(torch.empty(0, 9, device="cuda"),
                              torch.empty(9, device="cuda")).shape == (0, 9))
raises("weight length mismatch",
       lambda: rn.rmsnorm(torch.randn(2, 3, device="cuda"),
                          torch.randn(4, device="cuda")), ValueError)
raises("float64 rejected",
       lambda: rn.rmsnorm(torch.randn(2, 3, device="cuda", dtype=torch.float64),
                          torch.randn(3, device="cuda", dtype=torch.float64)), TypeError)
raises("host tensors rejected",
       lambda: rn.rmsnorm(torch.randn(2, 3), torch.randn(3)), ValueError)
raises("hidden above the block limit",
       lambda: rn.rmsnorm(torch.randn(1, rn.MAX_COLS + 1, device="cuda"),
                          torch.randn(rn.MAX_COLS + 1, device="cuda")), ValueError)

print(f"{'PASS' if not failures else 'FAIL'}: {checks - failures}/{checks} rmsnorm checks")
sys.exit(1 if failures else 0)
