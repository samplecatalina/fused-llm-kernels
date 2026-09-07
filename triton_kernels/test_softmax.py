"""Checks for softmax beyond the benchmark shapes.

Row and column tails, both sides of torch.softmax's dispatch boundary, large
logits, rows with -inf entries (attention masks), all -inf rows (NaN, like
torch.softmax), an output tensor passed in, views, empty inputs and the
argument checks of the wrapper. The reference is computed on the host in
double precision.

Usage: python -m triton_kernels.test_softmax
"""
import sys

import torch

from . import softmax as sm
from . import verify as vf

failures = 0
checks = 0


def check(name, cond):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {name}")


def compare(name, fn, x):
    got = fn(x)
    torch.cuda.synchronize()
    max_rel, fro_rel = vf.verify(sm.reference(x), got)
    check(f"{name} max_rel={max_rel:.2e} fro_rel={fro_rel:.2e}",
          vf.passed(max_rel, fro_rel))


def raises(name, fn, exc):
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name + " (did not raise)", False)


gen = torch.Generator().manual_seed(13)
shapes = [(1, 1), (1, 3), (5, 1), (17, 33), (129, 1000), (5, 2048), (5, 2049),
          (3, 65537), (1025, 7)]
for rows, cols in shapes:
    for scale in (1.0, 30.0):
        x = ((torch.rand(rows, cols, generator=gen) * 2 - 1) * scale).cuda()
        for nw in (2, 4, 8, 16):
            compare(f"{rows}x{cols} scale={scale} num_warps={nw}",
                    lambda x, nw=nw: sm.softmax(x, num_warps=nw), x)

# Large logits: subtracting the row maximum keeps exp finite.
x = torch.randn(8, 300, device="cuda") * 1e4
compare("logits around 1e4", sm.softmax, x)

# Masked entries (-inf) give exactly 0; the rest still sum to 1.
x = torch.randn(16, 257, device="cuda")
x[:, ::3] = -float("inf")
got = sm.softmax(x)
check("-inf entries give 0", bool((got[:, ::3] == 0).all()))
compare("rows with -inf entries", sm.softmax, x)

# An all -inf row is NaN in torch.softmax; the kernel must agree.
x = torch.randn(4, 50, device="cuda")
x[2] = -float("inf")
got = sm.softmax(x)
ref = torch.softmax(x, dim=-1)
torch.cuda.synchronize()
check("all -inf row: NaN where torch.softmax is NaN",
      torch.equal(torch.isnan(got), torch.isnan(ref)))
check("other rows unaffected",
      vf.passed(*vf.verify(sm.reference(x[[0, 1, 3]]), got[[0, 1, 3]])))

x = torch.randn(33, 517, device="cuda")
out = torch.full_like(x, 123.0)
compare("out= prefilled", lambda x: sm.softmax(x, out=out), x)

big = torch.randn(40, 700, device="cuda")
compare("every third row", sm.softmax, big[::3])
compare("row and column window", sm.softmax, big[::3, 100:613])
raises("column step rejected", lambda: sm.softmax(big[:, ::2]), ValueError)
raises("transposed rows rejected", lambda: sm.softmax(big.t()), ValueError)

x = torch.randn(64, 3000, device="cuda")
compare("eager", sm.eager, x)
compare("torch.softmax", sm.native, x)

check("zero rows", sm.softmax(torch.empty(0, 9, device="cuda")).shape == (0, 9))
raises("float64 rejected",
       lambda: sm.softmax(torch.randn(2, 3, device="cuda", dtype=torch.float64)), TypeError)
raises("host tensor rejected", lambda: sm.softmax(torch.randn(2, 3)), ValueError)
raises("1-D input rejected", lambda: sm.softmax(torch.randn(3, device="cuda")), ValueError)
raises("hidden above the block limit",
       lambda: sm.softmax(torch.randn(1, sm.MAX_COLS + 1, device="cuda")), ValueError)

print(f"{'PASS' if not failures else 'FAIL'}: {checks - failures}/{checks} softmax checks")
sys.exit(1 if failures else 0)
