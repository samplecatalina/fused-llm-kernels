"""Checks for the RMSNorm backward kernel.

Both gradients against a double-precision host reference, row and column
tails, views, and the conditioning limit at hidden 1, where dx is a
difference of two nearly equal terms and no float32 implementation is
accurate - there the kernel is only required to be no worse than PyTorch's
own backward.

Usage: python -m triton_kernels.test_rmsnorm_bwd
"""
import sys

import torch

from . import rmsnorm_backward as rb
from . import verify as vf

failures = 0
checks = 0


def check(name, cond):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {name}")


def compare(name, x, w, dy, num_warps=rb.DEFAULT_NUM_WARPS, ref=None):
    dx, dw = rb.rmsnorm_backward(x, w, dy, num_warps=num_warps)
    torch.cuda.synchronize()
    rdx, rdw = ref if ref is not None else rb.reference(x, w, dy)
    a, b = vf.verify(rdx, dx), vf.verify(rdw, dw)
    check(f"{name} dx={a[0]:.1e}/{a[1]:.1e} dw={b[0]:.1e}/{b[1]:.1e}",
          vf.passed(*a) and vf.passed(*b))


def raises(name, fn, exc):
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name + " (did not raise)", False)


gen = torch.Generator().manual_seed(17)
# The host reference is double precision, so the largest shape here is kept
# to a size the CPU can do quickly; the benchmark shapes are checked through
# `bench --preset correctness`.
shapes = [(1, 2), (3, 5), (17, 33), (129, 1000), (3, 8191), (1025, 7),
          (512, 1024)]
for rows, cols in shapes:
    for scale in (1.0, 20.0):
        x = ((torch.rand(rows, cols, generator=gen) * 2 - 1) * scale).cuda()
        w = (torch.rand(cols, generator=gen) * 2 - 1).cuda()
        dy = ((torch.rand(rows, cols, generator=gen) * 2 - 1) * scale).cuda()
        ref = rb.reference(x, w, dy)          # once per shape, not per warp count
        for nw in (2, 4, 8, 16):
            compare(f"{rows}x{cols} scale={scale} num_warps={nw}", x, w, dy, nw, ref)

# Both PyTorch paths agree with the reference on well-conditioned shapes.
x = torch.randn(64, 500, device="cuda")
w = torch.randn(500, device="cuda")
dy = torch.randn(64, 500, device="cuda")
rdx, rdw = rb.reference(x, w, dy)
for name, fn in (("eager", rb.autograd_backward_fn(rb.eager_forward)),
                 ("F.rms_norm", rb.autograd_backward_fn(rb.native_forward))):
    gx, gw = fn(x, w, dy)
    torch.cuda.synchronize()
    check(f"{name} dx", vf.passed(*vf.verify(rdx, gx)))
    check(f"{name} dw", vf.passed(*vf.verify(rdw, gw)))

# hidden 1: dx = g*r*eps/(ms+eps), so the two terms cancel. Every float32
# implementation loses most of the result; the kernel must not be worse than
# PyTorch's backward.
x = torch.randn(64, 1, device="cuda")
w = torch.randn(1, device="cuda")
dy = torch.randn(64, 1, device="cuda")
rdx, _ = rb.reference(x, w, dy)
tdx, _ = rb.rmsnorm_backward(x, w, dy)
edx, _ = rb.autograd_backward_fn(rb.eager_forward)(x, w, dy)
torch.cuda.synchronize()
t_err, e_err = vf.verify(rdx, tdx)[0], vf.verify(rdx, edx)[0]
check(f"hidden 1 ill-conditioned (amplification {rb.conditioning(x):.1e}): "
      f"kernel {t_err:.1e} <= eager {e_err:.1e} * 2", t_err <= e_err * 2)

# Views: row stride and offset are honoured, column steps rejected.
big = torch.randn(40, 700, device="cuda")
bigdy = torch.randn(40, 700, device="cuda")
compare("every third row", big[::3], torch.randn(700, device="cuda"), bigdy[::3])
raises("column step rejected",
       lambda: rb.rmsnorm_backward(big[:, ::2], torch.randn(350, device="cuda"),
                                   bigdy[:, ::2]), ValueError)

dx, dw = rb.rmsnorm_backward(torch.empty(0, 9, device="cuda"),
                             torch.empty(9, device="cuda"),
                             torch.empty(0, 9, device="cuda"))
check("zero rows", dx.shape == (0, 9) and bool((dw == 0).all()))
raises("shape mismatch",
       lambda: rb.rmsnorm_backward(torch.randn(2, 3, device="cuda"),
                                   torch.randn(3, device="cuda"),
                                   torch.randn(2, 4, device="cuda")), ValueError)
raises("hidden above the supported maximum",
       lambda: rb.rmsnorm_backward(torch.randn(1, rb.MAX_COLS + 1, device="cuda"),
                                   torch.randn(rb.MAX_COLS + 1, device="cuda"),
                                   torch.randn(1, rb.MAX_COLS + 1, device="cuda")),
       ValueError)
raises("float64 rejected",
       lambda: rb.rmsnorm_backward(torch.randn(2, 3, device="cuda", dtype=torch.float64),
                                   torch.randn(3, device="cuda", dtype=torch.float64),
                                   torch.randn(2, 3, device="cuda", dtype=torch.float64)),
       TypeError)

print(f"{'PASS' if not failures else 'FAIL'}: {checks - failures}/{checks} "
      "rmsnorm backward checks")
sys.exit(1 if failures else 0)
