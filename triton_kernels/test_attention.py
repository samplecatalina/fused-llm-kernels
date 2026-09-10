"""Checks for the fused causal attention kernel.

Structure (a query may only see keys up to its own position), agreement with
an FP32 reference, and agreement with PyTorch's flash backend, which is the
meaningful bound on element-wise error: both produce FP16 outputs, and a
single rounding of one of those against an FP32 reference is already a few
percent relative once the sequence is long (the reference's RMS falls with
sequence length while the rounding does not).

Usage: python -m triton_kernels.test_attention
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import attention as at
from . import verify as vf

FRO_TOL = 2e-3
failures = 0
checks = 0


def check(name, cond):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL {name}")


def flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def inputs(batch, heads, seq, dim, seed):
    g = torch.Generator().manual_seed(seed)
    return tuple(((torch.rand(batch, heads, seq, dim, generator=g) * 2 - 1) * 0.5)
                 .half().cuda() for _ in range(3))


def raises(name, fn, exc):
    try:
        fn()
    except exc:
        check(name, True)
        return
    check(name + " (did not raise)", False)


shapes = [(1, 1, 1, 64), (1, 2, 17, 64), (1, 2, 64, 64), (2, 3, 129, 64),
          (1, 2, 200, 128), (1, 4, 1000, 64), (1, 2, 1024, 128)]
for seed, (b, h, s, d) in enumerate(shapes):
    q, k, v = inputs(b, h, s, d, seed)
    ref = at.reference(q, k, v).cpu().double()
    fl = flash(q, k, v)
    fl_max, _ = vf.verify(ref, fl)
    for bm, bn in ((64, 64), (64, 32), (128, 64)):
        got = at.attention(q, k, v, bm=bm, bn=bn)
        torch.cuda.synchronize()
        mx, fro = vf.verify(ref, got)
        check(f"{b}x{h}x{s}x{d} bm={bm} bn={bn}: fro={fro:.1e} < {FRO_TOL}",
              fro < FRO_TOL)
        check(f"{b}x{h}x{s}x{d} bm={bm} bn={bn}: max={mx:.1e} <= flash "
              f"{fl_max:.1e} * 1.05", mx <= max(fl_max * 1.05, 1e-6))

# Causality: the output of row i must not change when rows after i change.
q, k, v = inputs(1, 2, 96, 64, 99)
before = at.attention(q, k, v).clone()
k2, v2 = k.clone(), v.clone()
k2[:, :, 50:] = torch.randn_like(k2[:, :, 50:])
v2[:, :, 50:] = torch.randn_like(v2[:, :, 50:])
after = at.attention(q, k2, v2)
torch.cuda.synchronize()
check("rows before the change are untouched",
      torch.equal(before[:, :, :50], after[:, :, :50]))
check("rows after the change do change",
      not torch.equal(before[:, :, 50:], after[:, :, 50:]))

# The first query attends to one key only, so its output is that value row.
q, k, v = inputs(1, 1, 8, 64, 7)
out = at.attention(q, k, v)
torch.cuda.synchronize()
check("first row equals v[0]", torch.allclose(out[0, 0, 0], v[0, 0, 0],
                                              atol=2e-3))

# Non-contiguous but plane-regular inputs (a head slice) are accepted.
q, k, v = inputs(2, 4, 64, 64, 11)
qs, ks, vs = (t[:, :2] for t in (q, k, v))
ref = at.reference(qs, ks, vs).cpu().double()
got = at.attention(qs.contiguous(), ks.contiguous(), vs.contiguous())
torch.cuda.synchronize()
check("head slice", vf.verify(ref, got)[1] < FRO_TOL)

raises("float32 rejected",
       lambda: at.attention(*(t.float() for t in inputs(1, 1, 8, 64, 3))), TypeError)
raises("head_dim 32 rejected",
       lambda: at.attention(*inputs(1, 1, 8, 32, 3)), ValueError)
raises("3-D input rejected",
       lambda: at.attention(*(t[0] for t in inputs(1, 1, 8, 64, 3))), ValueError)
raises("mismatched shapes",
       lambda: at.attention(inputs(1, 1, 8, 64, 3)[0], *inputs(1, 1, 16, 64, 4)[1:]),
       ValueError)

print(f"{'PASS' if not failures else 'FAIL'}: {checks - failures}/{checks} "
      "attention checks")
sys.exit(1 if failures else 0)
