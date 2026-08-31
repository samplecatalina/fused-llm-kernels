"""The two-tolerance check of csrc/common/verify.h, for host tensors.

max_rel divides by |ref| plus 5% of the RMS of ref; fro_rel is the relative
Frobenius error. Both must hold: max_rel < 1e-3 and fro_rel < 1e-5.
"""
import math

import torch

MAX_REL_FLOOR_OF_RMS = 5e-2
MAX_TOL = 1e-3
FRO_TOL = 1e-5


def verify(ref, got):
    ref = ref.double().flatten()
    got = got.detach().cpu().double().flatten()
    den = float((ref * ref).sum())
    rms = math.sqrt(den / ref.numel()) if ref.numel() else 0.0
    floor = MAX_REL_FLOOR_OF_RMS * rms if rms > 0 else 1e-5
    d = ref - got
    if not torch.isfinite(got).all():
        return math.inf, math.inf
    max_rel = float((d.abs() / (ref.abs() + floor)).max()) if ref.numel() else 0.0
    num = math.sqrt(float((d * d).sum()))
    fro_rel = num / math.sqrt(den) if den > 0 else num
    return max_rel, fro_rel


def passed(max_rel, fro_rel):
    return max_rel < MAX_TOL and fro_rel < FRO_TOL


def reference_bias_silu(x, b):
    """SiLU(x + b) on the host in double precision."""
    t = x.detach().cpu().double() + b.detach().cpu().double()
    return t * torch.sigmoid(t)
