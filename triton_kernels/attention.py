"""Fused causal attention, forward only.

O = softmax(Q Kt / sqrt(D) + causal mask) V, with Q, K, V of shape
(batch, heads, seq, head_dim). One program owns a block of BM queries of one
(batch, head): it holds that block of Q in registers and walks the key blocks
it is allowed to see, keeping a running maximum and normalizer.

This is where the online-softmax recurrence earns its place. A row of the
attention matrix is seq elements long and is produced a block at a time, so
each new block rescales what has been accumulated:

    m_new = max(m, rowmax(s))
    alpha = exp(m - m_new)
    l     = l * alpha + rowsum(exp(s - m_new))
    acc   = acc * alpha + exp(s - m_new) @ V_block

The attention matrix never reaches memory: global traffic is Q, K, V and O,
against the seq^2 per (batch, head) that an explicit implementation writes and
reads back.

Scope, fixed: forward only, causal, no dropout, head_dim 64 or 128, FP16 in
with FP32 accumulation.
"""
import torch
import triton
import triton.language as tl

DEFAULT_BM = 64
DEFAULT_BN = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


@triton.jit
def attn_fwd_kernel(Q, K, V, O, q_plane, k_plane, v_plane, o_plane,
                    q_row, k_row, v_row, o_row, seq, scale,
                    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    qb = tl.program_id(0)
    bh = tl.program_id(1)          # flattened (batch, head)
    qm = qb * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q + bh * q_plane + qm[:, None] * q_row + d[None, :],
                mask=qm[:, None] < seq, other=0.0)

    m = tl.full([BM], -float("inf"), dtype=tl.float32)
    l = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    # Causal: a query at row i may attend to keys 0..i, so the key blocks
    # visited are those covering positions 0..(qb+1)*BM-1. Counting them in
    # query blocks would be wrong whenever BM and BN differ.
    for kb in range(0, tl.cdiv((qb + 1) * BM, BN)):
        kn = kb * BN + tl.arange(0, BN)
        k = tl.load(K + bh * k_plane + kn[:, None] * k_row + d[None, :],
                    mask=kn[:, None] < seq, other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        s = tl.where((kn[None, :] <= qm[:, None]) & (kn[None, :] < seq),
                     s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        # A block entirely masked out leaves m at -inf; exp(-inf - -inf) is
        # NaN, so an all-masked row is handled by the guard below.
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(V + bh * v_plane + kn[:, None] * v_row + d[None, :],
                    mask=kn[:, None] < seq, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m = m_new

    out = acc / l[:, None]
    tl.store(O + bh * o_plane + qm[:, None] * o_row + d[None, :],
             out.to(O.dtype.element_ty), mask=qm[:, None] < seq)


def attention(q, k, v, bm=DEFAULT_BM, bn=DEFAULT_BN, num_warps=DEFAULT_NUM_WARPS,
              num_stages=DEFAULT_NUM_STAGES, out=None):
    if not (q.dim() == k.dim() == v.dim() == 4):
        raise ValueError("expected (batch, heads, seq, head_dim)")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k and v must have the same shape")
    b, h, s, d = q.shape
    if d not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if q.dtype != torch.float16:
        raise TypeError("float16 only")
    for t in (q, k, v):
        if not t.is_cuda or t.stride(-1) != 1:
            raise ValueError("q, k and v must be on the GPU with contiguous rows")
        # (batch, head) is flattened into one program axis, so consecutive
        # planes must be a fixed stride apart.
        if t.stride(0) != h * t.stride(1):
            raise ValueError("expected the head axis to tile the batch axis")
    if out is None:
        out = torch.empty_like(q)
    grid = (triton.cdiv(s, bm), b * h)
    attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(1), k.stride(1), v.stride(1), out.stride(1),
        q.stride(2), k.stride(2), v.stride(2), out.stride(2),
        s, d ** -0.5, BM=bm, BN=bn, D=d,
        num_warps=num_warps, num_stages=num_stages)
    return out


def reference(q, k, v):
    """Float32 attention on the GPU, the answer the FP16 kernels approximate."""
    qs, ks, vs = (t.detach().float() for t in (q, k, v))
    s = (qs @ ks.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    causal = torch.ones(q.shape[-2], q.shape[-2], device=q.device,
                        dtype=torch.bool).tril()
    s = s.masked_fill(~causal, float("-inf"))
    return torch.softmax(s, dim=-1) @ vs
