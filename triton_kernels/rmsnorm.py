"""RMSNorm forward: y = x * rsqrt(mean(x^2 over the row) + eps) * w.

One Triton program per row. The program loads the row once, reduces it to
its mean square, and scales the loaded values: the row is not read from
memory a second time. Column offsets are tl.arange(0, BLOCK) with
BLOCK = next_pow2(n_cols); masked positions load as 0 and add nothing to the
sum, and the mean divides by n_cols.

Eager PyTorch written out by hand launches six kernels (pow, mean, add eps,
rsqrt, and two multiplies) and makes two full-size intermediates.
torch.nn.RMSNorm and torch.compile each launch one kernel.
"""
import torch
import triton
import triton.language as tl

MAX_COLS = 131072
EPS = 1e-6
# Bounded search over 2, 4, 8, 16 at 4096x4096 (tuning_triton_rmsnorm.csv):
# 4 was fastest; all four were within 1.1% of each other.
DEFAULT_NUM_WARPS = 4


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, x_stride, y_stride, n_cols, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_stride + col, mask=mask, other=0.0)
    ms = tl.sum(x * x) / n_cols
    r = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + col, mask=mask, other=0.0)
    tl.store(y_ptr + row * y_stride + col, x * r * w, mask=mask)


def rmsnorm(x, w, eps=EPS, out=None, num_warps=DEFAULT_NUM_WARPS):
    if x.dim() != 2 or w.dim() != 1:
        raise ValueError("expected x of shape (rows, hidden) and w of shape (hidden,)")
    rows, cols = x.shape
    if w.shape[0] != cols:
        raise ValueError(f"weight length {w.shape[0]} != hidden {cols}")
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        raise TypeError("float32 only")
    if not (x.is_cuda and w.is_cuda):
        raise ValueError("x and w must be on the GPU")
    # A silent .contiguous() would add a copy the benchmark does not show.
    if x.stride(1) != 1 or not w.is_contiguous():
        raise ValueError("x rows and w must be contiguous")
    if cols > MAX_COLS:
        raise ValueError(f"hidden {cols} exceeds the Triton block limit {MAX_COLS}")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.stride(1) != 1:
        raise ValueError("out must match x and have contiguous rows")
    if rows == 0 or cols == 0:
        return out
    rmsnorm_kernel[(rows,)](x, w, out, x.stride(0), out.stride(0), cols, eps,
                            BLOCK=triton.next_power_of_2(cols),
                            num_warps=num_warps)
    return out


def eager(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def native_module(w, eps=EPS):
    """torch.nn.RMSNorm holding a copy of w. The weight does not require a
    gradient, so no autograd state is recorded while timing."""
    mod = torch.nn.RMSNorm(w.shape[0], eps=eps, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        mod.weight.copy_(w)
    mod.weight.requires_grad_(False)
    return mod


def reference(x, w, eps=EPS):
    """The same formula on the host in double precision."""
    xd = x.detach().cpu().double()
    wd = w.detach().cpu().double()
    ms = (xd * xd).mean(dim=-1, keepdim=True)
    return xd * torch.rsqrt(ms + eps) * wd
