"""Row-wise softmax: y = exp(x - m) / sum(exp(x - m)), m the row maximum.

One Triton program per row. The row is loaded once; its maximum and the
normalizer are parallel reductions over the loaded values, so neither costs
another pass over memory. That is what the online normalizer achieves with a
sequential recurrence; here the recurrence is unnecessary. Masked positions
load as -inf: they do not change the maximum, contribute exp(-inf) = 0 to the
sum, and are not stored.

The naive eager form launches five kernels (max, subtract, exp, sum, divide)
and writes three full-size tensors. torch.softmax and torch.compile each
launch one kernel.
"""
import torch
import triton
import triton.language as tl

MAX_COLS = 131072
# Provisional; replaced by the result of the bounded num_warps search.
DEFAULT_NUM_WARPS = 4


@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_stride, y_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_stride + col, mask=mask, other=-float("inf"))
    z = tl.exp(x - tl.max(x, axis=0))
    tl.store(y_ptr + row * y_stride + col, z / tl.sum(z, axis=0), mask=mask)


def softmax(x, out=None, num_warps=DEFAULT_NUM_WARPS):
    if x.dim() != 2:
        raise ValueError("expected x of shape (rows, hidden)")
    rows, cols = x.shape
    if x.dtype != torch.float32:
        raise TypeError("float32 only")
    if not x.is_cuda:
        raise ValueError("x must be on the GPU")
    # A silent .contiguous() would add a copy the benchmark does not show.
    if x.stride(1) != 1:
        raise ValueError("x rows must be contiguous")
    if cols > MAX_COLS:
        raise ValueError(f"hidden {cols} exceeds the Triton block limit {MAX_COLS}")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.stride(1) != 1:
        raise ValueError("out must match x and have contiguous rows")
    if rows == 0 or cols == 0:
        return out
    softmax_kernel[(rows,)](x, out, x.stride(0), out.stride(0), cols,
                            BLOCK=triton.next_power_of_2(cols),
                            num_warps=num_warps)
    return out


def eager(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=-1, keepdim=True)


def native(x):
    return torch.softmax(x, dim=-1)


def reference(x):
    """The same formula on the host in double precision."""
    xd = x.detach().cpu().double()
    e = torch.exp(xd - xd.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)
