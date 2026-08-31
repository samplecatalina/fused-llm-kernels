"""y = SiLU(x + b): bias broadcast over the rows of x, then SiLU.

One Triton program per row. The column offsets of a program are
tl.arange(0, BLOCK) with BLOCK = next_pow2(n_cols), masked to n_cols, so any
hidden size works. The bias is indexed by the column alone: every program
reads the same contiguous stretch of b.

Unfused, the same result takes two launches (add, silu) and an intermediate
tensor, or three (add, sigmoid, mul) when written by hand. Fused, each element
is loaded once from x, once from b, and stored once into y.
"""
import torch
import triton
import triton.language as tl

# Largest block a single Triton program may use.
MAX_COLS = 131072
# Provisional; replaced by the result of the bounded num_warps search.
DEFAULT_NUM_WARPS = 4


@triton.jit
def bias_silu_kernel(x_ptr, b_ptr, y_ptr, x_stride, y_stride, n_cols,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    x = tl.load(x_ptr + row * x_stride + col, mask=mask, other=0.0)
    b = tl.load(b_ptr + col, mask=mask, other=0.0)
    t = x + b
    # sigmoid(t) underflows to 0 for very negative t, giving -0, never NaN.
    y = t * tl.sigmoid(t)
    tl.store(y_ptr + row * y_stride + col, y, mask=mask)


def bias_silu(x, b, out=None, num_warps=DEFAULT_NUM_WARPS):
    if x.dim() != 2 or b.dim() != 1:
        raise ValueError("expected x of shape (rows, hidden) and b of shape (hidden,)")
    rows, cols = x.shape
    if b.shape[0] != cols:
        raise ValueError(f"bias length {b.shape[0]} != hidden {cols}")
    if x.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("float32 only")
    if not (x.is_cuda and b.is_cuda):
        raise ValueError("x and b must be on the GPU")
    # A silent .contiguous() would add a copy the benchmark does not show.
    if x.stride(1) != 1 or not b.is_contiguous():
        raise ValueError("x rows and b must be contiguous")
    if cols > MAX_COLS:
        raise ValueError(f"hidden {cols} exceeds the Triton block limit {MAX_COLS}")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.stride(1) != 1:
        raise ValueError("out must match x and have contiguous rows")
    if rows == 0 or cols == 0:
        return out
    bias_silu_kernel[(rows,)](x, b, out, x.stride(0), out.stride(0), cols,
                              BLOCK=triton.next_power_of_2(cols),
                              num_warps=num_warps)
    return out


def eager_native(x, b):
    return torch.nn.functional.silu(x + b)


def eager_composite(x, b):
    t = x + b
    return t * torch.sigmoid(t)
