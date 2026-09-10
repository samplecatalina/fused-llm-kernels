"""RMSNorm backward: dx and dw from dy, x and w.

With r = rsqrt(mean(x^2) + eps) and y = x * r * w, writing g = dy * w:

    dx_ij = r_i * g_ij - (r_i^3 / H) * x_ij * sum_k x_ik * g_ik     (row-wise)
    dw_j  = sum_i dy_ij * x_ij * r_i                                (column-wise)

dx needs a reduction along the row, dw one along the column. Both read dy and
x, so one kernel does both: each program owns a strip of rows, writes their
dx, and accumulates its own partial dw; a small sum over the partials
finishes dw. That keeps global traffic at read dy + read x + write dx.

r is recomputed rather than saved by the forward pass: it costs one more
row-wise reduction over values already loaded, and saves reading a rows-long
vector back.
"""
import torch
import triton
import triton.language as tl

# The forward kernel accepts blocks up to Triton's 131072 limit. This one
# holds three block-sized vectors and a loop, and its compile time grows
# steeply with the block: 0.2 s at hidden 8192, 0.6 s at 16384, 3.0 s at
# 32768, minutes beyond. 32768 is the supported maximum.
MAX_COLS = 32768
EPS = 1e-6
# Bounded search over 2, 4, 8, 16 at 4096x4096 (tuning_triton_rmsnorm_bwd.csv):
# 4 was fastest; all four were within 3.1% of each other.
DEFAULT_NUM_WARPS = 4
# Programs are sized so that the partial-dw matrix stays small: with 24 SMs,
# a few hundred programs keep every SM busy and the matrix under a megabyte.
TARGET_PROGRAMS = 192


@triton.jit
def rmsnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dwp_ptr, x_stride,
                       dy_stride, dx_stride, n_rows, n_cols, eps,
                       ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < n_cols
    w = tl.load(w_ptr + col, mask=mask, other=0.0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(ROWS):
        row = pid * ROWS + k
        if row < n_rows:
            x = tl.load(x_ptr + row * x_stride + col, mask=mask, other=0.0)
            dy = tl.load(dy_ptr + row * dy_stride + col, mask=mask, other=0.0)
            r = 1.0 / tl.sqrt(tl.sum(x * x) / n_cols + eps)
            g = dy * w
            dx = r * g - (r * r * r / n_cols) * x * tl.sum(x * g)
            tl.store(dx_ptr + row * dx_stride + col, dx, mask=mask)
            acc += dy * x * r
    tl.store(dwp_ptr + pid * BLOCK + col, acc, mask=mask)


def rmsnorm_backward(x, w, dy, eps=EPS, num_warps=DEFAULT_NUM_WARPS):
    """Returns (dx, dw)."""
    if x.dim() != 2 or w.dim() != 1 or dy.shape != x.shape:
        raise ValueError("expected x, dy of shape (rows, hidden) and w of shape (hidden,)")
    rows, cols = x.shape
    if w.shape[0] != cols:
        raise ValueError(f"weight length {w.shape[0]} != hidden {cols}")
    if not (x.dtype == w.dtype == dy.dtype == torch.float32):
        raise TypeError("float32 only")
    if not (x.is_cuda and w.is_cuda and dy.is_cuda):
        raise ValueError("x, w and dy must be on the GPU")
    if x.stride(1) != 1 or dy.stride(1) != 1 or not w.is_contiguous():
        raise ValueError("x and dy rows and w must be contiguous")
    if cols > MAX_COLS:
        raise ValueError(f"hidden {cols} exceeds the Triton block limit {MAX_COLS}")
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w)
    if rows == 0 or cols == 0:
        return dx, dw
    block = triton.next_power_of_2(cols)
    rows_per_program = max(1, triton.cdiv(rows, TARGET_PROGRAMS))
    programs = triton.cdiv(rows, rows_per_program)
    partial = torch.empty((programs, block), device=x.device, dtype=torch.float32)
    rmsnorm_bwd_kernel[(programs,)](
        x, w, dy, dx, partial, x.stride(0), dy.stride(0), dx.stride(0),
        rows, cols, eps, ROWS=rows_per_program, BLOCK=block,
        num_warps=num_warps)
    torch.sum(partial[:, :cols], dim=0, out=dw)
    return dx, dw


def autograd_backward_fn(forward, eps=EPS):
    """Time the backward pass alone: the forward graph is built once per set
    of inputs, outside the timed region, and every call replays it.

    `forward(x, w)` is the expression under test (the eager formula, or an
    nn.RMSNorm module). Gradients are cleared before each call, so the cost
    is one backward pass plus the two gradient allocations, which is what the
    handwritten kernel also produces.
    """
    state = {}

    def run(x, w, dy):
        key = (x.data_ptr(), w.data_ptr(), x.shape)
        if key not in state:
            state.clear()
            xg = x.detach().requires_grad_(True)
            wg = w.detach().requires_grad_(True)
            state[key] = (xg, wg, forward(xg, wg))
        xg, wg, y = state[key]
        xg.grad = None
        wg.grad = None
        y.backward(dy, retain_graph=True)
        return xg.grad, wg.grad
    return run


def eager_forward(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def native_forward(x, w, eps=EPS):
    """PyTorch's fused RMSNorm. torch.nn.RMSNorm dispatches to this same
    path (_fused_rms_norm); taking the weight as an argument lets autograd
    deliver its gradient to the tensor under test."""
    return torch.nn.functional.rms_norm(x, (w.shape[0],), w, eps)


def conditioning(x, eps=EPS):
    """How much a float32 rounding error in the two terms of dx is amplified.

    dx = r*g - (r^3/H) * x * sum(x*g). At H = 1 the two terms nearly cancel:
    dx = g*r*eps/(ms+eps), so the result is smaller than either term by
    (ms+eps)/eps - about 1e6 for unit-scale inputs and eps = 1e-6. No float32
    implementation is accurate there; PyTorch's own backward is not either
    (measured 1.6e-3 to 7.2e-2 relative error against a double reference at
    hidden 1, against 2.9e-2 for this kernel).
    """
    ms = (x.detach().double() ** 2).mean(dim=-1)
    return float(((ms + eps) / eps).max())


def reference(x, w, dy, eps=EPS):
    """dx and dw on the host in double precision, from the closed form."""
    xd = x.detach().cpu().double()
    wd = w.detach().cpu().double()
    dyd = dy.detach().cpu().double()
    cols = xd.shape[1]
    r = torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + eps)
    g = dyd * wd
    dx = r * g - (r ** 3 / cols) * xd * (xd * g).sum(dim=-1, keepdim=True)
    dw = (dyd * xd * r).sum(dim=0)
    return dx, dw
