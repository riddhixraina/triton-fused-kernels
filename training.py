"""
Training versions of the fused kernels: forward and backward passes wired into autograd.

    rmsnorm_train(x, w)                  y = x / sqrt(mean(x^2) + eps) * w
    fused_add_rmsnorm_train(x, res, w)   res = x + res, then y = rmsnorm(res, w)
    silu_and_mul_train(x)                y = silu(x[..., :d]) * x[..., d:]   (SwiGLU)

Each one matches the eager PyTorch op it replaces, including its gradients, so it can
be dropped into a training loop. The forward kernels also save what the backward pass
needs (one fp32 rstd per row for the norms), and each backward pass is a single kernel:
eager autograd runs several kernels and saves several intermediate tensors for the same
gradients.

How the weight gradient of RMSNorm is reduced: dw sums dy * x * rstd over every row. One
program per row would need a (rows, N) buffer for the partial sums, which costs as much
memory traffic as the kernel itself. Instead a fixed number of programs (one per GPU
streaming multiprocessor) each loop over a slice of rows and keep their partial sum in
registers, so the buffer is only (programs, N).
"""

import os

import torch
import triton
import triton.language as tl

from kernels import silu_and_mul

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_WARP_CONFIGS = [triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)]


def _autotune_warps(fn):
    return fn if _INTERPRET else triton.autotune(configs=_WARP_CONFIGS, key=["N"])(fn)


def _num_programs(rows: int, device: torch.device) -> int:
    if device.type == "cuda":
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        sms = 4                                   # the CPU interpreter runs programs one by one
    return max(1, min(rows, sms))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@_autotune_warps
@triton.jit
def _rmsnorm_fwd_kernel(X, R, W, Y, R_OUT, RSTD, stride_x, stride_r, stride_y, stride_ro,
                        N, eps, HAS_RES: tl.constexpr, BLOCK_N: tl.constexpr):
    # One row per program. With HAS_RES, normalizes x + residual and writes the sum to
    # R_OUT. Saves rstd for the backward pass.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        s = (x + r).to(R_OUT.dtype.element_ty)    # residual is kept in the model dtype
        tl.store(R_OUT + row * stride_ro + cols, s, mask=mask)
        x = s.to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(RSTD + row, rstd)
    tl.store(Y + row * stride_y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


@_autotune_warps
@triton.jit
def _rmsnorm_bwd_kernel(DY, X, W, RSTD, DRES, DX, DW_PART,
                        stride_dy, stride_x, stride_dres, stride_dx,
                        M, N, rows_per_program, HAS_DRES: tl.constexpr, BLOCK_N: tl.constexpr):
    # With g = dy * w and c = mean(g * x):
    #   dx = rstd * g - x * rstd^3 * c       (plus the residual's own gradient, if any)
    #   dw = sum over rows of dy * x * rstd
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)

    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)

        g = dy * w
        c = tl.sum(g * x, axis=0) / N
        dx = rstd * g - x * (rstd * rstd * rstd) * c
        if HAS_DRES:
            dx += tl.load(DRES + row * stride_dres + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)
        dw += dy * x * rstd

    tl.store(DW_PART + pid * N + cols, dw, mask=mask)


@triton.jit
def _silu_and_mul_bwd_kernel(X, DY, DX, stride_x, stride_dy, stride_dx, D, BLOCK_D: tl.constexpr):
    # y = silu(g) * u with silu(g) = g * sigmoid(g), so
    #   du = dy * silu(g)
    #   dg = dy * u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    g = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(X + row * stride_x + D + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    du = dy * g * sig
    dg = dy * u * sig * (1.0 + g * (1.0 - sig))

    tl.store(DX + row * stride_dx + cols, dg.to(DX.dtype.element_ty), mask=mask)
    tl.store(DX + row * stride_dx + D + cols, du.to(DX.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------

def _rows(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1]).contiguous()


def _norm_backward(dy, x, w, rstd, dres=None):
    """Shared backward for both norms. x is what was normalized."""
    rows, n = x.shape
    dx = torch.empty_like(x)
    programs = _num_programs(rows, x.device)
    dw_part = torch.empty((programs, n), dtype=torch.float32, device=x.device)
    _rmsnorm_bwd_kernel[(programs,)](
        dy, x, w, rstd, dres if dres is not None else dy, dx, dw_part,
        dy.stride(0), x.stride(0), dres.stride(0) if dres is not None else 0, dx.stride(0),
        rows, n, triton.cdiv(rows, programs),
        HAS_DRES=dres is not None, BLOCK_N=triton.next_power_of_2(n))
    return dx, dw_part.sum(0).to(w.dtype)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = _rows(x)
        rows, n = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        _rmsnorm_fwd_kernel[(rows,)](x2, x2, w, y, y, rstd,
                                     x2.stride(0), 0, y.stride(0), 0, n, eps,
                                     HAS_RES=False, BLOCK_N=triton.next_power_of_2(n))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        dx, dw = _norm_backward(_rows(dy), x2, w, rstd)
        return dx.reshape(ctx.shape), dw, None


class _FusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, w, eps):
        shape = x.shape
        x2, r2 = _rows(x), _rows(residual)
        rows, n = x2.shape
        y = torch.empty_like(x2)
        r_out = torch.empty_like(r2)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        _rmsnorm_fwd_kernel[(rows,)](x2, r2, w, y, r_out, rstd,
                                     x2.stride(0), r2.stride(0), y.stride(0), r_out.stride(0),
                                     n, eps, HAS_RES=True, BLOCK_N=triton.next_power_of_2(n))
        ctx.save_for_backward(r_out, w, rstd)     # backward needs the sum, not x and res
        ctx.shape = shape
        return y.reshape(shape), r_out.reshape(shape)

    @staticmethod
    def backward(ctx, dy, dres):
        s, w, rstd = ctx.saved_tensors
        dres = None if dres is None else _rows(dres)
        ds, dw = _norm_backward(_rows(dy), s, w, rstd, dres)
        ds = ds.reshape(ctx.shape)
        return ds, ds, dw, None                  # x and the residual get the same gradient


class _SiLUAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block_d):
        ctx.save_for_backward(x)                # the backward recomputes silu from x
        ctx.block_d = block_d
        return silu_and_mul(x, block_d)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        d = x.shape[-1] // 2
        x2, dy2 = _rows(x), _rows(dy)
        rows = x2.shape[0]
        dx = torch.empty_like(x2)
        grid = (rows, triton.cdiv(d, ctx.block_d))
        _silu_and_mul_bwd_kernel[grid](x2, dy2, dx, x2.stride(0), dy2.stride(0), dx.stride(0), d,
                                       BLOCK_D=ctx.block_d, num_warps=4)
        return dx.reshape(x.shape), None


def rmsnorm_train(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _RMSNorm.apply(x, w, eps)


def fused_add_rmsnorm_train(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor,
                            eps: float = 1e-6):
    """Returns (normed, new_residual), both differentiable."""
    return _FusedAddRMSNorm.apply(x, residual, w, eps)


def silu_and_mul_train(x: torch.Tensor, block_d: int = 1024) -> torch.Tensor:
    return _SiLUAndMul.apply(x, block_d)
