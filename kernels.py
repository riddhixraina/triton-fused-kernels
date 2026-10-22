"""
Fused Triton kernels for the memory-bound ops in a Llama/Qwen decoder layer.

Each kernel reads its inputs once and writes its outputs once. The PyTorch eager
versions below launch several kernels and round-trip intermediates through GPU
memory, which is why fusing them helps: these ops are limited by memory bandwidth,
not by arithmetic.

    rmsnorm(x, w)                    y = x / sqrt(mean(x^2) + eps) * w
    fused_add_rmsnorm(x, res, w)     res = x + res, then y = rmsnorm(res, w)
    silu_and_mul(x)                  y = silu(x[..., :d]) * x[..., d:]   (SwiGLU)

All math runs in fp32 inside the kernel and is cast back to the input dtype on store.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reference implementations (what Hugging Face Llama/Qwen do in eager mode)
# ---------------------------------------------------------------------------

def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * y).to(x.dtype)


def fused_add_rmsnorm_ref(x, residual, w, eps: float = 1e-6):
    new_residual = (x.float() + residual.float()).to(x.dtype)
    return rmsnorm_ref(new_residual, w, eps), new_residual


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_kernel(X, W, Y, stride_x, stride_y, N, eps, BLOCK_N: tl.constexpr):
    # One program normalizes one row. The whole row fits in registers
    # (BLOCK_N is the next power of two >= N), so x is read from memory once.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * rstd * w

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(X, R, W, Y, stride_x, stride_r, stride_y, N, eps,
                              BLOCK_N: tl.constexpr):
    # Residual add and RMSNorm in one pass: eager PyTorch writes x + residual
    # to memory and reads it back for the norm. Here it never leaves registers,
    # apart from the updated residual the next layer needs.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    s = x + r
    s_rounded = s.to(R.dtype.element_ty)          # residual is kept in the model dtype
    tl.store(R + row * stride_r + cols, s_rounded, mask=mask)

    s = s_rounded.to(tl.float32)                  # normalize what the next layer will see
    mean_sq = tl.sum(s * s, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = s * rstd * w

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, stride_y, D, BLOCK_D: tl.constexpr):
    # x holds [gate | up] side by side, the layout of a merged gate_up projection.
    # Grid: (rows, column blocks).
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_x + D + cols, mask=mask, other=0.0).to(tl.float32)
    y = gate * tl.sigmoid(gate) * up

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _rows(t: torch.Tensor) -> torch.Tensor:
    """View any (..., N) tensor as a contiguous 2-D (rows, N) tensor."""
    return t.reshape(-1, t.shape[-1]).contiguous()


def _num_warps(block: int) -> int:
    # More warps for wider rows, so each thread holds a manageable slice.
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = _rows(x)
    rows, n = x2.shape
    assert w.shape == (n,), f"weight must be ({n},), got {tuple(w.shape)}"
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(n)
    _rmsnorm_kernel[(rows,)](x2, w, y, x2.stride(0), y.stride(0), n, eps,
                             BLOCK_N=block, num_warps=_num_warps(block))
    return y.reshape(shape)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor,
                      eps: float = 1e-6):
    """Returns (normed, new_residual). Unlike vLLM's version, inputs are not modified."""
    assert x.shape == residual.shape
    shape = x.shape
    x2 = _rows(x)
    r2 = _rows(residual).clone()                  # kernel writes the new residual here
    rows, n = x2.shape
    assert w.shape == (n,)
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(n)
    _fused_add_rmsnorm_kernel[(rows,)](x2, r2, w, y, x2.stride(0), r2.stride(0), y.stride(0),
                                       n, eps, BLOCK_N=block, num_warps=_num_warps(block))
    return y.reshape(shape), r2.reshape(shape)


def silu_and_mul(x: torch.Tensor, block_d: int = 1024) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, "last dim must hold [gate | up]"
    d = x.shape[-1] // 2
    x2 = _rows(x)
    rows = x2.shape[0]
    y = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    grid = (rows, triton.cdiv(d, block_d))
    _silu_and_mul_kernel[grid](x2, y, x2.stride(0), y.stride(0), d,
                               BLOCK_D=block_d, num_warps=4)
    return y.reshape(*x.shape[:-1], d)
