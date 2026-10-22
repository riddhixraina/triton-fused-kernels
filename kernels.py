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

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Let Triton time a few warp counts per hidden size and keep the fastest. Skipped in
# the CPU interpreter, where there is nothing to time.
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_WARP_CONFIGS = [triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)]


def _autotune_warps(fn):
    return fn if _INTERPRET else triton.autotune(configs=_WARP_CONFIGS, key=["N"])(fn)


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

@_autotune_warps
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


@_autotune_warps
@triton.jit
def _fused_add_rmsnorm_kernel(X, R, W, Y, R_OUT, stride_x, stride_r, stride_y, stride_ro,
                              N, eps, BLOCK_N: tl.constexpr):
    # Residual add and RMSNorm in one pass: eager PyTorch writes x + residual
    # to memory and reads it back for the norm. Here it never leaves registers,
    # apart from the updated residual the next layer needs, which goes to R_OUT.
    # Inputs are never written, so autotuning can safely rerun the kernel.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    s = x + r
    s_rounded = s.to(R_OUT.dtype.element_ty)      # residual is kept in the model dtype
    tl.store(R_OUT + row * stride_ro + cols, s_rounded, mask=mask)

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


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = _rows(x)
    rows, n = x2.shape
    assert w.shape == (n,), f"weight must be ({n},), got {tuple(w.shape)}"
    y = torch.empty_like(x2)
    _rmsnorm_kernel[(rows,)](x2, w, y, x2.stride(0), y.stride(0), n, eps,
                             BLOCK_N=triton.next_power_of_2(n))
    return y.reshape(shape)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor,
                      eps: float = 1e-6):
    """Returns (normed, new_residual). Inputs are not modified.

    The new residual goes to a fresh buffer. An earlier version cloned the residual
    and updated the clone in place, which added a full extra read and write of the
    tensor: about 50% more memory traffic than the kernel itself.
    """
    assert x.shape == residual.shape
    shape = x.shape
    x2 = _rows(x)
    r2 = _rows(residual)
    rows, n = x2.shape
    assert w.shape == (n,)
    y = torch.empty_like(x2)
    r_out = torch.empty_like(r2)
    _fused_add_rmsnorm_kernel[(rows,)](x2, r2, w, y, r_out,
                                       x2.stride(0), r2.stride(0), y.stride(0), r_out.stride(0),
                                       n, eps, BLOCK_N=triton.next_power_of_2(n))
    return y.reshape(shape), r_out.reshape(shape)


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
