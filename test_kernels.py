"""Correctness tests: every fused kernel must match the PyTorch reference.

    pytest -q                       # on a GPU: fp32, fp16 and bf16
    pytest -q                       # without a GPU: fp32 in Triton's interpreter (slower)
"""

import pytest
import torch

from kernels import (fused_add_rmsnorm, fused_add_rmsnorm_ref, rmsnorm, rmsnorm_ref,
                     silu_and_mul, silu_and_mul_ref)

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

# Hidden sizes that are not powers of two (1000, 3584) check the masking.
SHAPES = [(1, 64), (7, 1000), (33, 3584), (4, 4096)] if CUDA else [(1, 64), (7, 1000), (3, 3584)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16] if CUDA else [torch.float32]
TOL = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def rand(*shape, dtype):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm(shape, dtype):
    x = rand(*shape, dtype=dtype)
    w = rand(shape[-1], dtype=dtype)
    atol, rtol = TOL[dtype]
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_ref(x, w), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm(shape, dtype):
    x = rand(*shape, dtype=dtype)
    res = rand(*shape, dtype=dtype)
    w = rand(shape[-1], dtype=dtype)
    res_before = res.clone()

    y, new_res = fused_add_rmsnorm(x, res, w)
    y_ref, new_res_ref = fused_add_rmsnorm_ref(x, res, w)

    atol, rtol = TOL[dtype]
    torch.testing.assert_close(new_res, new_res_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(res, res_before)          # inputs are left untouched


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(shape, dtype):
    rows, d = shape
    x = rand(rows, 2 * d, dtype=dtype)
    atol, rtol = TOL[dtype]
    torch.testing.assert_close(silu_and_mul(x), silu_and_mul_ref(x), atol=atol, rtol=rtol)


def test_3d_input_keeps_shape():
    x = rand(2, 5, 1000, dtype=torch.float32)
    w = rand(1000, dtype=torch.float32)
    assert rmsnorm(x, w).shape == x.shape
    assert silu_and_mul(rand(2, 5, 2000, dtype=torch.float32)).shape == (2, 5, 1000)
