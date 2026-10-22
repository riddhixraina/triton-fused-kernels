"""Gradient tests: every training kernel must match eager PyTorch autograd.

    pytest -q test_training.py      # on a GPU: fp32, fp16 and bf16
                                    # without a GPU: fp32 in Triton's interpreter
"""

import pytest
import torch

from kernels import fused_add_rmsnorm_ref, rmsnorm_ref, silu_and_mul_ref
from training import fused_add_rmsnorm_train, rmsnorm_train, silu_and_mul_train

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

# More rows than the GPU has SMs (or the interpreter's 4 programs) checks that every
# program's slice of rows is covered and the weight-gradient partials add up.
SHAPES = [(1, 64), (7, 1000), (300, 3584), (37, 4096)] if CUDA else [(1, 64), (7, 1000), (9, 3584)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16] if CUDA else [torch.float32]
TOL = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}


def leaf(*shape, dtype):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(dtype).requires_grad_()


def close(actual, expected, dtype):
    """Compare relative to the tensor's scale, since weight gradients sum over many rows."""
    scale = expected.float().abs().max().clamp_min(1.0)
    torch.testing.assert_close(actual.float() / scale, expected.float() / scale,
                               atol=TOL[dtype], rtol=TOL[dtype])


def grads(outputs, inputs, grad_outputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_backward(shape, dtype):
    x, w = leaf(*shape, dtype=dtype), leaf(shape[-1], dtype=dtype)
    dy = torch.randn(shape, device=DEVICE).to(dtype)

    y = rmsnorm_train(x, w)
    y_ref = rmsnorm_ref(x, w)
    close(y, y_ref, dtype)
    for got, want in zip(grads(y, (x, w), dy), grads(y_ref, (x, w), dy)):
        close(got, want, dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("residual_used", [True, False])
def test_fused_add_rmsnorm_backward(shape, dtype, residual_used):
    x, res, w = leaf(*shape, dtype=dtype), leaf(*shape, dtype=dtype), leaf(shape[-1], dtype=dtype)
    dy = torch.randn(shape, device=DEVICE).to(dtype)
    dres = torch.randn(shape, device=DEVICE).to(dtype)

    y, new_res = fused_add_rmsnorm_train(x, res, w)
    y_ref, new_res_ref = fused_add_rmsnorm_ref(x, res, w)
    close(y, y_ref, dtype)
    close(new_res, new_res_ref, dtype)

    # The new residual feeds the next layer, so it usually has its own gradient too.
    outs, outs_ref = ([y, new_res], [y_ref, new_res_ref]) if residual_used else ([y], [y_ref])
    gouts = [dy, dres] if residual_used else [dy]
    for got, want in zip(grads(outs, (x, res, w), gouts), grads(outs_ref, (x, res, w), gouts)):
        close(got, want, dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul_backward(shape, dtype):
    rows, d = shape
    x = leaf(rows, 2 * d, dtype=dtype)
    dy = torch.randn(rows, d, device=DEVICE).to(dtype)

    y = silu_and_mul_train(x)
    y_ref = silu_and_mul_ref(x)
    close(y, y_ref, dtype)
    close(grads(y, x, dy)[0], grads(y_ref, x, dy)[0], dtype)


def test_3d_input_backward():
    x, w = leaf(2, 5, 1000, dtype=torch.float32), leaf(1000, dtype=torch.float32)
    dx, dw = grads(rmsnorm_train(x, w), (x, w), torch.ones(2, 5, 1000, device=DEVICE))
    assert dx.shape == x.shape and dw.shape == w.shape
