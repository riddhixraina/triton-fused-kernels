"""Correctness of the CUDA C++ RMSNorm against the PyTorch reference. Needs a GPU.

    pytest -q test_cuda.py
"""

import pytest
import torch

from kernels import rmsnorm_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

# 1000 and 3584 take the 16-byte vector path, 1001 the one-value-at-a-time path.
SHAPES = [(1, 64), (7, 1000), (33, 3584), (4, 4096), (5, 1001), (300, 4096)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_cuda(shape, dtype):
    from cuda_rmsnorm import rmsnorm_cuda
    x = torch.randn(*shape, device="cuda").to(dtype)
    w = torch.randn(shape[-1], device="cuda").to(dtype)
    atol, rtol = TOL[dtype]
    torch.testing.assert_close(rmsnorm_cuda(x, w), rmsnorm_ref(x, w), atol=atol, rtol=rtol)


def test_non_contiguous_and_3d():
    from cuda_rmsnorm import rmsnorm_cuda
    base = torch.randn(6, 2 * 1024, device="cuda", dtype=torch.bfloat16)
    x = base[:, :1024]                                   # every other block of 1024: strided rows
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(rmsnorm_cuda(x, w), rmsnorm_ref(x, w), atol=2e-2, rtol=2e-2)
    x3 = torch.randn(2, 5, 1024, device="cuda", dtype=torch.bfloat16)
    assert rmsnorm_cuda(x3, w).shape == x3.shape
