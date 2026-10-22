"""
RMSNorm written in CUDA C++ (csrc/rmsnorm.cu), for comparison with the Triton kernel.

    from cuda_rmsnorm import rmsnorm_cuda
    y = rmsnorm_cuda(x, w)

The extension compiles the first time it's used, which takes about a minute. That needs
nvcc and ninja. Colab GPU runtimes have nvcc, and ninja is in requirements.txt.
"""

import os

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "rmsnorm.cu")
_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        try:
            _ext = load(name="cuda_rmsnorm", sources=[_SRC], extra_cuda_cflags=["-O3"])
        except Exception as err:
            raise RuntimeError(
                f"Could not build the CUDA extension ({err}). It needs nvcc from a CUDA toolkit "
                f"whose major version matches PyTorch's (torch.version.cuda = {torch.version.cuda})."
            ) from err
    return _ext


def rmsnorm_cuda(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _load().rmsnorm(x, w, eps)
