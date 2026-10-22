"""Without a GPU, run the Triton kernels in Triton's CPU interpreter.

TRITON_INTERPRET must be set before triton is imported, which is why it lives here.
"""

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
