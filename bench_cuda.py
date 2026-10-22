"""
RMSNorm three ways: PyTorch eager, the Triton kernel, and the CUDA C++ kernel.

    python bench_cuda.py --dtype bf16
    python bench_cuda.py --dtype bf16 --model llama3-8b --csv cuda_llama_bf16.csv
"""

import argparse
import csv
import sys

import torch

from bench import DTYPES, MODELS, peak_bandwidth, time_ms
from cuda_rmsnorm import rmsnorm_cuda
from kernels import rmsnorm, rmsnorm_ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 512, 2048, 8192])
    p.add_argument("--peak-gbps", type=float, help="override the GPU's peak bandwidth")
    p.add_argument("--csv", help="also write the results to this CSV file")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench_cuda.py needs an NVIDIA GPU.")

    dtype = DTYPES[args.dtype]
    hidden, _ = MODELS[args.model]
    peak = peak_bandwidth(args.peak_gbps)
    e = torch.tensor([], dtype=dtype).element_size()
    print(f"GPU: {torch.cuda.get_device_name()}   peak bandwidth: {peak or 'unknown'} GB/s")
    print(f"RMSNorm, hidden {hidden} ({args.model}), dtype {args.dtype}\n")

    header = ["tokens", "eager_ms", "triton_ms", "cuda_ms", "triton_pct_peak", "cuda_pct_peak",
              "cuda_vs_triton"]
    rows = []
    for tokens in args.tokens:
        x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
        w = torch.randn(hidden, device="cuda", dtype=dtype)
        torch.testing.assert_close(rmsnorm_cuda(x, w), rmsnorm_ref(x, w), atol=2e-2, rtol=2e-2)
        nbytes = 2 * tokens * hidden * e                    # read x, write y
        eager_ms = time_ms(lambda: rmsnorm_ref(x, w))
        triton_ms = time_ms(lambda: rmsnorm(x, w))
        cuda_ms = time_ms(lambda: rmsnorm_cuda(x, w))
        pct = (lambda ms: 100 * nbytes / (ms * 1e-3) / 1e9 / peak) if peak else (lambda ms: None)
        rows.append([tokens, eager_ms, triton_ms, cuda_ms, pct(triton_ms), pct(cuda_ms),
                     triton_ms / cuda_ms])

    fmt = "{:>6} {:>9} {:>10} {:>9} {:>14} {:>12} {:>15}"
    print(fmt.format("tokens", "eager ms", "triton ms", "cuda ms", "triton % peak", "cuda % peak",
                     "cuda vs triton"))
    show = lambda v: "-" if v is None else f"{v:.0f}%"
    for r in rows:
        print(fmt.format(r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}", show(r[4]), show(r[5]),
                         f"{r[6]:.2f}x"))
    print("\ncuda vs triton above 1.00x means the CUDA kernel is faster.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
