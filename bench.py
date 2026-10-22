"""
Benchmark the fused Triton kernels against PyTorch eager and torch.compile on a GPU.

    python bench.py                          # Qwen2.5-7B shapes, fp16
    python bench.py --model llama3-8b --dtype bf16
    python bench.py --no-compile --csv results.csv

For each op and token count it reports time per call and the memory bandwidth the
Triton kernel achieves, as a share of the GPU's peak. These ops are memory-bound, so
bandwidth (not FLOPs) is the number that says how close to optimal a kernel is.
"""

import argparse
import csv
import sys

import torch
import triton
import triton.testing

from kernels import (fused_add_rmsnorm, fused_add_rmsnorm_ref, rmsnorm, rmsnorm_ref,
                     silu_and_mul, silu_and_mul_ref)

MODELS = {  # hidden size, MLP intermediate size
    "qwen2.5-7b": (3584, 18944),
    "llama3-8b": (4096, 14336),
    "qwen2.5-1.5b": (1536, 8960),
}

# Peak HBM bandwidth in GB/s, from vendor spec sheets. Pass --peak-gbps to override.
PEAK_GBPS = {
    "H100": 3350, "A100-SXM4-80GB": 2039, "A100 80GB": 1935, "A100-SXM4-40GB": 1555,
    "A100-PCIE-40GB": 1555, "L4": 300, "T4": 320, "V100": 900, "A10G": 600,
    "RTX 4090": 1008, "RTX 3090": 936,
}

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def peak_bandwidth(override):
    if override:
        return override
    name = torch.cuda.get_device_name()
    for key, gbps in PEAK_GBPS.items():
        if key in name:
            return gbps
    return None


def time_ms(fn):
    """Median time of one call, in milliseconds."""
    try:
        return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
    except TypeError:  # older Triton without return_mode
        return triton.testing.do_bench(fn, warmup=25, rep=100)


def cases(hidden, inter, tokens, dtype):
    """(name, bytes the fused kernel must move, eager fn, fused fn) for one token count."""
    e = torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    res = torch.randn_like(x)
    w = torch.randn(hidden, device="cuda", dtype=dtype)
    gate_up = torch.randn(tokens, 2 * inter, device="cuda", dtype=dtype)
    return [
        # read x, write y (the weight is tiny and stays in cache)
        ("rmsnorm", 2 * tokens * hidden * e,
         lambda: rmsnorm_ref(x, w), lambda: rmsnorm(x, w)),
        # read x and residual, write the new residual and y
        ("add+rmsnorm", 4 * tokens * hidden * e,
         lambda: fused_add_rmsnorm_ref(x, res, w), lambda: fused_add_rmsnorm(x, res, w)),
        # read gate and up, write y
        ("silu_and_mul", 3 * tokens * inter * e,
         lambda: silu_and_mul_ref(gate_up), lambda: silu_and_mul(gate_up)),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    p.add_argument("--dtype", choices=DTYPES, default="fp16")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 512, 2048, 8192],
                   help="rows per call: 1 is single-token decode, large values are prefill")
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    p.add_argument("--peak-gbps", type=float, help="override the GPU's peak bandwidth")
    p.add_argument("--csv", help="also write the results to this CSV file")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench.py needs an NVIDIA GPU. Use pytest to check correctness without one.")

    dtype = DTYPES[args.dtype]
    hidden, inter = MODELS[args.model]
    peak = peak_bandwidth(args.peak_gbps)
    print(f"GPU: {torch.cuda.get_device_name()}   peak bandwidth: {peak or 'unknown'} GB/s")
    print(f"Model shapes: {args.model} (hidden {hidden}, intermediate {inter}), dtype {args.dtype}\n")

    header = ["op", "tokens", "eager_ms", "compile_ms", "triton_ms",
              "speedup_vs_eager", "triton_GBps", "pct_of_peak"]
    rows = []
    for tokens in args.tokens:
        for name, nbytes, eager_fn, fused_fn in cases(hidden, inter, tokens, dtype):
            eager_ms = time_ms(eager_fn)
            compile_ms = None
            if not args.no_compile:
                compiled = torch.compile(eager_fn)
                compiled()                                  # compile once before timing
                compile_ms = time_ms(compiled)
            triton_ms = time_ms(fused_fn)
            gbps = nbytes / (triton_ms * 1e-3) / 1e9
            rows.append([name, tokens, eager_ms, compile_ms, triton_ms, eager_ms / triton_ms,
                         gbps, 100 * gbps / peak if peak else None])

    fmt = "{:<13} {:>6} {:>10} {:>11} {:>10} {:>9} {:>9} {:>8}"
    print(fmt.format("op", "tokens", "eager ms", "compile ms", "triton ms", "speedup", "GB/s", "% peak"))
    for r in rows:
        print(fmt.format(r[0], r[1], f"{r[2]:.4f}", "-" if r[3] is None else f"{r[3]:.4f}",
                         f"{r[4]:.4f}", f"{r[5]:.2f}x", f"{r[6]:.0f}",
                         "-" if r[7] is None else f"{r[7]:.0f}%"))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
