"""
Benchmark the training kernels: forward plus backward, against eager autograd and torch.compile.

    python bench_train.py --dtype bf16                     # Qwen2.5-7B shapes
    python bench_train.py --dtype bf16 --model llama3-8b --csv train_llama_bf16.csv

Same layout as bench.py. "% peak" counts the bytes the fused forward and backward
kernels must move between them, so it says how close the whole training step for
that op gets to the GPU's memory bandwidth.
"""

import argparse
import csv
import sys

import torch

from bench import DTYPES, MODELS, peak_bandwidth, time_ms
from kernels import fused_add_rmsnorm_ref, rmsnorm_ref, silu_and_mul_ref
from training import fused_add_rmsnorm_train, rmsnorm_train, silu_and_mul_train


def leaf(*shape, dtype):
    return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)


def step(fn, inputs, grad_outputs):
    """One forward and backward pass. autograd.grad avoids accumulating into .grad."""
    def run():
        out = fn(*inputs)
        outs = list(out) if isinstance(out, tuple) else [out]
        torch.autograd.grad(outs, inputs, grad_outputs)
    return run


def cases(hidden, inter, tokens, dtype):
    """(name, bytes the fused forward and backward must move, eager fn, fused fn, inputs, grads)."""
    e = torch.tensor([], dtype=dtype).element_size()
    x, res, w = leaf(tokens, hidden, dtype=dtype), leaf(tokens, hidden, dtype=dtype), leaf(hidden, dtype=dtype)
    dy, dres = torch.randn_like(x), torch.randn_like(x)
    gate_up = leaf(tokens, 2 * inter, dtype=dtype)
    dact = torch.randn(tokens, inter, device="cuda", dtype=dtype)
    return [
        # forward: read x, write y. backward: read dy and x, write dx.
        ("rmsnorm", 5 * tokens * hidden * e, rmsnorm_ref, rmsnorm_train, (x, w), [dy]),
        # forward: read x and residual, write y and the new residual.
        # backward: read dy, the residual's gradient and the saved sum, write one gradient.
        ("add+rmsnorm", 8 * tokens * hidden * e, fused_add_rmsnorm_ref, fused_add_rmsnorm_train,
         (x, res, w), [dy, dres]),
        # forward: read gate and up, write y. backward: read gate, up and dy, write both gradients.
        ("silu_and_mul", 8 * tokens * inter * e, silu_and_mul_ref, silu_and_mul_train,
         (gate_up,), [dact]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="qwen2.5-7b")
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 8192, 16384],
                   help="rows per call, for example batch size times sequence length")
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    p.add_argument("--peak-gbps", type=float, help="override the GPU's peak bandwidth")
    p.add_argument("--csv", help="also write the results to this CSV file")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench_train.py needs an NVIDIA GPU. Use pytest to check gradients without one.")

    dtype = DTYPES[args.dtype]
    hidden, inter = MODELS[args.model]
    peak = peak_bandwidth(args.peak_gbps)
    print(f"GPU: {torch.cuda.get_device_name()}   peak bandwidth: {peak or 'unknown'} GB/s")
    print(f"Model shapes: {args.model} (hidden {hidden}, intermediate {inter}), dtype {args.dtype}")
    print("Times are one forward plus backward pass.\n")

    header = ["op", "tokens", "eager_ms", "compile_ms", "triton_ms",
              "speedup_vs_eager", "speedup_vs_compile", "triton_GBps", "pct_of_peak"]
    rows = []
    for tokens in args.tokens:
        for name, nbytes, ref, fused, inputs, gouts in cases(hidden, inter, tokens, dtype):
            eager_ms = time_ms(step(ref, inputs, gouts))
            compile_ms = None
            if not args.no_compile:
                compiled = step(torch.compile(ref), inputs, gouts)
                compiled()                                  # compile once before timing
                compile_ms = time_ms(compiled)
            triton_ms = time_ms(step(fused, inputs, gouts))
            gbps = nbytes / (triton_ms * 1e-3) / 1e9
            rows.append([name, tokens, eager_ms, compile_ms, triton_ms, eager_ms / triton_ms,
                         compile_ms / triton_ms if compile_ms else None,
                         gbps, 100 * gbps / peak if peak else None])

    fmt = "{:<13} {:>6} {:>9} {:>10} {:>9} {:>9} {:>11} {:>7} {:>7}"
    print(fmt.format("op", "tokens", "eager ms", "compile ms", "triton ms",
                     "vs eager", "vs compile", "GB/s", "% peak"))
    for r in rows:
        print(fmt.format(r[0], r[1], f"{r[2]:.4f}", "-" if r[3] is None else f"{r[3]:.4f}",
                         f"{r[4]:.4f}", f"{r[5]:.2f}x", "-" if r[6] is None else f"{r[6]:.2f}x",
                         f"{r[7]:.0f}", "-" if r[8] is None else f"{r[8]:.0f}%"))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
