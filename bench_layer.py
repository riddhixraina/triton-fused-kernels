"""
Time one training step of a full Llama or Qwen decoder layer, with eager norms and SwiGLU
versus the fused Triton kernels. Everything else (the linear layers and attention) is the
same in both, so this shows how much the fused kernels speed up real training.

    python bench_layer.py                                   # Llama 3 8B layer, 4 x 2048 tokens
    python bench_layer.py --model qwen2.5-7b --batch 8

The layer follows the Llama layout: RMSNorm, attention with grouped-query heads, residual
add and RMSNorm, then the SwiGLU MLP and a second residual add. It leaves out rotary
embeddings and biases, which are the same cost in both versions.
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import DTYPES
from kernels import rmsnorm_ref, silu_and_mul_ref
from training import fused_add_rmsnorm_train, rmsnorm_train, silu_and_mul_train

LAYERS = {  # hidden, attention heads, key/value heads, MLP intermediate
    "llama3-8b": (4096, 32, 8, 14336),
    "qwen2.5-7b": (3584, 28, 4, 18944),
}


def _supports_gqa():
    """enable_gqa arrived in PyTorch 2.5. Older versions repeat the key/value heads instead."""
    t = torch.zeros(1, 2, 1, 4)
    try:
        F.scaled_dot_product_attention(t, t[:, :1], t[:, :1], enable_gqa=True)
        return True
    except TypeError:
        return False


_GQA = _supports_gqa()


class DecoderLayer(nn.Module):
    def __init__(self, hidden, heads, kv_heads, inter, fused):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.fused = fused
        self.q = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.kv = nn.Linear(hidden, 2 * kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.gate_up = nn.Linear(hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.norm1 = nn.Parameter(torch.ones(hidden))
        self.norm2 = nn.Parameter(torch.ones(hidden))

    def attention(self, h):
        b, s, _ = h.shape
        q = self.q(h).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(h).view(b, s, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if _GQA:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            rep = self.heads // self.kv_heads
            out = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1),
                                                 v.repeat_interleave(rep, 1), is_causal=True)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))

    def forward(self, x):
        if self.fused:
            a = self.attention(rmsnorm_train(x, self.norm1))
            y, res = fused_add_rmsnorm_train(a, x, self.norm2)
            return res + self.down(silu_and_mul_train(self.gate_up(y)))
        a = self.attention(rmsnorm_ref(x, self.norm1))
        res = x + a
        y = rmsnorm_ref(res, self.norm2)
        return res + self.down(silu_and_mul_ref(self.gate_up(y)))


def training_step(layer, x, dout):
    params = list(layer.parameters())

    def run():
        for p in params:                 # what optimizer.zero_grad(set_to_none=True) does
            p.grad = None
        x.grad = None
        layer(x).backward(dout)
    return run


def time_step_ms(fn, iters=20):
    """Median of iters timed steps. A layer step takes tens of ms, so time each one."""
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def peak_memory_gb(fn):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=LAYERS, default="llama3-8b")
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench_layer.py needs an NVIDIA GPU.")

    dtype = DTYPES[args.dtype]
    hidden, heads, kv_heads, inter = LAYERS[args.model]
    torch.manual_seed(0)
    eager = DecoderLayer(hidden, heads, kv_heads, inter, fused=False).to("cuda", dtype)
    fused = DecoderLayer(hidden, heads, kv_heads, inter, fused=True).to("cuda", dtype)
    fused.load_state_dict(eager.state_dict())   # same weights, so outputs can be compared

    x = torch.randn(args.batch, args.seq, hidden, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn_like(x)

    with torch.no_grad():
        diff = (fused(x) - eager(x)).abs().max().item()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{args.model} decoder layer, {args.batch} x {args.seq} = {args.batch * args.seq:,} tokens, "
          f"{args.dtype}. Max output difference, fused vs eager: {diff:.3g}\n")

    variants = [("eager", training_step(eager, x, dout)),
                ("triton kernels", training_step(fused, x, dout))]
    if not args.no_compile:
        variants.append(("torch.compile", training_step(torch.compile(eager), x, dout)))

    results = []
    for name, fn in variants:
        for _ in range(3):                      # warm up (and compile)
            fn()
        results.append((name, time_step_ms(fn), peak_memory_gb(fn)))

    base_ms, base_gb = results[0][1], results[0][2]
    print(f"{'version':<16} {'step ms':>9} {'vs eager':>9} {'peak GB':>8} {'vs eager':>9}")
    for name, ms, gb in results:
        print(f"{name:<16} {ms:>9.2f} {base_ms / ms:>8.2f}x {gb:>8.2f} {100 * (gb / base_gb - 1):>+8.1f}%")
    print("\nstep ms is one forward plus backward pass. peak GB is the most GPU memory allocated "
          "during a step, including weights and gradients.")


if __name__ == "__main__":
    main()
