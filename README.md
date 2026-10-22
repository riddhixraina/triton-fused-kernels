# Fused Transformer Kernels in Triton

Three fused GPU kernels for the memory-bound ops in every Llama and Qwen decoder layer, benchmarked against PyTorch eager and `torch.compile`. Each one also has a backward pass for training, and RMSNorm has a CUDA C++ version for comparison with Triton.

| Kernel | What it computes | Where it runs in the model |
|---|---|---|
| `rmsnorm` | `x / sqrt(mean(x²) + eps) * w` | Before attention and before the MLP |
| `fused_add_rmsnorm` | residual add, then RMSNorm | Every layer boundary |
| `silu_and_mul` | `silu(gate) * up` (SwiGLU) | Inside the MLP |

## Why fuse them

These ops do almost no arithmetic per byte, so their speed is set by memory bandwidth. Eager PyTorch runs RMSNorm as about six separate kernels (cast to fp32, square, mean, rsqrt, multiply, multiply by the weight, cast back), and each one reads and writes the whole tensor in GPU memory. The fused kernel reads `x` once and writes `y` once. The residual-add version also keeps `x + residual` in registers instead of writing it out and reading it back.

So the goal is not "faster than PyTorch" in the abstract. It's **how close each kernel gets to the GPU's peak memory bandwidth**, which `bench.py` reports directly.

## Run it

You need an NVIDIA GPU. A free Colab T4 runs the inference benchmarks. The training and CUDA benchmarks use bf16, so they need an A100 or newer.

```bash
pip install -r requirements.txt

pytest -q                                   # correctness against the PyTorch reference
python bench.py                             # Qwen2.5-7B shapes, fp16
python bench.py --model llama3-8b --dtype bf16 --csv llama_bf16.csv   # bf16 needs A100 or newer
```

Training kernels, a full decoder layer, and the CUDA version (bf16, so an A100):

```bash
python bench_train.py --dtype bf16          # forward plus backward, per op
python bench_layer.py --model llama3-8b     # one training step of a whole decoder layer
python bench_cuda.py --dtype bf16           # RMSNorm: eager vs Triton vs CUDA C++
```

Without a GPU, `pytest` still runs the Triton kernels and their gradients in Triton's CPU interpreter (fp32 only), so you can check correctness on a laptop. The CUDA tests are skipped.

On Colab: Runtime → Change runtime type → T4 GPU, then in a cell:

```python
!git clone https://github.com/riddhixraina/triton-fused-kernels && cd triton-fused-kernels && pip install -q -r requirements.txt && pytest -q && python bench.py --dtype fp16
```

## Reading the output

```
op            tokens   eager ms  compile ms  triton ms   speedup      GB/s   % peak
rmsnorm         2048     ...
```

- **tokens = 1** is a single decode step. The tensors are tiny, so kernel launch overhead dominates and nothing gets near peak bandwidth.
- **tokens = 2048 and up** looks like prefill. This is where the fused kernels should approach peak bandwidth, and where the speedup over eager is clearest.
- **torch.compile** also fuses these ops, so expect it to be close to Triton. 

## Results

NVIDIA A100-SXM4-40GB (peak 1,555 GB/s), bf16, Qwen2.5-7B shapes (hidden 3584, intermediate 18944).
Times are the median of 100 runs. `% peak` is the memory bandwidth the Triton kernel achieves.

| Op | Tokens | Eager (ms) | torch.compile (ms) | Triton (ms) | Speedup vs eager | GB/s | % peak |
|---|---|---|---|---|---|---|---|
| rmsnorm | 1 | 0.0471 | 0.0102 | 0.0082 | 5.75x | 2 | 0% |
| rmsnorm | 16 | 0.0461 | 0.0072 | 0.0072 | 6.43x | 32 | 2% |
| rmsnorm | 128 | 0.0532 | 0.0082 | 0.0082 | 6.50x | 224 | 14% |
| rmsnorm | 512 | 0.0758 | 0.0102 | 0.0102 | 7.40x | 717 | 46% |
| rmsnorm | 2048 | 0.2560 | 0.0297 | 0.0246 | 10.42x | 1195 | 77% |
| rmsnorm | 8192 | 0.9400 | 0.0942 | 0.0942 | 9.98x | 1247 | 80% |
| add+rmsnorm | 1 | 0.0522 | 0.0082 | 0.0082 | 6.37x | 4 | 0% |
| add+rmsnorm | 16 | 0.0614 | 0.0082 | 0.0082 | 7.50x | 56 | 4% |
| add+rmsnorm | 128 | 0.0717 | 0.0102 | 0.0092 | 7.78x | 398 | 26% |
| add+rmsnorm | 512 | 0.1147 | 0.0184 | 0.0174 | 6.59x | 843 | 54% |
| add+rmsnorm | 2048 | 0.4485 | 0.0512 | 0.0512 | 8.76x | 1147 | 74% |
| add+rmsnorm | 8192 | 1.6568 | 0.1833 | 0.1782 | 9.30x | 1318 | 85% |
| silu_and_mul | 1 | 0.0102 | 0.0072 | 0.0072 | 1.43x | 16 | 1% |
| silu_and_mul | 16 | 0.0174 | 0.0092 | 0.0092 | 1.89x | 197 | 13% |
| silu_and_mul | 128 | 0.0358 | 0.0195 | 0.0195 | 1.84x | 748 | 48% |
| silu_and_mul | 512 | 0.1116 | 0.0573 | 0.0553 | 2.02x | 1052 | 68% |
| silu_and_mul | 2048 | 0.3922 | 0.1874 | 0.1823 | 2.15x | 1277 | 82% |
| silu_and_mul | 8192 | 1.5135 | 0.7066 | 0.6932 | 2.18x | 1343 | 86% |

Llama 3 8B shapes (hidden 4096, intermediate 14336) give the same picture at 8,192 tokens:
rmsnorm 10.02x at 81% of peak, add+rmsnorm 9.34x at 86%, silu_and_mul 2.19x at 86%.
Full results: `results_bf16.csv`, `results_llama_bf16.csv`, `results_fp16.csv`.

### Reading the results

- **1 to 16 tokens** is a single decode step. Every version takes 7 to 10 µs, which is the fixed cost of launching a kernel, so bandwidth stays under 5% of peak. This is why inference servers like vLLM capture decode steps in CUDA graphs.
- **2,048 tokens and up** looks like prefill. The fused kernels reach 77 to 86% of peak bandwidth, and RMSNorm runs about 10x faster than eager PyTorch, which launches around six kernels and re-reads the tensor for each.
- **SwiGLU gains less (about 2.2x)** because eager PyTorch already does it in two kernels, so there's less wasted memory traffic to remove.
- **torch.compile** also fuses these ops, and the Triton kernels match it to within 3%.
- **add+rmsnorm** first reached only 56% of peak, 48% slower than torch.compile. The wrapper cloned the residual before the kernel ran, a hidden extra read and write of the whole tensor. Writing the new residual to a fresh buffer instead lifted it to 85%, 3% faster than torch.compile.

## Training kernels

`training.py` wraps each kernel in a `torch.autograd.Function`, so `rmsnorm_train`, `fused_add_rmsnorm_train` and `silu_and_mul_train` can replace the eager ops in a training loop. `test_training.py` checks every output and gradient against eager autograd.

- **Forward** works like the inference kernels, and the norms also save one fp32 `rstd` value per row for the backward pass. Eager autograd saves several full-size intermediate tensors instead.
- **Backward** is one kernel per op. For RMSNorm, with `g = dy * w`, the input gradient is `rstd * g - x * rstd³ * mean(g * x)`, computed from the saved `x` and `rstd`.
- **The weight gradient** sums `dy * x * rstd` over every row. One program per row would need a buffer as large as the input to hold the partial sums. Instead, one program per GPU streaming multiprocessor loops over a slice of rows and keeps its partial sum in registers, so the buffer is only 108 rows on an A100.
- **add+rmsnorm** saves the summed residual rather than both inputs, and both inputs get the same gradient: the norm's gradient plus the gradient flowing back through the residual stream.

`bench_train.py` times one forward plus backward pass per op. `bench_layer.py` times a training step of a whole Llama or Qwen decoder layer (attention, MLP, norms and residuals, without rotary embeddings) with the eager ops versus the Triton kernels, and also reports peak GPU memory, since saving less for the backward pass leaves more memory for activations.

## CUDA C++ version

`csrc/rmsnorm.cu` implements the same RMSNorm in CUDA C++, built at runtime by `cuda_rmsnorm.py` with `torch.utils.cpp_extension`. One thread block handles one row. Each thread loads 16 bytes at a time (8 bf16 values), sums squares in fp32, and the block combines the per-thread sums with warp shuffles. A second pass rereads the row, mostly from cache, and writes the output. `bench_cuda.py` compares it with the Triton kernel and eager PyTorch at the same shapes.

