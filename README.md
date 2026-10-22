# Fused Transformer Kernels in Triton

Three fused GPU kernels for the memory-bound ops in every Llama and Qwen decoder layer, benchmarked against PyTorch eager and `torch.compile`.

| Kernel | What it computes | Where it runs in the model |
|---|---|---|
| `rmsnorm` | `x / sqrt(mean(x²) + eps) * w` | Before attention and before the MLP |
| `fused_add_rmsnorm` | residual add, then RMSNorm | Every layer boundary |
| `silu_and_mul` | `silu(gate) * up` (SwiGLU) | Inside the MLP |

## Why fuse them

These ops do almost no arithmetic per byte, so their speed is set by memory bandwidth. Eager PyTorch runs RMSNorm as about six separate kernels (cast to fp32, square, mean, rsqrt, multiply, multiply by the weight, cast back), and each one reads and writes the whole tensor in GPU memory. The fused kernel reads `x` once and writes `y` once. The residual-add version also keeps `x + residual` in registers instead of writing it out and reading it back.

So the goal is not "faster than PyTorch" in the abstract. It's **how close each kernel gets to the GPU's peak memory bandwidth**, which `bench.py` reports directly.

## Run it

You need an NVIDIA GPU. A free Colab T4 works; NYU HPC's A100s are better.

```bash
pip install -r requirements.txt

pytest -q                                   # correctness against the PyTorch reference
python bench.py                             # Qwen2.5-7B shapes, fp16
python bench.py --model llama3-8b --dtype bf16 --csv llama_bf16.csv   # bf16 needs A100 or newer
```

Without a GPU, `pytest` still runs the kernels in Triton's CPU interpreter (fp32 only), so you can check correctness on a laptop.

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
