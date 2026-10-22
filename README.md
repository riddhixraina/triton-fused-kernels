# LLM Inference Quantization Benchmark on vLLM

How much faster does a quantized LLM serve, and what does it cost in quality? This benchmarks Qwen2.5 on vLLM in three forms:

| Label | Weights | Model |
|---|---|---|
| `bf16` | 16-bit (baseline) | `Qwen/Qwen2.5-<size>-Instruct` |
| `awq-int4` | 4-bit AWQ | `Qwen/Qwen2.5-<size>-Instruct-AWQ` |
| `gptq-int8` | 8-bit GPTQ | `Qwen/Qwen2.5-<size>-Instruct-GPTQ-Int8` |

The baseline runs in bfloat16, the model's native type. On a T4, which has no bfloat16, it runs in float16 and is labeled `fp16` instead.

For each one it measures:
- **Throughput** in output tokens per second at batch sizes 1, 4, 16 and 64, with 512-token prompts and 128 generated tokens.
- **Latency** for a single request: time to first token, and time per output token after that.
- **Quality** as perplexity on WikiText-2. Lower is better, and the interesting number is how much it rises versus the 16-bit baseline.

## Results

Qwen2.5-7B on one A100 (40 GB), 512-token prompts, 128 output tokens:

| Run | Quantization | Tok/s @ batch 1 | Tok/s @ batch 4 | Tok/s @ batch 16 | Tok/s @ batch 64 | TTFT (ms) | ms / token | Perplexity |
|---|---|---|---|---|---|---|---|---|
| bf16 | none | 81 | 311 | 987 | 2,148 | 37.0 | 12.04 | 8.123 |
| awq-int4 | auto_awq | 183 (2.24x) | 603 (1.94x) | 1,502 (1.52x) | 2,189 (1.02x) | 42.9 | 5.18 | 8.665 (+6.7%) |
| gptq-int8 | auto_gptq | 130 (1.60x) | 444 (1.43x) | 1,199 (1.21x) | 1,971 (0.92x) | 53.7 | 7.32 | 8.133 (+0.1%) |

Ratios and perplexity changes are relative to `bf16`.

- **At batch 1, AWQ 4-bit runs 2.2× faster than BF16.** Decoding one request is memory-bound, and 4-bit weights are a quarter the size. Time per output token drops from 12.0 ms to 5.2 ms.
- **The gain shrinks as the batch grows**, from 2.2× at batch 1 to 1.02× at batch 64, and 8-bit GPTQ ends up 8% slower than BF16 there. Large batches are compute-bound, so unpacking quantized weights becomes overhead.
- **Both quantized models are slower to the first token**: 16% slower for AWQ and 45% slower for GPTQ. Prefill is compute-bound too, so smaller weights don't help it, and the unpacking still costs time.
- **AWQ raises perplexity by 6.7%. 8-bit GPTQ is near-lossless** at +0.1%.

### Long prompts

The same models with 2048-token prompts (perplexity skipped):

| Run | Quantization | Tok/s @ batch 1 | Tok/s @ batch 4 | Tok/s @ batch 16 | Tok/s @ batch 64 | TTFT (ms) | ms / token |
|---|---|---|---|---|---|---|---|
| bf16 | none | 77 | 251 | 566 | 800 | 126.4 | 12.09 |
| awq-int4 | auto_awq | 156 (2.02x) | 381 (1.52x) | 613 (1.08x) | 688 (0.86x) | 160.4 | 5.22 |
| gptq-int8 | auto_gptq | 114 (1.48x) | 301 (1.20x) | 524 (0.93x) | 620 (0.77x) | 183.5 | 7.38 |

- **Longer prompts make 4-bit lose sooner.** Each request now carries 4× as much prefill, so the run becomes compute-bound at smaller batches. AWQ is 14% slower than BF16 at batch 64, and GPTQ is 23% slower.
- **The time to first token gap grows with prompt length** for AWQ, from 16% slower at 512 tokens to 27% slower at 2048.
- **Decoding is unchanged.** Time per output token stays at about 12.1 ms for BF16 and 5.2 ms for AWQ, since one decode step reads the same weights whatever the prompt length.

So 4-bit pays off for low-traffic, latency-sensitive serving with short prompts, and stops paying off once batches or prompts are large.

Raw results are in [`results/`](results) and [`results-2048/`](results-2048).

## Run it

You need an NVIDIA GPU. The 1.5B model fits a free Colab T4 (16 GB). The 7B model needs about 24 GB or more, such as an A100.

```bash
pip install -r requirements.txt
./run_all.sh 1.5B        # or 7B on an A100
```

This runs each model in its own process (vLLM doesn't fully free GPU memory between models in one process), saves `results/<label>.json`, and writes `results/summary.md`:

```
| Run       | Quantization | Tok/s @ batch 1 | ... | TTFT (ms) | ms / token | Perplexity    |
| bf16      | none         | ...             |     |           |            | ...           |
| awq-int4  | auto_awq     | ... (2.24x)     |     |           |            | ... (+6.7%)   |
```

To run a single model: `python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --label awq-int4`. On a T4, add `--dtype half`. `run_all.sh` handles this for you.

Anything after the model size is passed to every run. To test long prompts and keep those results apart from the first run:

```bash
OUT=results-2048 ./run_all.sh 7B --input-len 2048 --skip-ppl
```

## What to expect, and why

- **Batch 1 is where 4-bit should win most.** Generating one token at a time is memory-bound: each step reads every weight once to do very little math. 4-bit weights are a quarter the size of 16-bit ones, so each step moves a quarter of the bytes.
- **The gain should shrink at batch 64.** With many requests per step, each weight read is reused across the batch, so the GPU becomes compute-bound, and unpacking 4-bit weights back to 16-bit becomes overhead.
- **Time to first token shouldn't improve, and can get worse.** Prefill runs the whole prompt through the model in one pass, so it's compute-bound, and weight-only quantization doesn't reduce the math. Run with `--input-len 2048` to see the gap grow with prompt length.
- **Smaller weights leave more memory for the KV cache**, so the server can hold more requests at once. That shows up in larger batches.
- **Kernels matter.** vLLM uses the fast Marlin kernels for AWQ and GPTQ on GPUs that support them, such as the A100, and falls back to slower kernels elsewhere. So the gap between 16-bit and 4-bit will look different on a T4. Recent vLLM versions report the method as `auto_awq` or `auto_gptq` and pick the kernel themselves.

## Measurement notes

- **Prefix caching is off.** vLLM turns it on by default. With it on, a prompt the engine has already seen skips most of its prefill. Timing the same prompt three times then measures a cache hit, which on an A100 shows up as a time to first token below what a real 512-token prefill of a 7B model can physically take (about 6.7 TFLOP, so at least 21 ms at the A100's 312 TFLOPS peak).
- **Every batch size gets its own prompts**, so no throughput run reuses another run's prompts.
- **Prompts are random token IDs** of exactly the requested length, and every request generates exactly 128 tokens (`ignore_eos`), so all three models do the same work.
