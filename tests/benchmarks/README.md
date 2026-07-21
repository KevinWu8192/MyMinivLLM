# Benchmark Catalog

Benchmark filenames use `benchmark_<scope>.py`; committed result filenames use
`stress_<suite>.json` under `tests/results/<model>/`.

## Qwen3-32B stress benchmark

`benchmark_qwen3_32b_stress.py` runs four independent TP=2 workloads:

* `prefix-cache`: 8 concurrent 16K-token prompts with cold, seed, and hot shared-prefix phases.
* `high-concurrency`: 16 concurrent 4K-token prompts with 256 generated tokens per request.
* `long-decode`: 2 concurrent 8K-token prompts with 4K generated tokens per request.
* `max-output`: one 8K-token prompt with exactly 32K generated tokens.

Example:

```bash
uv run python tests/benchmarks/benchmark_qwen3_32b_stress.py \
  --suite high-concurrency \
  --json-output tests/results/qwen3_32b/stress_high_concurrency.json
```

Use `--suite all` to run every workload. Each suite gets a separate engine so
its CUDA Graph and KV-cache shape match the requested concurrency and context.

## Attention microbenchmarks

`benchmark_attention_decode.py` imports the production old and large-scale
Paged Attention Decode implementations. It provides correctness, timing, and
CUDA-profiler modes:

```bash
uv run python tests/benchmarks/benchmark_attention_decode.py --mode correctness
uv run python tests/benchmarks/benchmark_attention_decode.py --mode benchmark
uv run python tests/benchmarks/benchmark_attention_decode.py --mode profile --iterations 1
```

`benchmark_attention_prefill.py` compares PyTorch attention, a small naive
Triton implementation, and the Flash Attention Prefill algorithm across short
and long sequence lengths:

```bash
uv run python tests/benchmarks/benchmark_attention_prefill.py
```

## Engine comparison

`benchmark_engine_tps.py` compares MiniVLLM, upstream vLLM, and Transformers on
the same Qwen3-0.6B prompts:

```bash
uv run python tests/benchmarks/benchmark_engine_tps.py
```

All benchmarks require an NVIDIA CUDA environment. The Qwen3-32B stress suite
also requires two GPUs and a locally accessible Qwen3-32B checkpoint.
