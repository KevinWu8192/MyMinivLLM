# Qwen3-32B Stress Results

These results were produced by
`tests/benchmarks/benchmark_qwen3_32b_stress.py` with exact-length generation
(`ignore_eos=True`). Raw JSON is preserved alongside this document.

## Test hardware

* Container image: `vllm-project/vllm/vllm-0.20.0cv1`
* GPU: 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GiB each
* Tensor Parallel size: 2
* CPU: 50 vCPU, Intel Xeon Platinum 8470Q
* Host memory: 240 GiB
* Model: `Qwen/Qwen3-32B`, BF16
* CUDA Graphs: enabled
* Position capacity: 40,960 tokens; documented native context: 32,768 tokens

## Workloads and results

| Suite | Requests | Prompt / request | Output / request | Prefill tok/s | Decode tok/s | End-to-end output tok/s | Total time |
|---|---:|---:|---:|---:|---:|---:|---:|
| High concurrency | 16 | 4,075 | 256 | 5,954.27 | 435.13 | 201.48 | 20.33 s |
| Long decode | 2 | 8,169 | 4,096 | 4,430.35 | 68.64 | 66.57 | 123.05 s |
| Maximum output | 1 | 8,169 | 32,768 | 3,726.97 | 35.60 | 35.50 | 923.03 s |

`Decode tok/s` is aggregate throughput across all concurrent requests. The
16-request workload therefore corresponds to about 27.20 tok/s per request;
the 2-request long-decode workload corresponds to about 34.32 tok/s per
request. The maximum-output run completed all 32,768 requested tokens at a
total sequence length of 40,937, 23 tokens below the configured 40,960 limit.
Its Decode phase took 920.45 seconds (15 minutes 20 seconds).

## Prefix-cache result

| Phase | Cache hit rate | Computed Prefill tokens | Cached tokens | Prefill time | Decode tok/s | Total time | End-to-end output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold unique | 0.00% | 130,856 | 0 | 31.902 s | 198.15 | 42.201 s | 48.53 |
| Cache seed | 0.00% | 16,334 | 0 | 4.577 s | 26.61 | 4.841 s | 1.65 |
| Hot shared | 98.69% | 1,712 | 129,024 | 1.087 s | 210.26 | 10.792 s | 189.77 |

The hot phase reuses 129,024 tokens: 16,128 tokens per request, exactly 63
complete 256-token cache blocks. Only 214 request-specific tokens per request
are recomputed. Relative to the cold phase, Prefill time falls by about 29.35×,
total time falls by 3.91×, and end-to-end output throughput rises by 3.91×.

## Raw results

* [`stress_high_concurrency.json`](stress_high_concurrency.json)
* [`stress_long_decode.json`](stress_long_decode.json)
* [`stress_max_output.json`](stress_max_output.json)
* [`stress_prefix_cache.json`](stress_prefix_cache.json)
