"""Compare old and large-scale Qwen3-32B decode kernels end to end.

Each kernel runs in a fresh TP=2 engine with identical prompts and sampling
parameters. The reported decode tok/s counts generated tokens across all
concurrent requests and excludes Prefill time.

Example:
    uv run python benchmark_qwen32_decode_kernels.py \
        --kernel both --json-output qwen32-kernel-comparison.json
"""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmark_qwen32_stress import (
    OFFICIAL_MAX_POSITION,
    make_workloads,
    print_result,
    run_workload,
)
from main_qwen32 import config as qwen32_config
from myvllm.engine.llm_engine import LLMEngine


KERNELS = {
    "old": False,
    "large-scale": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen3-32B old and large-scale decode token throughput."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--kernel",
        choices=["both", *KERNELS],
        default="both",
        help="Run both kernels or only one implementation.",
    )
    parser.add_argument(
        "--order",
        choices=["old-first", "large-scale-first"],
        default="old-first",
        help="Execution order when --kernel=both.",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=8_192)
    parser.add_argument("--output-tokens", type=int, default=4_096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graphs for both kernels.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write configuration, raw results, and speedup to JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")
    if args.prompt_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("prompt-tokens and output-tokens must be greater than 0")
    if args.prompt_tokens + args.output_tokens > OFFICIAL_MAX_POSITION:
        raise ValueError(
            "prompt-tokens + output-tokens exceeds the Qwen3-32B capacity "
            f"of {OFFICIAL_MAX_POSITION}"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")


def selected_kernels(args: argparse.Namespace) -> list[str]:
    if args.kernel != "both":
        return [args.kernel]
    if args.order == "large-scale-first":
        return ["large-scale", "old"]
    return ["old", "large-scale"]


def run_kernel(
    args: argparse.Namespace,
    tokenizer,
    prompts: list[str],
    kernel_name: str,
) -> dict:
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    actual_max_prompt = max(prompt_lengths)
    engine_max_model_length = actual_max_prompt + args.output_tokens
    if engine_max_model_length > OFFICIAL_MAX_POSITION:
        raise ValueError(
            f"Rendered prompt plus output is {engine_max_model_length} tokens; "
            f"the maximum is {OFFICIAL_MAX_POSITION}"
        )

    config = dict(qwen32_config)
    config.update(
        model_name_or_path=args.model,
        world_size=2,
        block_size=args.block_size,
        max_num_sequences=args.concurrency,
        max_num_batched_tokens=sum(prompt_lengths),
        max_position=OFFICIAL_MAX_POSITION,
        max_model_length=engine_max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        use_large_scale_attention=KERNELS[kernel_name],
    )

    print(f"\n{'=' * 72}")
    print(f"Decode kernel:             {kernel_name}")
    print(f"model:                     {args.model}")
    print("tensor parallel size:      2")
    print(f"concurrency:               {args.concurrency}")
    print(f"actual max prompt tokens:  {actual_max_prompt:,}")
    print(f"output tokens/request:     {args.output_tokens:,}")
    print(f"engine max model length:   {engine_max_model_length:,}")
    print(f"CUDA Graphs enabled:       {not args.enforce_eager}")

    engine = LLMEngine(config=config)
    try:
        result = run_workload(
            engine,
            f"decode-{kernel_name}",
            prompts,
            args.output_tokens,
        )
    finally:
        engine.exit()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_result(result)
    per_request_tps = result.decode_tokens_per_second / result.requests
    print(f"decode tok/s/request:      {per_request_tps:,.2f}")
    return {
        "kernel": kernel_name,
        "use_large_scale_attention": KERNELS[kernel_name],
        "decode_tokens_per_second_per_request": per_request_tps,
        "result": asdict(result),
    }


def build_comparison(results: list[dict]) -> dict | None:
    by_kernel = {result["kernel"]: result for result in results}
    if set(by_kernel) != set(KERNELS):
        return None

    old = by_kernel["old"]["result"]
    large_scale = by_kernel["large-scale"]["result"]
    old_tps = old["decode_tokens_per_second"]
    large_scale_tps = large_scale["decode_tokens_per_second"]
    return {
        "old_decode_tokens_per_second": old_tps,
        "large_scale_decode_tokens_per_second": large_scale_tps,
        "decode_tps_speedup": large_scale_tps / old_tps,
        "decode_latency_reduction": (
            1 - large_scale["decode_seconds"] / old["decode_seconds"]
        ),
        "end_to_end_output_tps_speedup": (
            large_scale["output_tokens_per_second"]
            / old["output_tokens_per_second"]
        ),
    }


def print_comparison(comparison: dict) -> None:
    print(f"\n{'=' * 72}")
    print("Decode kernel comparison")
    print(
        "old decode tok/s:             "
        f"{comparison['old_decode_tokens_per_second']:,.2f}"
    )
    print(
        "large-scale decode tok/s:     "
        f"{comparison['large_scale_decode_tokens_per_second']:,.2f}"
    )
    print(
        "large-scale decode speedup:   "
        f"{comparison['decode_tps_speedup']:.3f}x"
    )
    print(
        "decode latency reduction:     "
        f"{comparison['decode_latency_reduction']:.2%}"
    )
    print(
        "end-to-end TPS speedup:        "
        f"{comparison['end_to_end_output_tps_speedup']:.3f}x"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, _, _ = make_workloads(
        tokenizer,
        args.concurrency,
        args.prompt_tokens,
    )

    results = [
        run_kernel(args, tokenizer, prompts, kernel_name)
        for kernel_name in selected_kernels(args)
    ]
    comparison = build_comparison(results)
    if comparison is not None:
        print_comparison(comparison)

    summary = {
        "configuration": {
            "model": args.model,
            "tensor_parallel_size": 2,
            "concurrency": args.concurrency,
            "target_prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "block_size": args.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "kernel_order": selected_kernels(args),
        },
        "kernels": results,
        "comparison": comparison,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\nJSON result written to {args.json_output}")


if __name__ == "__main__":
    main()
