#!/usr/bin/env python3
"""Throughput benchmark for Qwen3.8 speculative decoding with a block drafter.

Sweeps context length and batch size, optionally running the speculative and
non-speculative arms back to back so each cell reports a speedup.
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from engine_example import (
    chat_template,
    detect_drafter,
    encode_prompt,
    register_cuda_plugin,
    speculative_metrics,
)

# Repeated to synthesize a prompt of a requested token length. The content is
# irrelevant to the measurement; only its length is.
FILLER_SENTENCE = (
    "A paged key-value cache stores attention state in fixed-size blocks so that "
    "concurrent requests share one pool without copying it. "
)


def build_prompt(tokenizer, template_str: str | None, target_tokens: int) -> tuple[str, np.ndarray]:
    """Largest whole-sentence prompt whose chat-formatted encoding fits in target_tokens."""
    encoded_lengths: dict[int, int] = {}

    def encode(repeats: int) -> np.ndarray:
        return encode_prompt(tokenizer, FILLER_SENTENCE * repeats, template_str)

    def length(repeats: int) -> int:
        if repeats not in encoded_lengths:
            encoded_lengths[repeats] = len(encode(repeats))
        return encoded_lengths[repeats]

    if length(1) > target_tokens:
        raise ValueError(
            f"--context-length {target_tokens} is below the chat template overhead "
            f"({length(1)} tokens for the shortest prompt)."
        )

    low, high = 1, 2
    while length(high) <= target_tokens:
        low = high
        high *= 2
        if high > 1 << 20:
            break
    while low + 1 < high:
        middle = (low + high) // 2
        if length(middle) <= target_tokens:
            low = middle
        else:
            high = middle
    return FILLER_SENTENCE * low, encode(low)


def gpu_memory_mib() -> float | None:
    """Used memory on the first visible device, or None when NVML is unavailable."""
    try:
        pynvml = importlib.import_module("pynvml")
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        pynvml.nvmlShutdown()
        return used / (1024 * 1024)
    except Exception:
        # NVML is a diagnostic here, never a reason to fail a benchmark.
        return None


def run_cell(
    engine: og.Engine,
    model: og.Model,
    event_buffer,
    input_ids: np.ndarray,
    batch_size: int,
    generate_length: int,
    speculative: bool,
    max_draft_tokens: int,
    stats_baseline: dict | None,
) -> dict:
    """One (context, batch) point. Every request generates exactly generate_length tokens."""
    requests: dict = {}
    token_times: dict[int, list[float]] = {index: [] for index in range(batch_size)}
    submitted_at: dict[int, float] = {}
    outputs: dict[int, int] = dict.fromkeys(range(batch_size), 0)
    closed: set[int] = set()

    started_at = time.perf_counter()
    for index in range(batch_size):
        params = og.GeneratorParams(model)
        # max_length only. Setting min_length would suppress drafting for the whole run:
        # the engine requires `min_length <= current sequence length` before it will draft
        # (src/engine/engine.cpp, Request::DraftTokenValidationError).
        params.set_search_options(
            do_sample=False,
            max_length=len(input_ids) + generate_length,
        )
        if speculative:
            params.set_speculative_options(max_draft_tokens=max_draft_tokens)
        request = engine.create_request(params)
        submitted_at[index] = time.perf_counter()
        request.begin_turn(input_ids)
        requests[request] = index

    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.request is None:
                    if event.flags & og.EngineEventFlags.FAILED:
                        raise RuntimeError(f"Engine failed; flags={event.flags}, error_code={event.error_code}")
                    continue
                index = requests[event.request]
                if event.flags & og.EngineEventFlags.TOKEN:
                    outputs[index] += 1
                    token_times[index].append(time.perf_counter())
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Request {index} failed; error_code={event.error_code}")
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    event.request.close()
                    closed.add(index)
    finally:
        for request, index in requests.items():
            if index not in closed:
                request.close()

    first_token_at = {index: times[0] for index, times in token_times.items() if times}
    if len(first_token_at) != batch_size:
        raise RuntimeError("Some requests produced no tokens; cannot report throughput.")
    last_first_token_at = max(first_token_at.values())
    last_token_at = max(times[-1] for times in token_times.values())
    steady_tokens = sum(timestamp > last_first_token_at for times in token_times.values() for timestamp in times)
    decode_seconds = last_token_at - last_first_token_at
    output_tokens = sum(outputs.values())

    cell = {
        "prompt_tokens": len(input_ids),
        "batch_size": batch_size,
        "generate_length": generate_length,
        "output_tokens": output_tokens,
        # False when a request hit EOS early, which shortens the measured decode window.
        "completed_full_length": all(count == generate_length for count in outputs.values()),
        "ttft_seconds_mean": sum(first_token_at[index] - submitted_at[index] for index in range(batch_size))
        / batch_size,
        "ttft_seconds_max": max(first_token_at[index] - submitted_at[index] for index in range(batch_size)),
        "decode_tokens_per_second": steady_tokens / decode_seconds if decode_seconds > 0 else None,
        "wall_seconds": last_token_at - started_at,
        "gpu_used_mib": gpu_memory_mib(),
    }
    if speculative:
        cell["speculative"] = speculative_metrics(engine, output_tokens, stats_baseline)
    return cell


def build_engine(model_dir: Path, args, drafter: str | None, speculative: bool):
    """Fresh model/engine for one arm. The drafter overlay is a load-time decision."""
    batching: dict = {"max_batch_size": args.max_batch_size}
    if args.num_blocks is not None:
        batching["num_blocks"] = args.num_blocks
    if args.gpu_utilization is not None:
        batching["gpu_utilization_factor"] = args.gpu_utilization
    overlay: dict = {"engine": {"dynamic_batching": batching}}
    if not speculative and drafter is not None:
        overlay["model"] = {drafter: {"filename": ""}}

    config = og.Config(str(model_dir))
    config.overlay(json.dumps(overlay))
    config.clear_providers()
    config.append_provider("cuda")
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(args.event_buffer_capacity)
    return model, tokenizer, engine, event_buffer


def format_table(rows: list[dict], baseline_key: dict | None) -> str:
    header = "| ctx | batch | arm | prompt tok | TTFT s | decode tok/s | accept | tok/target fwd | speedup |"
    separator = "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, separator]
    for row in rows:
        spec = row.get("speculative") or {}
        acceptance = spec.get("acceptance_rate")
        per_forward = spec.get("output_tokens_per_target_forward")
        speedup = ""
        if baseline_key is not None and row["arm"] == "speculative":
            base = baseline_key.get((row["context_length"], row["batch_size"]))
            if base and base.get("decode_tokens_per_second"):
                speedup = f"{row['decode_tokens_per_second'] / base['decode_tokens_per_second']:.3f}x"
        lines.append(
            "| {ctx} | {batch} | {arm} | {ptok} | {ttft} | {dec} | {acc} | {fwd} | {sp} |".format(
                ctx=row["context_length"],
                batch=row["batch_size"],
                arm=row["arm"],
                ptok=row["prompt_tokens"],
                ttft=f"{row['ttft_seconds_max']:.3f}",
                dec=f"{row['decode_tokens_per_second']:.1f}" if row["decode_tokens_per_second"] else "-",
                acc=f"{acceptance:.3f}" if acceptance is not None else "-",
                fwd=f"{per_forward:.2f}" if per_forward is not None else "-",
                sp=speedup or "-",
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding throughput across contexts and batch sizes."
    )
    parser.add_argument("--model", type=Path, required=True, help="Model directory.")
    parser.add_argument(
        "--context-length",
        type=int,
        action="append",
        dest="context_lengths",
        help="Prompt length in tokens. Repeat to sweep. Default: 512 2048 8192.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="append",
        dest="batch_sizes",
        help="Concurrent requests. Repeat to sweep. Default: 1 4.",
    )
    parser.add_argument("--generate-length", type=int, default=256)
    parser.add_argument(
        "--arms",
        choices=("speculative", "baseline", "both"),
        default="both",
        help="Run the drafter arm, the --no-drafter arm, or both for a speedup column.",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        help="Draft tokens per target forward. Defaults to the model's num_draft_tokens.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="engine.dynamic_batching.max_batch_size. Defaults to the largest --batch-size.",
    )
    parser.add_argument("--num-blocks", type=int, help="Explicit paged KV pool size.")
    parser.add_argument("--gpu-utilization", type=float)
    parser.add_argument("--event-buffer-capacity", type=int, default=16)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the throwaway request. Lazy prepack then lands on the first measured cell.",
    )
    parser.add_argument("--json-out", type=Path, help="Write the full results as JSON.")
    plugin = parser.add_mutually_exclusive_group()
    plugin.add_argument("--cuda-plugin", action="store_true")
    plugin.add_argument("--cuda-plugin-library", type=Path)
    args = parser.parse_args()

    context_lengths = args.context_lengths or [512, 2048, 8192]
    batch_sizes = args.batch_sizes or [1, 4]
    if any(value <= 0 for value in context_lengths + batch_sizes):
        parser.error("--context-length and --batch-size must be positive")
    if args.generate_length <= 0:
        parser.error("--generate-length must be positive")
    if args.max_batch_size is None:
        args.max_batch_size = max(batch_sizes)
    elif args.max_batch_size < max(batch_sizes):
        parser.error("--max-batch-size is smaller than the largest --batch-size")

    model_dir = args.model.expanduser().resolve()
    drafter, configured_draft_tokens = detect_drafter(model_dir)
    max_draft_tokens = args.max_draft_tokens if args.max_draft_tokens is not None else configured_draft_tokens
    if drafter is None and args.arms != "baseline":
        parser.error(f"'{model_dir}' declares no drafter; use --arms baseline")

    register_cuda_plugin(args.cuda_plugin, args.cuda_plugin_library)

    arms = ["speculative", "baseline"] if args.arms == "both" else [args.arms]
    if drafter is None or max_draft_tokens <= 0:
        arms = ["baseline"]
    print(
        f"Model: {model_dir}\n"
        f"Drafter: {drafter or 'none'} (max_draft_tokens={max_draft_tokens})\n"
        f"Arms: {', '.join(arms)}"
    )

    rows: list[dict] = []
    for arm in arms:
        speculative = arm == "speculative"
        model, tokenizer, engine, event_buffer = build_engine(model_dir, args, drafter, speculative)
        template_str = chat_template(model_dir, thinking=False, reasoning_effort="low")
        print(f"\n=== arm: {arm} ===", flush=True)
        stats_baseline = None
        if not args.no_warmup:
            # Warm up at the largest shape, and generate enough tokens to cover several
            # speculative rounds. An 8-token warmup leaves first-touch cost on the first
            # measured cell, which showed up as a 2x-inflated wall time there.
            _, warm_ids = build_prompt(tokenizer, template_str, max(context_lengths))
            run_cell(
                engine,
                model,
                event_buffer,
                warm_ids,
                max(batch_sizes),
                min(32, args.generate_length),
                speculative,
                max_draft_tokens,
                None,
            )
            if speculative:
                stats_baseline = engine.get_speculative_stats()

        for context_length in context_lengths:
            _, input_ids = build_prompt(tokenizer, template_str, context_length)
            for batch_size in batch_sizes:
                cell = run_cell(
                    engine,
                    model,
                    event_buffer,
                    input_ids,
                    batch_size,
                    args.generate_length,
                    speculative,
                    max_draft_tokens,
                    stats_baseline,
                )
                if speculative:
                    stats_baseline = engine.get_speculative_stats()
                cell["context_length"] = context_length
                cell["arm"] = arm
                rows.append(cell)
                short = "" if cell["completed_full_length"] else "  [early EOS]"
                print(
                    f"ctx={context_length:<6} batch={batch_size:<3} "
                    f"decode={cell['decode_tokens_per_second']:.1f} tok/s "
                    f"ttft={cell['ttft_seconds_max']:.3f}s{short}",
                    flush=True,
                )
        del engine, model, tokenizer, event_buffer

    baseline_index = {(row["context_length"], row["batch_size"]): row for row in rows if row["arm"] == "baseline"}
    print("\n" + format_table(rows, baseline_index if "baseline" in arms else None))

    if args.json_out:
        payload = {
            "model_dir": str(model_dir),
            "drafter": drafter,
            "max_draft_tokens": max_draft_tokens,
            "generate_length": args.generate_length,
            "max_batch_size": args.max_batch_size,
            "num_blocks": args.num_blocks,
            "cells": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
