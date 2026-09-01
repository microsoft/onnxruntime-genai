#!/usr/bin/env python3
"""Throughput benchmark for Qwen3.8 speculative decoding with a block drafter.

Sweeps context length and batch size, optionally running the speculative and
non-speculative arms back to back so each cell reports a speedup.
"""

import argparse
import importlib
import json
import re
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

# Acceptance rate depends on how predictable the context and the generated text are, so the
# content of a synthetic prompt is part of the measurement, not just its length. These paragraphs
# are deliberately varied: a single sentence repeated to length is close to a best case, because a
# periodic context lets the drafter copy from earlier repeats. Use --context-file to measure
# against real data, which is the only way to get a representative acceptance number.
CONTEXT_PARAGRAPHS = (
    "A paged key-value cache stores attention state in fixed-size blocks rather than one "
    "contiguous allocation per sequence. The scheduler hands out blocks from a shared pool, so a "
    "long conversation and a short one draw from the same memory without either reserving its "
    "worst case up front. The cost is an indirection: every attention kernel reads a block table "
    "to find where the next chunk of keys and values actually lives.",
    "Time to first token and steady decode throughput pull in opposite directions. Larger batches "
    "amortize weight reads across more sequences and raise aggregate tokens per second, but a "
    "request that arrives just after a batch is formed waits for the next scheduling boundary. "
    "Continuous batching narrows that gap by admitting new work between steps instead of draining "
    "the batch first.",
    "Quantizing a key-value cache to eight bits halves its footprint, which is usually the "
    "difference between a long context fitting on one accelerator and spilling. The arithmetic is "
    "not free: the kernel dequantizes on the fly, and a per-channel scale costs an extra multiply "
    "per element. Whether that shows up as lost throughput depends on whether the kernel was "
    "bound by memory bandwidth or by compute to begin with.",
    "A unit test that passes alone and fails in the full suite is almost always describing shared "
    "state. Module-level caches, a random seed set once at import, an environment variable another "
    "test mutated, or a temporary directory that a previous case left populated are the usual "
    "culprits. Running the suite in a randomized order surfaces the dependency faster than reading "
    "the tests in file order.",
    "Speculative decoding trades arithmetic for latency. A small drafter proposes several tokens, "
    "the large model verifies them in one forward pass, and every token the verifier agrees with "
    "is a token the large model did not have to generate serially. Under greedy decoding the "
    "accepted tokens are exactly what the target would have produced, so the output is unchanged "
    "and only the wall clock moves.",
    "Profilers disagree with each other for structural reasons. A sampling profiler attributes "
    "time to whatever was on the stack when the timer fired, so it under-reports short frequent "
    "calls. An instrumenting profiler sees every call but perturbs the thing it measures, "
    "especially for functions that would otherwise inline. Neither is wrong; they answer different "
    "questions, and the disagreement itself is often the interesting signal.",
    "Floating point addition is not associative, so changing how a reduction is tiled changes the "
    "result in the last few bits. That is usually harmless, but it means a kernel rewrite cannot "
    "be validated by a bitwise comparison against the old one. The honest gate is a tolerance "
    "derived from the accumulated error of the operation, plus a check that the difference does "
    "not grow with input size.",
    "Cache eviction policies are easy to reason about in isolation and hard to reason about under "
    "load. Least-recently-used degrades badly when a scan touches every entry once, because the "
    "scan evicts exactly the working set that was about to be reused. Admission policies help more "
    "than eviction policies here: deciding what never enters the cache is cheaper than deciding "
    "what leaves it.",
    "A distributed collective is a synchronization point wearing the costume of a data transfer. "
    "Its measured duration includes however long the slowest participant took to arrive, so a "
    "collective that looks slow is often reporting imbalance somewhere upstream. Timing the "
    "operation in isolation on an idle cluster produces a number that never appears in "
    "production.",
    "Reproducing a bug is usually more than half the work of fixing it. A reproduction that runs "
    "in seconds instead of minutes changes what kinds of hypotheses are affordable to test, and a "
    "deterministic one turns bisection from a statistical exercise into a mechanical one. Time "
    "spent shrinking a failing case is almost never wasted, even when the eventual fix is a single "
    "line.",
)

# Appended after the context so the model answers a question about the material instead of
# continuing filler text. Degenerate continuations have unrepresentative acceptance.
CONTEXT_INSTRUCTION = (
    "\n\nUsing only the material above, explain the most important trade-off it describes and "
    "what you would measure to decide it."
)


def split_sentences(text: str) -> list[str]:
    """Sentence-ish split. Sentences are the granularity at which prompt length is fitted."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        raise ValueError("Context source contains no usable text.")
    return sentences


def compose_context(sentences: list[str], count: int) -> str:
    """Take `count` sentences, rotating the pool on each pass so the text is not exactly periodic."""
    pool_size = len(sentences)
    picked = []
    for position in range(count):
        cycle, index = divmod(position, pool_size)
        picked.append(sentences[(index + cycle) % pool_size])
    return " ".join(picked)


def build_prompt(
    tokenizer,
    template_str: str | None,
    target_tokens: int,
    sentences: list[str],
) -> tuple[str, np.ndarray]:
    """Longest whole-sentence prompt whose chat-formatted encoding fits in target_tokens."""
    encoded_lengths: dict[int, int] = {}

    def encode(count: int) -> np.ndarray:
        return encode_prompt(tokenizer, compose_context(sentences, count) + CONTEXT_INSTRUCTION, template_str)

    def length(count: int) -> int:
        if count not in encoded_lengths:
            encoded_lengths[count] = len(encode(count))
        return encoded_lengths[count]

    if length(1) > target_tokens:
        raise ValueError(
            f"--context-length {target_tokens} is below the chat template overhead plus one "
            f"sentence ({length(1)} tokens)."
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
    return compose_context(sentences, low) + CONTEXT_INSTRUCTION, encode(low)


# Both SPEED-Bench (NVIDIA, parquet) and Spec-Bench (ACL 2024 Findings, jsonl) store one record
# per prompt with a `category` and a `turns` list, so one loader serves both.
DATASET_HELP = (
    "SPEED-Bench: https://huggingface.co/datasets/nvidia/SPEED-Bench (qualitative split for "
    "acceptance by domain, throughput split for fixed 1K-32K input lengths). "
    "Spec-Bench: https://github.com/hemingkx/Spec-Bench (data/spec_bench/question.jsonl). "
    "Neither is redistributed here; check the dataset's own license before use."
)


def _read_records(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        try:
            pq = importlib.import_module("pyarrow.parquet")
        except ImportError as error:
            raise RuntimeError(
                f"Reading '{path}' needs pyarrow (pip install pyarrow), or convert the file to JSON Lines first."
            ) from error
        return pq.read_table(path).to_pylist()
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read dataset '{path}': {error}") from error


def load_prompt_dataset(path: Path, limit: int, categories: list[str] | None) -> dict[str, list[str]]:
    """Group dataset prompts by category, keeping the first turn of each conversation.

    Multi-turn records are truncated to their first turn, so acceptance here is not directly
    comparable to a published leaderboard that replays the whole conversation. What it does buy
    is a realistic prompt distribution instead of synthetic filler.
    """
    records = _read_records(path)
    if not records:
        raise ValueError(f"Dataset '{path}' is empty.")

    grouped: dict[str, list[str]] = {}
    for record in records:
        category = record.get("category")
        turns = record.get("turns")
        if category is None or not turns:
            raise ValueError(
                f"'{path}' is not in SPEED-Bench / Spec-Bench format; each record needs a "
                f"'category' and a non-empty 'turns' list.\n{DATASET_HELP}"
            )
        if categories and category not in categories:
            continue
        first_turn = turns[0]
        # A placeholder marks a sample the dataset cannot redistribute; it has to be fetched
        # with the publisher's own build script before it is usable.
        if isinstance(first_turn, str) and first_turn.strip():
            grouped.setdefault(category, []).append(first_turn)

    if not grouped:
        available = sorted({str(record.get("category")) for record in records})
        raise ValueError(f"No usable prompts matched. Available categories: {', '.join(available)}")
    return {category: prompts[:limit] for category, prompts in sorted(grouped.items())}


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
    prompt_ids: list[np.ndarray],
    generate_length: int,
    speculative: bool,
    max_draft_tokens: int,
    stats_baseline: dict | None,
) -> dict:
    """One measured point: one concurrent request per entry in prompt_ids."""
    batch_size = len(prompt_ids)
    requests: dict = {}
    token_times: dict[int, list[float]] = {index: [] for index in range(batch_size)}
    submitted_at: dict[int, float] = {}
    outputs: dict[int, int] = dict.fromkeys(range(batch_size), 0)
    closed: set[int] = set()

    started_at = time.perf_counter()
    for index, input_ids in enumerate(prompt_ids):
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
        "prompt_tokens_mean": sum(len(ids) for ids in prompt_ids) / batch_size,
        "prompt_tokens_max": max(len(ids) for ids in prompt_ids),
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
    header = "| workload | batch | arm | prompt tok | TTFT s | decode tok/s | accept | tok/target fwd | speedup |"
    separator = "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, separator]
    for row in rows:
        spec = row.get("speculative") or {}
        acceptance = spec.get("acceptance_rate")
        per_forward = spec.get("output_tokens_per_target_forward")
        speedup = ""
        if baseline_key is not None and row["arm"] == "speculative":
            base = baseline_key.get((row["workload"], row["batch_size"]))
            if base and base.get("decode_tokens_per_second"):
                speedup = f"{row['decode_tokens_per_second'] / base['decode_tokens_per_second']:.3f}x"
        lines.append(
            "| {workload} | {batch} | {arm} | {ptok} | {ttft} | {dec} | {acc} | {fwd} | {sp} |".format(
                workload=row["workload"],
                batch=row["batch_size"],
                arm=row["arm"],
                ptok=round(row["prompt_tokens_mean"]),
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
        "--context-file",
        type=Path,
        help="Plain-text file to synthesize prompts from. Acceptance is workload-specific, so "
        "real data gives a far more representative number than the built-in passages.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="SPEED-Bench or Spec-Bench file (.parquet or .jsonl). Replaces the synthetic "
        "context sweep with real prompts, reported per category. " + DATASET_HELP,
    )
    parser.add_argument(
        "--dataset-category",
        action="append",
        dest="dataset_categories",
        help="Restrict --dataset to these categories. Repeat to select several.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=8,
        help="Prompts per category. Raise for a tighter acceptance estimate at more wall time.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure with the reasoning block enabled. Off by default so --generate-length "
        "bounds a plain answer; acceptance differs between the two modes.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="low",
    )
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

    if args.dataset is not None:
        source = args.dataset.expanduser().resolve()
        try:
            dataset = load_prompt_dataset(source, args.dataset_limit, args.dataset_categories)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        sentences = None
        context_label = f"{source.name} ({len(dataset)} categories, <={args.dataset_limit}/category)"
    else:
        dataset = None
        if args.context_file is not None:
            source = args.context_file.expanduser().resolve()
            try:
                sentences = split_sentences(source.read_text(encoding="utf-8"))
            except OSError as error:
                parser.error(f"Could not read --context-file '{source}': {error}")
            context_label = f"{source} ({len(sentences)} sentences)"
        else:
            sentences = split_sentences(" ".join(CONTEXT_PARAGRAPHS))
            context_label = f"built-in passages ({len(sentences)} sentences, synthetic; see --dataset for real prompts)"

    arms = ["speculative", "baseline"] if args.arms == "both" else [args.arms]
    if drafter is None or max_draft_tokens <= 0:
        arms = ["baseline"]
    print(
        f"Model: {model_dir}\n"
        f"Drafter: {drafter or 'none'} (max_draft_tokens={max_draft_tokens})\n"
        f"Arms: {', '.join(arms)}\n"
        f"Context source: {context_label}\n"
        f"Reasoning: {'on (effort=' + args.reasoning_effort + ')' if args.thinking else 'off'}"
    )

    rows: list[dict] = []
    for arm in arms:
        speculative = arm == "speculative"
        model, tokenizer, engine, event_buffer = build_engine(model_dir, args, drafter, speculative)
        template_str = chat_template(model_dir, args.thinking, args.reasoning_effort)
        print(f"\n=== arm: {arm} ===", flush=True)
        stats_baseline = None
        if not args.no_warmup:
            # Warm up at the largest shape, and generate enough tokens to cover several
            # speculative rounds. An 8-token warmup leaves first-touch cost on the first
            # measured cell, which showed up as a 2x-inflated wall time there.
            if dataset is not None:
                longest = max((p for prompts in dataset.values() for p in prompts), key=len)
                warm_ids = encode_prompt(tokenizer, longest, template_str)
            else:
                _, warm_ids = build_prompt(tokenizer, template_str, max(context_lengths), sentences)
            run_cell(
                engine,
                model,
                event_buffer,
                [warm_ids] * max(batch_sizes),
                min(32, args.generate_length),
                speculative,
                max_draft_tokens,
                None,
            )
            if speculative:
                stats_baseline = engine.get_speculative_stats()

        if dataset is not None:
            workloads = [
                (category, [encode_prompt(tokenizer, prompt, template_str) for prompt in prompts])
                for category, prompts in dataset.items()
            ]
        else:
            workloads = [
                (str(length), [build_prompt(tokenizer, template_str, length, sentences)[1]])
                for length in context_lengths
            ]

        for label, encoded in workloads:
            for batch_size in batch_sizes:
                # Cycle the pool so each request in a batch gets a distinct prompt when the
                # dataset supplies enough of them.
                prompt_ids = [encoded[index % len(encoded)] for index in range(batch_size)]
                cell = run_cell(
                    engine,
                    model,
                    event_buffer,
                    prompt_ids,
                    args.generate_length,
                    speculative,
                    max_draft_tokens,
                    stats_baseline,
                )
                if speculative:
                    stats_baseline = engine.get_speculative_stats()
                cell["workload"] = label
                cell["arm"] = arm
                rows.append(cell)
                short = "" if cell["completed_full_length"] else "  [early EOS]"
                print(
                    f"{label:<16} batch={batch_size:<3} "
                    f"prompt={round(cell['prompt_tokens_mean']):<6} "
                    f"decode={cell['decode_tokens_per_second']:.1f} tok/s "
                    f"ttft={cell['ttft_seconds_max']:.3f}s{short}",
                    flush=True,
                )
        del engine, model, tokenizer, event_buffer

    baseline_index = {(row["workload"], row["batch_size"]): row for row in rows if row["arm"] == "baseline"}
    print("\n" + format_table(rows, baseline_index if "baseline" in arms else None))

    if args.json_out:
        payload = {
            "model_dir": str(model_dir),
            "drafter": drafter,
            "max_draft_tokens": max_draft_tokens,
            "generate_length": args.generate_length,
            "max_batch_size": args.max_batch_size,
            "num_blocks": args.num_blocks,
            "context_source": context_label,
            "thinking": args.thinking,
            "cells": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
