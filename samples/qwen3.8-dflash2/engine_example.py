#!/usr/bin/env python3
"""Speculative-decoding Engine example for Qwen3.8 with a block drafter."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

# Block drafters that a genai_config.json may declare under "model".
DRAFTER_KEYS = ("dflash2", "dspark", "mtp")


def detect_drafter(model_dir: Path) -> tuple[str | None, int]:
    """Return the (section name, configured draft width) of the drafter in genai_config.json."""
    config_path = model_dir / "genai_config.json"
    try:
        model = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read '{config_path}': {error}") from error
    except KeyError as error:
        raise ValueError(f"'{config_path}' has no 'model' section.") from error

    declared = [key for key in DRAFTER_KEYS if model.get(key, {}).get("filename")]
    if len(declared) > 1:
        raise ValueError(
            f"'{config_path}' declares more than one drafter ({', '.join(declared)}). "
            "Exactly one is supported."
        )
    if not declared:
        return None, 0
    name = declared[0]
    return name, int(model[name].get("num_draft_tokens", 0))


def chat_template(model_dir: Path, thinking: bool, reasoning_effort: str) -> str | None:
    """Pin the Qwen3.8 template's reasoning mode.

    The genai ``apply_chat_template`` binding cannot pass extra template variables, so
    the two ``enable_thinking`` guards are rewritten to their constant outcomes.
    """
    path = model_dir / "chat_template.jinja"
    if not path.is_file():
        return None
    template = path.read_text(encoding="utf-8")
    template = template.replace(
        "enable_thinking is undefined or enable_thinking is true",
        "true" if thinking else "false",
    ).replace(
        "enable_thinking is defined and enable_thinking is false",
        "false" if thinking else "true",
    )
    if thinking:
        template = template.replace(
            "reasoning_effort|default('xhigh')",
            f"reasoning_effort|default('{reasoning_effort}')",
        )
    return template


def encode_prompt(
    tokenizer: og.Tokenizer,
    prompt: str,
    template_str: str | None,
) -> np.ndarray:
    messages = json.dumps([{"role": "user", "content": prompt}])
    rendered = tokenizer.apply_chat_template(
        messages=messages,
        template_str=template_str,
        add_generation_prompt=True,
    )
    return np.asarray(tokenizer.encode(rendered), dtype=np.int32)


def load_prompts(path: Path) -> list[str]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read prompt file '{path}': {error}") from error
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(
            f"Prompt file '{path}' must contain a non-empty JSON array of non-empty strings."
        )
    return values


def register_cuda_plugin(
    use_installed_package: bool,
    library_path: Path | None,
) -> None:
    if use_installed_package:
        try:
            plugin = importlib.import_module("onnxruntime_ep_cuda")
        except ImportError as error:
            raise RuntimeError(
                "The CUDA plugin package is not installed. Install the feed package "
                "that provides the 'onnxruntime_ep_cuda' module, or use "
                "--cuda-plugin-library."
            ) from error
        registration_name = plugin.get_ep_name()
        resolved_library = Path(plugin.get_library_path()).resolve()
    else:
        if library_path is None:
            return
        registration_name = "CUDA.GenAI"
        resolved_library = library_path.expanduser().resolve()

    if not resolved_library.is_file():
        raise FileNotFoundError(
            f"CUDA plugin EP library was not found: {resolved_library}"
        )
    og.register_execution_provider_library(
        registration_name,
        str(resolved_library),
    )
    print(
        f"Registered CUDA plugin EP '{registration_name}' from "
        f"{resolved_library}"
    )


def build_overlay(args, drafter: str | None) -> dict:
    """Config overrides for batching capacity and for disabling the drafter."""
    overlay: dict = {}
    batching = {}
    if args.max_batch_size is not None:
        batching["max_batch_size"] = args.max_batch_size
    if args.num_blocks is not None:
        batching["num_blocks"] = args.num_blocks
    if args.gpu_utilization is not None:
        batching["gpu_utilization_factor"] = args.gpu_utilization
    if batching:
        overlay["engine"] = {"dynamic_batching": batching}
    if args.no_drafter and drafter is not None:
        # An empty filename drops the drafter session, giving a same-model baseline.
        overlay["model"] = {drafter: {"filename": ""}}
    return overlay


def run_warmup(
    engine: og.Engine,
    model: og.Model,
    event_buffer,
    input_ids: np.ndarray,
    speculative: bool,
    max_draft_tokens: int,
) -> None:
    """Absorb lazy weight prepack and first-run allocations before the measured requests."""
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(input_ids) + 8)
    if speculative:
        params.set_speculative_options(max_draft_tokens=max_draft_tokens)
    request = engine.create_request(params)
    request.begin_turn(input_ids)
    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(
                        f"Warmup request failed; error_code={event.error_code}"
                    )
    finally:
        request.close()


def speculative_metrics(
    engine: og.Engine,
    output_tokens: int,
    baseline: dict | None = None,
) -> dict:
    """Drafter statistics for the measured requests, net of any warmup already counted."""
    stats = engine.get_speculative_stats()

    def delta(key: str) -> int:
        current = stats.get(key) or 0
        previous = (baseline.get(key) or 0) if baseline else 0
        return current - previous

    counters = (
        "rounds",
        "draft_tokens_proposed",
        "draft_tokens_evaluated",
        "draft_tokens_accepted",
        "draft_forward_passes",
        "target_forward_passes",
    )
    metrics = {key: delta(key) for key in counters}

    evaluated = metrics["draft_tokens_evaluated"]
    metrics["acceptance_rate"] = (
        metrics["draft_tokens_accepted"] / evaluated if evaluated else None
    )
    target_forwards = metrics["target_forward_passes"]
    metrics["output_tokens_per_target_forward"] = (
        output_tokens / target_forwards if target_forwards else None
    )

    histogram = stats.get("acceptance_length_histogram") or []
    base_histogram = (baseline.get("acceptance_length_histogram") or []) if baseline else []
    metrics["acceptance_length_histogram"] = [
        bucket - (base_histogram[index] if index < len(base_histogram) else 0)
        for index, bucket in enumerate(histogram)
    ]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompts through the continuous-batching Engine with a block drafter."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Model directory. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Custom prompt. Repeat to provide a prompt pool; overrides bundled prompts.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSON array of prompts. May be combined with repeatable --prompt values.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        help="Number of concurrent requests. Prompts are cycled when needed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a reasoning block before the answer. On by default, as the export is.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="low",
        help="Reasoning budget when thinking is on. The export defaults to xhigh.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
        help="Reject longer tokenized prompts. Raise deliberately for long-context tests.",
    )
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        help="Draft tokens proposed per target forward. Defaults to the model's num_draft_tokens.",
    )
    parser.add_argument(
        "--no-drafter",
        action="store_true",
        help="Disable the drafter to get a non-speculative baseline from the same model.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample instead of decoding greedily. Greedy keeps acceptance a lossless-match rate.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="Override engine.dynamic_batching.max_batch_size. Also sizes the fixed state pool.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="Override engine.dynamic_batching.num_blocks (explicit paged KV pool size).",
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        help="Override engine.dynamic_batching.gpu_utilization_factor.",
    )
    parser.add_argument("--event-buffer-capacity", type=int, default=16)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one short throwaway request first so lazy allocations are not timed.",
    )
    plugin = parser.add_mutually_exclusive_group()
    plugin.add_argument(
        "--cuda-plugin",
        action="store_true",
        help="Register CUDA from the installed 'onnxruntime_ep_cuda' package.",
    )
    plugin.add_argument(
        "--cuda-plugin-library",
        type=Path,
        help="Register a CUDA plugin EP library from an explicit path.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print TTFT, throughput, and drafter acceptance statistics.",
    )
    args = parser.parse_args()

    prompt_pool = []
    if args.prompt_file is not None:
        prompt_pool.extend(load_prompts(args.prompt_file.expanduser().resolve()))
    if args.prompts:
        prompt_pool.extend(args.prompts)
    if not prompt_pool:
        prompt_pool = load_prompts(Path(__file__).with_name("prompts.json"))

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.prompt_file is not None or args.prompts:
        batch_size = len(prompt_pool)
    else:
        batch_size = 1
    if batch_size <= 0:
        parser.error("--batch-size must be positive")
    prompts = [
        prompt_pool[index % len(prompt_pool)]
        for index in range(batch_size)
    ]

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be positive")
    if args.event_buffer_capacity <= 0:
        parser.error("--event-buffer-capacity must be positive")
    if args.max_draft_tokens is not None and args.max_draft_tokens <= 0:
        parser.error("--max-draft-tokens must be positive")

    model_dir = args.model.expanduser().resolve()
    drafter, configured_draft_tokens = detect_drafter(model_dir)
    max_draft_tokens = (
        args.max_draft_tokens
        if args.max_draft_tokens is not None
        else configured_draft_tokens
    )
    speculative = drafter is not None and not args.no_drafter and max_draft_tokens > 0
    if args.no_drafter and drafter is None:
        parser.error(f"--no-drafter was passed but '{model_dir}' declares no drafter")
    if speculative:
        print(f"Drafter: {drafter} (max_draft_tokens={max_draft_tokens})")
    else:
        print("Drafter: none (non-speculative baseline)")
    if args.thinking:
        print(f"Reasoning: on (effort={args.reasoning_effort})")
    else:
        print("Reasoning: off")

    register_cuda_plugin(args.cuda_plugin, args.cuda_plugin_library)

    config = og.Config(str(model_dir))
    overlay = build_overlay(args, drafter)
    if overlay:
        config.overlay(json.dumps(overlay))
    config.clear_providers()
    config.append_provider("cuda")
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(args.event_buffer_capacity)

    template_str = chat_template(model_dir, args.thinking, args.reasoning_effort)
    input_ids_by_index = {
        index: encode_prompt(tokenizer, prompt, template_str)
        for index, prompt in enumerate(prompts)
    }
    oversized = {
        index: len(input_ids)
        for index, input_ids in input_ids_by_index.items()
        if len(input_ids) > args.max_prompt_tokens
    }
    if oversized:
        details = ", ".join(
            f"request {index}: {length}"
            for index, length in oversized.items()
        )
        parser.error(
            f"Prompt token count exceeds --max-prompt-tokens={args.max_prompt_tokens} "
            f"({details}). Raise the limit explicitly for a long-context test."
        )
    requests = {}
    outputs = {}
    token_times = {index: [] for index in range(len(prompts))}
    submitted_at = {}
    finished_at = {}
    closed = set()
    stats_baseline = None
    if args.warmup:
        run_warmup(
            engine,
            model,
            event_buffer,
            input_ids_by_index[0],
            speculative,
            max_draft_tokens,
        )
        if speculative:
            stats_baseline = engine.get_speculative_stats()
    benchmark_start = time.perf_counter()
    for index in range(len(prompts)):
        input_ids = input_ids_by_index[index]
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=args.do_sample,
            max_length=len(input_ids) + args.max_new_tokens,
        )
        if speculative:
            params.set_speculative_options(max_draft_tokens=max_draft_tokens)
        request = engine.create_request(params)
        submitted_at[index] = time.perf_counter()
        request.begin_turn(input_ids)
        requests[request] = index
        outputs[index] = []

    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.request is None:
                    # Engine-level notices such as CAPACITY_BLOCKED carry no request.
                    if event.flags & og.EngineEventFlags.FAILED:
                        raise RuntimeError(
                            f"Engine failed; flags={event.flags}, "
                            f"error_code={event.error_code}"
                        )
                    continue
                index = requests[event.request]
                if event.flags & og.EngineEventFlags.TOKEN:
                    outputs[index].append(int(event.token))
                    token_times[index].append(time.perf_counter())
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(
                        f"Request {index} failed; error_code={event.error_code}"
                    )
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finished_at[index] = time.perf_counter()
                    event.request.close()
                    closed.add(index)
    finally:
        for request, index in requests.items():
            if index not in closed:
                request.close()

    for index, prompt in enumerate(prompts):
        print(f"\nPrompt {index + 1}: {prompt}")
        print(f"Response {index + 1}: {tokenizer.decode(outputs[index])}")

    if args.metrics:
        first_token_at = {
            index: times[0] for index, times in token_times.items() if times
        }
        last_token_at = {
            index: times[-1] for index, times in token_times.items() if times
        }
        all_requests_produced_tokens = len(first_token_at) == len(prompts)
        last_first_token_at = (
            max(first_token_at.values()) if all_requests_produced_tokens else None
        )
        prompt_seconds = (
            last_first_token_at - benchmark_start
            if last_first_token_at is not None
            else None
        )
        steady_decode_tokens = (
            sum(
                timestamp > last_first_token_at
                for times in token_times.values()
                for timestamp in times
            )
            if last_first_token_at is not None
            else 0
        )
        decode_seconds = (
            max(last_token_at.values()) - last_first_token_at
            if last_first_token_at is not None and last_token_at
            else None
        )
        output_tokens = sum(map(len, outputs.values()))
        metrics = {
            "drafter": drafter if speculative else None,
            "max_draft_tokens": max_draft_tokens if speculative else 0,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort if args.thinking else None,
            "requests": len(prompts),
            "input_tokens": sum(map(len, input_ids_by_index.values())),
            "output_tokens": output_tokens,
            "ttft_seconds": {
                str(index): (
                    first_token_at[index] - submitted_at[index]
                    if index in first_token_at
                    else None
                )
                for index in range(len(prompts))
            },
            "all_requests_ttft_seconds": prompt_seconds,
            "effective_prompt_tokens_per_second": (
                sum(map(len, input_ids_by_index.values())) / prompt_seconds
                if prompt_seconds is not None and prompt_seconds > 0
                else None
            ),
            "steady_decode_tokens_per_second": (
                steady_decode_tokens / decode_seconds
                if decode_seconds is not None and decode_seconds > 0
                else None
            ),
            "total_seconds": (
                max(finished_at.values()) - benchmark_start
                if len(finished_at) == len(prompts)
                else None
            ),
        }
        if speculative:
            metrics["speculative"] = speculative_metrics(
                engine, output_tokens, stats_baseline
            )
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
