#!/usr/bin/env python3
"""Minimal continuous-batching Engine example for ONNX Runtime GenAI."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


def encode_prompt(tokenizer: og.Tokenizer, prompt: str) -> np.ndarray:
    messages = json.dumps([{"role": "user", "content": prompt}])
    rendered = tokenizer.apply_chat_template(
        messages=messages,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more prompts with the continuous-batching Engine."
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
        help="Reject longer tokenized prompts. Raise deliberately for long-context tests.",
    )
    parser.add_argument("--event-buffer-capacity", type=int, default=16)
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
        help="Print TTFT, effective prompt throughput, and steady decode throughput.",
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

    register_cuda_plugin(args.cuda_plugin, args.cuda_plugin_library)

    config = og.Config(str(args.model))
    config.clear_providers()
    config.append_provider("cuda")
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(args.event_buffer_capacity)

    input_ids_by_index = {
        index: encode_prompt(tokenizer, prompt)
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
    benchmark_start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        input_ids = input_ids_by_index[index]
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=len(input_ids) + args.max_new_tokens,
        )
        request = engine.create_request(params)
        submitted_at[index] = time.perf_counter()
        request.begin_turn(input_ids)
        requests[request] = index
        outputs[index] = []

    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.request is None:
                    raise RuntimeError(
                        f"Engine event has no request; flags={event.flags}, "
                        f"error_code={event.error_code}"
                    )
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
    finally:
        for request in requests:
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
        metrics = {
            "requests": len(prompts),
            "input_tokens": sum(map(len, input_ids_by_index.values())),
            "output_tokens": sum(map(len, outputs.values())),
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
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
