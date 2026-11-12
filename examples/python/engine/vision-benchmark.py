# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Reproduce the eager packed-Qwen vision qualification timings with fresh Engine trials."""

import argparse
import gc
import json
import time
from itertools import pairwise
from statistics import median

import onnxruntime_genai as og


def measure(model, tokenizer, kind, image_path):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(4096)
    request = engine.create_request(options=request_options)
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_min_generated_tokens(16)
    options.set_max_generated_tokens(32)
    images = og.Images.open(image_path) if kind == "image" else None
    content = (
        "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the vehicles and visible colors in a few sentences."
        if images is not None
        else "Give a concise numbered list of steps for diagnosing a failing software build."
    )
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps([{"role": "user", "content": content}]), add_generation_prompt=True
    )
    if not prompt.endswith("<think>\n"):
        raise ValueError("This benchmark requires the Qwen qualification chat template.")
    prompt += "\n</think>\n\n"
    tokens, token_times, encoder_ms = [], [], []
    try:
        admitted_at = time.perf_counter()
        request.begin_turn(prompt, options, images=images)
        admission_ms = (time.perf_counter() - admitted_at) * 1000
        buffer = engine.create_event_buffer(1)
        for _ in range(4096):
            if not engine.has_pending_requests():
                break
            before = engine.get_speculative_stats()["target_forward_passes"]
            run_at = time.perf_counter()
            events = engine.run(buffer)
            elapsed_ms = (time.perf_counter() - run_at) * 1000
            if before == engine.get_speculative_stats()["target_forward_passes"] and not events:
                encoder_ms.append(elapsed_ms)
            for event in events:
                if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                    raise RuntimeError(f"Engine request failed or exceeded capacity: {event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    token_times.append(time.perf_counter())
                    tokens.append(event.token)
        else:
            raise RuntimeError("The benchmark request did not make bounded progress.")
        if len(tokens) < 2:
            raise RuntimeError("The benchmark needs at least two generated tokens for decode timing.")
        return {
            "kind": kind,
            "admission_ms": admission_ms,
            "encoder_run_ms": encoder_ms,
            "ttft_ms": (token_times[0] - admitted_at) * 1000,
            "median_decode_ms": median((right - left) * 1000 for left, right in pairwise(token_times)),
            "tokens": len(tokens),
            "answer": tokenizer.decode(tokens),
        }
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Complete packed Qwen vision package with eager target-only settings"
    )
    parser.add_argument(
        "--image", required=True, help="Use test/images/cars.jpg to reproduce the qualification workload"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Trials per workload after one text warmup (default: 3)")
    parser.add_argument(
        "--profile", help="Optional ORT profiling output prefix; profiling timings are not benchmark results"
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    config = og.Config(args.model)
    if args.profile:
        config.overlay(json.dumps({"model": {"decoder": {"session_options": {"enable_profiling": args.profile}}}}))
    start = time.perf_counter()
    model = og.Model(config)
    print(json.dumps({"model_load_seconds": time.perf_counter() - start}), flush=True)
    tokenizer = og.Tokenizer(model)
    for kind in ["warmup", *(["text"] * args.repeats), *(["image"] * args.repeats)]:
        print(json.dumps(measure(model, tokenizer, kind, args.image)), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
