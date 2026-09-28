#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compare distinct DFlash prompts sharing a long prefix with and without reuse."""

import argparse
import json
import time

import numpy as np
import onnxruntime_genai as og


def run_model(model_path, prompts, generated_tokens):
    model = og.Model(str(model_path))
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(64)
    runs = []
    for prompt in prompts:
        request = engine.create_request()
        options = og.TurnOptions(request)
        options.set_max_generated_tokens(generated_tokens)
        options.set_do_sample(False)
        before = dict(engine.get_speculative_stats())
        started = time.perf_counter()
        request.begin_turn(prompt, options)
        tokens = []
        first_token_ms = None
        usage = None
        for _ in range(4096):
            if not engine.has_pending_requests():
                break
            for event in engine.run(buffer):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine failed: {event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    tokens.append(event.token)
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    usage = {
                        "cached_prompt_tokens": event.usage.cached_prompt_tokens,
                        "generated_tokens": event.usage.generated_tokens,
                    }
        else:
            raise RuntimeError("Engine did not finish within 4096 calls")
        if usage is None:
            raise RuntimeError("Engine produced no completion event")
        after = dict(engine.get_speculative_stats())
        runs.append({
            "prompt_tokens": len(prompt),
            "ttft_ms": first_token_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
            "usage": usage,
            "draft_forwards": after["draft_forward_passes"] - before["draft_forward_passes"],
            "accepted_drafts": after["draft_tokens_accepted"] - before["draft_tokens_accepted"],
            "fallback_steps": after["standard_fallback_steps"] - before["standard_fallback_steps"],
            "tokens": tokens,
        })
        request.close()
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled-model", required=True)
    parser.add_argument("--disabled-model", required=True)
    parser.add_argument("--shared-prefix-tokens", type=int, default=40000)
    parser.add_argument("--cache-block-size", type=int, default=256)
    parser.add_argument("--generated-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if (args.shared_prefix_tokens < 2 or args.cache_block_size < 1 or
            args.generated_tokens < 1 or args.repetitions < 2):
        parser.error("Use a positive block size, at least two shared tokens and prompts, and one generated token")

    tokenizer_model = og.Model(args.disabled_model)
    tokenizer = og.Tokenizer(tokenizer_model)
    text = "Explain how prefix caching interacts with speculative decoding. " * max(
        1, args.shared_prefix_tokens // 16
    )
    encoded = tokenizer.encode(text)
    while len(encoded) < args.shared_prefix_tokens:
        text += text
        encoded = tokenizer.encode(text)
    prefix = np.asarray(encoded[:args.shared_prefix_tokens], dtype=np.int32)
    prompts = [
        np.concatenate((
            prefix,
            np.asarray(tokenizer.encode(
                f" Distinct request {i}: describe a different cache workload."
            ), dtype=np.int32),
        ))
        for i in range(args.repetitions)
    ]
    if any(len(prompt) <= len(prefix) for prompt in prompts):
        raise RuntimeError("The tokenizer produced an empty prompt suffix")
    if len({tuple(prompt[len(prefix):]) for prompt in prompts}) != len(prompts):
        raise RuntimeError("The tokenizer produced identical prompt suffixes")
    del tokenizer, tokenizer_model

    disabled = run_model(args.disabled_model, prompts, args.generated_tokens)
    enabled = run_model(args.enabled_model, prompts, args.generated_tokens)
    for index, (cold, cached) in enumerate(zip(disabled, enabled, strict=True)):
        if cold["tokens"] != cached["tokens"]:
            raise RuntimeError(
                f"Prompt {index} produced different greedy tokens with caching on and off: "
                f"uncached={cold}, cached={cached}"
            )
    min_cached = args.shared_prefix_tokens // args.cache_block_size * args.cache_block_size
    if any(run["usage"]["cached_prompt_tokens"] < min_cached for run in enabled[1:]):
        raise RuntimeError(f"Warm prompts did not reuse at least {min_cached} prefix tokens")
    for runs in (disabled, enabled):
        for run in runs:
            del run["tokens"]
    print(json.dumps({
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "disabled": disabled,
        "enabled": enabled,
    }, indent=2))


if __name__ == "__main__":
    main()
