#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Smoke test / minimal usage example for a qwen38_flash_graph package.

usage: smoke.py <package dir> [--tokens N] [--prompt TEXT] [--no-mtp] [--provider PATH]
"""

import argparse
import json
import os
import time

import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="package directory (genai_config.json with model.type qwen38_flash_graph)")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    parser.add_argument("--no-mtp", action="store_true", help="plain S=1 CUDA-graph decode")
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()

    if args.provider:
        og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    if args.no_mtp:
        config.overlay(json.dumps({"model": {"qwen38_graph": {"mtp": {"enabled": False}}}}))
    t0 = time.perf_counter()
    model = og.Model(config)
    print(f"model load {time.perf_counter() - t0:.1f} s", flush=True)
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()

    messages = json.dumps([{"role": "user", "content": args.prompt}])
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)
    print(f"prompt tokens: {len(tokens)}", flush=True)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tokens) + args.tokens, do_sample=False)
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    generator.append_tokens(tokens)
    count = 0
    t_first = None
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        if t_first is None:
            t_first = time.perf_counter()
        count += 1
        print(stream.decode(token), end="", flush=True)
    t1 = time.perf_counter()
    print()
    print(
        f"prefill+first token {t_first - t0:.3f} s; {count} tokens; decode {(count - 1) / (t1 - t_first):.1f} tokens/s"
    )
    print("stats:", generator.get_speculative_stats())


if __name__ == "__main__":
    main()
