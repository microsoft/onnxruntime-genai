#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Smoke test / minimal usage example of a qwen38_flash_graph package through the Engine API.

usage: smoke.py <package dir> [--tokens N] [--prompt TEXT] [--no-mtp] [--provider PATH]
"""

import argparse
import json
import time

import numpy as np
import onnxruntime_genai as og
from qwen38_engine_util import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="package directory (genai_config.json with model.type qwen38_flash_graph)")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    parser.add_argument("--no-mtp", action="store_true", help="plain S=1 CUDA-graph decode")
    parser.add_argument("--provider")
    args = parser.parse_args()

    overlay = {"model": {"qwen38_graph": {"mtp": {"enabled": False}}}} if args.no_mtp else None
    t0 = time.perf_counter()
    model = load_model(args.model, args.provider, overlay)
    print(f"model load {time.perf_counter() - t0:.1f} s", flush=True)
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()
    messages = json.dumps([{"role": "user", "content": args.prompt}])
    tokens = tokenizer.encode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
    print(f"prompt tokens: {len(tokens)}", flush=True)

    engine = og.Engine(model)
    print(
        "capabilities:",
        engine.get_capabilities().configured_max_batch_size,
        engine.get_capabilities().max_request_length,
    )
    buffer = engine.create_event_buffer(16)
    request = engine.create_request()
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_max_generated_tokens(args.tokens)
    t0 = time.perf_counter()
    request.begin_turn(np.asarray(tokens, dtype=np.int32), turn)
    count, t_first, finish = 0, None, None
    while engine.has_pending_requests():
        for event in engine.run(buffer):
            if event.flags & og.EngineEventFlags.TOKEN:
                if t_first is None:
                    t_first = time.perf_counter()
                count += 1
                print(stream.decode(event.token), end="", flush=True)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                finish = event.finish_reason
    t1 = time.perf_counter()
    print()
    print(
        f"first token after {t_first - t0:.3f} s; {count} tokens; decode {(count - 1) / (t1 - t_first):.1f} tokens/s; finish {finish}"
    )
    print("stats:", engine.get_speculative_stats())


if __name__ == "__main__":
    main()
