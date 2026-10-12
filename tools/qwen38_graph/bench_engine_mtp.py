#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Decode benchmark of the Engine-hosted qwen38_flash_graph backend.

Protocol: prompts of an exact token length cut from a natural-text file (--text-file; its content decides the draft
acceptance rate, so use real prose, not repeated or random tokens), --new generated tokens, one warm-up and --reps measured
runs on distinct prompts. TPS = (n-1) generated tokens / (time the last token was delivered - time the first token was
delivered), measured around Engine.run:

    request.begin_turn(prompt, turn); while engine.has_pending_requests(): events = engine.run(buffer)

For a fixed-length run the model must not stop early: use a package copy whose eos_token_id list holds ids the model does not
generate (the script reports the number of tokens and the finish reason of every run, and flags short runs).

usage: bench_engine_mtp.py <package> --text-file TEXT --out <json> [--contexts 2048,8192,32768] [--no-mtp] [--reps 3] [--new 256]
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import (
    add_common_arguments,
    load_model,
    load_text_tokens,
    mtp_overlay,
    run_turn,
    text_prompt,
    token_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", default="2048,8192,32768")
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--no-head-prefill", action="store_true", help="start the MTP head without prompt context")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    model = load_model(args.model, args.provider, mtp_overlay(not args.no_mtp, not args.no_head_prefill))
    text_tokens = load_text_tokens(og.Tokenizer(model), args.text_file)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    result = {"args": {k: str(v) for k, v in vars(args).items()}, "contexts": {}}
    for n in [int(x) for x in args.contexts.split(",")]:
        seeds = [1000 + n] + [i * 7 + n for i in range(args.reps)]
        runs = []
        for index, seed in enumerate(seeds):
            prompt = text_prompt(text_tokens, n, seed)
            before = engine.get_speculative_stats()
            request = engine.create_request()
            turn, _ = run_turn(engine, buffer, request, prompt, args.new)
            request.close()
            after = engine.get_speculative_stats()
            count = len(turn.tokens)
            if count < 2:
                raise RuntimeError(f"context {n}: only {count} token(s) generated ({turn.finish_reason})")
            decode_s = turn.times[-1] - turn.times[0]
            rounds = after["rounds"] - before["rounds"]
            accepted = after["draft_tokens_accepted"] - before["draft_tokens_accepted"]
            record = {
                "label": "warmup" if index == 0 else f"run{index}",
                "seed": seed,
                "tokens": count,
                "complete": count == args.new,
                "finish_reason": str(turn.finish_reason),
                "hash": token_hash(turn.tokens),
                "first_token_s": turn.times[0] - turn.start,
                "decode_s": decode_s,
                "tps": (count - 1) / decode_s,
                "cycles": rounds,
                "accepted": accepted,
                "accept_rate": accepted / rounds if rounds else None,
                "ms_per_cycle": decode_s / rounds * 1e3 if rounds else None,
                "engine_runs": turn.runs,
            }
            runs.append(record)
            print(json.dumps({"ctx": n, **record}), flush=True)
        measured = sorted((r for r in runs if r["label"] != "warmup"), key=lambda r: r["tps"])
        median = measured[len(measured) // 2]
        result["contexts"][n] = {"median": median, "runs": runs}
        short = sum(1 for r in measured if not r["complete"])
        print(
            f"ctx {n}: decode {median['tps']:.2f} tps (median of {len(measured)}), accept {median['accept_rate']}"
            + (f"  WARNING: {short} run(s) stopped before --new tokens" if short else ""),
            flush=True,
        )
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
