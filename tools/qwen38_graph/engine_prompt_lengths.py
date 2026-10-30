#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Prompt-length and chunk-boundary check of the Engine-hosted qwen38_flash_graph backend.

Runs greedy turns of --new tokens for prompts of many lengths (1 token, odd / even lengths, dense chunk boundaries 8191 / 8192 /
8193, two and three chunks) and writes the token streams. Run it twice on the same package, once with --prefill steps (the
bit-exact S=2-step prompt path) and once with --prefill fast (dense chunks + hand-off; --chunk N changes the dense chunk), then
compare:

  engine_prompt_lengths.py compare steps.json fast.json

Pass / fail is structural: every turn must end with the token limit after --new tokens (or with the model's own EOS before) and report matching usage. The fast
prefill is tolerance level, so the agreement with the reference (first token equal, shared prefix) is reported, not required: the
prompts are arbitrary cuts of a text, and for tiny or unnatural prompts the top-1 margin is often inside the numerical noise.

usage: engine_prompt_lengths.py run <package> --prefill steps|fast --out <json> [--lengths 1,2,...] [--new 32] [--chunk N]
                                    [--text-pool FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import Prompts, add_prompt_arguments, load_model, run_turn, token_hash

LENGTHS = "1,2,3,4,5,7,8,31,127,128,129,1000,2047,2048,2049,4097,8191,8192,8193,12000,16384,16385,20000"


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    prefill = {"enabled": args.prefill == "fast"}
    if args.chunk:
        prefill["chunk"] = args.chunk
    model = load_model(args.model, args.provider, {"model": {"qwen38_graph": {"prefill": prefill}}})
    prompts = Prompts.from_args(model, args)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)
    results = []
    for n in [int(x) for x in args.lengths.split(",")]:
        prompt = prompts.make("text", max(n, 64), 77 + n)[:n]
        request = engine.create_request()
        turn, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        entry = {
            "length": n,
            "tokens": turn.tokens,
            "hash": token_hash(turn.tokens),
            "finish_reason": str(turn.finish_reason),
            "usage_ok": turn.usage.prompt_tokens == n
            and turn.usage.generated_tokens == len(turn.tokens)
            and (
                (turn.finish_reason == og.FinishReason.MAX_GENERATED_TOKENS and len(turn.tokens) == args.new)
                or (turn.finish_reason == og.FinishReason.EOS and len(turn.tokens) < args.new)
            ),
            "engine_runs": turn.runs,
        }
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("length", "finish_reason", "usage_ok", "engine_runs")}), flush=True)
    args.out.write_text(
        json.dumps({"prefill": args.prefill, "chunk": args.chunk, "new": args.new, "results": results}, indent=1)
    )


def compare(args):
    a = {r["length"]: r for r in json.loads(Path(args.reference).read_text())["results"]}
    b = {r["length"]: r for r in json.loads(Path(args.candidate).read_text())["results"]}
    ok = True
    first_equal = 0
    for n in sorted(a):
        x, y = a[n]["tokens"], b[n]["tokens"]
        common = 0
        while common < min(len(x), len(y)) and x[common] == y[common]:
            common += 1
        structural = a[n]["usage_ok"] and b[n]["usage_ok"]
        ok &= structural
        first_equal += x[:1] == y[:1]
        print(
            f"length {n:6d}: first token equal {x[:1] == y[:1]}, identical prefix {common}/{len(x)}, structural ok {structural}"
        )
    print(f"first token equal for {first_equal} of {len(a)} lengths")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--prefill", choices=("steps", "fast"), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lengths", default=LENGTHS)
    r.add_argument("--new", type=int, default=32)
    r.add_argument("--chunk", type=int, default=0, help="overlay prefill.chunk (dense chunk size)")
    r.add_argument("--provider")
    add_prompt_arguments(r)
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    args = parser.parse_args()
    return run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
