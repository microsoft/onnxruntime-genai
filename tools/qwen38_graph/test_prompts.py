#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Functional tests of the fast prefill path through the public API: prompt lengths around the QSA block (4), chunk (8192) and
append boundaries, chunked appends of a long prompt, state leakage between consecutive generators.

Dense prefill numerics are not bit-reproducible, so the checks are: no error, expected token counts, first token equal to the
steps path (S=2 graph steps, bit-exact) for short prompts, logits agreement (relative L2 against the steps path, and run-to-run for
the chunked append), acceptance rate of the decode that follows.

usage: test_prompts.py <package> --out <json> [--long]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

RUNNER = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
sys.path.insert(0, str(RUNNER))
import prompts  # noqa: E402


def run(model, prompt_chunks, new=48):
    prompt = [t for c in prompt_chunks for t in c]
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + new, do_sample=False)
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    for chunk in prompt_chunks:
        generator.append_tokens(chunk)
    logits = np.asarray(generator.get_logits(), dtype=np.float32).reshape(-1)
    tokens = []
    while not generator.is_done():
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
    stats = generator.get_speculative_stats()
    seq = [int(x) for x in generator.get_sequence(0)]
    del generator
    return {
        "tokens": tokens,
        "logits": logits,
        "seconds": time.perf_counter() - t0,
        "seq_ok": seq == prompt + tokens[: len(seq) - len(prompt)],
        "accept": stats["draft_tokens_accepted"] / max(1, stats["rounds"]),
    }


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--long", action="store_true", help="also 8191/8192/8193/16385-token prompts and a 12000-token chunked append"
    )
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)

    def load(fast):
        config = og.Config(args.model)
        config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": fast}}}}))
        return og.Model(config)

    results, ok = {}, True
    short = [1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 128, 129, 513]
    pool = prompts.make("text", 20000, 77)
    fast = load(True)
    fast_runs = {}
    for n in short:
        fast_runs[n] = run(fast, [pool[:n]])
    # state leakage: a longer prompt in between, then the 5-token prompt again
    run(fast, [pool[:4097]])
    again = run(fast, [pool[:5]])
    results["leakage_5_after_4097"] = {
        "first_equal": again["tokens"][0] == fast_runs[5]["tokens"][0],
        "logits_rel_l2": rel_l2(fast_runs[5]["logits"], again["logits"]),
    }
    long_results = {}
    if args.long:
        for n in (8191, 8192, 8193, 16385):
            r = run(fast, [pool[:n]])
            long_results[n] = {k: r[k] for k in ("seconds", "accept", "seq_ok")} | {"n_tokens": len(r["tokens"])}
            ok &= r["seq_ok"] and len(r["tokens"]) == 48
        single = run(fast, [pool[:12000]])
        chunked = run(fast, [pool[:5000], pool[5000:10000], pool[10000:12000]])
        single2 = run(fast, [pool[:12000]])
        results["chunked_12000"] = {
            "first_equal": single["tokens"][0] == chunked["tokens"][0],
            "logits_rel_l2_chunked_vs_single": rel_l2(single["logits"], chunked["logits"]),
            "logits_rel_l2_single_vs_single": rel_l2(single["logits"], single2["logits"]),
            "seq_ok": chunked["seq_ok"],
            "accept_chunked": chunked["accept"],
            "accept_single": single["accept"],
        }
        ok &= chunked["seq_ok"]
    del fast
    steps = load(False)
    per_length = {}
    for n in short:
        r = run(steps, [pool[:n]])
        f = fast_runs[n]
        per_length[n] = {
            "first_equal_steps": r["tokens"][0] == f["tokens"][0],
            "logits_rel_l2_vs_steps": rel_l2(r["logits"], f["logits"]),
            "accept_fast": f["accept"],
            "accept_steps": r["accept"],
            "tokens": len(f["tokens"]),
            "seq_ok": f["seq_ok"],
        }
        ok &= f["seq_ok"] and len(f["tokens"]) == 48
    results["lengths"] = per_length
    results["long"] = long_results
    results["all_ok"] = bool(ok)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
