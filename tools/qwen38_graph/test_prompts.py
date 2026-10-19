#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Functional tests of the fast prefill path through the public API: prompt lengths around the QSA block (4), chunk (8192) and
append boundaries, chunked appends of a long prompt, state leakage between consecutive generators.

Dense prefill numerics are not bit-reproducible, so the checks are: no error, expected token counts, first token equal to the
steps path (S=2 graph steps, bit-exact) for short prompts, logits agreement (relative L2 against the steps path, and run-to-run for
the chunked append), acceptance rate of the decode that follows.

usage: test_prompts.py <package> --prompts-module <prompts.py> --out <json> [--long]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


def reference_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.suffix != ".py":
        raise argparse.ArgumentTypeError(f"expected an existing Python reference module: {path}")
    return path


def load_prompts(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("prompts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "make", None)):
        raise ValueError(f"{path} must provide make(kind, length, seed)")
    return module


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
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    norm = np.linalg.norm(a.astype(np.float64))
    diff = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
    return float(diff / norm) if norm else (0.0 if diff == 0 else float("inf"))


def run_ok(result, expected=48):
    return result["seq_ok"] and len(result["tokens"]) == expected


def compare(reference, actual, tolerance):
    first_equal = bool(reference["tokens"] and actual["tokens"] and reference["tokens"][0] == actual["tokens"][0])
    error = rel_l2(reference["logits"], actual["logits"])
    return {
        "first_equal": first_equal,
        "logits_rel_l2": error,
        "pass": bool(run_ok(reference) and run_ok(actual) and first_equal and error <= tolerance),
    }


def positive_tolerance(value):
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise argparse.ArgumentTypeError("tolerance must be finite and positive")
    return tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prompts-module", type=reference_file, required=True, help="trusted prompts.py reference file"
    )
    parser.add_argument("--logits-rtol", type=positive_tolerance, default=1e-3, help="maximum relative L2 logits error")
    parser.add_argument(
        "--long", action="store_true", help="also 8191/8192/8193/16385-token prompts and a 12000-token chunked append"
    )
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    prompts = load_prompts(args.prompts_module)
    if args.provider:
        og.register_execution_provider_library("CUDAExecutionProvider", args.provider)

    def load(fast):
        config = og.Config(args.model)
        config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": fast}}}}))
        return og.Model(config)

    results, ok = {"logits_rtol": args.logits_rtol}, True
    short = [1, 2, 3, 4, 5, 6, 7, 8, 9, 127, 128, 129, 513]
    pool = prompts.make("text", 20000, 77)
    if len(pool) < 20000:
        raise ValueError("prompts.make('text', 20000, 77) must return at least 20000 token IDs")
    fast = load(True)
    fast_runs = {}
    for n in short:
        fast_runs[n] = run(fast, [pool[:n]])
    # state leakage: a longer prompt in between, then the 5-token prompt again
    intervening = run(fast, [pool[:4097]])
    again = run(fast, [pool[:5]])
    leakage = compare(fast_runs[5], again, args.logits_rtol)
    leakage["intervening_run_ok"] = bool(run_ok(intervening))
    leakage["pass"] &= leakage["intervening_run_ok"]
    results["leakage_5_after_4097"] = leakage
    ok &= leakage["pass"]
    long_results = {}
    if args.long:
        for n in (8191, 8192, 8193, 16385):
            r = run(fast, [pool[:n]])
            long_results[n] = {k: r[k] for k in ("seconds", "accept", "seq_ok")} | {"n_tokens": len(r["tokens"])}
            ok &= run_ok(r)
        single = run(fast, [pool[:12000]])
        chunked = run(fast, [pool[:5000], pool[5000:10000], pool[10000:12000]])
        single2 = run(fast, [pool[:12000]])
        chunked_comparison = compare(single, chunked, args.logits_rtol)
        repeated_comparison = compare(single, single2, args.logits_rtol)
        results["chunked_12000"] = {
            "first_equal": chunked_comparison["first_equal"],
            "first_equal_repeat": repeated_comparison["first_equal"],
            "logits_rel_l2_chunked_vs_single": chunked_comparison["logits_rel_l2"],
            "logits_rel_l2_single_vs_single": repeated_comparison["logits_rel_l2"],
            "seq_ok": chunked["seq_ok"],
            "token_counts": [len(r["tokens"]) for r in (single, chunked, single2)],
            "accept_chunked": chunked["accept"],
            "accept_single": single["accept"],
            "pass": chunked_comparison["pass"] and repeated_comparison["pass"],
        }
        ok &= results["chunked_12000"]["pass"]
    del fast
    steps = load(False)
    per_length = {}
    for n in short:
        r = run(steps, [pool[:n]])
        f = fast_runs[n]
        comparison = compare(r, f, args.logits_rtol)
        per_length[n] = {
            "first_equal_steps": comparison["first_equal"],
            "logits_rel_l2_vs_steps": comparison["logits_rel_l2"],
            "accept_fast": f["accept"],
            "accept_steps": r["accept"],
            "tokens": len(f["tokens"]),
            "seq_ok": f["seq_ok"],
            "tokens_steps": len(r["tokens"]),
            "seq_ok_steps": r["seq_ok"],
            "pass": comparison["pass"],
        }
        ok &= comparison["pass"]
    results["lengths"] = per_length
    results["long"] = long_results
    results["all_ok"] = bool(ok)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
