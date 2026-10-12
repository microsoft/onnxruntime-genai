#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Token gate of the v12 package (fold decode, optional fast prefill) through the public GenAI API.

Cases = the gate prompts of CG/mtp_runner/gate_mtp.py (text / prose / random 2k, three chat prompts), 256 new tokens.
  --prefill steps  prompt through the muted fold graph (S=2 steps, bit-exact path): the tokens must equal the research no-MTP
                   greedy hashes (CG/mtp_runner/runs/s2dev_gate_c2048_n256.json) and the Python fold runner (--python-ref, "old").
  --prefill fast   dense prefill + hand-off (tolerance level): first token and identical prefix against the Python PrefillEngine
                   (--python-ref, "new" / "new_rep1") and against the steps path of the research hashes.
Run on a package whose stop ids are unreachable (build_package_v12.py --stop-token-ids 248319) with --ignore-eos for the fixed-length
reference, or on the default package for EOS mode.

usage: gate_v12.py <package> --out <json> [--prefill steps|fast] [--ignore-eos] [--cases text,prose,random,chat] [--python-ref json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_genai as gg  # noqa: E402

prompts = gg.prompts


def common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill", default="fast", choices=("fast", "steps"))
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--cases", default="text,prose,random,chat")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--python-ref", type=Path, help="ref_v12.py json of the same mode (fixed|eos)")
    parser.add_argument("--no-research-hash", action="store_true",
                        help="v12b: the stored research hashes are the pre-v12b numerics; the gate is then the python reference only (research equality stays in the output)")
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    reference = {r["case"]: r for r in json.loads(gg.REF_JSON.read_text())["results"]}
    py = {}
    if args.python_ref:
        for c in json.loads(args.python_ref.read_text())["cases"]:
            py[c["case"]] = c

    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}}))
    model = og.Model(config)

    cases = []
    for case in args.cases.split(","):
        if case == "chat":
            for i, q in enumerate(gg.CHAT_QUESTIONS):
                cases.append((f"chat:{q[:24]}", f"chat:{i}", prompts.chat_prompt(q)))
        else:
            cases.append((f"{case}{args.context}", f"{case}:{args.context}:{1000 + args.context}",
                          prompts.make(case, args.context, 1000 + args.context)))

    results, ok = [], True
    for name, spec, prompt in cases:
        ref_entry = reference[name]["plain"][1 if args.ignore_eos else 0]
        entry = {"case": name, "spec": spec, "prompt_len": len(prompt), "ref_hash": ref_entry["hash"], "ref_n": ref_entry["n"]}
        runs = []
        for _ in range(args.repeat):
            run = gg.run_generation(model, prompt, args.new)
            runs.append(run)
        tokens = runs[0]["tokens"]
        entry.update({"n": len(tokens), "hash": gg.token_hash(tokens), "first": tokens[0], "tokens_head": tokens[:16],
                      "prefill_s": runs[0]["prefill_s"], "decode_s": runs[0]["decode_s"],
                      "tps": (len(tokens) - 1) / runs[0]["decode_s"] if runs[0]["decode_s"] > 0 else None,
                      "stats": runs[0]["stats"],
                      "sequence_matches_next_tokens": runs[0]["sequence"] == tokens[: len(runs[0]["sequence"])]})
        entry["equals_research_hash"] = entry["hash"] == ref_entry["hash"]
        if len(runs) > 1:
            entry["repeat_identical"] = [r["tokens"] == tokens for r in runs[1:]]
            entry["repeat_prefix"] = [common_prefix(tokens, r["tokens"]) for r in runs[1:]]
        if spec in py:
            p = py[spec]
            entry["python"] = {}
            for key in ("old", "new", "new_rep1"):
                if key in p:
                    pt = p[key]["tokens"]
                    entry["python"][key] = {"hash": p[key]["hash"], "identical": pt == tokens, "identical_prefix": common_prefix(pt, tokens),
                                            "first_equal": p[key]["first"] == tokens[0], "n": len(pt)}
        if args.prefill == "steps":
            ok &= (entry["equals_research_hash"] or args.no_research_hash) and entry["sequence_matches_next_tokens"]
            if args.no_research_hash and "old" not in entry.get("python", {}):
                ok = False
            if "old" in entry.get("python", {}):
                ok &= entry["python"]["old"]["identical"]
        results.append(entry)
        print(json.dumps({k: entry[k] for k in entry if k not in ("tokens_head", "stats", "ref_hash")}), flush=True)
    out = {"all_ok": ok, "prefill": args.prefill, "ignore_eos": args.ignore_eos, "context": args.context, "new": args.new,
           "results": results}
    args.out.write_text(json.dumps(out, indent=1))
    if args.prefill == "steps":
        print("GATE", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1
    print("GATE (fast prefill is tolerance level: no pass/fail, see the identical_prefix / first_equal fields)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
