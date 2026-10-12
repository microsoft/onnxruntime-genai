#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Correctness gate of the GenAI qwen38_flash_graph model against the research runner.

The prompts are those of CG/mtp_runner/gate_mtp.py (text/prose/random 2k, three chat prompts); the reference token hashes
are the no-MTP v10all/v11 S=1 greedy hashes recorded by the research gate (CG/mtp_runner/runs/s2dev_gate_c2048_n256.json).
Tokens are collected through the standard GenAI loop (append_tokens / generate_next_token / get_next_tokens) and also
cross-checked against generator.get_sequence().

usage: gate_genai.py <package> --out <json> [--no-mtp] [--ignore-eos] [--cases text,prose,random,chat]
       --ignore-eos reproduces the research "fixed length" mode (EOS is an ordinary token) by moving the stop token out of reach.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

RUNNER = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
sys.path.insert(0, str(RUNNER))
import prompts  # noqa: E402  (research prompt builders: text pools frozen under runs/)

REF_JSON = RUNNER / "runs/s2dev_gate_c2048_n256.json"
CHAT_QUESTIONS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the three primary colors in one short sentence.",
    "Write the word hello.",
]
NEVER_EOS = 248319  # a token id the model does not stop on; keeps the config valid (< vocab_size)


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def run_generation(model, prompt, new):
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + new, do_sample=False)
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    generator.append_tokens(prompt)
    tokens, stamps = [], []
    while not generator.is_done():
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
        stamps.append(time.perf_counter())
    sequence = generator.get_sequence(0)[len(prompt):]
    stats = generator.get_speculative_stats()
    del generator
    return {"tokens": tokens, "sequence": [int(x) for x in sequence], "prefill_s": stamps[0] - t0,
            "decode_s": stamps[-1] - stamps[0],
            "stats": {k: stats[k] for k in ("rounds", "draft_tokens_accepted", "tokens_emitted")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--cases", default="text,prose,random,chat")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    reference = {r["case"]: r for r in json.loads(REF_JSON.read_text())["results"]}
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    overlay = {}
    if args.no_mtp:
        overlay.setdefault("model", {})["qwen38_graph"] = {"mtp": {"enabled": False}}
    if args.ignore_eos:
        overlay.setdefault("model", {})["eos_token_id"] = [NEVER_EOS]
    if overlay:
        config.overlay(json.dumps(overlay))
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)

    cases = []
    for case in args.cases.split(","):
        if case == "chat":
            for q in CHAT_QUESTIONS:
                cases.append((f"chat:{q[:24]}", prompts.chat_prompt(q), q))
        else:
            cases.append((f"{case}{args.context}", prompts.make(case, args.context, 1000 + args.context), None))

    results, all_ok = [], True
    for name, prompt, question in cases:
        ref_entry = reference[name]["plain"][1 if args.ignore_eos else 0]
        entry = {"case": name, "prompt_len": len(prompt), "ref_hash": ref_entry["hash"], "ref_n": ref_entry["n"]}
        if question is not None:
            # chat template: GenAI default (thinking) vs HF enable_thinking=True, and GenAI with the non-thinking rewrite of the
            # template (the binding cannot pass enable_thinking) vs the HF enable_thinking=False prompt of the research runner
            messages = json.dumps([{"role": "user", "content": question}])
            hf = prompts.tokenizer()
            hf_default = hf.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                                add_generation_prompt=True)
            entry["chat_default_text_equal_hf_default"] = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True) == hf_default
            template = (Path(args.model) / "chat_template.jinja").read_text()
            template = template.replace("enable_thinking is undefined or enable_thinking is true", "false").replace(
                "enable_thinking is defined and enable_thinking is false", "true")
            genai_text = tokenizer.apply_chat_template(messages, template_str=template, add_generation_prompt=True)
            genai_ids = [int(x) for x in tokenizer.encode(genai_text)]
            entry["chat_nonthinking_ids_equal_hf"] = genai_ids == prompt
            entry["chat_nonthinking_tail"] = tokenizer.decode(np.asarray(genai_ids[-8:], dtype=np.int32))
        run = run_generation(model, prompt, args.new)
        tokens = run["tokens"]
        entry.update({"n": len(tokens), "hash": token_hash(tokens), "tokens_head": tokens[:16],
                      "ends_with_eos": bool(tokens) and tokens[-1] in (248046, 248044),
                      "sequence_matches_next_tokens": run["sequence"] == tokens[: len(run["sequence"])]
                      and len(tokens) - len(run["sequence"]) in (0, 1),
                      "prefill_s": run["prefill_s"], "decode_s": run["decode_s"],
                      "tps": (len(tokens) - 1) / run["decode_s"] if run["decode_s"] > 0 else None, "stats": run["stats"]})
        entry["tokens_identical"] = entry["hash"] == ref_entry["hash"]
        all_ok &= entry["tokens_identical"] and entry["sequence_matches_next_tokens"]
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("case", "n", "ref_n", "tokens_identical", "ends_with_eos", "tps")}), flush=True)
    out = {"all_ok": all_ok, "mtp": not args.no_mtp, "ignore_eos": args.ignore_eos, "context": args.context, "new": args.new,
           "results": results}
    args.out.write_text(json.dumps(out, indent=1))
    print("GATE", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
