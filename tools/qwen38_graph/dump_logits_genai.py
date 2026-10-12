#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""fp32 logits row of the last prompt token (generator.get_logits() right after append_tokens) of the GenAI qwen38_flash_graph model, for
the prompt specs of the python reference (v12/ref/ref_v12.py: kind:n:seed or chat:i), --repeat times each, saved as npz
(key "<spec with : -> _>__<rep>"), plus the first generated tokens. The python side writes the same keys with ref_v12.py --save-logits.

usage: dump_logits_genai.py <package> --out <npz> --cases text:2048:3048,chat:0 [--repeat 3] [--prefill fast|steps]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

RUNNER = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
sys.path.insert(0, str(RUNNER))
import prompts  # noqa: E402

CHAT = ["What is 2 + 3? Answer with only the number.", "Name the three primary colors in one short sentence.", "Write the word hello."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--prefill", default="fast", choices=("fast", "steps"))
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}}))
    model = og.Model(config)
    out = {}
    for spec in args.cases.split(","):
        kind, *rest = spec.split(":")
        prompt = prompts.chat_prompt(CHAT[int(rest[0])]) if kind == "chat" else [int(x) for x in prompts.make(kind, int(rest[0]), int(rest[1]))]
        for rep in range(args.repeat):
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=len(prompt) + 4, do_sample=False)
            generator = og.Generator(model, params)
            generator.append_tokens(prompt)
            logits = np.asarray(generator.get_logits(), dtype=np.float32).reshape(-1)
            generator.generate_next_token()
            first = int(generator.get_next_tokens()[0])
            out[f"{spec.replace(':', '_')}__{rep}"] = logits
            print(spec, rep, "argmax", int(np.argmax(logits)), "first", first, flush=True)
            del generator
    np.savez(args.out, **out)


if __name__ == "__main__":
    main()
