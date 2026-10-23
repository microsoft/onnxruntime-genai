# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Fail closed unless a built package generates through the DFlash pipeline."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


def verify(model_path, provider, report_path):
    og.register_execution_provider_library("CUDAExecutionProvider", str(provider))
    model = og.Model(og.Config(str(model_path)))
    tokenizer = og.Tokenizer(model)
    sanity_prompt = tokenizer.encode("The capital of France is")
    sanity_params = og.GeneratorParams(model)
    sanity_params.set_search_options(max_length=len(sanity_prompt) + 8, do_sample=False)
    sanity_generator = og.Generator(model, sanity_params)
    sanity_generator.append_tokens(sanity_prompt)
    sanity_tokens = []
    while not sanity_generator.is_done():
        sanity_generator.generate_next_token()
        sanity_tokens.append(int(sanity_generator.get_next_tokens()[0]))
    sanity_text = tokenizer.decode(sanity_tokens)
    if re.search(r"\bparis\b", sanity_text, re.IGNORECASE) is None:
        raise RuntimeError(f"Built target failed the basic factual completion sanity check: {sanity_text!r}")
    del sanity_generator
    messages = json.dumps([{"role": "user", "content": "Explain why the sky is blue in two sentences."}])
    prompt = tokenizer.encode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + 32, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(prompt)
    tokens = []
    while not generator.is_done():
        if len(tokens) % 8 == 0:
            logits = generator.get_logits()
            if logits.shape[-1] != 248320 or not np.isfinite(logits).all():
                raise RuntimeError("Built target produced invalid full logits")
        generator.generate_next_token()
        token = int(generator.get_next_tokens()[0])
        if not 0 <= token < 248320:
            raise RuntimeError(f"Built target produced an invalid token ID: {token}")
        tokens.append(token)
    stats = dict(generator.get_speculative_stats())
    if len(tokens) != 32 or stats["tokens_emitted"] != 32:
        raise RuntimeError(f"Build verification requires exactly 32 generated tokens, got {len(tokens)}")
    if stats["rounds"] <= 0 or stats["draft_forward_passes"] <= 0 or stats["target_verify_forward_passes"] <= 0:
        raise RuntimeError("Build verification did not exercise speculative target/draft execution")
    for name in (
        "mtp_failures",
        "dflash2_failures",
        "dflash2_disables",
        "dflash2_admission_misses",
        "standard_fallback_steps",
    ):
        if stats[name] != 0:
            raise RuntimeError(f"Build verification failed: {name}={stats[name]}")
    report = {
        "execution_check_passed": True,
        "completion_sanity_check_passed": True,
        "completion_sanity_text": sanity_text,
        "task_accuracy_qualified": False,
        "prompt_tokens": len(prompt),
        "output_tokens": tokens,
        "text": tokenizer.decode(tokens),
        "stats": stats,
    }
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.model, args.provider, args.report)


if __name__ == "__main__":
    main()
