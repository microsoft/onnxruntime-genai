# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Gemma 4 assistant-head self-speculative decoding example.

Gemma 4 pairs its decoder with a small *assistant* head. Unlike the Qwen3.6 MTP head,
which is handed the just-emitted token and looks the embedding up itself, the Gemma
assistant consumes the target's token embedding concatenated with the target's final
hidden state, and reads the target's present KV for a couple of layers instead of
keeping a cache of its own. ``og.MtpGenerator`` runs that draft/verify loop in-engine,
so the handoff stays on-device.

Requires a target and an assistant package built by the recipe in ``gemma-4-mtp.md``.
The pairing is driven entirely by the target's ``model.mtp`` config block; the runtime
picks this path when the draft model's ``model.type`` is ``gemma4_assistant``.

Pass ``--baseline`` to also decode the same prompt with a plain ``og.Generator`` and
report the end-to-end speedup.
"""

import argparse
import time

import numpy as np
import onnxruntime_genai as og


def run_speculative(target_model, assistant_model, prompt_tokens, args):
    """Decode with the assistant head and return (tokens, stats, seconds)."""
    params = og.GeneratorParams(target_model)
    params.set_search_options(max_length=args.max_length, do_sample=False)
    params.set_speculative_options(max_draft_tokens=args.max_draft_tokens)
    generator = og.MtpGenerator(target_model, assistant_model, params)

    n_prompt = len(prompt_tokens)
    generator.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))
    start = time.perf_counter()
    while not generator.is_done() and len(generator.get_sequence()) < n_prompt + args.max_new_tokens:
        generator.generate_next_token()
    elapsed = time.perf_counter() - start

    return generator.get_sequence().tolist()[n_prompt:], generator.get_stats(), elapsed


def run_baseline(target_model, prompt_tokens, args):
    """Decode the same prompt with plain greedy decoding, for a speedup reference."""
    params = og.GeneratorParams(target_model)
    params.set_search_options(max_length=args.max_length, do_sample=False)
    generator = og.Generator(target_model, params)

    n_prompt = len(prompt_tokens)
    generator.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))
    start = time.perf_counter()
    while not generator.is_done() and len(generator.get_sequence(0)) < n_prompt + args.max_new_tokens:
        generator.generate_next_token()
    elapsed = time.perf_counter() - start

    return generator.get_sequence(0).tolist()[n_prompt:], elapsed


def main(args):
    print("Loading target model...")
    target_model = og.Model(args.target_model_path)
    tokenizer = og.Tokenizer(target_model)
    print("Loading assistant head...")
    assistant_model = og.Model(args.assistant_model_path)

    prompts = args.prompts or ["Explain how photosynthesis works in plants, step by step."]
    for prompt in prompts:
        text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        prompt_tokens = tokenizer.encode(text)

        tokens, stats, elapsed = run_speculative(target_model, assistant_model, prompt_tokens, args)

        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print(tokenizer.decode(tokens))
        print("-" * 80)
        accepted, evaluated = stats["accepts"], stats["trials"]
        print(
            f"accept rate: {accepted / max(evaluated, 1):.1%} ({accepted}/{evaluated})  |  "
            f"tokens/forward: {len(tokens) / max(stats['forwards'], 1):.2f}  |  "
            f"{len(tokens)} tokens in {elapsed:.2f}s ({len(tokens) / elapsed:.1f} tok/s)"
        )

        if args.baseline:
            baseline_tokens, baseline_elapsed = run_baseline(target_model, prompt_tokens, args)
            print(
                f"baseline: {len(baseline_tokens)} tokens in {baseline_elapsed:.2f}s "
                f"({len(baseline_tokens) / baseline_elapsed:.1f} tok/s)  |  "
                f"speedup: {baseline_elapsed / elapsed:.2f}x"
            )
            # Greedy speculative decoding is lossless, so any divergence is a real bug
            # (or a floating-point near-tie in the batched verify forward).
            if tokens != baseline_tokens:
                print("WARNING: speculative output diverged from greedy baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 assistant-head speculative decoding")
    parser.add_argument(
        "-m",
        "--target_model_path",
        required=True,
        help="Path to the target model folder (its genai_config.json must declare a model.mtp block)",
    )
    parser.add_argument(
        "-d",
        "--assistant_model_path",
        required=True,
        help="Path to the assistant head folder (model.type must be gemma4_assistant)",
    )
    parser.add_argument("-n", "--max_new_tokens", type=int, default=128, help="Number of tokens to generate per prompt")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "-k",
        "--max_draft_tokens",
        type=int,
        default=4,
        help="Draft tokens proposed per round; must not exceed model.decoder.max_logits_sequence_length",
    )
    parser.add_argument("-p", "--prompts", nargs="*", default=None, help="Prompt(s) to run")
    parser.add_argument("--baseline", action="store_true", help="Also decode greedily and report the speedup")
    main(parser.parse_args())
