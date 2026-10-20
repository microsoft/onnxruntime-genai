#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""API behaviour tests of the qwen38_flash_graph model through the public GenAI Python API (one model load).

usage: test_api.py <package> --out <json> [--prefill fast|steps] [--provider PATH]

--prefill steps (prompt through the muted fold graph, bit-exact path) requires identical token streams between repeated runs; with the
fast dense prefill (default) the dense grouped GEMM is not bit-reproducible between runs, so the repeat checks require an equal first
token (the common prefix length is reported).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


def prefix(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def generate(model, prompt, new, chunks=1, generator=None, observer=None):
    if generator is None:
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(prompt) + new, do_sample=False)
        generator = og.Generator(model, params)
    if chunks == 1:
        generator.append_tokens(prompt)
    else:
        step = (len(prompt) + chunks - 1) // chunks
        for i in range(0, len(prompt), step):
            generator.append_tokens(prompt[i : i + step])
    tokens = []
    if observer:
        observer(generator, tokens)
    while not generator.is_done():
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
        if observer:
            observer(generator, tokens)
    return generator, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    parser.add_argument("--prefill", default="fast", choices=("fast", "steps"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    if args.provider:
        og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}}))
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    results = {}
    exact = args.prefill == "steps"

    def same(a, b):
        return a == b if exact else a[0] == b[0]

    def check(name, condition, detail=""):
        results[name] = {"pass": bool(condition), "detail": str(detail)}
        print(("PASS " if condition else "FAIL ") + name, detail, flush=True)

    def raises(fn, needle):
        try:
            fn()
        except Exception as error:
            return needle in str(error), str(error)[:200]
        return False, "no exception"

    messages = json.dumps([{"role": "user", "content": "Name the three primary colors in one short sentence."}])
    prompt = [int(x) for x in tokenizer.encode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))]
    new = 40

    def logits_observer(label):
        previous_stats = None
        expected_next = None
        accepted_pending = False
        observed_buffered = False

        def observe(generator, tokens):
            nonlocal previous_stats, expected_next, accepted_pending, observed_buffered
            count = len(tokens)
            stats_before = dict(generator.get_speculative_stats())
            sequence_before = np.asarray(generator.get_sequence(0)).copy()
            check(f"{label}_emitted_{count}", stats_before["tokens_emitted"] == count, stats_before)
            check(
                f"{label}_queued_accounting_{count}",
                stats_before["tokens_queued"]
                == stats_before["tokens_emitted"] + stats_before["tokens_buffered"] + stats_before["tokens_discarded"],
                stats_before,
            )
            if expected_next is not None:
                check(
                    f"{label}_logits_predict_next_emission_{count}",
                    tokens[-1] == expected_next,
                    f"argmax {expected_next}, emitted {tokens[-1]}",
                )
            if previous_stats is not None:
                if accepted_pending and stats_before["rounds"] == previous_stats["rounds"]:
                    observed_buffered = True
                accepted_pending = stats_before["draft_tokens_accepted"] > previous_stats["draft_tokens_accepted"]
            previous_stats = stats_before
            if generator.is_done():
                for repeat in range(2):
                    ok, detail = raises(generator.get_logits, "")
                    check(f"{label}_completed_logits_rejected_{repeat}", ok, detail)
                check(f"{label}_accepted_buffered_cycle_observed", observed_buffered, stats_before)
            else:
                logits = np.asarray(generator.get_logits()).copy()
                repeated = np.asarray(generator.get_logits()).copy()
                expected_next = int(np.argmax(logits))
                check(
                    f"{label}_logits_repeated_{count}",
                    logits.shape[-1] == 248320 and np.isfinite(logits).all() and np.array_equal(logits, repeated),
                    f"shape {logits.shape}",
                )
            check(f"{label}_get_logits_preserves_stats_{count}", generator.get_speculative_stats() == stats_before)
            check(
                f"{label}_get_logits_preserves_sequence_{count}",
                np.array_equal(np.asarray(generator.get_sequence(0)), sequence_before),
            )

        return observe

    generator, reference = generate(model, prompt, new, observer=logits_observer("initial"))
    check("generate_to_max_length", len(reference) == new and generator.is_done(), f"{len(reference)} tokens")
    sequence = [int(x) for x in generator.get_sequence(0)]
    check("sequence_is_prompt_plus_tokens", sequence == prompt + reference, f"len {len(sequence)}")
    check("token_count", generator.token_count() == len(prompt) + new, generator.token_count())
    stats = generator.get_speculative_stats()
    check("stats_tokens_emitted", stats["tokens_emitted"] == new, stats["tokens_emitted"])

    # second concurrent generator on the same model
    ok, detail = raises(lambda: og.Generator(model, og.GeneratorParams(model)), "already leased")
    check("second_concurrent_generator_rejected", ok, detail)

    ok, detail = raises(lambda generator=generator: generator.rewind_to(7), "RewindToLength(0)")
    check("rewind_to_nonzero_rejected", ok, detail)

    # rewind(0) reuse: same tokens again
    generator.rewind_to(0)
    reset_stats = generator.get_speculative_stats()
    check(
        "rewind0_clears_emission_and_acceptance_stats",
        all(
            reset_stats[key] == 0
            for key in (
                "tokens_emitted",
                "tokens_queued",
                "tokens_buffered",
                "tokens_discarded",
                "rounds",
                "draft_tokens_accepted",
            )
        ),
        reset_stats,
    )
    generator, again = generate(model, prompt, new, generator=generator, observer=logits_observer("rewound"))
    check("rewind0_reuse_identical", same(again, reference), f"common prefix {prefix(again, reference)}")
    del generator

    # append after generation started (room below max_length, so the model's own check is the one that fires)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + 100, do_sample=False)
    roomy = og.Generator(model, params)
    roomy.append_tokens(prompt)
    for _ in range(3):
        roomy.generate_next_token()
    ok, detail = raises(lambda roomy=roomy: roomy.append_tokens(prompt[:3]), "continuous decoding")
    check("append_after_generation_rejected", ok, detail)
    del roomy

    # fresh generator after deletion; chunked prefill equals single append
    g2, fresh = generate(model, prompt, new)
    check("new_generator_after_delete_identical", same(fresh, reference), f"common prefix {prefix(fresh, reference)}")
    del g2
    g3, chunked = generate(model, prompt, new, chunks=3)
    check("chunked_append_identical", same(chunked, reference), f"common prefix {prefix(chunked, reference)}")
    del g3

    # EOS stop: chat prompt that finishes early (stop token ends the loop, sequence excludes it)
    short = [
        int(x)
        for x in tokenizer.encode(
            tokenizer.apply_chat_template(
                json.dumps([{"role": "user", "content": "Write the word hello."}]), add_generation_prompt=True
            )
        )
    ]
    g4, early = generate(model, short, 200)
    check(
        "eos_stops_generation",
        len(early) < 200 and early[-1] in (248046, 248044),
        f"{len(early)} tokens, last {early[-1]}",
    )
    check("eos_not_in_sequence", [int(x) for x in g4.get_sequence(0)][len(short) :] == early[:-1])
    del g4

    # unsupported options are rejected at creation
    def sampled():
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=True, top_k=20, temperature=1.0, max_length=100)
        og.Generator(model, p)

    ok, detail = raises(sampled, "greedily")
    check("sampling_rejected", ok, detail)

    def penalty():
        p = og.GeneratorParams(model)
        p.set_search_options(repetition_penalty=1.2, max_length=100)
        og.Generator(model, p)

    ok, detail = raises(penalty, "repetition_penalty")
    check("repetition_penalty_rejected", ok, detail)

    def too_long():
        p = og.GeneratorParams(model)
        p.set_search_options(max_length=40000, do_sample=False)
        og.Generator(model, p)

    ok, detail = raises(too_long, "context_length")
    check("max_length_above_capacity_rejected", ok, detail)

    # generate before append
    def no_prompt():
        p = og.GeneratorParams(model)
        p.set_search_options(max_length=100, do_sample=False)
        g = og.Generator(model, p)
        g.generate_next_token()

    ok, detail = raises(no_prompt, "AppendTokens")
    check("generate_without_prompt_rejected", ok, detail)

    # streaming decode + chat text
    g5, toks = generate(model, short, 60)
    text = tokenizer.decode(np.asarray(toks, dtype=np.int32))
    check("decode_text_nonempty", len(text) > 0, repr(text[:80]))
    del g5

    # logits are available right after the prompt (standard GenAI contract)
    p = og.GeneratorParams(model)
    p.set_search_options(max_length=len(prompt) + 8, do_sample=False)
    g6 = og.Generator(model, p)
    g6.append_tokens(prompt)
    logits = np.asarray(g6.get_logits())
    check(
        "logits_after_prompt_argmax_is_first_token",
        logits.shape[-1] == 248320 and int(np.argmax(logits)) == reference[0],
        f"argmax {int(np.argmax(logits))} vs first token {reference[0]}",
    )
    del g6

    out = {"all_pass": all(v["pass"] for v in results.values()), "results": results}
    args.out.write_text(json.dumps(out, indent=1))
    print("ALL PASS" if out["all_pass"] else "SOME FAILED")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
