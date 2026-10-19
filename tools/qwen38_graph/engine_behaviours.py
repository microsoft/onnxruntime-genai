#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Engine behaviours of the qwen38_flash_graph backend on the real package (the GPU counterpart of the C++ tests in
test/cpp/engine/qwen38_graph_executor_tests.cpp): stop tokens, limits, stop strings, cancel (during prefill and decode),
queueing of a second request, turn continuation, rewind, speculative statistics and the capability errors.

--prefill steps|fast overlays prefill.enabled (the v12b fast prefill is tolerance level and not run-to-run reproducible, so the
"identical to the reference" checks are exact only with --prefill steps, the bit-exact muted-fold prompt path).

usage: engine_behaviours.py <package> --out <json> [--prefill steps|fast] [--provider PATH] --text-pool FILE
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from qwen38_engine_util import Prompts, TurnResult, add_prompt_arguments, load_model, pump, run_turn

CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  [{detail}]"), flush=True)


def expect_error(name, action, fragment):
    try:
        action()
    except Exception as error:
        check(name, fragment in str(error), str(error)[:200])
        return
    check(name, False, "no error raised")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill", choices=("steps", "fast"), default=None)
    parser.add_argument("--provider")
    add_prompt_arguments(parser)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    overlay = {"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}} if args.prefill else None
    model = load_model(args.model, args.provider, overlay)
    tokenizer = og.Tokenizer(model)
    prompts = Prompts.from_args(model, args)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)
    finish = og.FinishReason

    capabilities = engine.get_capabilities()
    check("capabilities: one row", capabilities.configured_max_batch_size == 1, capabilities.configured_max_batch_size)
    check(
        "capabilities: max_request_length is the KV capacity - 4",
        capabilities.max_request_length == 33024,
        capabilities.max_request_length,
    )
    check("no caller draft tokens", engine.max_draft_tokens_per_proposal() == 0)

    text = prompts.make("text", 512, 11)
    hello = prompts.chat_prompt("Write the word hello.")

    # reference run: a plain greedy turn of 24 tokens
    request = engine.create_request()
    reference, _ = run_turn(engine, buffer, request, text, 24)
    request.close()
    check(
        "max_generated_tokens ends the turn",
        reference.finish_reason == finish.MAX_GENERATED_TOKENS and len(reference.tokens) == 24,
        f"{reference.finish_reason} n={len(reference.tokens)}",
    )
    check(
        "usage reports prompt and generated tokens",
        reference.usage.prompt_tokens == len(text) and reference.usage.generated_tokens == 24,
    )

    # stop token: the model stops by itself after "hello" (EOS is not delivered)
    request = engine.create_request()
    eos_run, _ = run_turn(engine, buffer, request, hello, 256)
    request.close()
    check(
        "EOS ends the turn without being delivered",
        eos_run.finish_reason == finish.EOS and 0 < len(eos_run.tokens) < 256,
        f"{eos_run.finish_reason} n={len(eos_run.tokens)} -> {tokenizer.decode(np.asarray(eos_run.tokens, dtype=np.int32))!r}",
    )

    # session limit
    session = og.RequestOptions()
    session.set_max_session_tokens(len(text) + 9)
    request = engine.create_request(options=session)
    limited, _ = run_turn(engine, buffer, request, text, 100)
    request.close()
    check(
        "max_session_tokens ends the turn",
        limited.finish_reason == finish.MAX_SESSION_TOKENS and len(limited.tokens) == 9,
        f"{limited.finish_reason} n={len(limited.tokens)}",
    )
    check("session-limited tokens are a prefix of the reference", limited.tokens == reference.tokens[:9])

    # stop string: a 5-character window of the reference text (with a visible character) that ends after the 12th token
    pieces = [tokenizer.decode(np.asarray([t], dtype=np.int32)) for t in reference.tokens]
    full = "".join(pieces)
    offset = len("".join(pieces[:12]))
    stop = next(
        full[i : i + 5]
        for i in range(offset, len(full) - 5)
        if full[i : i + 5].strip() and full.find(full[i : i + 5]) == i
    )
    request = engine.create_request()
    stopped, _ = run_turn(engine, buffer, request, text, 100, stop_strings=[stop])
    request.close()
    produced = "".join(tokenizer.decode(np.asarray([t], dtype=np.int32)) for t in stopped.tokens)
    before_last = "".join(tokenizer.decode(np.asarray([t], dtype=np.int32)) for t in stopped.tokens[:-1])
    # the Engine delivers whole tokens: the turn ends with the first token whose text completes the match
    check(
        "stop string ends the turn at its first occurrence",
        stopped.finish_reason == finish.STOP_STRING
        and stop in produced
        and stop not in before_last
        and stopped.tokens == reference.tokens[: len(stopped.tokens)]
        and stopped.matched_stop_string_index == 0,
        f"{stopped.finish_reason} stop={stop!r} n={len(stopped.tokens)}",
    )
    # the look-ahead cycle that was in flight must not disturb the next request
    request = engine.create_request()
    again, _ = run_turn(engine, buffer, request, text, 24)
    request.close()
    check("a request after a stop-string turn is identical to the reference", again.tokens == reference.tokens)

    # cancel with a cycle in flight
    request = engine.create_request()
    partial = TurnResult()
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_max_generated_tokens(200)
    turn_id = request.begin_turn(np.asarray(text, dtype=np.int32), turn)
    pump(engine, buffer, partial, until=lambda r: len(r.tokens) >= 10)
    cancelled = request.cancel_turn(turn_id)
    pump(engine, buffer, partial)
    check(
        "cancel during decode",
        cancelled and partial.finish_reason == finish.CANCELLED and not engine.has_pending_requests(),
        f"{partial.finish_reason} n={len(partial.tokens)}",
    )
    request.close()
    request = engine.create_request()
    after_cancel, _ = run_turn(engine, buffer, request, text, 24)
    request.close()
    check("a request after a cancel is identical to the reference", after_cancel.tokens == reference.tokens)

    # cancel during prefill (the 20k prompt is ingested in several Engine steps: 8192-token dense chunks, 2048-token S=2 steps)
    long_prompt = prompts.make("text", 20000, 5)
    request = engine.create_request()
    partial = TurnResult()
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_max_generated_tokens(50)
    turn_id = request.begin_turn(np.asarray(long_prompt, dtype=np.int32), turn)
    events = engine.run(buffer)
    check("a prefill step delivers no event", len(events) == 0 and engine.has_pending_requests())
    cancelled = request.cancel_turn(turn_id)
    pump(engine, buffer, partial)
    check(
        "cancel during prefill",
        cancelled and partial.finish_reason == finish.CANCELLED and not partial.tokens,
        str(partial.finish_reason),
    )
    request.close()

    # a second request queues behind the first and both are correct
    first = engine.create_request()
    second = engine.create_request()
    t1, t2 = og.TurnOptions(first), og.TurnOptions(second)
    for options in (t1, t2):
        options.set_do_sample(False)
        options.set_max_generated_tokens(24)
    first.begin_turn(np.asarray(text, dtype=np.int32), t1)
    second.begin_turn(np.asarray(text, dtype=np.int32), t2)
    order, tokens = [], {0: [], 1: []}
    while engine.has_pending_requests():
        for event in engine.run(buffer):
            if event.flags & og.EngineEventFlags.TOKEN:
                which = 0 if event.request == first else 1
                order.append(which)
                tokens[which].append(int(event.token))
    check(
        "second request queues and runs after the first",
        tokens[0] == reference.tokens and tokens[1] == reference.tokens and order == sorted(order),
        f"n0={len(tokens[0])} n1={len(tokens[1])}",
    )
    expect_error(
        "the first request cannot continue after the second took the slot",
        lambda: first.begin_turn(np.asarray([1000, 1001], dtype=np.int32), t1),
        "",
    )
    first.close()
    second.close()

    # continuation and rewind: the history is re-ingested; the result must equal a fresh run over the same history
    request = engine.create_request()
    turn1, _ = run_turn(engine, buffer, request, text[:256], 12)
    follow = [int(x) for x in tokenizer.encode(" Continue.")]
    turn2, turn2_id = run_turn(engine, buffer, request, follow, 12)
    request.rewind_to_start_of_turn(turn2_id)
    check("rewind to the start of a turn releases the slot", not engine.has_pending_requests())
    replay, _ = run_turn(engine, buffer, request, follow, 12)
    check("replay after rewind is identical", replay.tokens == turn2.tokens)
    request.close()
    history = text[:256] + turn1.tokens + follow
    fresh = engine.create_request()
    fresh_run, _ = run_turn(engine, buffer, fresh, history, 12)
    fresh.close()
    check(
        "a continued turn equals a fresh run over the same history",
        turn2.tokens == fresh_run.tokens,
        f"{turn2.tokens[:6]} vs {fresh_run.tokens[:6]}",
    )

    # unsupported options are rejected with a clear message before the request changes
    for label, setup, fragment in (
        ("sampling", lambda o: (o.set_do_sample(True), o.set_top_k(5), o.set_temperature(0.7)), "greedily"),
        ("repetition_penalty", lambda o: o.set_repetition_penalty(1.2), "repetition_penalty"),
        ("min_generated_tokens", lambda o: o.set_min_generated_tokens(3), "min_generated_tokens"),
        ("seed", lambda o: o.set_seed(3), "seed"),
    ):
        request = engine.create_request()
        options = og.TurnOptions(request)
        options.set_do_sample(False)
        setup(options)
        expect_error(
            f"rejected: {label}",
            lambda request=request, options=options: request.begin_turn(np.asarray(text[:16], dtype=np.int32), options),
            fragment,
        )
        request.close()
    too_long = og.RequestOptions()
    too_long.set_max_session_tokens(40000)
    expect_error(
        "rejected: session longer than the KV capacity",
        lambda: engine.create_request(options=too_long),
        "max_request_length",
    )

    stats = engine.get_speculative_stats()
    check(
        "speculative statistics are populated",
        stats["rounds"] > 0
        and stats["draft_tokens_proposed"] >= stats["rounds"]
        and 0 < stats["draft_tokens_accepted"] < stats["draft_tokens_proposed"]
        and stats["tokens_emitted"] > stats["rounds"],
        {k: stats[k] for k in ("rounds", "draft_tokens_accepted", "tokens_emitted", "acceptance_rate")},
    )

    # the model's CUDA-graph state belongs to one Engine at a time
    expect_error("a second Engine over the same model is rejected", lambda: og.Engine(model), "another Engine")
    del buffer, engine
    gc.collect()
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)
    request = engine.create_request()
    reborn, _ = run_turn(engine, buffer, request, text, 24)
    request.close()
    check(
        "a new Engine works after the first was destroyed and is identical to the reference",
        reborn.tokens == reference.tokens,
    )

    ok = all(c["ok"] for c in CHECKS)
    args.out.write_text(json.dumps({"all_ok": ok, "checks": CHECKS}, indent=1))
    print("BEHAVIOURS", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
