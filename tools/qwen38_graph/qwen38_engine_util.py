# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers of the qwen38_flash_graph Engine qualification scripts (smoke, gate, behaviours, benchmark).

The scripts drive a package of model.type "qwen38_flash_graph" (see docs/qwen38_flash_graph.md) through the public Engine
API (og.Engine / og.Request). They need no reference data from any other project: prompts are cut from a natural-text file
given on the command line, and correctness is checked by comparing runs of the same package.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

PROVIDER_ENV = "QWEN38_CUDA_EP_LIBRARY"
DEFAULT_KV_CAPACITY = 33028
RESERVED_TAIL = 4  # positions at the end of the KV store that are never filled


def add_common_arguments(parser):
    parser.add_argument("model", help="package directory (genai_config.json with model.type qwen38_flash_graph)")
    parser.add_argument(
        "--provider",
        default=os.environ.get(PROVIDER_ENV),
        help=f"plugin CUDA execution provider library to register (default: ${PROVIDER_ENV}; none if unset)",
    )


def package_info(package):
    """The facts of the package config the scripts need: the stop tokens and the KV capacity."""
    config = json.loads((Path(package) / "genai_config.json").read_text())["model"]
    eos = config.get("eos_token_id", [])
    return {
        "eos_token_ids": [int(x) for x in (eos if isinstance(eos, list) else [eos])],
        "kv_capacity": int(config.get("qwen38_graph", {}).get("kv_capacity", DEFAULT_KV_CAPACITY)),
    }


def mtp_overlay(enabled, prefill_head=True):
    return {"model": {"qwen38_graph": {"mtp": {"enabled": enabled, "prefill_head": prefill_head}}}}


def load_model(package, provider=None, overlay=None):
    """Registers the plugin CUDA EP library `provider` (when given) and loads the package."""
    if provider:
        og.register_execution_provider_library("CUDAExecutionProvider", provider)
    config = og.Config(str(package))
    if overlay:
        config.overlay(json.dumps(overlay))
    return og.Model(config)


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def load_text_tokens(tokenizer, text_file):
    return [int(x) for x in tokenizer.encode(Path(text_file).read_text(encoding="utf-8", errors="replace"))]


def text_prompt(text_tokens, length, seed):
    """`length` consecutive tokens of the text, starting at an offset derived from `seed` (distinct prompts per seed)."""
    if len(text_tokens) < length:
        raise SystemExit(
            f"--text-file has {len(text_tokens)} tokens but a {length}-token prompt is needed; pass a longer file"
        )
    spare = len(text_tokens) - length
    offset = (seed * 7919) % max(spare, 1)
    return text_tokens[offset : offset + length]


def chat_prompt(tokenizer, question):
    messages = json.dumps([{"role": "user", "content": question}])
    return [int(x) for x in tokenizer.encode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))]


class TurnResult:
    def __init__(self):
        self.tokens = []
        self.times = []  # perf_counter() after the Engine.run call that delivered each token
        self.finish_reason = None
        self.usage = None
        self.matched_stop_string_index = None
        self.runs = 0
        self.events = 0
        self.start = None


def run_turn(engine, buffer, request, prompt, max_new, *, stop_strings=None, turn_setup=None, until=None):
    """Begins a greedy turn and runs the Engine until the Engine has no work (or `until(result)` returns True).

    Returns the TurnResult and the turn id. Raises on a failed or capacity-blocked event.
    """
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    if max_new is not None:
        turn.set_max_generated_tokens(max_new)
    if stop_strings:
        turn.set_stop_strings(stop_strings)
    if turn_setup:
        turn_setup(turn)
    result = TurnResult()
    result.start = time.perf_counter()
    turn_id = request.begin_turn(np.asarray(prompt, dtype=np.int32), turn)
    pump(engine, buffer, result, until)
    return result, turn_id


def pump(engine, buffer, result, until=None):
    while engine.has_pending_requests():
        events = engine.run(buffer)
        now = time.perf_counter()
        result.runs += 1
        for event in events:
            result.events += 1
            if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                raise RuntimeError(f"Engine failure: flags={event.flags} error={event.error_code}")
            if event.flags & og.EngineEventFlags.TOKEN:
                result.tokens.append(int(event.token))
                result.times.append(now)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                result.finish_reason = event.finish_reason
                result.usage = event.usage
                result.matched_stop_string_index = event.matched_stop_string_index
        if until is not None and until(result):
            break
    return result
