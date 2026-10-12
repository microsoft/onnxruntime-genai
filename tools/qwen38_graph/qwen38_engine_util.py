# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Shared helpers of the qwen38_flash_graph Engine qualification scripts (smoke, gate, behaviours, benchmark).

The scripts drive a package built by build_package.py through the public Engine API (og.Engine / og.Request). The
reference prompts and hashes come from the research runner (CG/mtp_runner); point QWEN38_RUNNER_DIR at it.
"""

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

RUNNER_DIR = Path(
    os.environ.get("QWEN38_RUNNER_DIR", "/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
)
EOS_IDS = (248046, 248044)


def research_prompts():
    """The research prompt builders (text pools frozen under runs/)."""
    sys.path.insert(0, str(RUNNER_DIR))
    return importlib.import_module("prompts")


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def load_model(package, provider=None, overlay=None):
    """Registers the plugin CUDA EP (`provider` path, default $QWEN_PROVIDER) and loads the package."""
    provider = provider or os.environ.get("QWEN_PROVIDER")
    if provider:
        og.register_execution_provider_library("CUDAExecutionProvider", provider)
    config = og.Config(str(package))
    if overlay:
        config.overlay(json.dumps(overlay))
    return og.Model(config)


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
