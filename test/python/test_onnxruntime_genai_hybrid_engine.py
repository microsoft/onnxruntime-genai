# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for packed paged-plus-fixed Engine state."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import register_plugin_providers

register_plugin_providers(logging.getLogger(__name__))

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-composite"
_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])


class _Sink:
    def __init__(self):
        self.tokens = []


@pytest.fixture(params=_DEVICES)
def model(request):
    config = og.Config(str(_MODEL_DIR))
    config.clear_providers()
    if request.param != "cpu":
        config.append_provider(request.param)
    return og.Model(config)


def _request(engine, model, prompt, max_new_tokens, sinks):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(prompt) + max_new_tokens)
    request = engine.create_request(params)
    sink = _Sink()
    sinks[request] = sink
    request.begin_turn(np.asarray(prompt, dtype=np.int32))
    return request, sink


def _run(engine, sinks):
    steps = 0
    event_buffer = engine.create_event_buffer(8)
    while engine.has_pending_requests():
        for event in engine.run(event_buffer):
            sink = sinks[event.request]
            if event.flags & og.EngineEventFlags.TOKEN:
                sink.tokens.append(event.token)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                event.request.close()
        steps += 1
        assert steps < 100


def test_fixture_declares_sparse_paged_and_fixed_groups():
    config = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    groups = config["model"]["decoder"]["state_groups"]
    assert config["model"]["decoder"]["inputs"]["position_ids"] == "position_ids"
    assert [group["kind"] for group in groups] == ["fixed_conv", "paged_kv", "fixed_recurrent"]
    assert [group["layer_ids"] for group in groups] == [[0, 3], [1, 4], [2, 5]]


def test_mixed_unequal_requests_match_isolated_execution(model):
    prompts = [[2, 3, 4], [7], [9, 10]]
    max_new_tokens = 3

    expected = []
    for prompt in prompts:
        engine = og.Engine(model)
        sinks = {}
        _, sink = _request(engine, model, prompt, max_new_tokens, sinks)
        _run(engine, sinks)
        expected.append(sink.tokens)
    # The first request's fixed convolution state contributes 0, 6, then 12 to successive scores.
    # Without fixed output binding/commit/re-gather this would be [9, 15, 22].
    assert expected[0] == [9, 21, 40]

    engine = og.Engine(model)
    sinks = {}
    requests = [_request(engine, model, prompt, max_new_tokens, sinks) for prompt in prompts]
    _run(engine, sinks)
    assert [sink.tokens for _, sink in requests] == expected
