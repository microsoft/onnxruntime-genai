# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for the paged Engine execution path.

The synthetic graph reads and writes KV slots through the Engine-provided block
table. It detects row or length mixing, overlapping allocations, and block
drift across decode steps. CUDA runs this synthetic graph through the CUDA EP;
the real PagedAttention operator is covered by the real-model integration test.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

# Tiny checked-in fixture; no external model download or --test_models needed.
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged"

# These mirror create_synthetic_paged_model.py. block_size 4 makes short prompts
# span several blocks, which is what stresses block-table addressing.
_VOCAB_SIZE = 64
_BLOCK_SIZE = 4
_EOS_TOKEN_ID = 1

# Upper bound on engine.step() calls per generation so a scheduling bug fails
# fast instead of hanging.
_MAX_STEPS = 10_000

# CPU covers the paged transaction path on every build; CUDA adds the executable
# paged-attention graph when a GPU build is present.
_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])

# Distinct prompts (tokens < vocab). They differ in first token and length, so
# their generations diverge and mixing them in a batch is detectable. Lengths 3
# and 4 sit inside one block; length 6 spans two 4-slot blocks.
_PROMPT_A = [5, 9, 13]
_PROMPT_B = [7, 2, 20, 4]
_PROMPT_LONG = [3, 8, 2, 15, 6, 11]


def predicted_tokens(prompt, max_new_tokens):
    """Greedy tokens the synthetic graph makes the engine emit for one request.

    Chaining the graph's per-row rule over a request's own tokens gives

        g_0   = (prompt[0] + prompt[-1] + len(prompt)) % vocab
        g_k   = (prompt[0] + g_{k-1}   + len(prompt) + k) % vocab   for k >= 1

    The engine stops when it samples the EOS token and does not surface it, which
    this mirrors by breaking before appending an EOS token.
    """
    first, prev, length = prompt[0], prompt[-1], len(prompt)
    tokens = []
    for step in range(max_new_tokens):
        nxt = (first + prev + length + step) % _VOCAB_SIZE
        if nxt == _EOS_TOKEN_ID:
            break
        tokens.append(nxt)
        prev = nxt
    return tokens


class _Sink:
    """Per-request token collector correlated back through opaque data."""

    __slots__ = ("tokens",)

    def __init__(self):
        self.tokens = []


@pytest.fixture(params=_DEVICES)
def device(request):
    return request.param


@pytest.fixture
def model(device):
    config = og.Config(str(_MODEL_DIR))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    return og.Model(config)


def _add_request(engine, model, prompt, max_new_tokens, sink):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(prompt) + max_new_tokens)
    request = og.Request(params)
    request.add_tokens(np.asarray(prompt, dtype=np.int32))
    request.set_opaque_data(sink)
    engine.add_request(request)
    return request


def _drain(ready):
    sink = ready.get_opaque_data()
    while ready.has_unseen_tokens():
        sink.tokens.append(ready.get_unseen_token())
    return ready.is_done()


def _step_once(engine):
    """Advance one transaction; drain and release a request that completed.

    Returns False when the engine had no work to do, else True.
    """
    ready = engine.step()
    if ready is None:
        return False
    if _drain(ready):
        engine.remove_request(ready)
    return True


def _run(engine, *, max_steps=_MAX_STEPS):
    steps = 0
    while engine.has_pending_requests():
        assert _step_once(engine), "engine.step() returned no request while work remained"
        steps += 1
        assert steps <= max_steps, "engine.step() exceeded the safety bound; possible non-termination"


def _generate_isolated(model, prompt, max_new_tokens):
    """Greedy output of one prompt on its own dedicated engine."""
    sink = _Sink()
    engine = og.Engine(model)
    _add_request(engine, model, prompt, max_new_tokens, sink)
    _run(engine)
    del engine
    gc.collect()
    return sink.tokens


def test_model_declares_paged_config():
    # The fixture must route through the paged cache, or none of the scenarios
    # below exercise it. This does not need a device.
    config = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    dynamic_batching = config.get("engine", {}).get("dynamic_batching")
    assert dynamic_batching, f"synthetic fixture must declare engine.dynamic_batching; got {config.get('engine')!r}"
    assert dynamic_batching["block_size"] == _BLOCK_SIZE
    assert config["model"]["vocab_size"] == _VOCAB_SIZE
    assert config["model"]["eos_token_id"] == _EOS_TOKEN_ID
    assert config["search"]["do_sample"] is False


def test_deterministic_tokens(model):
    # Exact greedy output, cross-checked against a concrete expected sequence so
    # the check cannot silently drift in lockstep with the helper.
    max_new = 12
    tokens = _generate_isolated(model, _PROMPT_A, max_new)

    assert tokens == [21, 30, 40, 51, 63, 12, 26, 41, 57, 10, 28, 47]
    assert tokens == predicted_tokens(_PROMPT_A, max_new)
    assert len(tokens) == max_new


def test_isolated_matches_simultaneous(model):
    # Two differing prompts sharing a batch must each keep the output they
    # produce alone: batching must not mix rows, lengths, or block tables.
    max_new = 16
    expected_a = predicted_tokens(_PROMPT_A, max_new)
    expected_b = predicted_tokens(_PROMPT_B, max_new)
    assert expected_a != expected_b, "prompts must diverge for the isolation check to mean anything"

    isolated_a = _generate_isolated(model, _PROMPT_A, max_new)
    isolated_b = _generate_isolated(model, _PROMPT_B, max_new)
    assert isolated_a == expected_a
    assert isolated_b == expected_b

    engine = og.Engine(model)
    sink_a, sink_b = _Sink(), _Sink()
    _add_request(engine, model, _PROMPT_A, max_new, sink_a)
    _add_request(engine, model, _PROMPT_B, max_new, sink_b)
    assert engine.has_pending_requests()
    _run(engine)

    assert sink_a.tokens == isolated_a, "request A diverged when batched with B"
    assert sink_b.tokens == isolated_b, "request B diverged when batched with A"


def test_staggered_admission(model):
    # Admitting a request while another is already in flight must not perturb
    # either: both still match their isolated runs.
    max_new = 16
    expected_a = predicted_tokens(_PROMPT_A, max_new)
    expected_b = predicted_tokens(_PROMPT_B, max_new)

    engine = og.Engine(model)
    sink_a = _Sink()
    _add_request(engine, model, _PROMPT_A, max_new, sink_a)

    # Advance A a few steps before B is admitted, so admission happens mid-flight.
    for _ in range(3):
        if not engine.has_pending_requests():
            break
        _step_once(engine)
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    _add_request(engine, model, _PROMPT_B, max_new, sink_b)
    _run(engine)

    assert sink_a.tokens == expected_a
    assert sink_b.tokens == expected_b


def test_max_length_stops(model):
    # An EOS-free horizon stops purely because the sequence reached max_length,
    # so the request emits exactly max_new tokens.
    max_new = 16
    expected = predicted_tokens(_PROMPT_LONG, max_new)
    assert len(expected) == max_new, "chosen prompt must not hit EOS within the horizon"

    tokens = _generate_isolated(model, _PROMPT_LONG, max_new)

    assert len(tokens) == max_new
    assert tokens == expected


def test_eos_terminates_before_max_length(model):
    # _PROMPT_A samples the EOS token at generated index 24, well before this
    # horizon, so the engine stops early and never surfaces the EOS token.
    max_new = 60
    expected = predicted_tokens(_PROMPT_A, max_new)
    assert len(expected) < max_new, "chosen prompt must reach EOS before the horizon"

    tokens = _generate_isolated(model, _PROMPT_A, max_new)

    assert tokens == expected
    assert len(tokens) < max_new, "generation did not stop early on EOS"
    # The token that would come next is exactly the EOS the engine stopped on.
    next_token = (_PROMPT_A[0] + tokens[-1] + len(_PROMPT_A) + len(tokens)) % _VOCAB_SIZE
    assert next_token == _EOS_TOKEN_ID


def test_completion_isolation(model):
    # A short request that finishes and is released mid-run must not truncate a
    # longer sibling: the survivor still matches its isolated run.
    short_new, long_new = 5, 20
    long_isolated = _generate_isolated(model, _PROMPT_LONG, long_new)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    _add_request(engine, model, _PROMPT_A, short_new, short_sink)
    _add_request(engine, model, _PROMPT_LONG, long_new, long_sink)
    _run(engine)

    assert short_sink.tokens == predicted_tokens(_PROMPT_A, short_new)
    assert long_sink.tokens == long_isolated, "survivor diverged after its sibling completed"


def test_remove_request_freezes_output(model):
    # Removing a request in flight stops its output there; the sibling still
    # completes and matches its isolated run.
    max_new = 40
    sibling_new = 16
    sibling_expected = predicted_tokens(_PROMPT_B, sibling_new)

    engine = og.Engine(model)
    sink_a, sink_b = _Sink(), _Sink()
    request_a = _add_request(engine, model, _PROMPT_A, max_new, sink_a)
    _add_request(engine, model, _PROMPT_B, sibling_new, sink_b)

    for _ in range(4):
        if not engine.has_pending_requests():
            break
        _step_once(engine)
    assert len(sink_a.tokens) > 0, "request A produced nothing before removal"

    engine.remove_request(request_a)
    frozen_a = list(sink_a.tokens)

    _run(engine)

    assert sink_a.tokens == frozen_a, "removed request kept producing tokens"
    assert sink_b.tokens == sibling_expected, "sibling did not complete after removal"


def test_engine_teardown_and_recreation(model):
    # A fresh engine on the same model must reload the paged cache and serve the
    # same deterministic output.
    max_new = 12
    expected = predicted_tokens(_PROMPT_A, max_new)

    first = og.Engine(model)
    sink1 = _Sink()
    _add_request(first, model, _PROMPT_A, max_new, sink1)
    _run(first)
    assert sink1.tokens == expected
    del first
    gc.collect()

    second = og.Engine(model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    _add_request(second, model, _PROMPT_A, max_new, sink2)
    _run(second)
    assert sink2.tokens == expected
