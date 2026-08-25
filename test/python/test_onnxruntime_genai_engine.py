# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for the paged Engine path."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged"

_VOCAB_SIZE = 64
_BLOCK_SIZE = 4
_EOS_TOKEN_ID = 1

_MAX_STEPS = 10_000

_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])

_PROMPT_A = [5, 9, 13]
_PROMPT_B = [7, 2, 20, 4]
_PROMPT_LONG = [3, 8, 2, 15, 6, 11]


def predicted_tokens(prompt, max_new_tokens):
    """Return the synthetic graph's greedy output, excluding EOS."""
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
    return ready.is_turn_complete()


def _step_once(engine):
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
    sink = _Sink()
    engine = og.Engine(model)
    request = _add_request(engine, model, prompt, max_new_tokens, sink)
    _run(engine)
    assert not request.is_turn_complete()
    del engine
    gc.collect()
    return sink.tokens


def test_model_declares_paged_config():
    config = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    dynamic_batching = config.get("engine", {}).get("dynamic_batching")
    assert dynamic_batching, f"synthetic fixture must declare engine.dynamic_batching; got {config.get('engine')!r}"
    assert dynamic_batching["block_size"] == _BLOCK_SIZE
    assert config["model"]["vocab_size"] == _VOCAB_SIZE
    assert config["model"]["eos_token_id"] == _EOS_TOKEN_ID
    assert config["model"]["decoder"]["inputs"]["attention_metadata"] == "attention_metadata"
    assert config["search"]["do_sample"] is False


def test_model_declares_three_value_attention_metadata():
    graph = onnx.load(_MODEL_DIR / "decoder.onnx", load_external_data=False).graph
    metadata = next(input_value for input_value in graph.input if input_value.name == "attention_metadata")
    tensor_type = metadata.type.tensor_type

    assert tensor_type.elem_type == onnx.TensorProto.INT32
    assert len(tensor_type.shape.dim) == 1
    assert tensor_type.shape.dim[0].dim_value == 3


def test_model_declares_per_request_fp16_logits():
    graph = onnx.load(_MODEL_DIR / "decoder.onnx", load_external_data=False).graph
    logits = next(output for output in graph.output if output.name == "logits")
    tensor_type = logits.type.tensor_type

    assert tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert len(tensor_type.shape.dim) == 2
    assert tensor_type.shape.dim[0].dim_param == "batch_size"
    assert tensor_type.shape.dim[1].dim_value == _VOCAB_SIZE


def test_deterministic_tokens(model):
    max_new = 12
    tokens = _generate_isolated(model, _PROMPT_A, max_new)

    assert tokens == [21, 30, 40, 51, 63, 12, 26, 41, 57, 10, 28, 47]
    assert tokens == predicted_tokens(_PROMPT_A, max_new)
    assert len(tokens) == max_new


def test_isolated_matches_simultaneous(model):
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
    requests = [
        _add_request(engine, model, _PROMPT_A, max_new, sink_a),
        _add_request(engine, model, _PROMPT_B, max_new, sink_b),
    ]
    assert engine.has_pending_requests()
    _run(engine)

    assert sink_a.tokens == isolated_a, "request A diverged when batched with B"
    assert sink_b.tokens == isolated_b, "request B diverged when batched with A"
    assert all(not request.is_turn_complete() for request in requests)


def test_staggered_admission(model):
    max_new = 16
    expected_a = predicted_tokens(_PROMPT_A, max_new)
    expected_b = predicted_tokens(_PROMPT_B, max_new)

    engine = og.Engine(model)
    sink_a = _Sink()
    request_a = _add_request(engine, model, _PROMPT_A, max_new, sink_a)

    for _ in range(3):
        if not engine.has_pending_requests():
            break
        _step_once(engine)
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    request_b = _add_request(engine, model, _PROMPT_B, max_new, sink_b)
    _run(engine)

    assert sink_a.tokens == expected_a
    assert sink_b.tokens == expected_b
    assert not request_a.is_turn_complete()
    assert not request_b.is_turn_complete()


def test_max_length_stops(model):
    max_new = 16
    expected = predicted_tokens(_PROMPT_LONG, max_new)
    assert len(expected) == max_new, "chosen prompt must not hit EOS within the horizon"

    tokens = _generate_isolated(model, _PROMPT_LONG, max_new)

    assert len(tokens) == max_new
    assert tokens == expected


def test_eos_terminates_before_max_length(model):
    max_new = 60
    expected = predicted_tokens(_PROMPT_A, max_new)
    assert len(expected) < max_new, "chosen prompt must reach EOS before the horizon"

    tokens = _generate_isolated(model, _PROMPT_A, max_new)

    assert tokens == expected
    assert len(tokens) < max_new, "generation did not stop early on EOS"
    next_token = (_PROMPT_A[0] + tokens[-1] + len(_PROMPT_A) + len(tokens)) % _VOCAB_SIZE
    assert next_token == _EOS_TOKEN_ID


def test_completion_isolation(model):
    short_new, long_new = 5, 20
    long_isolated = _generate_isolated(model, _PROMPT_LONG, long_new)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    short_request = _add_request(engine, model, _PROMPT_A, short_new, short_sink)
    long_request = _add_request(engine, model, _PROMPT_LONG, long_new, long_sink)
    _run(engine)

    assert short_sink.tokens == predicted_tokens(_PROMPT_A, short_new)
    assert long_sink.tokens == long_isolated, "survivor diverged after its sibling completed"
    assert not short_request.is_turn_complete()
    assert not long_request.is_turn_complete()


def test_continuation_while_peer_remains_active(model):
    short_max_new, long_max_new = 60, 80
    # EOS is valid input context here; Continue must reset the prior turn's done state rather than
    # treating an EOS token in the new prompt fragment as a newly generated stop.
    follow_up = [_EOS_TOKEN_ID, 12]

    reference_engine = og.Engine(model)
    reference_sink = _Sink()
    reference = _add_request(
        reference_engine, model, _PROMPT_A, short_max_new, reference_sink
    )
    while not reference.is_turn_complete():
        ready = reference_engine.step()
        assert ready is not None
        _drain(ready)
    reference.continue_with(np.asarray(follow_up, dtype=np.int32))
    _run(reference_engine)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    short = _add_request(engine, model, _PROMPT_A, short_max_new, short_sink)
    long = _add_request(engine, model, _PROMPT_LONG, long_max_new, long_sink)

    while not short.is_turn_complete():
        ready = engine.step()
        assert ready is not None
        if _drain(ready) and ready is not short:
            engine.remove_request(ready)

    assert not long.is_turn_complete(), "peer must remain active when continuation is appended"
    for _ in range(3):
        ready = engine.step()
        assert ready is not None
        _drain(ready)
    assert not long.is_turn_complete(), "peer must remain active during the continuation delay"

    short.continue_with(np.asarray(follow_up, dtype=np.int32))
    _run(engine)

    assert short_sink.tokens == reference_sink.tokens


def test_request_rejects_tokens_while_awaiting_admission(model):
    engine = og.Engine(model)
    sink = _Sink()
    request = _add_request(engine, model, _PROMPT_A, 8, sink)

    with pytest.raises(RuntimeError, match="initial input before submission"):
        request.add_tokens(np.asarray([12], dtype=np.int32))

    engine.remove_request(request)


def test_request_cannot_be_removed_from_another_engine(model):
    owner = og.Engine(model)
    other = og.Engine(model)
    sink = _Sink()
    request = _add_request(owner, model, _PROMPT_A, 8, sink)

    with pytest.raises(RuntimeError, match="does not belong"):
        other.remove_request(request)

    owner.remove_request(request)
    other.remove_request(request)
    assert not request.is_turn_complete()


def test_request_lifecycle_operations(model):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=64)
    request = og.Request(params)

    request.add_tokens(np.asarray(_PROMPT_A, dtype=np.int32))
    sink = _Sink()
    request.set_opaque_data(sink)
    engine = og.Engine(model)
    engine.add_request(request)

    ready = engine.step()
    assert ready is not None
    _drain(ready)
    assert not request.is_turn_complete()

    while not request.is_turn_complete():
        ready = engine.step()
        assert ready is not None
        _drain(ready)
    assert request.is_turn_complete()

    with pytest.raises(RuntimeError, match="use the continuation API"):
        request.add_tokens(np.asarray([12], dtype=np.int32))
    request.continue_with(np.asarray([12], dtype=np.int32))
    assert not request.is_turn_complete()

    engine.remove_request(request)
    assert not request.is_turn_complete()
    with pytest.raises(RuntimeError, match="closed request"):
        request.add_tokens(np.asarray([12], dtype=np.int32))
    with pytest.raises(RuntimeError, match="closed request"):
        request.continue_with(np.asarray([12], dtype=np.int32))
    engine.remove_request(request)


def test_last_handle_release_reclaims_retained_capacity(model):
    engine = og.Engine(model)
    sinks = [_Sink() for _ in range(8)]
    requests = [
        _add_request(engine, model, [5 + index, 9, 13], 1, sinks[index])
        for index in range(8)
    ]

    while not all(request.is_turn_complete() for request in requests):
        ready = engine.step()
        assert ready is not None
        _drain(ready)

    # Every TurnComplete request still owns one of the eight resident slots. Dropping all public
    # handles must mark them abandoned so the next admission can reclaim that capacity.
    requests.clear()
    del ready
    gc.collect()

    replacement_sink = _Sink()
    replacement = _add_request(engine, model, _PROMPT_A, 4, replacement_sink)
    _run(engine)

    assert replacement_sink.tokens == predicted_tokens(_PROMPT_A, 4)
    assert not replacement.is_turn_complete()


def test_remove_request_freezes_output(model):
    max_new = 40
    sibling_new = 16
    sibling_expected = predicted_tokens(_PROMPT_B, sibling_new)

    engine = og.Engine(model)
    sink_a, sink_b = _Sink(), _Sink()
    request_a = _add_request(engine, model, _PROMPT_A, max_new, sink_a)
    request_b = _add_request(engine, model, _PROMPT_B, sibling_new, sink_b)

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
    assert not request_b.is_turn_complete()


def test_engine_teardown_and_recreation(model):
    max_new = 12
    expected = predicted_tokens(_PROMPT_A, max_new)

    first = og.Engine(model)
    sink1 = _Sink()
    first_request = _add_request(first, model, _PROMPT_A, max_new, sink1)
    _run(first)
    assert sink1.tokens == expected
    assert not first_request.is_turn_complete()
    del first
    gc.collect()

    second = og.Engine(model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    second_request = _add_request(second, model, _PROMPT_A, max_new, sink2)
    _run(second)
    assert sink2.tokens == expected
    assert not second_request.is_turn_complete()
