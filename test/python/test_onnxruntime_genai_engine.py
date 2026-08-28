# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for the paged Engine path."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from _test_utils import register_plugin_providers

register_plugin_providers(logging.getLogger(__name__))

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
    __slots__ = ("tokens", "finish_reason", "usage")

    def __init__(self):
        self.tokens = []
        self.finish_reason = og.FinishReason.NONE
        self.usage = None


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


def _create_request(engine, model, prompt, max_new_tokens, sink, sinks):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(prompt) + max_new_tokens)
    request = engine.create_request(params)
    sinks[request] = sink
    request.begin_turn(np.asarray(prompt, dtype=np.int32))
    return request


def _drain(event, sinks):
    ready = event.request
    canonical = next((request for request in sinks if request is ready), None)
    assert canonical is not None, "EngineEvent.request must be the existing borrowed Request object"
    sink = sinks[ready]
    if event.flags & og.EngineEventFlags.TOKEN:
        sink.tokens.append(event.token)
    if event.flags & og.EngineEventFlags.TURN_FINISHED:
        sink.finish_reason = event.finish_reason
        sink.usage = event.usage
        return True
    return False


def _step_once(engine, sinks, *, close_completed=True):
    events = engine.run(8)
    for event in events:
        if _drain(event, sinks) and close_completed:
            event.request.close()
    return bool(events)


def _next_event(engine):
    while engine.has_pending_requests():
        events = engine.run()
        assert len(events) <= 1
        if events:
            return events[0]
    raise AssertionError("Engine completed without producing an event")


def _run(engine, sinks, *, max_steps=_MAX_STEPS, close_completed=True):
    steps = 0
    while engine.has_pending_requests():
        _step_once(engine, sinks, close_completed=close_completed)
        steps += 1
        assert steps <= max_steps, "engine.run() exceeded the safety bound; possible non-termination"


def _generate_isolated(model, prompt, max_new_tokens):
    sink = _Sink()
    engine = og.Engine(model)
    sinks = {}
    request = _create_request(engine, model, prompt, max_new_tokens, sink, sinks)
    _run(engine, sinks)
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


def test_run_returns_lists_for_default_zero_and_bulk_capacity(model):
    engine = og.Engine(model)
    sinks = {}
    first = _create_request(engine, model, _PROMPT_A, 1, _Sink(), sinks)
    second = _create_request(engine, model, _PROMPT_B, 1, _Sink(), sinks)

    assert engine.run(0) == []
    assert engine.has_pending_requests()
    with pytest.raises(ValueError, match="nonnegative"):
        engine.run(-1)
    with pytest.raises((OverflowError, TypeError)):
        engine.run(1 << 100)

    default_events = engine.run()
    assert isinstance(default_events, list)
    assert len(default_events) == 1
    assert default_events[0].request is first

    retained_events = engine.run(8)
    assert len(retained_events) == 1
    assert retained_events[0].request is second

    third = _create_request(engine, model, _PROMPT_A, 1, _Sink(), sinks)
    fourth = _create_request(engine, model, _PROMPT_B, 1, _Sink(), sinks)
    bulk_events = engine.run(8)
    assert len(bulk_events) == 2
    assert [event.request for event in bulk_events] == [third, fourth]

    for request in (first, second, third, fourth):
        request.close()


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
    sinks = {}
    requests = [
        _create_request(engine, model, _PROMPT_A, max_new, sink_a, sinks),
        _create_request(engine, model, _PROMPT_B, max_new, sink_b, sinks),
    ]
    assert engine.has_pending_requests()
    _run(engine, sinks)

    assert sink_a.tokens == isolated_a, "request A diverged when batched with B"
    assert sink_b.tokens == isolated_b, "request B diverged when batched with A"


def test_staggered_admission(model):
    max_new = 16
    expected_a = predicted_tokens(_PROMPT_A, max_new)
    expected_b = predicted_tokens(_PROMPT_B, max_new)

    engine = og.Engine(model)
    sink_a = _Sink()
    sinks = {}
    request_a = _create_request(engine, model, _PROMPT_A, max_new, sink_a, sinks)

    for _ in range(3):
        if not engine.has_pending_requests():
            break
        _step_once(engine, sinks)
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    request_b = _create_request(engine, model, _PROMPT_B, max_new, sink_b, sinks)
    _run(engine, sinks)

    assert sink_a.tokens == expected_a
    assert sink_b.tokens == expected_b


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


def test_per_turn_budget_is_independent_and_snapshotted(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=32)
    request = engine.create_request(params)
    sink = _Sink()
    sinks = {request: sink}
    prompt = np.asarray(_PROMPT_LONG, dtype=np.int32)

    turn_params = og.TurnParams(request)
    turn_params.set_max_generated_tokens(3)
    assert request.begin_turn(prompt, turn_params) == 0
    _run(engine, sinks, close_completed=False)

    assert sink.tokens == predicted_tokens(_PROMPT_LONG, 3)
    assert sink.finish_reason == og.FinishReason.MAX_GENERATED_TOKENS
    assert sink.usage.prompt_tokens == len(prompt)
    assert sink.usage.generated_tokens == 3

    continuation = np.asarray([12], dtype=np.int32)
    turn_params.set_max_generated_tokens(2)
    assert request.begin_turn(continuation, turn_params) == 1
    _run(engine, sinks, close_completed=False)

    assert len(sink.tokens) == 5
    request.close()


def test_request_total_limit_and_cancel_metadata(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=32)
    request = engine.create_request(params, max_session_tokens=len(_PROMPT_LONG) + 2)
    sink = _Sink()
    sinks = {request: sink}

    assert request.begin_turn(np.asarray(_PROMPT_LONG, dtype=np.int32)) == 0
    _run(engine, sinks, close_completed=False)
    assert len(sink.tokens) == 2
    assert sink.finish_reason == og.FinishReason.MAX_SESSION_TOKENS

    request.close()

    canceled = engine.create_request(params)
    turn_id = canceled.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32))
    assert canceled.cancel_turn(turn_id)
    assert not canceled.cancel_turn(turn_id)
    event = _next_event(engine)
    assert event.request is canceled
    assert event.finish_reason == og.FinishReason.CANCELLED
    canceled.close()


def test_zero_turn_budget_uses_default_limit(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=16)
    request = engine.create_request(params)
    prompt = np.asarray(_PROMPT_A, dtype=np.int32)

    turn_params = og.TurnParams(request)
    turn_params.set_max_generated_tokens(0)
    assert request.begin_turn(prompt, turn_params) == 0
    assert engine.has_pending_requests()
    assert _next_event(engine).request is request
    request.close()


def test_completion_isolation(model):
    short_new, long_new = 5, 20
    long_isolated = _generate_isolated(model, _PROMPT_LONG, long_new)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    sinks = {}
    short_request = _create_request(engine, model, _PROMPT_A, short_new, short_sink, sinks)
    long_request = _create_request(engine, model, _PROMPT_LONG, long_new, long_sink, sinks)
    _run(engine, sinks)

    assert short_sink.tokens == predicted_tokens(_PROMPT_A, short_new)
    assert long_sink.tokens == long_isolated, "survivor diverged after its sibling completed"


def test_continuation_while_peer_remains_active(model):
    short_max_new, long_max_new = 60, 80
    # EOS is valid input context here; begin_turn must reset the prior turn's done state rather than
    # treating an EOS token in the new prompt fragment as a newly generated stop.
    follow_up = [_EOS_TOKEN_ID, 12]

    reference_engine = og.Engine(model)
    reference_sink = _Sink()
    reference_sinks = {}
    reference = _create_request(reference_engine, model, _PROMPT_A, short_max_new, reference_sink, reference_sinks)
    reference_finished = False
    while not reference_finished:
        event = _next_event(reference_engine)
        assert event.request is reference
        reference_finished = _drain(event, reference_sinks)
    reference.begin_turn(np.asarray(follow_up, dtype=np.int32))
    _run(reference_engine, reference_sinks)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    sinks = {}
    short = _create_request(engine, model, _PROMPT_A, short_max_new, short_sink, sinks)
    long = _create_request(engine, model, _PROMPT_LONG, long_max_new, long_sink, sinks)

    short_finished = False
    while not short_finished:
        event = _next_event(engine)
        finished = _drain(event, sinks)
        short_finished = finished and event.request is short
        if finished and event.request is not short:
            event.request.close()

    for _ in range(3):
        event = _next_event(engine)
        assert event.flags != og.EngineEventFlags.NONE
        _drain(event, sinks)

    short.begin_turn(np.asarray(follow_up, dtype=np.int32))
    _run(engine, sinks)

    assert short_sink.tokens == reference_sink.tokens


def test_continuation_waits_for_ready_notification(model):
    engine = og.Engine(model)
    sinks = {}
    first = _create_request(engine, model, [5, 8, 57], 40, _Sink(), sinks)
    second = _create_request(engine, model, [6, 8, 56], 40, _Sink(), sinks)

    event = _next_event(engine)
    assert event.request is first
    assert _drain(event, sinks)

    continuation = np.asarray([12], dtype=np.int32)
    with pytest.raises(RuntimeError, match="event is pending"):
        second.begin_turn(continuation)

    event = _next_event(engine)
    assert event.request is second
    assert _drain(event, sinks)
    second.begin_turn(continuation)
    first.close()
    second.close()


def test_request_rejects_second_turn_while_active(model):
    engine = og.Engine(model)
    sinks = {}
    request = _create_request(engine, model, _PROMPT_A, 8, _Sink(), sinks)

    with pytest.raises(RuntimeError, match="new request or after the current turn is complete"):
        request.begin_turn(np.asarray([12], dtype=np.int32))

    request.close()


def test_request_lifecycle_operations(model):
    engine = og.Engine(model)
    sink = _Sink()
    sinks = {}
    request = _create_request(engine, model, _PROMPT_A, 61, sink, sinks)

    event = _next_event(engine)
    assert event.request is request
    finished = _drain(event, sinks)

    while not finished:
        event = _next_event(engine)
        finished = _drain(event, sinks)

    request.begin_turn(np.asarray([12], dtype=np.int32))

    request.close()
    with pytest.raises(RuntimeError, match="closed request"):
        request.begin_turn(np.asarray([12], dtype=np.int32))
    request.close()


def test_request_parameters_are_snapshotted(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(_PROMPT_LONG) + 4)
    request = engine.create_request(params)
    sink = _Sink()
    sinks = {request: sink}

    params.set_search_options(max_length=len(_PROMPT_LONG) + 12)
    request.begin_turn(np.asarray(_PROMPT_LONG, dtype=np.int32))
    _run(engine, sinks)

    assert sink.tokens == predicted_tokens(_PROMPT_LONG, 4)


def test_stop_sequences_accept_token_id_sequences_and_remain_unsupported(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=16)
    request = engine.create_request(params)
    turn_params = og.TurnParams(request)

    with pytest.raises(RuntimeError, match="not implemented"):
        turn_params.set_stop_sequences([[7, 8], [9]])

    request.close()


@pytest.mark.parametrize("state", ["created", "active", "turn-complete"])
def test_close_is_valid_and_idempotent_from_every_state(model, state):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(_PROMPT_LONG) + 8)
    request = engine.create_request(params)
    sinks = {request: _Sink()}

    if state != "created":
        request.begin_turn(np.asarray(_PROMPT_LONG, dtype=np.int32))
    if state == "active":
        event = _next_event(engine)
        assert event.request is request
        _drain(event, sinks)
    elif state == "turn-complete":
        finished = False
        while not finished:
            event = _next_event(engine)
            assert event.request is request
            finished = _drain(event, sinks)

    request.close()
    request.close()

    assert not engine.has_pending_requests()
    with pytest.raises(RuntimeError, match="closed request"):
        request.begin_turn(np.asarray([12], dtype=np.int32))


def test_events_deliver_tokens_across_turns(model):
    follow_up = np.asarray([_EOS_TOKEN_ID, 12], dtype=np.int32)

    def run_two_turns():
        engine = og.Engine(model)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=64)
        request = engine.create_request(params)
        request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32))
        tokens = []

        finished = False
        while not finished:
            event = _next_event(engine)
            assert event.request is request
            if event.token is not None:
                tokens.append(event.token)
            finished = bool(event.flags & og.EngineEventFlags.TURN_FINISHED)

        request.begin_turn(follow_up)
        finished = False
        while not finished:
            event = _next_event(engine)
            assert event.request is request
            if event.token is not None:
                tokens.append(event.token)
            finished = bool(event.flags & og.EngineEventFlags.TURN_FINISHED)

        request.close()
        return tokens

    assert run_two_turns()


def test_last_handle_release_reclaims_retained_capacity(model):
    engine = og.Engine(model)
    sinks = [_Sink() for _ in range(8)]
    sinks_by_request = {}
    requests = [
        _create_request(engine, model, [5 + index, 9, 13], 1, sinks[index], sinks_by_request) for index in range(8)
    ]

    completed = set()
    while len(completed) != len(requests):
        event = _next_event(engine)
        if _drain(event, sinks_by_request):
            completed.add(event.request)

    # Every TurnComplete request still owns one of the eight resident slots. Dropping all public
    # handles must mark them abandoned so the next admission can reclaim that capacity.
    sinks_by_request.clear()
    requests.clear()
    del event
    completed.clear()
    gc.collect()

    replacement_sink = _Sink()
    replacement_sinks = {}
    replacement = _create_request(engine, model, _PROMPT_A, 4, replacement_sink, replacement_sinks)
    _run(engine, replacement_sinks)

    assert replacement_sink.tokens == predicted_tokens(_PROMPT_A, 4)


def test_close_request_freezes_output(model):
    max_new = 40
    sibling_new = 16
    sibling_expected = predicted_tokens(_PROMPT_B, sibling_new)

    engine = og.Engine(model)
    sink_a, sink_b = _Sink(), _Sink()
    sinks = {}
    request_a = _create_request(engine, model, _PROMPT_A, max_new, sink_a, sinks)
    request_b = _create_request(engine, model, _PROMPT_B, sibling_new, sink_b, sinks)

    for _ in range(4):
        if not engine.has_pending_requests():
            break
        _step_once(engine, sinks)
    assert len(sink_a.tokens) > 0, "request A produced nothing before close"

    request_a.close()
    frozen_a = list(sink_a.tokens)

    _run(engine, sinks)

    assert sink_a.tokens == frozen_a, "closed request kept producing tokens"
    assert sink_b.tokens == sibling_expected, "sibling did not complete after close"


def test_engine_teardown_and_recreation(model):
    max_new = 12
    expected = predicted_tokens(_PROMPT_A, max_new)

    first = og.Engine(model)
    sink1 = _Sink()
    first_sinks = {}
    first_request = _create_request(first, model, _PROMPT_A, max_new, sink1, first_sinks)
    _run(first, first_sinks)
    assert sink1.tokens == expected
    del first
    gc.collect()

    second = og.Engine(model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    second_sinks = {}
    second_request = _create_request(second, model, _PROMPT_A, max_new, sink2, second_sinks)
    _run(second, second_sinks)
    assert sink2.tokens == expected
