# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Real-model correctness tests for the continuous-batching Engine."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

from . import models, resolver

pytestmark = pytest.mark.engine

_MODEL_ID = "qwen2.5-0.5b-instruct-paged"

_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "One plus one equals",
]

_MAX_STEPS = 20_000


class _Sink:
    __slots__ = ("tokens",)

    def __init__(self) -> None:
        self.tokens: list[int] = []


@dataclass
class _Bundle:
    model: og.Model
    tokenizer: og.Tokenizer
    config: dict
    device: str


@pytest.fixture
def paged_model_path(device, pytestconfig) -> Path:
    return resolver.get_path_for(
        _MODEL_ID,
        device,
        model_root=pytestconfig.getoption("--model-root"),
    )


@pytest.fixture
def bundle(device, paged_model_path) -> _Bundle:
    if device != "cuda":
        pytest.skip(f"Paged Engine test is CUDA only; device '{device}' is not supported.")
    if not og.is_cuda_available():
        pytest.skip("CUDA execution provider is not available in this build.")

    config = og.Config(str(paged_model_path))
    config.clear_providers()
    config.append_provider("cuda")

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    cfg = json.loads((paged_model_path / "genai_config.json").read_text())
    return _Bundle(model=model, tokenizer=tokenizer, config=cfg, device=device)


def _greedy_params(model: og.Model, prompt_len: int, max_new_tokens: int, min_new_tokens: int) -> og.GeneratorParams:
    params = og.GeneratorParams(model)
    options = {"do_sample": False, "max_length": prompt_len + max_new_tokens}
    if min_new_tokens:
        options["min_length"] = prompt_len + min_new_tokens
    params.set_search_options(**options)
    return params


def _create_request(engine, model, prompt_tokens, max_new_tokens, sink, sinks, *, min_new_tokens=0):
    params = _greedy_params(model, len(prompt_tokens), max_new_tokens, min_new_tokens)
    request = engine.create_request(params)
    sinks[request] = sink
    request.begin_turn(np.asarray(prompt_tokens, dtype=np.int32))
    return request


def _drain(ready, sinks) -> bool:
    canonical = next((request for request in sinks if request is ready), None)
    assert canonical is not None, "Engine.run() must return the existing borrowed Request object"
    sink = sinks[ready]
    while ready.has_unseen_tokens():
        sink.tokens.append(ready.get_unseen_token())
    return ready.is_turn_complete()


def _run(engine, sinks, *, max_steps=_MAX_STEPS) -> None:
    steps = 0
    while engine.has_pending_requests():
        ready = engine.run()
        assert ready is not None, "engine.run() returned no request while work remained"
        if _drain(ready, sinks):
            ready.close()
        steps += 1
        assert steps <= max_steps, "engine.run() exceeded the safety bound; possible non-termination"


def _generate_isolated(model, prompt_tokens, max_new_tokens, *, min_new_tokens=0) -> list[int]:
    sink = _Sink()
    engine = og.Engine(model)
    sinks = {}
    request = _create_request(engine, model, prompt_tokens, max_new_tokens, sink, sinks, min_new_tokens=min_new_tokens)
    _run(engine, sinks)
    assert not request.is_turn_complete()
    del engine
    gc.collect()
    return sink.tokens


def _eos_ids(bundle: _Bundle) -> set[int]:
    raw = bundle.config.get("model", {}).get("eos_token_id")
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {int(t) for t in raw}
    return {int(raw)}


def test_model_is_paged(bundle):
    engine_cfg = bundle.config.get("engine")
    assert engine_cfg and "dynamic_batching" in engine_cfg, (
        f"'{_MODEL_ID}' must declare engine.dynamic_batching to exercise the paged cache; got engine={engine_cfg!r}"
    )


def test_pinned_model_identity(paged_model_path):
    identity = models.pinned_identity(_MODEL_ID)
    if not identity:
        pytest.skip(f"No content identity recorded for '{_MODEL_ID}'.")
    for filename, expected in identity.items():
        path = paged_model_path / filename
        assert path.exists(), f"pinned artifact is missing '{filename}' at {paged_model_path}"
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        assert actual == expected, (
            f"'{_MODEL_ID}' {filename} sha256 {actual} != pinned {expected}. The artifact at the "
            f"pinned v{models.pinned_version(_MODEL_ID)} differs from the catalog; update "
            f"PINNED_IDENTITY in test/python/integration/models.py if this is an intentional re-export."
        )


def test_single_request_bounded_output(bundle):
    max_new = 32
    prompt_tokens = bundle.tokenizer.encode(_PROMPTS[0])

    tokens = _generate_isolated(bundle.model, prompt_tokens, max_new)

    assert 0 < len(tokens) <= max_new
    text = bundle.tokenizer.decode(np.asarray(tokens, dtype=np.int32))
    assert isinstance(text, str) and text.strip(), "decoded text was empty"


def test_simultaneous_requests(bundle):
    max_new = 24
    engine = og.Engine(bundle.model)
    sinks = [_Sink() for _ in _PROMPTS]
    sinks_by_request = {}
    requests = [
        _create_request(engine, bundle.model, bundle.tokenizer.encode(p), max_new, sink, sinks_by_request)
        for p, sink in zip(_PROMPTS, sinks, strict=True)
    ]
    assert engine.has_pending_requests()

    _run(engine, sinks_by_request)

    assert len(requests) == len(_PROMPTS)
    for prompt, sink in zip(_PROMPTS, sinks, strict=True):
        assert 0 < len(sink.tokens) <= max_new, f"no bounded output for {prompt!r}"


def test_staggered_admission(bundle):
    max_new = 32
    prompt_a = bundle.tokenizer.encode(_PROMPTS[0])
    prompt_b = bundle.tokenizer.encode(_PROMPTS[1])
    isolated_a = _generate_isolated(bundle.model, prompt_a, max_new)
    isolated_b = _generate_isolated(bundle.model, prompt_b, max_new)

    engine = og.Engine(bundle.model)

    sink_a = _Sink()
    sinks = {}
    request_a = _create_request(engine, bundle.model, prompt_a, max_new, sink_a, sinks)

    for _ in range(3):
        if not engine.has_pending_requests():
            break
        ready = engine.run()
        if ready is None:
            break
        if _drain(ready, sinks):
            ready.close()
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    request_b = _create_request(engine, bundle.model, prompt_b, max_new, sink_b, sinks)

    _run(engine, sinks)

    assert sink_a.tokens == isolated_a
    assert sink_b.tokens == isolated_b
    assert not request_a.is_turn_complete()
    assert not request_b.is_turn_complete()


def test_isolated_matches_batched(bundle):
    max_new = 16
    prompt = _PROMPTS[0]
    prompt_tokens = bundle.tokenizer.encode(prompt)

    isolated = _generate_isolated(bundle.model, prompt_tokens, max_new)

    engine = og.Engine(bundle.model)
    sinks = {p: _Sink() for p in _PROMPTS}
    sinks_by_request = {}
    requests = [
        _create_request(engine, bundle.model, bundle.tokenizer.encode(p), max_new, sink, sinks_by_request)
        for p, sink in sinks.items()
    ]
    _run(engine, sinks_by_request)

    assert sinks[prompt].tokens == isolated, "batched output diverged from the isolated run"
    assert all(not request.is_turn_complete() for request in requests)


def test_output_isolation(bundle):
    max_new = 16
    p0, p1 = _PROMPTS[0], _PROMPTS[2]

    isolated0 = _generate_isolated(bundle.model, bundle.tokenizer.encode(p0), max_new)
    isolated1 = _generate_isolated(bundle.model, bundle.tokenizer.encode(p1), max_new)
    assert isolated0 != isolated1, "distinct prompts produced identical greedy output; pick sharper prompts"

    engine = og.Engine(bundle.model)
    s0, s1 = _Sink(), _Sink()
    sinks = {}
    requests = [
        _create_request(engine, bundle.model, bundle.tokenizer.encode(p0), max_new, s0, sinks),
        _create_request(engine, bundle.model, bundle.tokenizer.encode(p1), max_new, s1, sinks),
    ]
    _run(engine, sinks)

    assert s0.tokens == isolated0
    assert s1.tokens == isolated1
    assert all(not request.is_turn_complete() for request in requests)


def test_completion_isolation(bundle):
    short_new, long_new = 6, 20
    short_prompt, long_prompt = _PROMPTS[3], _PROMPTS[1]

    long_isolated = _generate_isolated(bundle.model, bundle.tokenizer.encode(long_prompt), long_new)

    engine = og.Engine(bundle.model)
    short_sink, long_sink = _Sink(), _Sink()
    sinks = {}
    short_request = _create_request(
        engine,
        bundle.model,
        bundle.tokenizer.encode(short_prompt),
        short_new,
        short_sink,
        sinks,
        min_new_tokens=short_new,
    )
    long_request = _create_request(
        engine, bundle.model, bundle.tokenizer.encode(long_prompt), long_new, long_sink, sinks
    )
    _run(engine, sinks)

    assert len(short_sink.tokens) == short_new, "forced-length request did not stop at its bound"
    assert long_sink.tokens == long_isolated, "survivor diverged after its sibling completed"
    assert not short_request.is_turn_complete()
    assert not long_request.is_turn_complete()


def test_max_length_stops(bundle):
    max_new = 8
    prompt_tokens = bundle.tokenizer.encode(_PROMPTS[0])

    tokens = _generate_isolated(bundle.model, prompt_tokens, max_new, min_new_tokens=max_new)

    assert len(tokens) == max_new


def test_eos_gates_termination(bundle):
    eos_ids = _eos_ids(bundle)
    if not eos_ids:
        pytest.skip("Model config declares no eos_token_id.")

    messages = json.dumps([{"role": "user", "content": "Reply with the single word: ok"}])
    templated = bundle.tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)

    prompt_tokens = bundle.tokenizer.encode(templated)
    headroom = 200
    natural = _generate_isolated(bundle.model, prompt_tokens, headroom)

    if not natural or len(natural) >= headroom:
        pytest.skip("Greedy generation did not stop before max_length; EOS stop not reproducible here.")

    # Suppress EOS past the natural stop and verify the greedy prefix is stable.
    forced = _generate_isolated(
        bundle.model,
        prompt_tokens,
        headroom,
        min_new_tokens=min(len(natural) + 8, headroom - 1),
    )
    assert len(forced) > len(natural), "min_length did not suppress the EOS-gated stop"
    assert forced[: len(natural)] == natural, "forced continuation diverged from the natural greedy prefix"


def test_close_request_stops_output(bundle):
    max_new = 40
    sibling_prompt = bundle.tokenizer.encode(_PROMPTS[1])
    sibling_isolated = _generate_isolated(bundle.model, sibling_prompt, max_new)
    engine = og.Engine(bundle.model)

    sink_a, sink_b = _Sink(), _Sink()
    sinks = {}
    request_a = _create_request(engine, bundle.model, bundle.tokenizer.encode(_PROMPTS[0]), max_new, sink_a, sinks)
    request_b = _create_request(engine, bundle.model, sibling_prompt, max_new, sink_b, sinks)

    for _ in range(4):
        if not engine.has_pending_requests():
            break
        ready = engine.run()
        if ready is None:
            break
        if _drain(ready, sinks):
            ready.close()
    assert len(sink_a.tokens) > 0, "request A produced nothing before close"

    request_a.close()
    frozen_a = len(sink_a.tokens)

    _run(engine, sinks)

    assert len(sink_a.tokens) == frozen_a, "closed request kept producing tokens"
    assert sink_b.tokens == sibling_isolated, "sibling diverged after request close"
    assert not request_b.is_turn_complete()


def test_engine_teardown_and_recreation(bundle):
    max_new = 12
    prompt_tokens = bundle.tokenizer.encode(_PROMPTS[0])
    expected = _generate_isolated(bundle.model, prompt_tokens, max_new)

    first = og.Engine(bundle.model)
    sink1 = _Sink()
    first_sinks = {}
    first_request = _create_request(first, bundle.model, prompt_tokens, max_new, sink1, first_sinks)
    _run(first, first_sinks)
    assert sink1.tokens == expected
    assert not first_request.is_turn_complete()
    del first
    gc.collect()

    second = og.Engine(bundle.model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    second_sinks = {}
    second_request = _create_request(second, bundle.model, prompt_tokens, max_new, sink2, second_sinks)
    _run(second, second_sinks)
    assert sink2.tokens == expected
    assert not second_request.is_turn_complete()
