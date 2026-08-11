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

# Plain completion prompts with confident greedy continuations. They avoid
# chat templating so the batching-equality assertions do not hinge on a
# particular template.
_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "One plus one equals",
]

# Upper bound on engine.step() calls per generation, so a scheduling bug
# fails fast instead of hanging CI.
_MAX_STEPS = 20_000


class _Sink:
    """Per-request token collector correlated back through opaque data."""

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
    # PagedAttention is a CUDA-only operator, so this lane is CUDA only. Other
    # devices (including the default cpu) skip cleanly. The catalog also
    # declares cuda-only support, so resolution already skips them; this keeps
    # the requirement explicit at the model-load boundary.
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
        # min_length suppresses the stop token until the sequence reaches it,
        # making the stop condition deterministic regardless of the checkpoint.
        options["min_length"] = prompt_len + min_new_tokens
    params.set_search_options(**options)
    return params


def _add_request(engine, model, prompt_tokens, max_new_tokens, sink, *, min_new_tokens=0):
    params = _greedy_params(model, len(prompt_tokens), max_new_tokens, min_new_tokens)
    request = og.Request(params)
    request.add_tokens(prompt_tokens)
    request.set_opaque_data(sink)
    engine.add_request(request)
    return request


def _drain(ready) -> bool:
    sink = ready.get_opaque_data()
    while ready.has_unseen_tokens():
        sink.tokens.append(ready.get_unseen_token())
    return ready.is_done()


def _run(engine, *, max_steps=_MAX_STEPS) -> None:
    steps = 0
    while engine.has_pending_requests():
        ready = engine.step()
        assert ready is not None, "engine.step() returned no request while work remained"
        if _drain(ready):
            engine.remove_request(ready)
        steps += 1
        assert steps <= max_steps, "engine.step() exceeded the safety bound; possible non-termination"


def _generate_isolated(model, prompt_tokens, max_new_tokens, *, min_new_tokens=0) -> list[int]:
    """Greedy output of one prompt on its own dedicated engine."""
    sink = _Sink()
    engine = og.Engine(model)
    _add_request(engine, model, prompt_tokens, max_new_tokens, sink, min_new_tokens=min_new_tokens)
    _run(engine)
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
        f"'{_MODEL_ID}' must declare engine.dynamic_batching to exercise the paged cache; "
        f"got engine={engine_cfg!r}"
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
            f"pinned v{models.pinned_version(_MODEL_ID)} differs from the catalog; bump "
            f"PINNED_VERSIONS and PINNED_IDENTITY together if this is an intentional re-export."
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
    requests = [
        _add_request(engine, bundle.model, bundle.tokenizer.encode(p), max_new, sink)
        for p, sink in zip(_PROMPTS, sinks, strict=True)
    ]
    assert engine.has_pending_requests()

    _run(engine)

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
    _add_request(engine, bundle.model, prompt_a, max_new, sink_a)

    # Advance the first request a few steps before the second is admitted, so
    # admission happens while a request is already in flight.
    for _ in range(3):
        if not engine.has_pending_requests():
            break
        ready = engine.step()
        if ready is None:
            break
        if _drain(ready):
            engine.remove_request(ready)
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    _add_request(engine, bundle.model, prompt_b, max_new, sink_b)

    _run(engine)

    assert sink_a.tokens == isolated_a
    assert sink_b.tokens == isolated_b


def test_isolated_matches_batched(bundle):
    # Greedy decoding: batching a request alongside others must not change its
    # own output.
    max_new = 16
    prompt = _PROMPTS[0]
    prompt_tokens = bundle.tokenizer.encode(prompt)

    isolated = _generate_isolated(bundle.model, prompt_tokens, max_new)

    engine = og.Engine(bundle.model)
    sinks = {p: _Sink() for p in _PROMPTS}
    for p, sink in sinks.items():
        _add_request(engine, bundle.model, bundle.tokenizer.encode(p), max_new, sink)
    _run(engine)

    assert sinks[prompt].tokens == isolated, "batched output diverged from the isolated run"


def test_output_isolation(bundle):
    # Two different prompts sharing a batch must keep their own outputs.
    max_new = 16
    p0, p1 = _PROMPTS[0], _PROMPTS[2]

    isolated0 = _generate_isolated(bundle.model, bundle.tokenizer.encode(p0), max_new)
    isolated1 = _generate_isolated(bundle.model, bundle.tokenizer.encode(p1), max_new)
    assert isolated0 != isolated1, "distinct prompts produced identical greedy output; pick sharper prompts"

    engine = og.Engine(bundle.model)
    s0, s1 = _Sink(), _Sink()
    _add_request(engine, bundle.model, bundle.tokenizer.encode(p0), max_new, s0)
    _add_request(engine, bundle.model, bundle.tokenizer.encode(p1), max_new, s1)
    _run(engine)

    assert s0.tokens == isolated0
    assert s1.tokens == isolated1


def test_completion_isolation(bundle):
    # A short request that finishes and is removed mid-run must not truncate a
    # longer sibling: the survivor still matches its isolated run.
    short_new, long_new = 6, 20
    short_prompt, long_prompt = _PROMPTS[3], _PROMPTS[1]

    long_isolated = _generate_isolated(bundle.model, bundle.tokenizer.encode(long_prompt), long_new)

    engine = og.Engine(bundle.model)
    short_sink, long_sink = _Sink(), _Sink()
    _add_request(
        engine, bundle.model, bundle.tokenizer.encode(short_prompt), short_new, short_sink,
        min_new_tokens=short_new,
    )
    _add_request(engine, bundle.model, bundle.tokenizer.encode(long_prompt), long_new, long_sink)
    _run(engine)

    assert len(short_sink.tokens) == short_new, "forced-length request did not stop at its bound"
    assert long_sink.tokens == long_isolated, "survivor diverged after its sibling completed"


def test_max_length_stops(bundle):
    # min_length == max_length forces exactly max_new tokens, so the bound is
    # testable independent of when the checkpoint would emit a stop token.
    max_new = 8
    prompt_tokens = bundle.tokenizer.encode(_PROMPTS[0])

    tokens = _generate_isolated(bundle.model, prompt_tokens, max_new, min_new_tokens=max_new)

    assert len(tokens) == max_new


def test_eos_gates_termination(bundle):
    # The Engine stops a greedy request for exactly two reasons: it reaches
    # max_length, or the model emits an EOS token (the EOS token is consumed,
    # not surfaced as output). So a greedy run that stops *before* max_length
    # was terminated by EOS. Rather than asserting on a hardcoded token stream,
    # prove that positively: forcing generation past that point with min_length
    # suppresses the EOS and yields strictly more tokens with an identical
    # prefix.
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

    # Stopped early => EOS gated it. Forcing min_length past that point must
    # suppress the stop and extend the otherwise-identical greedy sequence.
    forced = _generate_isolated(
        bundle.model,
        prompt_tokens,
        headroom,
        min_new_tokens=min(len(natural) + 8, headroom - 1),
    )
    assert len(forced) > len(natural), "min_length did not suppress the EOS-gated stop"
    assert forced[: len(natural)] == natural, "forced continuation diverged from the natural greedy prefix"


def test_remove_request_stops_output(bundle):
    max_new = 40
    sibling_prompt = bundle.tokenizer.encode(_PROMPTS[1])
    sibling_isolated = _generate_isolated(bundle.model, sibling_prompt, max_new)
    engine = og.Engine(bundle.model)

    sink_a, sink_b = _Sink(), _Sink()
    request_a = _add_request(engine, bundle.model, bundle.tokenizer.encode(_PROMPTS[0]), max_new, sink_a)
    _add_request(engine, bundle.model, sibling_prompt, max_new, sink_b)

    # Let both requests generate a few tokens, then remove the first in flight.
    for _ in range(4):
        if not engine.has_pending_requests():
            break
        ready = engine.step()
        if ready is None:
            break
        if _drain(ready):
            engine.remove_request(ready)
    assert len(sink_a.tokens) > 0, "request A produced nothing before removal"

    engine.remove_request(request_a)
    frozen_a = len(sink_a.tokens)

    _run(engine)

    assert len(sink_a.tokens) == frozen_a, "removed request kept producing tokens"
    assert sink_b.tokens == sibling_isolated, "sibling diverged after request removal"


def test_engine_teardown_and_recreation(bundle):
    max_new = 12
    prompt_tokens = bundle.tokenizer.encode(_PROMPTS[0])
    expected = _generate_isolated(bundle.model, prompt_tokens, max_new)

    first = og.Engine(bundle.model)
    sink1 = _Sink()
    _add_request(first, bundle.model, prompt_tokens, max_new, sink1)
    _run(first)
    assert sink1.tokens == expected
    del first
    gc.collect()

    # A fresh engine on the same model must reload the paged cache and serve.
    second = og.Engine(bundle.model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    _add_request(second, bundle.model, prompt_tokens, max_new, sink2)
    _run(second)
    assert sink2.tokens == expected
