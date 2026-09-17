# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Eager versus CUDA-graph-captured parity for the paged Engine decode path.

Every test runs the same work twice against one model directory, differing only in the
``enable_cuda_graph`` provider option, and requires identical token streams. A captured graph
replays recorded device pointers and launch dimensions without re-running any host code, so a wrong
annotation key, a per-step buffer address, or a stale fixed-state bank shows up here as diverging
tokens rather than as a crash.

These tests need a paged model whose decoder the CUDA execution provider can actually capture. The
synthetic models under ``test/models/engine`` are shape-manipulation graphs that ORT partitions
across CPU and CUDA, and they fault on the first replay, so they cannot serve here. Point
``ORTGENAI_PAGED_CUDA_GRAPH_MODEL`` at a real paged export to run this file; it skips otherwise.

Coverage depends on what the supplied model declares:

* one logits row per packed token exercises uniform multi-token verification;
* packed ``position_ids`` exercise the persistent position buffer, in either the ``[num_tokens]``
  or the ``[3, num_tokens]`` geometry;
* ``fixed_conv``/``fixed_recurrent`` state groups exercise HybridDecoderIO, the persistent bank
  flip, and the staged-binding fallback.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import register_plugin_providers

register_plugin_providers(logging.getLogger(__name__))

_MODEL_DIR = os.environ.get("ORTGENAI_PAGED_CUDA_GRAPH_MODEL")

pytestmark = pytest.mark.skipif(
    not og.is_cuda_available() or not _MODEL_DIR or not Path(_MODEL_DIR).is_dir(),
    reason=(
        "set ORTGENAI_PAGED_CUDA_GRAPH_MODEL to a paged model directory whose decoder the CUDA "
        "execution provider can capture"
    ),
)

_MAX_STEPS = 10_000

_PROMPT_A = [5, 9, 13]
_PROMPT_B = [7, 2, 20, 4]
_PROMPT_C = [3, 8, 2, 15, 6, 11]


def _load_model(*, graph_capture):
    config = og.Config(_MODEL_DIR)
    config.clear_providers()
    config.append_provider("cuda")
    config.set_provider_option("cuda", "enable_cuda_graph", "1" if graph_capture else "0")
    return og.Model(config)


def _begin(engine, prompt, max_new_tokens):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(max_new_tokens)
    request.begin_turn(np.asarray(prompt, dtype=np.int32), turn_options)
    return request


def _drain(engine, streams, event_buffer):
    for event in engine.run(event_buffer):
        assert event.finish_reason != og.FinishReason.FAILED, f"engine step failed: {event.error_code}"
        if event.flags & og.EngineEventFlags.TOKEN:
            streams[event.request].append(event.token)


def _generate(model, prompts, max_new_tokens):
    """Token stream per prompt, with every prompt decoded in one batched Engine."""
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(8)
    requests = [_begin(engine, prompt, max_new_tokens) for prompt in prompts]
    streams = {request: [] for request in requests}

    steps = 0
    while engine.has_pending_requests():
        _drain(engine, streams, event_buffer)
        steps += 1
        assert steps <= _MAX_STEPS, "engine.run() exceeded the safety bound"

    tokens = [streams[request] for request in requests]
    for request in requests:
        request.close()
    del engine
    gc.collect()
    return tokens


def _generate_with_drafts(model, prompt, max_new_tokens, draft_width, reference):
    """Decode `prompt`, attaching a fixed-width accepted draft block to every step after the first.

    A uniform `1 + draft_width` token step is the shape the multi-token capture path exists for.
    The drafts replay the model's own greedy output, so every draft is accepted and every verify
    step carries the same token count.
    """
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(8)
    request = _begin(engine, prompt, max_new_tokens)
    streams = {request: []}

    steps = 0
    while engine.has_pending_requests():
        produced = len(streams[request])
        drafts = reference[produced : produced + draft_width]
        if produced and len(drafts) == draft_width:
            request.set_draft_tokens(np.asarray(drafts, dtype=np.int32))
        _drain(engine, streams, event_buffer)
        steps += 1
        assert steps <= _MAX_STEPS, "engine.run() exceeded the safety bound"

    tokens = list(streams[request])
    request.close()
    del engine
    gc.collect()
    return tokens


def test_captured_decode_matches_eager():
    """Repeated single-token decode, long enough to cross a block-table bucket boundary."""
    max_new_tokens = 32

    eager = _generate(_load_model(graph_capture=False), [_PROMPT_A], max_new_tokens)
    captured = _generate(_load_model(graph_capture=True), [_PROMPT_A], max_new_tokens)

    assert eager[0], "the eager run produced no tokens"
    assert captured == eager


def test_captured_batched_decode_matches_eager():
    """Concurrent requests: several batch widths, and prompts that finish at different steps."""
    prompts = [_PROMPT_A, _PROMPT_B, _PROMPT_C]
    max_new_tokens = 24

    eager = _generate(_load_model(graph_capture=False), prompts, max_new_tokens)
    captured = _generate(_load_model(graph_capture=True), prompts, max_new_tokens)

    assert captured == eager


def test_captured_multi_token_verification_matches_eager():
    """Uniform multi-token verify steps, which only a per-token-logits model can schedule."""
    max_new_tokens = 24
    draft_width = 2

    eager_model = _load_model(graph_capture=False)
    probe = og.Engine(eager_model)
    supported = probe.max_draft_tokens_per_proposal()
    del probe
    gc.collect()
    if supported < draft_width:
        pytest.skip("the supplied model cannot verify draft tokens")

    reference = _generate(eager_model, [_PROMPT_A], max_new_tokens)[0]
    eager = _generate_with_drafts(eager_model, _PROMPT_A, max_new_tokens, draft_width, reference)
    captured = _generate_with_drafts(
        _load_model(graph_capture=True), _PROMPT_A, max_new_tokens, draft_width, reference
    )

    assert eager, "the draft run produced no tokens"
    assert captured == eager


def test_two_engines_sharing_one_model_match_eager():
    """Annotation ids are per decoder but captured graphs live on the shared session.

    Two Engines over one Model use one ORT session. If both decoders named their graphs from the
    same id space, the second would replay the first's graph against its own buffers.
    """
    max_new_tokens = 24
    expected = _generate(_load_model(graph_capture=False), [_PROMPT_A, _PROMPT_B], max_new_tokens)

    model = _load_model(graph_capture=True)
    first, second = og.Engine(model), og.Engine(model)
    first_buffer, second_buffer = first.create_event_buffer(8), second.create_event_buffer(8)
    first_request = _begin(first, _PROMPT_A, max_new_tokens)
    second_request = _begin(second, _PROMPT_B, max_new_tokens)
    streams = {first_request: [], second_request: []}

    steps = 0
    while first.has_pending_requests() or second.has_pending_requests():
        if first.has_pending_requests():
            _drain(first, streams, first_buffer)
        if second.has_pending_requests():
            _drain(second, streams, second_buffer)
        steps += 1
        assert steps <= _MAX_STEPS, "engine.run() exceeded the safety bound"

    assert [streams[first_request], streams[second_request]] == expected
    first_request.close()
    second_request.close()


def test_engine_recreated_on_the_same_model_matches_eager():
    """A destroyed Engine must release its graphs, and the next one must not replay them.

    The second Engine allocates fresh persistent buffers. Reusing the first Engine's annotation ids
    would replay graphs recorded against memory that no longer exists.
    """
    max_new_tokens = 24
    expected = _generate(_load_model(graph_capture=False), [_PROMPT_A], max_new_tokens)

    model = _load_model(graph_capture=True)
    for _ in range(3):
        engine = og.Engine(model)
        event_buffer = engine.create_event_buffer(8)
        request = _begin(engine, _PROMPT_A, max_new_tokens)
        streams = {request: []}
        steps = 0
        while engine.has_pending_requests():
            _drain(engine, streams, event_buffer)
            steps += 1
            assert steps <= _MAX_STEPS, "engine.run() exceeded the safety bound"
        assert [streams[request]] == expected
        request.close()
        del engine
        gc.collect()
