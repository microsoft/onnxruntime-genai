# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""End-to-end WebGPU PagedAttention export coverage for Qwen 2.5 0.5B."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
import pytest
from _test_utils import register_webgpu_plugin

_NUM_BLOCKS = 8
_NUM_HIDDEN_LAYERS = 24
_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_PREFILL_TOKENS = np.asarray([1, 2, 3], dtype=np.int64)
_DECODE_TOKENS = np.asarray([4], dtype=np.int64)
_ENGINE_PROMPTS = [[1, 2, 3], [4, 5, 6, 7, 8]]
_MAX_NEW_TOKENS = 4
_BUILDER_PATH = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builder.py"


def _export_model(output_dir, precision, ep, paged=True, prune_lm_head=True):
    extra_options = [f"prune_lm_head={str(prune_lm_head).lower()}", "hf_token=false"]
    if paged:
        extra_options += [
            "use_paged_attention=true",
            "paged_block_size=256",
            "paged_chunk_size=256",
            f"num_blocks={_NUM_BLOCKS}",
            "max_batch_size=2",
            "max_scheduled_tokens=256",
        ]
    subprocess.run(
        [
            sys.executable,
            str(_BUILDER_PATH),
            "-m",
            _MODEL_ID,
            "-o",
            str(output_dir),
            "-p",
            precision,
            "-e",
            ep,
            "--extra_options",
            *extra_options,
        ],
        check=True,
    )


def _cache_shape(node_arg):
    shape = node_arg.shape
    assert len(shape) == 4, f"Unexpected paged KV-cache shape for {node_arg.name}: {shape}"
    assert shape[1] in ("block_size", 256), f"Unexpected PagedAttention block size for {node_arg.name}: {shape}"
    assert isinstance(shape[2], int) and isinstance(shape[3], int), (
        f"Expected concrete KV head dimensions for {node_arg.name}: {shape}"
    )
    return (_NUM_BLOCKS, 256, shape[2], shape[3])


def _cache_dtype(node_arg):
    assert node_arg.type in ("tensor(float)", "tensor(float16)")
    return np.float32 if node_arg.type == "tensor(float)" else np.float16


def _make_inputs(session, tokens, past_length, caches):
    inputs = {
        "input_ids": tokens,
        "cumulative_sequence_lengths": np.asarray([0, len(tokens)], dtype=np.int32),
        "past_sequence_lengths": np.asarray([past_length], dtype=np.int32),
        "block_table": np.asarray([[0]], dtype=np.int32),
        "attention_metadata": np.asarray(
            [len(tokens), past_length + len(tokens), past_length + len(tokens)], dtype=np.int32
        ),
    }
    if any(node_arg.name == "logits_indices" for node_arg in session.get_inputs()):
        inputs["logits_indices"] = np.asarray([len(tokens) - 1], dtype=np.int32)
    for node_arg in session.get_inputs():
        if not node_arg.name.startswith("past_key_values."):
            continue
        inputs[node_arg.name] = caches.get(
            node_arg.name, np.zeros(_cache_shape(node_arg), dtype=_cache_dtype(node_arg))
        )
    return inputs


def _run_step(session, tokens, past_length, caches):
    outputs = session.run(None, _make_inputs(session, tokens, past_length, caches))
    output_names = [node_arg.name for node_arg in session.get_outputs()]
    output_by_name = dict(zip(output_names, outputs, strict=True))
    next_caches = {
        input_name: output_by_name[input_name.replace("past_key_values", "present")]
        for input_name in caches
        if input_name.replace("past_key_values", "present") in output_by_name
    }
    assert next_caches, "Exported model did not return any paged KV caches"
    return output_by_name["logits"], next_caches


def _run_packed_prefills(session, prompts, caches):
    lengths = [len(prompt) for prompt in prompts]
    cumulative_lengths = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    inputs = {
        "input_ids": np.concatenate(prompts).astype(np.int64),
        "cumulative_sequence_lengths": cumulative_lengths,
        "past_sequence_lengths": np.zeros(len(prompts), dtype=np.int32),
        "block_table": np.arange(len(prompts), dtype=np.int32)[:, np.newaxis],
        "attention_metadata": np.asarray([max(lengths), max(lengths), max(lengths)], dtype=np.int32),
    }
    if any(node_arg.name == "logits_indices" for node_arg in session.get_inputs()):
        inputs["logits_indices"] = cumulative_lengths[1:] - 1
    inputs.update(caches)
    outputs = session.run(None, inputs)
    return dict(zip((node_arg.name for node_arg in session.get_outputs()), outputs, strict=True))["logits"]


def _create_engine_request(engine, prompt, max_new_tokens):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_do_sample(False)
    turn_options.set_min_generated_tokens(max_new_tokens)
    turn_options.set_max_generated_tokens(max_new_tokens)
    request.begin_turn(np.asarray(prompt, dtype=np.int32), turn_options)
    return request


def _record_engine_events(events, request_indices, outputs):
    for event in events:
        if event.flags & og.EngineEventFlags.TOKEN:
            outputs[request_indices[event.request]].append(event.token)
        if event.flags & og.EngineEventFlags.TURN_FINISHED:
            event.request.close()


def _run_engine(model, prompts, max_new_tokens):
    engine = og.Engine(model)
    outputs = [[] for _ in prompts]
    request_indices = {
        _create_engine_request(engine, prompt, max_new_tokens): index for index, prompt in enumerate(prompts)
    }

    event_buffer = engine.create_event_buffer(len(prompts))
    steps = 0
    while engine.has_pending_requests():
        _record_engine_events(engine.run(event_buffer), request_indices, outputs)
        steps += 1
        assert steps <= max_new_tokens + 2, "Engine generation exceeded the expected step count"
    return outputs


def _run_reference(model):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(_PREFILL_TOKENS) + len(_DECODE_TOKENS) + 1)
    generator = og.Generator(model, params)
    generator.append_tokens(_PREFILL_TOKENS[np.newaxis, :].astype(np.int32))
    prefill = np.array(generator.get_logits(), copy=True)
    generator.append_tokens(_DECODE_TOKENS[np.newaxis, :].astype(np.int32))
    return prefill, np.array(generator.get_logits(), copy=True)


def _assert_logits_match(actual, expected):
    assert actual.dtype == np.float16
    actual = actual.reshape(-1)
    expected = expected.astype(actual.dtype, copy=False).reshape(-1)
    assert np.argmax(actual) == np.argmax(expected)
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=1.5e-1)


@pytest.fixture(scope="module")
def webgpu_paged_models(tmp_path_factory):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    webgpu_ep = importlib.import_module("onnxruntime_ep_webgpu")

    webgpu_provider = webgpu_ep.get_ep_name()
    ort.register_execution_provider_library(webgpu_provider, webgpu_ep.get_library_path())

    tmp_path = tmp_path_factory.mktemp("webgpu-paged-export")
    output_dir = tmp_path / "webgpu-paged"
    _export_model(output_dir, "fp16", "webgpu")

    config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert config["engine"]["dynamic_batching"]["num_blocks"] == _NUM_BLOCKS
    assert config["engine"]["dynamic_batching"]["max_batch_size"] == 2
    assert config["model"]["decoder"]["num_hidden_layers"] == _NUM_HIDDEN_LAYERS
    assert config["model"]["decoder"]["inputs"]["attention_metadata"] == "attention_metadata"

    engine_output_dir = tmp_path / "webgpu-engine"
    _export_model(engine_output_dir, "fp16", "webgpu", prune_lm_head=False)
    engine_config = json.loads((engine_output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert engine_config["engine"]["dynamic_batching"]["max_batch_size"] == 2

    reference_output_dir = tmp_path / "webgpu-reference"
    _export_model(reference_output_dir, "fp16", "webgpu", paged=False)
    reference_config = og.Config(str(reference_output_dir))
    reference_config.clear_providers()
    reference_config.append_provider("webgpu")
    reference_model = og.Model(reference_config)

    return output_dir, engine_output_dir, config, webgpu_provider, reference_model, tmp_path


def _create_webgpu_session(output_dir, config, webgpu_provider, *, profile_prefix=None):
    model_path = output_dir / config["model"]["decoder"]["filename"]
    session_options = ort.SessionOptions()
    if profile_prefix is not None:
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(profile_prefix)
    webgpu_devices = [device for device in ort.get_ep_devices() if device.ep_name == webgpu_provider]
    assert webgpu_devices, f"No {webgpu_provider} device found; discovered: {[d.ep_name for d in ort.get_ep_devices()]}"
    session_options.add_provider_for_devices([webgpu_devices[0]], {})
    return ort.InferenceSession(str(model_path), sess_options=session_options)


def test_webgpu_paged_export_runs_prefill_and_decode(webgpu_paged_models):
    output_dir, _, config, webgpu_provider, reference_model, tmp_path = webgpu_paged_models

    webgpu_session = _create_webgpu_session(
        output_dir,
        config,
        webgpu_provider,
        profile_prefix=tmp_path / "webgpu-profile",
    )

    reference_prefill, reference_decode = _run_reference(reference_model)

    cache_inputs = {
        node_arg.name: np.zeros(_cache_shape(node_arg), dtype=_cache_dtype(node_arg))
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith("past_key_values.")
    }
    assert cache_inputs, "Exported model has no paged KV-cache inputs"

    webgpu_prefill, webgpu_caches = _run_step(webgpu_session, _PREFILL_TOKENS, 0, cache_inputs)
    _assert_logits_match(webgpu_prefill, reference_prefill)
    assert webgpu_prefill.size > 0
    assert np.isfinite(webgpu_prefill).all()
    assert all(np.isfinite(cache).all() for cache in webgpu_caches.values())
    assert any(np.any(cache != 0) for cache in webgpu_caches.values()), "Prefill did not update the paged KV cache"

    prefill_caches = webgpu_caches
    webgpu_decode, webgpu_caches = _run_step(webgpu_session, _DECODE_TOKENS, len(_PREFILL_TOKENS), webgpu_caches)
    _assert_logits_match(webgpu_decode, reference_decode)
    assert webgpu_decode.size > 0
    assert webgpu_decode.shape[-1] == webgpu_prefill.shape[-1]
    assert np.isfinite(webgpu_decode).all()
    assert all(webgpu_caches[name].shape == prefill_caches[name].shape for name in cache_inputs)
    assert all(np.isfinite(cache).all() for cache in webgpu_caches.values())
    assert any(
        np.any(webgpu_caches[name] != prefill_caches[name]) for name in cache_inputs
    ), "Decode did not update the paged KV cache"

    with open(webgpu_session.end_profiling(), encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    paged_attention_events = [
        event
        for event in profile
        if "PagedAttention" in event.get("name", "") and event.get("args", {}).get("provider") == webgpu_provider
    ]
    assert paged_attention_events, "PagedAttention was not assigned to WebGPUExecutionProvider"


def test_webgpu_paged_simultaneous_unequal_prefills(webgpu_paged_models):
    output_dir, _, config, webgpu_provider, reference_model, _ = webgpu_paged_models
    webgpu_session = _create_webgpu_session(output_dir, config, webgpu_provider)
    cache_inputs = {
        node_arg.name: np.zeros(_cache_shape(node_arg), dtype=_cache_dtype(node_arg))
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith("past_key_values.")
    }
    prompts = [np.asarray(prompt, dtype=np.int64) for prompt in _ENGINE_PROMPTS]
    actual = _run_packed_prefills(webgpu_session, prompts, cache_inputs)
    expected = np.concatenate([_run_reference_prefill(reference_model, prompt) for prompt in prompts])

    _assert_logits_match(actual, expected)


def _run_reference_prefill(model, tokens):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(tokens) + 1)
    generator = og.Generator(model, params)
    generator.append_tokens(tokens[np.newaxis, :].astype(np.int32))
    return np.array(generator.get_logits(), copy=True)


def test_webgpu_paged_staggered_admission(webgpu_paged_models):
    _, engine_output_dir, _, _, _, _ = webgpu_paged_models
    engine_config = og.Config(str(engine_output_dir))
    engine_config.clear_providers()
    engine_config.append_provider("webgpu")
    engine_model = og.Model(engine_config)
    isolated = [_run_engine(engine_model, [prompt], _MAX_NEW_TOKENS)[0] for prompt in _ENGINE_PROMPTS]
    assert all(len(tokens) == _MAX_NEW_TOKENS for tokens in isolated)

    engine = og.Engine(engine_model)
    outputs = [[] for _ in _ENGINE_PROMPTS]
    first_request = _create_engine_request(engine, _ENGINE_PROMPTS[0], _MAX_NEW_TOKENS)
    request_indices = {first_request: 0}
    event_buffer = engine.create_event_buffer(len(_ENGINE_PROMPTS))

    _record_engine_events(engine.run(event_buffer), request_indices, outputs)
    assert outputs[0], "First request produced no token before staggered admission"
    assert engine.has_pending_requests(), "First request finished before staggered admission"

    second_request = _create_engine_request(engine, _ENGINE_PROMPTS[1], _MAX_NEW_TOKENS)
    request_indices[second_request] = 1
    first_token_count = len(outputs[0])
    _record_engine_events(engine.run(event_buffer), request_indices, outputs)
    assert len(outputs[0]) == first_token_count + 1, "First request did not decode during staggered admission"
    assert outputs[1], "Second request did not prefill during staggered admission"

    steps = 2
    while engine.has_pending_requests():
        _record_engine_events(engine.run(event_buffer), request_indices, outputs)
        steps += 1
        assert steps <= _MAX_NEW_TOKENS + 2, "Engine generation exceeded the expected step count"

    assert outputs == isolated
