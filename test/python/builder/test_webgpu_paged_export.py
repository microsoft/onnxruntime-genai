# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""End-to-end WebGPU paged attention and hybrid-state export coverage for Qwen."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import re
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
_BUILDER_PATH = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builder.py"


def _export_model(output_dir, precision, ep, paged=True, structured=False, model_id=_MODEL_ID, model_options=()):
    extra_options = ["prune_lm_head=true", "hf_token=false", *model_options]
    runtime_options = []
    if paged:
        extra_options += [
            "use_paged_attention=true",
            "paged_block_size=256",
            "paged_chunk_size=256",
            "max_batch_size=2",
            "max_scheduled_tokens=256",
        ]
        if structured:
            runtime_options = [
                "--runtime_config",
                json.dumps({"engine": {"dynamic_batching": {"num_blocks": _NUM_BLOCKS}}}),
            ]
        else:
            extra_options.append(f"num_blocks={_NUM_BLOCKS}")
    subprocess.run(
        [
            sys.executable,
            str(_BUILDER_PATH),
            "-m",
            model_id,
            "-o",
            str(output_dir),
            "-p",
            precision,
            "-e",
            ep,
            *runtime_options,
            "--extra_options",
            *extra_options,
        ],
        check=True,
    )


def _cache_shape(node_arg, paged=True, head_size=None):
    shape = node_arg.shape
    if node_arg.name.endswith(".conv"):
        assert len(shape) == 3, f"Unexpected convolution state rank for {node_arg.name}: {shape}"
        assert all(isinstance(dim, int) for dim in shape[1:]), f"Unexpected fixed state shape: {shape}"
        return [1, shape[1], shape[2]]
    if node_arg.name.endswith(".recurrent"):
        assert len(shape) == 4, f"Unexpected recurrent state rank for {node_arg.name}: {shape}"
        assert all(isinstance(dim, int) for dim in shape[1:]), f"Unexpected fixed state shape: {shape}"
        return [1, shape[1], shape[2], shape[3]]
    if not paged:
        return [1, shape[1], 0, head_size]
    assert len(shape) == 4, f"Unexpected paged KV-cache shape for {node_arg.name}: {shape}"
    assert shape[1] in ("block_size", 256), f"Unexpected PagedAttention block size for {node_arg.name}: {shape}"
    assert isinstance(shape[2], int) and isinstance(shape[3], int), (
        f"Expected concrete KV head dimensions for {node_arg.name}: {shape}"
    )
    return [_NUM_BLOCKS, 256, shape[2], shape[3]]


def _cache_dtype(node_arg):
    assert node_arg.type in ("tensor(float)", "tensor(float16)")
    return np.float32 if node_arg.type == "tensor(float)" else np.float16


def _make_inputs(session, tokens, past_length, caches):
    input_names = {node_arg.name for node_arg in session.get_inputs()}
    paged = "cumulative_sequence_lengths" in input_names
    if paged:
        inputs = {
            "input_ids": tokens,
            "cumulative_sequence_lengths": np.asarray([0, len(tokens)], dtype=np.int32),
            "past_sequence_lengths": np.asarray([past_length], dtype=np.int32),
            "block_table": np.asarray([[0]], dtype=np.int32),
            "attention_metadata": np.asarray(
                [len(tokens), past_length + len(tokens), past_length + len(tokens)], dtype=np.int32
            ),
        }
    else:
        inputs = {
            "input_ids": tokens[np.newaxis, :],
            "attention_mask": np.ones((1, past_length + len(tokens)), dtype=np.int64),
        }
    if "position_ids" in input_names:
        positions = np.tile(np.arange(past_length, past_length + len(tokens), dtype=np.int64), (3, 1))
        inputs["position_ids"] = positions if paged else positions[:, np.newaxis, :]
    if "logits_indices" in input_names:
        inputs["logits_indices"] = np.asarray([len(tokens) - 1], dtype=np.int32)
    for node_arg in session.get_inputs():
        if not node_arg.name.startswith(("past_key_values.", "past.")):
            continue
        inputs[node_arg.name] = caches[node_arg.name]
    return inputs


def _run_step(session, tokens, past_length, caches):
    outputs = session.run(None, _make_inputs(session, tokens, past_length, caches))
    output_names = [node_arg.name for node_arg in session.get_outputs()]
    output_by_name = dict(zip(output_names, outputs, strict=True))
    next_caches = {
        input_name: output_by_name[input_name.replace("past_key_values.", "present.").replace("past.", "present.")]
        for input_name in caches
        if input_name.replace("past_key_values.", "present.").replace("past.", "present.") in output_by_name
    }
    assert next_caches, "Exported model did not return any paged KV caches"
    return output_by_name["logits"], next_caches


def _run_engine(model, prompts, max_new_tokens):
    engine = og.Engine(model)
    outputs = [[] for _ in prompts]
    requests = {}
    for index, prompt in enumerate(prompts):
        request_options = og.RequestOptions()
        request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
        request = engine.create_request(options=request_options)
        requests[request] = index
        turn_options = og.TurnOptions(request)
        turn_options.set_do_sample(False)
        turn_options.set_min_generated_tokens(max_new_tokens)
        turn_options.set_max_generated_tokens(max_new_tokens)
        request.begin_turn(np.asarray(prompt, dtype=np.int32), turn_options)

    event_buffer = engine.create_event_buffer(len(prompts))
    steps = 0
    while engine.has_pending_requests():
        for event in engine.run(event_buffer):
            if event.flags & og.EngineEventFlags.TOKEN:
                outputs[requests[event.request]].append(event.token)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                event.request.close()
        steps += 1
        assert steps <= max_new_tokens + 2, "Engine generation exceeded the expected step count"
    return outputs


def _run_cpu_reference(model):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=len(_PREFILL_TOKENS) + len(_DECODE_TOKENS) + 1)
    generator = og.Generator(model, params)
    generator.append_tokens(_PREFILL_TOKENS[np.newaxis, :].astype(np.int32))
    prefill = np.array(generator.get_logits(), copy=True)
    generator.append_tokens(_DECODE_TOKENS[np.newaxis, :].astype(np.int32))
    return prefill, np.array(generator.get_logits(), copy=True)


def _assert_logits_match(actual, expected):
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    assert np.argmax(actual) == np.argmax(expected)
    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=1.5e-1)


def _run_accumulation_probe(output_dir, batching, max_batch_size, explicit_value):
    assert register_webgpu_plugin()
    config = og.Config(str(output_dir))
    config.overlay(json.dumps({"engine": {batching: {"max_batch_size": max_batch_size}}}))
    if explicit_value is not None:
        config.set_provider_option("webgpu", "enableMatmulFp32Accumulation", explicit_value)
    model = og.Model(config)
    if batching == "dynamic_batching":
        assert len(_run_engine(model, [[1, 2, 3]], 1)[0]) == 1
    else:
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=4)
        generator = og.Generator(model, params)
        generator.append_tokens(np.asarray([[1, 2, 3]], dtype=np.int32))
        generator.generate_next_token()
        assert len(generator.get_sequence(0)) == 4


@pytest.fixture(scope="module", params=["dynamic_batching", "static_batching"])
def accumulation_model(request, tmp_path_factory):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    output_dir = tmp_path_factory.mktemp(f"accumulation-{request.param}")
    paged = request.param == "dynamic_batching"
    _export_model(output_dir, "fp16", "webgpu", paged=paged, structured=paged)
    return output_dir, request.param


@pytest.mark.parametrize("max_batch_size", [1, 2])
@pytest.mark.parametrize("explicit_value", [None, "0", "1"], ids=["default", "explicit-off", "explicit-on"])
def test_webgpu_genai_session_accumulation(accumulation_model, tmp_path, monkeypatch, max_batch_size, explicit_value):
    output_dir, batching = accumulation_model
    shader_path = tmp_path / "shaders.wgsl"
    monkeypatch.setenv("ORT_WEBGPU_EP_SHADER_DUMP_FILE", str(shader_path))
    process = multiprocessing.get_context("spawn").Process(
        target=_run_accumulation_probe,
        args=(output_dir, batching, max_batch_size, explicit_value),
    )
    process.start()
    process.join()
    exitcode = process.exitcode
    process.close()
    assert exitcode == 0
    # Version-specific shader probe for the CI pins: ORT 1.32.0.dev20261009003 and
    # onnxruntime-ep-webgpu 0.5.0.dev20261009200253 (test/python/webgpu/ort/requirements.txt).
    # The f16 input alias identifies FP16 MatMul; a vec4<f32> accumulator means its
    # packed kernel widens accumulation to FP32 without changing model input precision.
    # These are internal WGSL/debug-dump markers, not a public ORT contract. On an ORT
    # upgrade, check dump availability, delimiters, and accumulator naming before
    # interpreting a failure as a GenAI provider-option propagation regression.
    shaders = re.findall(
        r"=== WebGPU Shader code \[MatMul,.*? Start ===\n(.*?)\n=== WebGPU Shader code \[MatMul,.*? End ===",
        shader_path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    fp16_shaders = [shader for shader in shaders if "alias a_element_t = f16;" in shader]
    assert fp16_shaders, "GenAI inference did not compile any FP16 WebGPU MatMul shaders"
    uses_fp32 = any("var acc: array<vec4<f32>" in shader for shader in fp16_shaders)
    expected = explicit_value == "1" if explicit_value is not None else max_batch_size > 1
    assert uses_fp32 == expected


@pytest.fixture(scope="module")
def webgpu_plugin():
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    webgpu_ep = importlib.import_module("onnxruntime_ep_webgpu")
    webgpu_provider = webgpu_ep.get_ep_name()
    ort.register_execution_provider_library(webgpu_provider, webgpu_ep.get_library_path())
    return webgpu_ep


@pytest.mark.parametrize("structured", [False, True], ids=["legacy", "structured-runtime"])
def test_webgpu_paged_export_runs_prefill_and_decode(tmp_path, structured, webgpu_plugin):
    webgpu_ep = webgpu_plugin
    webgpu_provider = webgpu_ep.get_ep_name()

    output_dir = tmp_path / "webgpu-paged"
    _export_model(output_dir, "fp16", "webgpu", structured=structured)

    config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert config["engine"]["dynamic_batching"]["num_blocks"] == _NUM_BLOCKS
    assert "gpu_utilization_factor" not in config["engine"]["dynamic_batching"]
    assert config["model"]["decoder"]["num_hidden_layers"] == _NUM_HIDDEN_LAYERS
    assert config["model"]["decoder"]["inputs"]["attention_metadata"] == "attention_metadata"

    model_path = output_dir / config["model"]["decoder"]["filename"]
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(tmp_path / "webgpu-profile")
    # GenAI enables FP32 accumulation for max_batch_size > 1 (2 in this export).
    # This direct ORT session bypasses GenAI's config handling, so enable it explicitly.
    session_options.add_session_config_entry("ep.webgpuexecutionprovider.enableMatmulFp32Accumulation", "1")
    webgpu_devices = [device for device in ort.get_ep_devices() if device.ep_name == webgpu_provider]
    assert webgpu_devices, (
        f"No {webgpu_provider} device found after registering {webgpu_ep.get_library_path()}; "
        f"discovered providers: {[device.ep_name for device in ort.get_ep_devices()]}"
    )
    session_options.add_provider_for_devices([webgpu_devices[0]], {})
    webgpu_session = ort.InferenceSession(str(model_path), sess_options=session_options)

    cpu_output_dir = tmp_path / "cpu-reference"
    _export_model(cpu_output_dir, "fp32", "cpu", paged=False)
    cpu_model = og.Model(str(cpu_output_dir))
    cpu_prefill, cpu_decode = _run_cpu_reference(cpu_model)

    cache_inputs = {
        node_arg.name: np.zeros(_cache_shape(node_arg), dtype=_cache_dtype(node_arg))
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith("past_key_values.")
    }
    assert cache_inputs, "Exported model has no paged KV-cache inputs"

    webgpu_prefill, webgpu_caches = _run_step(webgpu_session, _PREFILL_TOKENS, 0, cache_inputs)
    _assert_logits_match(webgpu_prefill, cpu_prefill)
    assert webgpu_prefill.size > 0
    assert np.isfinite(webgpu_prefill).all()
    assert all(np.isfinite(cache).all() for cache in webgpu_caches.values())
    assert any(np.any(cache != 0) for cache in webgpu_caches.values()), "Prefill did not update the paged KV cache"

    prefill_caches = webgpu_caches
    webgpu_decode, webgpu_caches = _run_step(webgpu_session, _DECODE_TOKENS, len(_PREFILL_TOKENS), webgpu_caches)
    _assert_logits_match(webgpu_decode, cpu_decode)
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

    engine_config = og.Config(str(output_dir))
    engine_config.clear_providers()
    engine_config.append_provider("webgpu")
    engine_model = og.Model(engine_config)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    max_new_tokens = 4
    isolated = [_run_engine(engine_model, [prompt], max_new_tokens)[0] for prompt in prompts]
    assert all(len(tokens) == max_new_tokens for tokens in isolated)
    assert _run_engine(engine_model, prompts, max_new_tokens) == isolated


@pytest.mark.parametrize(
    ("cache_options", "error"),
    [
        ([], "WebGPU paged attention requires num_blocks to be a positive integer"),
        ([f"num_blocks={_NUM_BLOCKS}"], "num_blocks and gpu_utilization_factor are mutually exclusive"),
    ],
    ids=["missing-fixed-capacity", "conflicting-cache-sizing"],
)
def test_webgpu_paged_export_requires_final_num_blocks(tmp_path, cache_options, error):
    result = subprocess.run(
        [
            sys.executable,
            str(_BUILDER_PATH),
            "-m",
            _MODEL_ID,
            "-o",
            str(tmp_path),
            "-p",
            "fp16",
            "-e",
            "webgpu",
            "--runtime_config",
            json.dumps({"engine": {"dynamic_batching": {"gpu_utilization_factor": 0.6}}}),
            "--extra_options",
            "use_paged_attention=true",
            *cache_options,
            "config_only=true",
            "hf_token=false",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert error in result.stderr


def test_webgpu_qwen35_paged_export_runs_prefill_and_decode(tmp_path, webgpu_plugin):
    model_id = "Qwen/Qwen3.5-0.8B"
    output_dir = tmp_path / "qwen35-webgpu-paged"
    cpu_output_dir = tmp_path / "qwen35-cpu-reference"
    model_options = ("exclude_mtp=true",)
    _export_model(output_dir, "fp16", "webgpu", structured=True, model_id=model_id, model_options=model_options)
    _export_model(cpu_output_dir, "fp32", "cpu", paged=False, model_id=model_id, model_options=model_options)

    config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(tmp_path / "qwen35-webgpu-profile")
    session_options.add_session_config_entry("ep.webgpuexecutionprovider.enableMatmulFp32Accumulation", "1")
    provider = webgpu_plugin.get_ep_name()
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert devices, "No registered WebGPU device found"
    session_options.add_provider_for_devices([devices[0]], {})
    webgpu_session = ort.InferenceSession(
        str(output_dir / config["model"]["decoder"]["filename"]), sess_options=session_options
    )
    cpu_config = json.loads((cpu_output_dir / "genai_config.json").read_text(encoding="utf-8"))
    cpu_session = ort.InferenceSession(
        str(cpu_output_dir / cpu_config["model"]["decoder"]["filename"]), providers=["CPUExecutionProvider"]
    )
    webgpu_caches = {
        node_arg.name: np.zeros(_cache_shape(node_arg), dtype=_cache_dtype(node_arg))
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith(("past_key_values.", "past."))
    }
    cpu_caches = {
        node_arg.name: np.zeros(
            _cache_shape(node_arg, paged=False, head_size=cpu_config["model"]["decoder"]["head_size"]),
            dtype=_cache_dtype(node_arg),
        )
        for node_arg in cpu_session.get_inputs()
        if node_arg.name.startswith(("past_key_values.", "past."))
    }
    past_length = 0
    for tokens in (_PREFILL_TOKENS, _DECODE_TOKENS):
        webgpu_logits, webgpu_caches = _run_step(webgpu_session, tokens, past_length, webgpu_caches)
        cpu_logits, cpu_caches = _run_step(cpu_session, tokens, past_length, cpu_caches)
        _assert_logits_match(webgpu_logits, cpu_logits)
        past_length += len(tokens)

    with open(webgpu_session.end_profiling(), encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    for operator in ("GatedDeltaNet", "VarlenCausalConvWithState", "PagedAttention"):
        assert any(
            operator in event.get("name", "") and event.get("args", {}).get("provider") == provider
            for event in profile
        ), f"{operator} was not assigned to WebGPU"
