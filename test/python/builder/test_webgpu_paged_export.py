# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""End-to-end WebGPU PagedAttention export coverage for Qwen 2.5 0.5B."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from _test_utils import register_webgpu_plugin


_NUM_BLOCKS = 8
_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_PREFILL_TOKENS = np.asarray([1, 2, 3], dtype=np.int64)
_DECODE_TOKENS = np.asarray([4], dtype=np.int64)
_BUILDER_PATH = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builder.py"


def _cache_shape(node_arg):
    shape = node_arg.shape
    assert len(shape) == 4, f"Unexpected paged KV-cache shape for {node_arg.name}: {shape}"
    assert shape[1] == 256, f"Unexpected PagedAttention block size for {node_arg.name}: {shape}"
    assert isinstance(shape[2], int) and isinstance(shape[3], int), (
        f"Expected concrete KV head dimensions for {node_arg.name}: {shape}"
    )
    return (_NUM_BLOCKS, shape[1], shape[2], shape[3])


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
    for node_arg in session.get_inputs():
        if not node_arg.name.startswith("past_key_values."):
            continue
        inputs[node_arg.name] = caches.get(node_arg.name, np.zeros(_cache_shape(node_arg), dtype=np.float16))
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


def _assert_close(webgpu_value, cpu_value, label):
    np.testing.assert_allclose(webgpu_value, cpu_value, rtol=2e-2, atol=2e-2, err_msg=label)


def test_webgpu_paged_export_runs_prefill_and_decode_with_cpu_reference(tmp_path):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    import onnxruntime_ep_webgpu as webgpu_ep

    webgpu_provider = webgpu_ep.get_ep_name()
    ort.register_execution_provider_library(webgpu_provider, webgpu_ep.get_library_path())

    output_dir = tmp_path / "webgpu-paged"
    subprocess.run(
        [
            sys.executable,
            str(_BUILDER_PATH),
            "-m",
            _MODEL_ID,
            "-o",
            str(output_dir),
            "-p",
            "fp16",
            "-e",
            "webgpu",
            "--extra_options",
            "use_paged_attention=true",
            "prune_lm_head=true",
            "paged_block_size=256",
            "paged_chunk_size=256",
            f"num_blocks={_NUM_BLOCKS}",
            "max_batch_size=1",
            "max_scheduled_tokens=256",
            "num_hidden_layers=2",
            "hf_token=false",
        ],
        check=True,
    )

    config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert config["engine"]["dynamic_batching"]["num_blocks"] == _NUM_BLOCKS
    assert config["model"]["decoder"]["inputs"]["attention_metadata"] == "attention_metadata"

    model_path = output_dir / config["model"]["decoder"]["filename"]
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(tmp_path / "webgpu-profile")
    webgpu_session = ort.InferenceSession(str(model_path), session_options, [webgpu_provider])
    cpu_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    cache_inputs = {
        node_arg.name: np.zeros(_cache_shape(node_arg), dtype=np.float16)
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith("past_key_values.")
    }
    assert cache_inputs, "Exported model has no paged KV-cache inputs"

    webgpu_prefill, webgpu_caches = _run_step(webgpu_session, _PREFILL_TOKENS, 0, cache_inputs)
    cpu_prefill, cpu_caches = _run_step(cpu_session, _PREFILL_TOKENS, 0, cache_inputs)
    _assert_close(webgpu_prefill, cpu_prefill, "prefill logits")
    for name in cache_inputs:
        _assert_close(webgpu_caches[name], cpu_caches[name], f"prefill cache {name}")

    webgpu_decode, webgpu_caches = _run_step(webgpu_session, _DECODE_TOKENS, len(_PREFILL_TOKENS), webgpu_caches)
    cpu_decode, cpu_caches = _run_step(cpu_session, _DECODE_TOKENS, len(_PREFILL_TOKENS), cpu_caches)
    _assert_close(webgpu_decode, cpu_decode, "decode logits")
    for name in cache_inputs:
        _assert_close(webgpu_caches[name], cpu_caches[name], f"decode cache {name}")

    with open(webgpu_session.end_profiling(), encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    paged_attention_events = [
        event
        for event in profile
        if "PagedAttention" in event.get("name", "") and event.get("args", {}).get("provider") == webgpu_provider
    ]
    assert paged_attention_events, "PagedAttention was not assigned to WebGPUExecutionProvider"