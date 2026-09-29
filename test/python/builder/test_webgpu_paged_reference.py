# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""End-to-end WebGPU PagedAttention export coverage for Qwen 2.5 0.5B."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from _test_utils import register_webgpu_plugin

_NUM_BLOCKS = 8
_BLOCK_SIZE = 256
_BLOCK_TABLE = np.asarray([[3]], dtype=np.int32)
_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_PREFILL_TOKENS = np.asarray([1, 2, 3], dtype=np.int64)
_DECODE_TOKENS = np.asarray([4], dtype=np.int64)
_BUILDER_PATH = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builder.py"


def _cache_shape(node_arg, block_size):
    shape = node_arg.shape
    assert len(shape) == 4, f"Unexpected paged KV-cache shape for {node_arg.name}: {shape}"
    assert not isinstance(shape[1], int) or shape[1] == block_size, (
        f"Unexpected PagedAttention block size for {node_arg.name}: {shape}"
    )
    assert isinstance(shape[2], int) and isinstance(shape[3], int), (
        f"Expected concrete KV head dimensions for {node_arg.name}: {shape}"
    )
    return (_NUM_BLOCKS, block_size, shape[2], shape[3])


def _make_inputs(tokens, past_length, caches):
    inputs = {
        "input_ids": tokens,
        "cumulative_sequence_lengths": np.asarray([0, len(tokens)], dtype=np.int32),
        "past_sequence_lengths": np.asarray([past_length], dtype=np.int32),
        "block_table": _BLOCK_TABLE,
        "attention_metadata": np.asarray(
            [len(tokens), past_length + len(tokens), past_length + len(tokens)], dtype=np.int32
        ),
    }
    inputs.update(caches)
    return inputs


def _run_step(session, tokens, past_length, caches):
    outputs = session.run(None, _make_inputs(tokens, past_length, caches))
    assert all(np.isfinite(output).all() for output in outputs), "Non-finite logits or KV cache"
    output_names = [node_arg.name for node_arg in session.get_outputs()]
    output_by_name = dict(zip(output_names, outputs, strict=True))
    next_caches = {
        input_name: output_by_name[input_name.replace("past_key_values", "present")]
        for input_name in caches
        if input_name.replace("past_key_values", "present") in output_by_name
    }
    assert next_caches.keys() == caches.keys(), "Model did not return every KV cache"
    return output_by_name, next_caches


def _make_cpu_references(model_path):
    # Capture the exported attention boundaries so both kernels consume identical Q/K/V.
    model = onnx.load(model_path, load_external_data=False)
    values = {value.name: value for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    values.update(
        {
            tensor.name: onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
            for tensor in model.graph.initializer
        }
    )
    output_names = {value.name for value in model.graph.output}
    references = []
    axes = "cpu_reference_batch_axis"
    for node in model.graph.node:
        if node.op_type != "PagedAttention":
            continue
        assert node.domain == "com.microsoft"
        attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        attributes["causal"] = attributes.pop("is_causal", 1)
        assert attributes.keys() <= {
            "num_heads",
            "kv_num_heads",
            "scale",
            "local_window_size",
            "softcap",
            "do_rotary",
            "rotary_interleaved",
            "qk_norm_epsilon",
            "causal",
        }, f"Unsupported CPU reference attributes: {attributes}"
        inputs = list(node.input) + [""] * (17 - len(node.input))
        assert not inputs[10], "The CPU reference does not implement slot_mapping"
        nodes = []
        qkv = []
        for name in inputs[:3]:
            batched = f"{name}/cpu_batched" if name else ""
            if name:
                nodes.append(onnx.helper.make_node("Unsqueeze", [name, axes], [batched]))
            qkv.append(batched)
        batched_output = f"{node.output[0]}/cpu_batched"
        nodes.append(
            onnx.helper.make_node(
                "GroupQueryAttention",
                [
                    *qkv,
                    *inputs[3:5],
                    "seqlens_k",
                    "total_sequence_length",
                    *inputs[8:10],
                    "",
                    "",
                    inputs[11],
                    *inputs[14:16],
                    *inputs[12:14],
                ],
                [batched_output, *node.output[1:]],
                name=f"{node.name}/cpu_reference",
                domain="com.microsoft",
                **attributes,
            )
        )
        nodes.append(onnx.helper.make_node("Squeeze", [batched_output, axes], [node.output[0]]))
        reference_input_names = [name for name in [*inputs[:5], *inputs[8:10], *inputs[11:16]] if name]
        reference_inputs = [copy.deepcopy(values[name]) for name in reference_input_names]
        reference_inputs.extend(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT32, [1])
            for name in ("seqlens_k", "total_sequence_length")
        )
        reference_outputs = [copy.deepcopy(values[name]) for name in node.output]
        cache_names = {*inputs[3:5], *node.output[1:]}
        for value in (*reference_inputs, *reference_outputs):
            if value.name not in cache_names:
                continue
            shape = value.type.tensor_type.shape
            heads, head_size = shape.dim[2].dim_value, shape.dim[3].dim_value
            assert heads > 0 and head_size > 0
            shape.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    value.name, onnx.TensorProto.FLOAT16, [1, heads, "cache_length", head_size]
                ).type.tensor_type.shape
            )
        reference = onnx.helper.make_model(
            onnx.helper.make_graph(
                nodes,
                f"{node.name}/cpu_reference",
                reference_inputs,
                reference_outputs,
                [onnx.helper.make_tensor(axes, onnx.TensorProto.INT64, [1], [0])],
            ),
            opset_imports=model.opset_import,
            ir_version=model.ir_version,
        )
        onnx.checker.check_model(reference)
        session = ort.InferenceSession(reference.SerializeToString(), providers=["CPUExecutionProvider"])
        references.append((node, session))
        for name in (*reference_input_names, node.output[0]):
            if name not in cache_names and name not in output_names:
                model.graph.output.append(values[name])
                output_names.add(name)
    assert references, "Exported graph has no PagedAttention nodes"
    capture_path = model_path.with_name("attention-outputs.onnx")
    onnx.save(model, capture_path)
    return capture_path, references


def _active_cache(paged_cache, length):
    block_size = paged_cache.shape[1]
    blocks = _BLOCK_TABLE[0, : (length + block_size - 1) // block_size]
    # [blocks, block_size, heads, head_size] -> [batch, heads, sequence, head_size].
    return paged_cache[blocks].reshape(-1, *paged_cache.shape[2:])[:length].transpose(1, 0, 2)[None]


def _assert_close(webgpu_value, cpu_value, label):
    assert np.isfinite(webgpu_value).all() and np.isfinite(cpu_value).all(), f"Non-finite {label}"
    np.testing.assert_allclose(webgpu_value, cpu_value, rtol=2e-2, atol=2e-2, equal_nan=False, err_msg=label)
    max_error = np.max(np.abs(webgpu_value.astype(np.float32) - cpu_value.astype(np.float32)))
    print(f"{label}: max absolute error {max_error:.6g}")


def test_webgpu_paged_export_runs_prefill_and_decode_with_cpu_reference(tmp_path):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    webgpu_ep = pytest.importorskip("onnxruntime_ep_webgpu")

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
            f"num_blocks={_NUM_BLOCKS}",
            f"paged_block_size={_BLOCK_SIZE}",
            "num_hidden_layers=2",
            "hf_token=false",
        ],
        check=True,
    )

    config = json.loads((output_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert config["engine"]["dynamic_batching"]["num_blocks"] == _NUM_BLOCKS
    assert config["model"]["decoder"]["inputs"]["attention_metadata"] == "attention_metadata"
    block_size = config["engine"]["dynamic_batching"]["block_size"]
    assert block_size == _BLOCK_SIZE

    model_path = output_dir / config["model"]["decoder"]["filename"]
    capture_path, references = _make_cpu_references(model_path)
    devices = [device for device in ort.get_ep_devices() if device.ep_name == webgpu_provider]
    assert devices, "Registered WebGPU plugin exposes no devices"
    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices([devices[0]], {})
    capture_session = ort.InferenceSession(str(capture_path), session_options, providers=[])
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(tmp_path / "webgpu-profile")
    webgpu_session = ort.InferenceSession(str(model_path), session_options, providers=[])
    assert webgpu_provider in webgpu_session.get_providers(), "WebGPU was not selected for the session"
    assert webgpu_provider in capture_session.get_providers(), "WebGPU was not selected for attention capture"
    print(f"Selected {devices[0].ep_name}, device {devices[0].device.device_id}")

    cache_inputs = {
        node_arg.name: np.zeros(_cache_shape(node_arg, block_size), dtype=np.float16)
        for node_arg in webgpu_session.get_inputs()
        if node_arg.name.startswith("past_key_values.")
    }
    assert cache_inputs, "Exported model has no paged KV-cache inputs"
    cpu_cache_inputs = {
        name: np.zeros((1, cache.shape[2], 0, cache.shape[3]), dtype=cache.dtype)
        for name, cache in cache_inputs.items()
    }

    webgpu_caches = cache_inputs
    capture_caches = cache_inputs
    for label, tokens, past_length in (
        ("prefill", _PREFILL_TOKENS, 0),
        ("decode", _DECODE_TOKENS, len(_PREFILL_TOKENS)),
    ):
        model_outputs, webgpu_caches = _run_step(webgpu_session, tokens, past_length, webgpu_caches)
        outputs, capture_caches = _run_step(capture_session, tokens, past_length, capture_caches)
        _assert_close(model_outputs["logits"], outputs["logits"], f"{label} original versus capture logits")
        for name in cache_inputs:
            _assert_close(webgpu_caches[name], capture_caches[name], f"{label} original versus capture {name}")
        total_length = past_length + len(tokens)
        for node, reference in references:
            reference_inputs = {
                arg.name: cpu_cache_inputs[arg.name] if arg.name in cpu_cache_inputs else outputs[arg.name]
                for arg in reference.get_inputs()
                if arg.name not in ("seqlens_k", "total_sequence_length")
            }
            reference_inputs["seqlens_k"] = np.asarray([total_length - 1], dtype=np.int32)
            reference_inputs["total_sequence_length"] = np.asarray([total_length], dtype=np.int32)
            attention, key, value = reference.run(None, reference_inputs)
            _assert_close(outputs[node.output[0]], attention, f"{label} attention {node.name}")
            for name, cache in zip(node.input[3:5], (key, value), strict=True):
                _assert_close(_active_cache(webgpu_caches[name], total_length), cache, f"{label} cache {name}")
                cpu_cache_inputs[name] = cache

    with open(webgpu_session.end_profiling(), encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    paged_attention_events = [
        event
        for event in profile
        if "PagedAttention" in event.get("name", "") and event.get("args", {}).get("provider") == webgpu_provider
    ]
    assert paged_attention_events, "PagedAttention was not assigned to WebGPUExecutionProvider"
    assert len(paged_attention_events) == 4, "Expected two PagedAttention layers for both prefill and decode"
    print(
        f"Verified {len(paged_attention_events)} WebGPU PagedAttention executions; attention and caches match CPU GQA"
    )
