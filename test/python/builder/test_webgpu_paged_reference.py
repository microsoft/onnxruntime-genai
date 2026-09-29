# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""WebGPU PagedAttention coverage for exported attention graphs and Qwen 2.5."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import types
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


def _active_cache(paged_cache, length, block_row=None):
    block_size = paged_cache.shape[1]
    block_row = _BLOCK_TABLE[0] if block_row is None else block_row
    blocks = block_row[: (length + block_size - 1) // block_size]
    # [blocks, block_size, heads, head_size] -> [batch, heads, sequence, head_size].
    return paged_cache[blocks].reshape(-1, *paged_cache.shape[2:])[:length].transpose(1, 0, 2)[None]


def _assert_close(webgpu_value, cpu_value, label):
    assert np.isfinite(webgpu_value).all() and np.isfinite(cpu_value).all(), f"Non-finite {label}"
    np.testing.assert_allclose(webgpu_value, cpu_value, rtol=2e-2, atol=2e-2, equal_nan=False, err_msg=label)
    max_error = np.max(np.abs(webgpu_value.astype(np.float32) - cpu_value.astype(np.float32)))
    print(f"{label}: max absolute error {max_error:.6g}")


def _export_attention_graph(tmp_path, monkeypatch, *, drafter):
    # Exercise the actual emitters without loading a checkpoint or executing projections.
    import onnx_ir as ir  # noqa: PLC0415
    import torch  # noqa: PLC0415

    models_dir = _BUILDER_PATH.parent
    monkeypatch.syspath_prepend(str(models_dir))
    package_name = "_webgpu_attention_builders"
    package = types.ModuleType(package_name)
    package.__path__ = [str(models_dir / "builders")]
    monkeypatch.setitem(sys.modules, package_name, package)
    base = importlib.import_module(f"{package_name}.base")
    block = importlib.import_module(f"{package_name}.block_drafter")
    num_heads, kv_heads, head_size = (32, 8, 128) if drafter else (8, 2, 256)
    if drafter:
        cls = importlib.import_module(f"{package_name}.dflash2").DFlash2Builder
        builder = cls.__new__(cls)
    else:
        builder = base.Model.__new__(base.Model)
    block.BlockDrafterBuilder.make_graph(builder, "attention_regression", "regression")
    builder.io_dtype = ir.DataType.FLOAT16
    builder.head_size = head_size
    builder.num_kv_heads = kv_heads
    if drafter:
        builder.num_heads = num_heads
        builder.hidden_size = 1
        builder.paged_block_size = _BLOCK_SIZE
        builder.include_attention_metadata = True
        builder.sliding_window = 2048
        builder.is_causal = False
        builder.rms_eps = 1e-5
        prefix = "layers.0.self_attn"
        builder.weights = {
            f"{prefix}.{projection}_proj.weight": torch.zeros(heads * head_size, 1)
            for projection, heads in (("q", num_heads), ("k", kv_heads), ("v", kv_heads))
        }
        builder.weights[f"{prefix}.o_proj.weight"] = torch.zeros(1, num_heads * head_size)
        for kind in ("q", "k"):
            builder.weights[f"{prefix}.{kind}_norm.weight"] = torch.ones(head_size)
        builder._make_attention(0, "hidden", ("context_key", "context_value"), "num_tokens")
    else:
        builder.num_attn_heads = num_heads
        builder.window_size = -1
        builder.kv_cache_attrs = {"quant_scheme": "none"}
        builder.rope_attrs = {"interleaved": 0}
        builder.attention_attrs = {"scale": head_size**-0.5, "softcap": 0.0, "use_rope_in_attn": 0}
        builder.make_paged_attention(
            "/regression/PagedAttention",
            q_path="query",
            k_path="key",
            v_path="value",
            past_k="past_key_values.0.key",
            past_v="past_key_values.0.value",
            present_k="present.0.key",
            present_v="present.0.value",
            cumulative_sequence_lengths="cumulative_sequence_lengths",
            past_sequence_lengths="past_sequence_lengths",
            block_table="block_table",
            attention_metadata="attention_metadata",
        )
    nodes = [node for node in ir.to_proto(builder.model).graph.node if node.op_type == "PagedAttention"]
    assert len(nodes) == 1
    node = nodes[0]
    attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    assert attributes["num_heads"] == num_heads and attributes["kv_num_heads"] == kv_heads
    if drafter:
        assert attributes["is_causal"] == 0 and attributes["local_window_size"] == 2048
        assert node.input[12] and node.input[13] and attributes["qk_norm_epsilon"] == pytest.approx(1e-5)
    cache_shape = ["num_blocks", _BLOCK_SIZE, kv_heads, head_size]
    shapes = {
        0: ["num_tokens", num_heads * head_size],
        1: ["num_tokens", kv_heads * head_size],
        2: ["num_tokens", kv_heads * head_size],
        3: cache_shape,
        4: cache_shape,
        5: ["batch_plus_one"],
        6: ["batch"],
        7: ["batch", "max_blocks"],
        8: ["max_position", head_size // 2],
        9: ["max_position", head_size // 2],
        12: [head_size],
        13: [head_size],
        16: [3],
    }
    inputs = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.INT32 if index in (5, 6, 7, 16) else onnx.TensorProto.FLOAT16, shapes[index]
        )
        for index, name in enumerate(node.input)
        if name
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, shapes[0] if index == 0 else cache_shape)
        for index, name in enumerate(node.output)
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph([node], "exported_attention", inputs, outputs),
        opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "attention.onnx"
    onnx.save(model, path)
    return path, node


def _profiled_attention_session(model_path, tmp_path):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    webgpu_ep = pytest.importorskip("onnxruntime_ep_webgpu")

    provider = webgpu_ep.get_ep_name()
    if not any(device.ep_name == provider for device in ort.get_ep_devices()):
        ort.register_execution_provider_library(provider, webgpu_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert devices, "Registered WebGPU plugin exposes no OrtEpDevice"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_provider_for_devices([devices[0]], {})
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "attention-profile")
    session = ort.InferenceSession(str(model_path), options, providers=[])
    assert provider in session.get_providers()
    return session, provider


def _assert_attention_profile(session, provider):
    profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    events = [
        event
        for event in profile
        if event.get("args", {}).get("op_name") == "PagedAttention"
        and event.get("args", {}).get("provider") == provider
    ]
    assert len(events) == 2, f"Expected two PagedAttention executions on {provider}, got {len(events)}"


def _assert_cache_guards(before, after, block_table, past_lengths, query_lengths):
    written = np.zeros(before.shape[:2], dtype=bool)
    for blocks, past, query in zip(block_table, past_lengths, query_lengths, strict=True):
        positions = np.arange(past, past + query)
        written[blocks[positions // _BLOCK_SIZE], positions % _BLOCK_SIZE] = True
    # Includes old active cache rows, unused tails, and entirely unallocated blocks.
    np.testing.assert_array_equal(after[~written], before[~written], err_msg="PA overwrote untouched cache slots")


def test_webgpu_paged_attention_head256_reordered_requests(tmp_path, monkeypatch):
    path, node = _export_attention_graph(tmp_path, monkeypatch, drafter=False)
    _, references = _make_cpu_references(path)
    reference = references[0][1]
    session, provider = _profiled_attention_session(path, tmp_path)
    rng = np.random.default_rng(256)
    caches = [rng.normal(offset, 0.4, (_NUM_BLOCKS, _BLOCK_SIZE, 2, 256)).astype(np.float16) for offset in (0.1, -0.3)]
    block_rows = np.asarray([[5, 1], [3, 7]], dtype=np.int32)
    past_lengths = np.asarray([255, 17], dtype=np.int32)
    cpu_caches = [
        [_active_cache(cache, past, row).copy() for cache in caches]
        for past, row in zip(past_lengths, block_rows, strict=True)
    ]
    for step, (order, query_lengths) in enumerate((([0, 1], [3, 1]), ([1, 0], [2, 1]))):
        cumulative = np.asarray([0, *np.cumsum(query_lengths)], dtype=np.int32)
        past = past_lengths[order]
        table = block_rows[order]
        total = past + query_lengths
        qkv = [
            rng.normal(0.2 * (index + step), 0.7, (cumulative[-1], heads * 256)).astype(np.float16)
            for index, heads in enumerate((8, 2, 2))
        ]
        feeds = dict(zip(node.input[:5], [*qkv, *[cache.copy() for cache in caches]], strict=True))
        feeds.update(
            dict(zip(node.input[5:8], [cumulative, past, table], strict=True)),
            attention_metadata=np.asarray([max(query_lengths), max(total), min(total)], dtype=np.int32),
        )
        attention, *next_caches = session.run(None, feeds)
        for row, request in enumerate(order):
            start, end = cumulative[row : row + 2]
            cpu_feeds = dict(
                zip(node.input[:5], [*[value[start:end] for value in qkv], *cpu_caches[request]], strict=True)
            )
            cpu_feeds["seqlens_k"] = np.asarray([total[row] - 1], dtype=np.int32)
            cpu_feeds["total_sequence_length"] = np.asarray([total[row]], dtype=np.int32)
            expected, *cpu_caches[request] = reference.run(None, cpu_feeds)
            _assert_close(attention[start:end], expected, f"step {step} request {request} attention")
            for cache, expected_cache in zip(next_caches, cpu_caches[request], strict=True):
                _assert_close(_active_cache(cache, total[row], table[row]), expected_cache, "request cache")
            past_lengths[request] = total[row]
        for before, after in zip(caches, next_caches, strict=True):
            _assert_cache_guards(before, after, table, past, query_lengths)
        caches = next_caches
    _assert_attention_profile(session, provider)


def _rms_norm(value, weight, epsilon):
    value = value.astype(np.float32)
    return (value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + epsilon) * weight).astype(np.float16)


def _numpy_drafter_attention(query, key, value, past_key, past_value, q_weight, k_weight):
    query = _rms_norm(query, q_weight, 1e-5).astype(np.float32)
    key = np.concatenate((past_key, _rms_norm(key, k_weight, 1e-5)))
    value = np.concatenate((past_value, value))
    # ORT 052bf660f1 flash_attention.wgsl.template uses a query-relative left
    # bound (query_end - window), including for noncausal PA. CPU GQA's schema
    # requires local_window_size == -1 when causal == 0, so it is not this oracle.
    logits = np.einsum("qhd,khd->qhk", query, np.repeat(key.astype(np.float32), 4, axis=1)) / np.sqrt(np.float32(128))
    left = np.maximum(0, len(past_key) + np.arange(len(query)) + 1 - 2048)
    logits = np.where(np.arange(len(key))[None, None, :] >= left[:, None, None], logits, -np.inf)
    probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    output = np.einsum("qhk,khd->qhd", probability, np.repeat(value.astype(np.float32), 4, axis=1))
    return output.reshape(len(query), -1), key, value


def test_webgpu_paged_attention_drafter_window_qk_norm(tmp_path, monkeypatch):
    path, node = _export_attention_graph(tmp_path, monkeypatch, drafter=True)
    session, provider = _profiled_attention_session(path, tmp_path)
    rng = np.random.default_rng(2048)
    table = np.asarray([[9, 2, 7, 0, 10, 5, 1, 8, 4, 6]], dtype=np.int32)
    past = 2303
    q_weight = np.linspace(0.4, 1.8, 128, dtype=np.float16)
    k_weight = np.linspace(1.7, 0.3, 128, dtype=np.float16)
    caches = [rng.normal(offset, 0.5, (12, _BLOCK_SIZE, 8, 128)).astype(np.float16) for offset in (0.1, -0.2)]
    # Expired values make dropping the local window an unmistakable failure.
    caches[1][table[0, 0]] += np.float16(40)
    # High-scoring keys at the moving left edge distinguish a query-relative
    # window from incorrectly anchoring every row's window at the end of the block.
    caches[0][table[0, 1], :3] = _rms_norm(np.ones((3, 8, 128), dtype=np.float16), k_weight, 1e-5)
    caches[1][table[0, 1], :3] = np.asarray([-12, 15, -9], dtype=np.float16)[:, None, None]
    cpu_caches = [_active_cache(cache, past, table[0])[0].transpose(1, 0, 2).copy() for cache in caches]
    for step, count in enumerate((3, 2)):
        qkv = [rng.normal(1.0, 0.15, (count, heads, 128)).astype(np.float16) for heads in (32, 8, 8)]
        qkv[2] = (rng.normal(0, 0.4, qkv[2].shape) + np.arange(count)[:, None, None] * 3 + step).astype(np.float16)
        feeds = dict(
            zip(
                node.input[:5], [*[value.reshape(count, -1) for value in qkv], *[c.copy() for c in caches]], strict=True
            )
        )
        feeds.update(
            dict(
                zip(
                    node.input[5:10],
                    [
                        np.asarray([0, count], dtype=np.int32),
                        np.asarray([past], dtype=np.int32),
                        table,
                        np.ones((2560, 64), dtype=np.float16),
                        np.zeros((2560, 64), dtype=np.float16),
                    ],
                    strict=True,
                )
            )
        )
        feeds[node.input[12]] = q_weight
        feeds[node.input[13]] = k_weight
        feeds[node.input[16]] = np.asarray([count, past + count, past + count], dtype=np.int32)
        attention, *next_caches = session.run(None, feeds)
        expected, *cpu_caches = _numpy_drafter_attention(*qkv, *cpu_caches, q_weight, k_weight)
        _assert_close(attention, expected, f"drafter step {step} attention")
        for before, after, expected_cache in zip(caches, next_caches, cpu_caches, strict=True):
            _assert_close(
                _active_cache(after, past + count, table[0])[0].transpose(1, 0, 2), expected_cache, "drafter cache"
            )
            _assert_cache_guards(before, after, table, [past], [count])
        caches = next_caches
        past += count
    _assert_attention_profile(session, provider)


def test_webgpu_paged_export_runs_prefill_and_decode_with_cpu_reference(tmp_path):
    if not register_webgpu_plugin():
        pytest.skip("onnxruntime-ep-webgpu plugin package is not installed.")
    webgpu_ep = pytest.importorskip("onnxruntime_ep_webgpu")

    webgpu_provider = webgpu_ep.get_ep_name()
    if not any(device.ep_name == webgpu_provider for device in ort.get_ep_devices()):
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
