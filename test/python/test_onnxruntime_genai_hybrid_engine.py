# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for packed paged-plus-fixed Engine state."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from _test_utils import register_plugin_providers, register_webgpu_plugin

_LOG = logging.getLogger(__name__)
register_plugin_providers(_LOG)
_WEBGPU_AVAILABLE = register_webgpu_plugin(_LOG)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-composite"
_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])


class _Sink:
    def __init__(self):
        self.tokens = []


@pytest.fixture(params=_DEVICES)
def model(request):
    config = og.Config(str(_MODEL_DIR))
    config.clear_providers()
    if request.param != "cpu":
        config.append_provider(request.param)
    return og.Model(config)


@pytest.fixture
def webgpu_model(tmp_path):
    if not _WEBGPU_AVAILABLE:
        pytest.skip("WebGPU execution provider plug-in is not installed.")

    model_dir = tmp_path / "synthetic-composite-no-state-updates"
    model_dir.mkdir()
    config_data = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    for group in config_data["model"]["decoder"]["state_groups"]:
        group.pop("state_update", None)
    (model_dir / "genai_config.json").write_text(json.dumps(config_data))

    decoder = onnx.load(_MODEL_DIR / "decoder.onnx")
    logits = next(value for value in decoder.graph.output if value.name == "logits")
    logits.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    inputs = [
        value
        for value in decoder.graph.input
        if value.name != "state_update_capture_count"
    ]
    decoder.graph.ClearField("input")
    decoder.graph.input.extend(inputs)
    onnx.save(decoder, model_dir / "decoder.onnx")

    config = og.Config(str(model_dir))
    config.clear_providers()
    config.append_provider("webgpu")
    return og.Model(config)


def _request(engine, prompt, max_new_tokens, sinks):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks[request] = sink
    request.begin_turn(np.asarray(prompt, dtype=np.int32))
    return request, sink


def _run(engine, sinks):
    steps = 0
    event_buffer = engine.create_event_buffer(8)
    while engine.has_pending_requests():
        for event in engine.run(event_buffer):
            sink = sinks[event.request]
            if event.flags & og.EngineEventFlags.TOKEN:
                sink.tokens.append(event.token)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                event.request.close()
        steps += 1
        assert steps < 100


def test_fixture_declares_sparse_paged_and_fixed_groups():
    config = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    groups = config["model"]["decoder"]["state_groups"]
    assert config["model"]["decoder"]["inputs"]["position_ids"] == "position_ids"
    assert [group["kind"] for group in groups] == ["fixed_conv", "paged_kv", "fixed_recurrent"]
    assert [group["layer_ids"] for group in groups] == [[0, 3], [1, 4], [2, 5]]


@pytest.fixture(params=_DEVICES + (["cuda_graph"] if og.is_cuda_available() else []))
def external_engram_model(tmp_path, request):
    config_data = json.loads((_MODEL_DIR / "genai_config.json").read_text())
    vocab_size = config_data["model"]["vocab_size"]
    config_data["model"]["decoder"]["inputs"]["engram_embeddings"] = "engram_embeddings"
    config_data["model"]["decoder"]["ple_token_pad_id"] = 0
    config_data["model"]["engram"] = {
        "filename": "engram.onnx",
        "session_options": {"provider_options": [{"cpu": {}}]},
        "inputs": {"input_ids": "input_ids", "past_tokens": "past_ple_tokens"},
        "outputs": {"embeddings": "engram_embeddings", "present_tokens": "present_ple_tokens"},
    }
    decoder = onnx.load(_MODEL_DIR / "decoder.onnx")
    logits = next(value for value in decoder.graph.output if value.name == "logits")
    logits.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    decoder_nodes = []
    for node in decoder.graph.node:
        if node.op_type == "Range" and list(node.output) == ["token_index"]:
            input_ids = config_data["model"]["decoder"]["inputs"]["input_ids"]
            decoder_nodes.extend([
                onnx.helper.make_node("Equal", [input_ids, input_ids], ["token_index_mask"]),
                onnx.helper.make_node("Cast", ["token_index_mask"], ["token_index_ones"], to=onnx.TensorProto.INT64),
                onnx.helper.make_node("CumSum", ["token_index_ones", node.input[0]], ["token_index_offsets"]),
                onnx.helper.make_node("Sub", ["token_index_offsets", node.input[2]], ["token_index"]),
            ])
            continue
        for index, output in enumerate(node.output):
            if output == "logits":
                node.output[index] = "original_logits"
        decoder_nodes.append(node)
    decoder.graph.ClearField("node")
    decoder.graph.node.extend(decoder_nodes)
    decoder.graph.input.extend([
        onnx.helper.make_tensor_value_info("engram_embeddings", onnx.TensorProto.FLOAT, ["num_tokens", 1]),
        onnx.helper.make_tensor_value_info("present_ple_tokens", onnx.TensorProto.INT64, ["batch_size", 2]),
    ])
    constants = {
        "engram_slice_start": np.array([1], dtype=np.int64),
        "engram_slice_end": np.array([2**63 - 1], dtype=np.int64),
        "engram_axis": np.array([1], dtype=np.int64),
        "engram_one": np.array(1, dtype=np.int32),
        "engram_vocab": np.array(vocab_size, dtype=np.float32),
        "engram_classes": np.arange(vocab_size, dtype=np.float32),
    }
    decoder.graph.initializer.extend(onnx.numpy_helper.from_array(value, name) for name, value in constants.items())
    cumulative = config_data["model"]["decoder"]["inputs"]["cumulative_sequence_lengths"]
    decoder.graph.node.extend([
        onnx.helper.make_node("Slice", [cumulative, "engram_slice_start", "engram_slice_end"], ["engram_ends"]),
        onnx.helper.make_node("Sub", ["engram_ends", "engram_one"], ["engram_last_rows"]),
        onnx.helper.make_node("Gather", ["engram_embeddings", "engram_last_rows"], ["engram_last"], axis=0),
        onnx.helper.make_node("Squeeze", ["engram_last", "engram_axis"], ["engram_last_ids"]),
        onnx.helper.make_node("Cast", ["present_ple_tokens"], ["engram_tokens_float"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("ReduceSum", ["engram_tokens_float", "engram_axis"], ["engram_history_float"], keepdims=0),
        onnx.helper.make_node("Add", ["engram_last_ids", "engram_history_float"], ["engram_score"]),
        onnx.helper.make_node("Mod", ["engram_score", "engram_vocab"], ["engram_label"], fmod=1),
        onnx.helper.make_node("Unsqueeze", ["engram_label", "engram_axis"], ["engram_label_column"]),
        onnx.helper.make_node("Equal", ["engram_label_column", "engram_classes"], ["engram_selected"]),
        onnx.helper.make_node("Cast", ["engram_selected"], ["logits"], to=onnx.TensorProto.FLOAT16),
    ])
    required_values = {value.name for value in decoder.graph.output}
    live_nodes = []
    for node in reversed(decoder.graph.node):
        if required_values.intersection(node.output):
            live_nodes.append(node)
            required_values.update(node.input)
    decoder.graph.ClearField("node")
    decoder.graph.node.extend(reversed(live_nodes))
    onnx.checker.check_model(decoder)
    onnx.save(decoder, tmp_path / "decoder.onnx")

    engram = onnx.helper.make_graph([
        onnx.helper.make_node("Cast", ["input_ids"], ["ids_float"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Unsqueeze", ["ids_float", "embedding_axis"], ["engram_embeddings"]),
        onnx.helper.make_node("Concat", ["past_ple_tokens", "input_ids"], ["token_history"], axis=1),
        onnx.helper.make_node("Slice", ["token_history", "tail_start", "tail_end", "history_axis"], ["present_ple_tokens"]),
    ], "external_engram", [
        onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["batch_size", "sequence_length"]),
        onnx.helper.make_tensor_value_info("past_ple_tokens", onnx.TensorProto.INT64, ["batch_size", 2]),
    ], [
        onnx.helper.make_tensor_value_info("engram_embeddings", onnx.TensorProto.FLOAT, ["batch_size", "sequence_length", 1]),
        onnx.helper.make_tensor_value_info("present_ple_tokens", onnx.TensorProto.INT64, ["batch_size", 2]),
    ], [
        onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "embedding_axis"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "history_axis"),
        onnx.numpy_helper.from_array(np.array([-2], dtype=np.int64), "tail_start"),
        onnx.numpy_helper.from_array(np.array([2**63 - 1], dtype=np.int64), "tail_end"),
    ])
    engram_model = onnx.helper.make_model(engram, opset_imports=[onnx.helper.make_opsetid("", 21)], ir_version=10)
    onnx.checker.check_model(engram_model)
    onnx.save(engram_model, tmp_path / "engram.onnx")
    (tmp_path / "genai_config.json").write_text(json.dumps(config_data))
    config = og.Config(str(tmp_path))
    config.clear_providers()
    if request.param != "cpu":
        config.append_provider("cuda")
        config.set_provider_option("cuda", "enable_cuda_graph", "1" if request.param == "cuda_graph" else "0")
    return og.Model(config)


def test_external_engram_outputs_drive_batched_decoding(external_engram_model):
    engine = og.Engine(external_engram_model)
    sinks = {}
    requests = [_request(engine, prompt, 10, sinks) for prompt in ([2, 3, 4], [7])]

    _run(engine, sinks)

    assert [sink.tokens for _, sink in requests] == [
        [11, 26, 63, 24, 47, 54, 27, 44, 51, 18],
        [14, 35, 20, 11, 42, 31, 40, 47, 6, 59],
    ]


def test_mixed_unequal_requests_match_isolated_execution(model):
    prompts = [[2, 3, 4], [7], [9, 10]]
    max_new_tokens = 3

    expected = []
    for prompt in prompts:
        engine = og.Engine(model)
        sinks = {}
        _, sink = _request(engine, prompt, max_new_tokens, sinks)
        _run(engine, sinks)
        expected.append(sink.tokens)
    # The first request's fixed convolution state contributes 0, 6, then 12 to successive scores.
    # Without fixed output binding/commit/re-gather this would be [9, 15, 22].
    assert expected[0] == [9, 21, 40]

    engine = og.Engine(model)
    sinks = {}
    requests = [_request(engine, prompt, max_new_tokens, sinks) for prompt in prompts]
    _run(engine, sinks)
    assert [sink.tokens for _, sink in requests] == expected


def test_webgpu_fixed_state_staging_persists_across_steps(webgpu_model):
    engine = og.Engine(webgpu_model)
    sinks = {}
    _, sink = _request(engine, [2, 3, 4], 2, sinks)

    _run(engine, sinks)

    # The first fixed-convolution output commits six 1s. Gathering that state on the
    # second step changes the second token from the stateless value 15 to 21.
    assert sink.tokens == [9, 21]
