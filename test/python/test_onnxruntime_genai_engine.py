# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Deterministic integration tests for the paged Engine path."""

from __future__ import annotations

import gc
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from _test_utils import register_plugin_providers
from onnxruntime.capi import _pybind_state

register_plugin_providers(logging.getLogger(__name__))

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged"
_DRAFT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged-per-token"
_SELECTED_LOGITS_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged-selected-logits"
)

_VOCAB_SIZE = 64
_BLOCK_SIZE = 4
_EOS_TOKEN_ID = 1

_MAX_STEPS = 10_000

_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])

_PROMPT_A = [5, 9, 13]
_PROMPT_B = [7, 2, 20, 4]
_PROMPT_LONG = [3, 8, 2, 15, 6, 11]


def _has_indexer_merge():
    return any(schema.name == "PackedSparseAttentionIndexerMerge" for schema in _pybind_state.get_all_operator_schema())


def _make_indexshare_mtp_model(
    root, enabled=True, draft_count=7, merge_capacity=8, capture=False, persistent_state=False
):
    helper = onnx.helper
    tensor = onnx.TensorProto
    target = onnx.load(_DRAFT_MODEL_DIR / "decoder.onnx")
    for value in list(target.graph.input) + list(target.graph.output):
        if value.name.startswith(("past_key_values.", "present.")):
            dimension = value.type.tensor_type.shape.dim[0]
            dimension.ClearField("dim_value")
            dimension.dim_param = "num_blocks"
    next(value for value in target.graph.initializer if value.name == "cache_shape").CopyFrom(
        helper.make_tensor("cache_shape", tensor.INT64, [4], [-1, 4, 1, 1])
    )
    del target.graph.value_info[:]
    onnx.save_model(target, root / "decoder.onnx")
    config = json.loads((_DRAFT_MODEL_DIR / "genai_config.json").read_text())
    config["model"]["decoder"]["session_options"]["provider_options"] = [{"cuda": {}}]
    config["speculative"] = {"max_draft_tokens": draft_count}
    config["model"]["mtp"] = {
        "filename": "mtp.onnx",
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_size": 1,
        "main_hidden_states": "hidden_states",
        "inputs": {"hidden_states": "hidden_states"},
        "outputs": {"hidden_states": "hidden_states_out"},
        "session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "1" if capture else "0"}}]},
        "index_share": {
            "enabled": enabled,
            "base_capacity": 3,
            "max_draft_tokens": 7,
            "indices_output": "indexshare.present_indices",
            "counts_output": "indexshare.present_counts",
        },
    }
    logits = np.zeros((_VOCAB_SIZE, _VOCAB_SIZE), dtype=np.float16)
    logits[:, 17] = 10
    initializers = [
        onnx.numpy_helper.from_array(logits, "constant_logits"),
        helper.make_tensor("first_column", tensor.INT64, [1], [0]),
        helper.make_tensor("squeeze_axis", tensor.INT64, [1], [1]),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", tensor.INT64, ["num_tokens"]),
        helper.make_tensor_value_info("hidden_states", tensor.FLOAT16, ["num_tokens", 1]),
        helper.make_tensor_value_info("block_table", tensor.INT32, ["batch_size", "columns"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", tensor.INT32, ["batch_plus_one"]),
        helper.make_tensor_value_info("past_sequence_lengths", tensor.INT32, ["num_tokens"]),
        helper.make_tensor_value_info("attention_metadata", tensor.INT32, [3]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", tensor.FLOAT16, ["num_tokens", _VOCAB_SIZE]),
        helper.make_tensor_value_info("hidden_states_out", tensor.FLOAT16, ["num_tokens", 1]),
    ]
    nodes = [
        helper.make_node("Gather", ["constant_logits", "input_ids"], ["raw_logits"], axis=0),
        helper.make_node("Gather", ["cumulative_sequence_lengths", "first_column"], ["first_offset"], axis=0),
        helper.make_node("Gather", ["past_sequence_lengths", "first_column"], ["first_past"], axis=0),
        helper.make_node("Gather", ["block_table", "first_column"], ["first_block_column"], axis=1),
        helper.make_node("Gather", ["first_block_column", "first_column"], ["first_block"], axis=0),
        helper.make_node("Squeeze", ["first_block", "squeeze_axis"], ["block_scalar"]),
        helper.make_node("Gather", ["attention_metadata", "first_column"], ["first_metadata"], axis=0),
        helper.make_node("Sub", ["block_scalar", "block_scalar"], ["block_zero"]),
        helper.make_node("Sub", ["first_metadata", "first_metadata"], ["attention_zero"]),
        helper.make_node("Sub", ["first_past", "first_past"], ["past_zero"]),
        helper.make_node("Add", ["first_offset", "past_zero"], ["sequence_zero"]),
        helper.make_node("Add", ["block_zero", "attention_zero"], ["cache_zero"]),
        helper.make_node("Add", ["sequence_zero", "cache_zero"], ["metadata_zero"]),
        helper.make_node("Cast", ["metadata_zero"], ["logit_zero"], to=tensor.FLOAT16),
        helper.make_node("Add", ["raw_logits", "logit_zero"], ["logits"]),
        helper.make_node("Identity", ["hidden_states"], ["hidden_states_out"]),
    ]
    for kind in ("key", "value"):
        name = f"past_key_values.0.{kind}"
        present = f"present.0.{kind}"
        inputs.append(helper.make_tensor_value_info(name, tensor.FLOAT16, ["num_blocks", 4, 1, 1]))
        outputs.append(helper.make_tensor_value_info(present, tensor.FLOAT16, ["num_blocks", 4, 1, 1]))
        nodes.append(helper.make_node("Identity", [name], [present]))

    def save(filename, graph_nodes, graph_inputs, graph_outputs):
        model = helper.make_model(
            helper.make_graph(graph_nodes, filename, graph_inputs, graph_outputs, initializers),
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        onnx.save_model(model, root / filename)

    packed_inputs = [
        *inputs,
        helper.make_tensor_value_info("indexshare.mode", tensor.INT32, [1]),
        helper.make_tensor_value_info("indexshare.projection_rows", tensor.INT64, ["projection_rows"]),
        helper.make_tensor_value_info("indexshare.past_indices", tensor.INT32, ["rows", 3]),
        helper.make_tensor_value_info("indexshare.past_counts", tensor.INT32, ["rows"]),
        helper.make_tensor_value_info("indexshare.base_row_indices", tensor.INT32, ["merge_rows"]),
        helper.make_tensor_value_info("indexshare.range_starts", tensor.INT32, ["merge_rows"]),
        helper.make_tensor_value_info("indexshare.range_ends", tensor.INT32, ["merge_rows"]),
    ]
    initializers.extend(
        [
            onnx.numpy_helper.from_array(np.zeros((1, 4), dtype=np.float16), "index_weight"),
            onnx.numpy_helper.from_array(np.ones(2, dtype=np.float16), "index_norm"),
            onnx.numpy_helper.from_array(np.ones((64, 2), dtype=np.float16), "index_cos"),
            onnx.numpy_helper.from_array(np.zeros((64, 2), dtype=np.float16), "index_sin"),
            helper.make_tensor("key_geometry", tensor.INT64, [2], [4, 2]),
            helper.make_tensor("buffer_geometry", tensor.INT64, [2], [3, 2]),
            helper.make_tensor("length_geometry", tensor.INT64, [1], [2]),
            helper.make_tensor("range_extra", tensor.INT32, [], [0 if merge_capacity == 8 else 64]),
        ]
    )
    packed_nodes = [
        *nodes,
        helper.make_node("Gather", ["hidden_states", "indexshare.projection_rows"], ["projected_hidden"], axis=0),
        helper.make_node("MatMul", ["projected_hidden", "index_weight"], ["packed_qk"]),
        helper.make_node("Shape", ["past_sequence_lengths"], ["batch_shape"]),
        helper.make_node("Concat", ["batch_shape", "key_geometry"], ["key_shape"], axis=0),
        helper.make_node("Concat", ["batch_shape", "buffer_geometry"], ["buffer_shape"], axis=0),
        helper.make_node("Concat", ["batch_shape", "length_geometry"], ["length_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["key_shape"],
            ["key_state"],
            value=onnx.numpy_helper.from_array(np.zeros(1, dtype=np.float16)),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["buffer_shape"],
            ["kv_buffer"],
            value=onnx.numpy_helper.from_array(np.zeros(1, dtype=np.float16)),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["length_shape"],
            ["state_lengths"],
            value=helper.make_tensor("", tensor.INT32, [1], [0]),
        ),
        helper.make_node("Add", ["indexshare.range_ends", "range_extra"], ["merge_ends"]),
        helper.make_node(
            "PackedSparseAttentionIndexerMerge",
            [
                "indexshare.past_indices",
                "indexshare.past_counts",
                "indexshare.base_row_indices",
                "indexshare.range_starts",
                "merge_ends",
            ],
            ["merged", "merged_count", "merge_status"],
            domain="com.microsoft",
            policy_mode="append_range",
            max_output_entries=9,
        ),
        helper.make_node(
            "PackedSparseAttentionIndexer",
            [
                "packed_qk",
                "",
                "index_norm",
                "index_norm",
                "index_cos",
                "index_sin",
                "cumulative_sequence_lengths",
                "past_sequence_lengths",
                "",
                "",
                "",
                "",
                "key_state",
                "kv_buffer",
                "",
                "state_lengths",
                "",
                "",
                "indexshare.mode",
                "merged",
                "merged_count",
                "merge_status",
            ],
            [
                "indexshare.present_indices",
                "indexshare.present_counts",
                "present_index_keys",
                "present_index_buffer",
                "",
                "present_index_lengths",
                "",
                "indexshare.status",
            ],
            domain="com.microsoft",
            policy_mode="qsa",
            compress_ratio=2,
            state_capacity=4,
            token_budget=2,
            max_output_entries=9,
        ),
    ]
    packed_outputs = [
        *outputs,
        helper.make_tensor_value_info("indexshare.present_indices", tensor.INT32, ["num_tokens", 9]),
        helper.make_tensor_value_info("indexshare.present_counts", tensor.INT32, ["num_tokens"]),
        helper.make_tensor_value_info("indexshare.status", tensor.INT32, ["num_tokens"]),
    ]
    if persistent_state:
        state_names = {
            "key_state": ("indexer_key", tensor.FLOAT16, [4, 2]),
            "kv_buffer": ("indexer_kv_buffer", tensor.FLOAT16, [3, 2]),
            "state_lengths": ("indexer_state_lengths", tensor.INT32, [2]),
        }
        state_outputs = {
            "key_state": "present_index_keys",
            "kv_buffer": "present_index_buffer",
            "state_lengths": "present_index_lengths",
        }
        renames = {}
        for name, (suffix, dtype, dimensions) in state_names.items():
            past, present = f"past.0.{suffix}", f"present.0.{suffix}"
            shape = ["batch_size", *dimensions]
            past_info = helper.make_tensor_value_info(past, dtype, shape)
            present_info = helper.make_tensor_value_info(present, dtype, shape)
            packed_inputs.append(past_info)
            packed_outputs.append(present_info)
            target.graph.input.append(past_info)
            target.graph.output.append(present_info)
            target.graph.node.append(helper.make_node("Identity", [past], [present]))
            renames[name] = past
            renames[state_outputs[name]] = present
        packed_nodes = [
            node for node in packed_nodes if not (node.op_type == "ConstantOfShape" and node.output[0] in state_names)
        ]
        for node in packed_nodes:
            for names in (node.input, node.output):
                for index, name in enumerate(names):
                    names[index] = renames.get(name, name)
        decoder = config["model"]["decoder"]
        for field, name in (("inputs", "past"), ("outputs", "present")):
            bindings = decoder.setdefault(field, {})
            for key in state_names:
                suffix = state_names[key][0]
                config_field = {
                    "key_state": "indexer_names",
                    "kv_buffer": "indexer_kv_buffer_names",
                    "state_lengths": "indexer_state_lengths_names",
                }[key]
                bindings[f"{name}_{config_field}"] = f"{name}.%d.{suffix}"
        decoder.setdefault("state_groups", []).append(
            {
                "kind": "fixed_indexer",
                "layer_ids": [0],
                "state_update": {"capacity": 7, "compress_ratio": 2},
            }
        )
        decoder["state_update_capacity"] = 7
        decoder["inputs"]["state_update_capture_count"] = "state_update_capture_count"
        decoder["inputs"]["state_update_active"] = "state_update_active"
        decoder["outputs"]["state_update_indexer_names"] = "state_update.%d.indexer"
        config["model"]["mtp"]["inputs"]["past_indexer_names"] = "past.%d.indexer_key"
        config["model"]["mtp"]["outputs"]["present_indexer_names"] = "present.%d.indexer_key"
        capture_input = helper.make_tensor_value_info("state_update_capture_count", tensor.INT32, ["batch_size"])
        active_input = helper.make_tensor_value_info("state_update_active", tensor.INT32, [1])
        snapshot_output = helper.make_tensor_value_info("state_update.0.indexer", tensor.FLOAT16, ["batch_size", 7, 2])
        target.graph.input.extend([capture_input, active_input])
        target.graph.output.append(snapshot_output)
        target.graph.initializer.extend(
            [
                helper.make_tensor("indexer_batch_axis", tensor.INT64, [1], [0]),
                helper.make_tensor("snapshot_geometry", tensor.INT64, [2], [7, 2]),
            ]
        )
        target.graph.node.extend(
            [
                helper.make_node("Shape", ["past.0.indexer_key"], ["indexer_key_shape"]),
                helper.make_node("Gather", ["indexer_key_shape", "indexer_batch_axis"], ["indexer_batch"], axis=0),
                helper.make_node("Concat", ["indexer_batch", "snapshot_geometry"], ["snapshot_shape"], axis=0),
                helper.make_node(
                    "ConstantOfShape",
                    ["snapshot_shape"],
                    ["state_update.0.indexer"],
                    value=onnx.numpy_helper.from_array(np.zeros(1, dtype=np.float16)),
                ),
            ]
        )
        onnx.save_model(target, root / "decoder.onnx")
        packed_inputs.extend([capture_input, active_input])
        packed_outputs.append(snapshot_output)
        indexer = next(node for node in packed_nodes if node.op_type == "PackedSparseAttentionIndexer")
        indexer.input[16:18] = ["state_update_capture_count", "state_update_active"]
        indexer.output[6] = "state_update.0.indexer"
        indexer.attribute.append(helper.make_attribute("state_update_capacity", 7))
    save("mtp.onnx", packed_nodes, packed_inputs, packed_outputs)
    (root / "genai_config.json").write_text(json.dumps(config))
    return og.Model(str(root))


@pytest.mark.skipif(not og.is_cuda_available(), reason="Requires CUDA")
@pytest.mark.parametrize("draft_count", range(1, 8))
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_indexshare_single_mtp_default_all_budgets(tmp_path, draft_count, capture, enabled):
    if not _has_indexer_merge():
        pytest.skip("Requires an ORT runtime with PackedSparseAttentionIndexerMerge")
    model = _make_indexshare_mtp_model(tmp_path, draft_count=draft_count, capture=capture, enabled=enabled)
    assert {path.name for path in tmp_path.glob("mtp*.onnx")} == {"mtp.onnx"}
    graph = onnx.load(tmp_path / "mtp.onnx")
    assert not any(node.op_type == "If" for node in graph.graph.node)
    assert sum(node.op_type == "PackedSparseAttentionIndexer" for node in graph.graph.node) == 1
    assert not any(node.op_type == "Split" for node in graph.graph.node)
    engine = og.Engine(model)
    sinks = {}
    first, second = _Sink(), _Sink()
    prompt_a = [3, 8, 4]
    _create_request(engine, prompt_a, 12, first, sinks)
    _create_request(engine, _PROMPT_B, 16, second, sinks)
    _run(engine, sinks)
    assert first.tokens == predicted_tokens(prompt_a, 12)
    assert second.tokens == predicted_tokens(_PROMPT_B, 16)
    stats = engine.get_speculative_stats()
    assert stats["draft_forward_passes"] > 0
    assert stats["mtp_failures"] == 0
    if draft_count > 1:
        assert stats["partial_accept_rounds"] > 0


@pytest.mark.skipif(not og.is_cuda_available(), reason="Requires CUDA")
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_indexshare_persistent_state_and_shrinking_batch(tmp_path, capture, enabled):
    if not _has_indexer_merge():
        pytest.skip("Requires IndexShare operators")
    model = _make_indexshare_mtp_model(tmp_path, enabled=enabled, capture=capture, persistent_state=True)
    engine = og.Engine(model)
    sinks = {}
    first, second = _Sink(), _Sink()
    _create_request(engine, _PROMPT_A, 6, first, sinks)
    _create_request(engine, _PROMPT_B, 16, second, sinks)
    _run(engine, sinks)
    assert first.tokens == predicted_tokens(_PROMPT_A, 6)
    assert second.tokens == predicted_tokens(_PROMPT_B, 16)
    assert engine.get_speculative_stats()["mtp_failures"] == 0


@pytest.mark.skipif(not og.is_cuda_available(), reason="Requires CUDA")
def test_indexshare_status_rejects_chain_before_publication(tmp_path):
    if not _has_indexer_merge():
        pytest.skip("Requires an ORT runtime with PackedSparseAttentionIndexerMerge")
    model = _make_indexshare_mtp_model(tmp_path, True, merge_capacity=1)
    engine = og.Engine(model)
    sink = _Sink()
    sinks = {}
    _create_request(engine, _PROMPT_A, 12, sink, sinks)
    try:
        _run(engine, sinks)
    except RuntimeError as error:
        assert "IndexShare indexer merge failed" in str(error)
    else:
        assert sink.finish_reason == og.FinishReason.FAILED
    assert len(sink.tokens) < 12
    assert sink.tokens == predicted_tokens(_PROMPT_A, len(sink.tokens))
    assert 17 not in sink.tokens


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
    __slots__ = ("finish_reason", "tokens", "usage")

    def __init__(self):
        self.tokens = []
        self.finish_reason = og.FinishReason.NONE
        self.usage = None


@dataclass(frozen=True)
class _UsageSnapshot:
    prompt_tokens: int
    generated_tokens: int
    cached_prompt_tokens: int


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


@pytest.fixture(params=[_DRAFT_MODEL_DIR, _SELECTED_LOGITS_MODEL_DIR], ids=["per-token", "selected-logits"])
def draft_model(device, request):
    config = og.Config(str(request.param))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    return og.Model(config)


@pytest.fixture
def selected_logits_model(device):
    config = og.Config(str(_SELECTED_LOGITS_MODEL_DIR))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    return og.Model(config)


def _create_request(engine, prompt, max_new_tokens, sink, sinks):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
    request = engine.create_request(options=request_options)
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
        sink.usage = _UsageSnapshot(
            prompt_tokens=event.usage.prompt_tokens,
            generated_tokens=event.usage.generated_tokens,
            cached_prompt_tokens=event.usage.cached_prompt_tokens,
        )
        return True
    return False


def _step_once(engine, sinks, *, close_completed=True, event_buffer=None):
    if event_buffer is None:
        event_buffer = engine.create_event_buffer(8)
    events = engine.run(event_buffer)
    for event in events:
        if _drain(event, sinks) and close_completed:
            event.request.close()
    return bool(events)


def _next_event(engine):
    event_buffer = engine.create_event_buffer(1)
    while engine.has_pending_requests():
        events = engine.run(event_buffer)
        assert len(events) <= 1
        if events:
            return events[0]
    raise AssertionError("Engine completed without producing an event")


def _run(engine, sinks, *, max_steps=_MAX_STEPS, close_completed=True):
    steps = 0
    event_buffer = engine.create_event_buffer(8)
    while engine.has_pending_requests():
        _step_once(
            engine,
            sinks,
            close_completed=close_completed,
            event_buffer=event_buffer,
        )
        steps += 1
        assert steps <= max_steps, "engine.run() exceeded the safety bound; possible non-termination"


def _generate_isolated(model, prompt, max_new_tokens):
    sink = _Sink()
    engine = og.Engine(model)
    sinks = {}
    _create_request(engine, prompt, max_new_tokens, sink, sinks)
    _run(engine, sinks)
    del engine
    gc.collect()
    return sink.tokens


def test_engine_capabilities(model):
    engine = og.Engine(model)

    capabilities = engine.get_capabilities()

    assert capabilities.configured_max_batch_size == 8
    assert capabilities.max_scheduled_tokens == 2048
    assert capabilities.max_request_length == 128

    off_thread_errors = []

    def read_capabilities_off_owner_thread():
        try:
            engine.get_capabilities()
        except Exception as error:
            off_thread_errors.append(error)

    thread = threading.Thread(target=read_capabilities_off_owner_thread)
    thread.start()
    thread.join()

    assert len(off_thread_errors) == 1
    assert isinstance(off_thread_errors[0], RuntimeError)
    assert "Engine operations must be called from the Engine owner thread" in str(off_thread_errors[0])


def test_engine_run_releases_gil(model):
    engine = og.Engine(model)
    prompt = _PROMPT_LONG * 20
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + 4)
    request = engine.create_request(options=request_options)
    request.begin_turn(np.asarray(prompt, dtype=np.int32))
    event_buffer = engine.create_event_buffer(1)

    worker_ready = threading.Event()
    allow_worker = threading.Event()
    worker_progressed = threading.Event()

    def worker():
        worker_ready.set()
        allow_worker.wait()
        worker_progressed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_ready.wait(timeout=5)

    previous_switch_interval = sys.getswitchinterval()
    try:
        # Prevent the interpreter's periodic thread switch from satisfying the assertion. The
        # worker can acquire the GIL here only while the native Engine Run has explicitly released
        # it.
        sys.setswitchinterval(10.0)
        deadline = time.monotonic() + 5.0
        allow_worker.set()
        # Releasing the GIL does not guarantee that the OS schedules the worker during one
        # short Run call. Keep offering native calls without a Python wait or thread switch.
        while True:
            engine.run(event_buffer)
            if worker_progressed.is_set() or time.monotonic() >= deadline:
                break
        assert worker_progressed.is_set()
    finally:
        sys.setswitchinterval(previous_switch_interval)
        thread.join(timeout=5)
        request.close()

    assert not thread.is_alive()


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


def test_strided_request_tokens_are_copied_contiguously(model):
    prompt_storage = np.asarray(
        [_PROMPT_A[0], 31, _PROMPT_A[1], 32, _PROMPT_A[2], 33],
        dtype=np.int32,
    )
    prompt = prompt_storage[::2]
    assert not prompt.flags.c_contiguous
    max_new_tokens = 4
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + max_new_tokens)
    request = engine.create_request(options=request_options)
    request.begin_turn(prompt)
    sink = _Sink()
    sinks = {request: sink}

    _run(engine, sinks)

    assert sink.tokens == predicted_tokens(_PROMPT_A, max_new_tokens)


def test_model_declares_per_request_fp16_logits():
    graph = onnx.load(_MODEL_DIR / "decoder.onnx", load_external_data=False).graph
    logits = next(output for output in graph.output if output.name == "logits")
    tensor_type = logits.type.tensor_type

    assert tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert len(tensor_type.shape.dim) == 2
    assert tensor_type.shape.dim[0].dim_param == "batch_size"
    assert tensor_type.shape.dim[1].dim_value == _VOCAB_SIZE


def test_run_reuses_buffer_for_zero_one_and_bulk_capacity(model):
    engine = og.Engine(model)
    sinks = {}
    first = _create_request(engine, _PROMPT_A, 1, _Sink(), sinks)
    second = _create_request(engine, _PROMPT_B, 1, _Sink(), sinks)

    zero_buffer = engine.create_event_buffer(0)
    assert engine.run(zero_buffer) is zero_buffer
    assert len(zero_buffer) == 0
    assert engine.has_pending_requests()
    with pytest.raises((OverflowError, TypeError)):
        engine.create_event_buffer(-1)
    with pytest.raises((OverflowError, TypeError)):
        engine.create_event_buffer(1 << 100)

    one_buffer = engine.create_event_buffer(1)
    other_engine = og.Engine(model)
    with pytest.raises(RuntimeError, match="Engine that created it"):
        other_engine.run(one_buffer)
    assert len(one_buffer) == 0

    assert engine.run(one_buffer) is one_buffer
    assert isinstance(one_buffer, og.EngineEventBuffer)
    assert len(one_buffer) == 1
    assert one_buffer[0].request is first
    assert one_buffer[-1].request is first
    with pytest.raises(IndexError):
        _ = one_buffer[1]

    retained_events = engine.run(one_buffer)
    assert len(retained_events) == 1
    assert retained_events[0].request is second

    third = _create_request(engine, _PROMPT_A, 1, _Sink(), sinks)
    fourth = _create_request(engine, _PROMPT_B, 1, _Sink(), sinks)
    bulk_buffer = engine.create_event_buffer(8)
    bulk_events = engine.run(bulk_buffer)
    assert len(bulk_events) == 2
    assert [event.request for event in bulk_events] == [third, fourth]
    borrowed_event = bulk_events[0]
    borrowed_usage = borrowed_event.usage
    del bulk_events, bulk_buffer
    gc.collect()
    assert borrowed_event.request is third
    assert borrowed_usage.prompt_tokens == len(_PROMPT_A)

    for request in (first, second, third, fourth):
        request.close()


def test_deterministic_tokens(model):
    max_new = 12
    tokens = _generate_isolated(model, _PROMPT_A, max_new)

    assert tokens == [21, 30, 40, 51, 63, 12, 26, 41, 57, 10, 28, 47]
    assert tokens == predicted_tokens(_PROMPT_A, max_new)
    assert len(tokens) == max_new


def test_request_rewind_replays_retained_prefix(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(24)
    request = engine.create_request(options=request_options)
    first_sink = _Sink()
    sinks = {request: first_sink}
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(3)

    assert request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options) == 1
    _run(engine, sinks, close_completed=False)
    assert first_sink.tokens == predicted_tokens(_PROMPT_A, 3)

    retained = _PROMPT_A + first_sink.tokens
    discarded_sink = _Sink()
    sinks[request] = discarded_sink
    discarded_turn = request.begin_turn(np.asarray([12], dtype=np.int32), turn_options)
    assert discarded_turn == 2
    _run(engine, sinks, close_completed=False)

    request.rewind_to_start_of_turn(discarded_turn)
    assert not engine.has_pending_requests()

    continuation = [13]
    second_sink = _Sink()
    sinks[request] = second_sink
    turn_options.set_max_generated_tokens(1)
    assert request.begin_turn(np.asarray(continuation, dtype=np.int32), turn_options) == 3
    _run(engine, sinks, close_completed=False)

    assert second_sink.tokens == predicted_tokens(retained + continuation, 1)
    assert second_sink.finish_reason == og.FinishReason.MAX_GENERATED_TOKENS
    assert second_sink.usage.prompt_tokens == len(continuation)
    assert second_sink.usage.generated_tokens == 1
    request.close()


def test_draft_proposal_public_api(draft_model):
    prompt = np.asarray(_PROMPT_A, dtype=np.int32)
    expected = predicted_tokens(_PROMPT_A, 4)
    engine = og.Engine(draft_model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + len(expected))
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(len(expected))
    request.begin_turn(prompt, turn_options)
    event_buffer = engine.create_event_buffer(3)

    first_events = engine.run(event_buffer)
    assert [event.token for event in first_events] == expected[:1]
    assert engine.max_draft_tokens_per_proposal() >= 2
    with pytest.raises(ValueError, match="one-dimensional"):
        request.set_draft_tokens(np.asarray([expected[1:3]], dtype=np.int32))

    request.set_draft_tokens(np.asarray(expected[1:3], dtype=np.int32))
    proposal_events = engine.run(event_buffer)
    assert [event.token for event in proposal_events] == expected[1:]
    assert all(event.flags & og.EngineEventFlags.TOKEN for event in proposal_events)
    assert proposal_events[-1].flags & og.EngineEventFlags.TURN_FINISHED
    request.close()


def test_rejected_draft_resamples_from_its_verification_row(draft_model):
    expected = predicted_tokens(_PROMPT_A, 4)
    engine = og.Engine(draft_model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_PROMPT_A) + len(expected))
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(len(expected))
    request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options)
    event_buffer = engine.create_event_buffer(3)

    assert [event.token for event in engine.run(event_buffer)] == expected[:1]
    wrong = (expected[2] + 1) % _VOCAB_SIZE
    assert wrong != _EOS_TOKEN_ID
    request.set_draft_tokens(np.asarray([expected[1], wrong], dtype=np.int32))

    assert [event.token for event in engine.run(event_buffer)] == expected[1:3]
    request.close()


def test_selected_logits_model_declares_indices_and_compact_logits():
    graph = onnx.load(_SELECTED_LOGITS_MODEL_DIR / "decoder.onnx", load_external_data=False).graph
    indices = next(value for value in graph.input if value.name == "logits_indices")
    logits = next(value for value in graph.output if value.name == "logits")
    config = json.loads((_SELECTED_LOGITS_MODEL_DIR / "genai_config.json").read_text())

    assert indices.type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert logits.type.tensor_type.shape.dim[0].dim_param == "num_logits"
    assert config["model"]["decoder"]["inputs"]["logits_indices"] == "logits_indices"


def test_selected_logits_follow_each_packed_request(selected_logits_model):
    max_new = 8
    prompts = [_PROMPT_A, _PROMPT_B, _PROMPT_LONG]
    engine = og.Engine(selected_logits_model)
    sinks = {}
    outputs = [_Sink() for _ in prompts]
    for prompt, sink in zip(prompts, outputs, strict=True):
        _create_request(engine, prompt, max_new, sink, sinks)
    _run(engine, sinks)

    for prompt, sink in zip(prompts, outputs, strict=True):
        assert sink.tokens == predicted_tokens(prompt, max_new)


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
    _create_request(engine, _PROMPT_A, max_new, sink_a, sinks)
    _create_request(engine, _PROMPT_B, max_new, sink_b, sinks)
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
    _create_request(engine, _PROMPT_A, max_new, sink_a, sinks)

    for _ in range(3):
        if not engine.has_pending_requests():
            break
        _step_once(engine, sinks)
    assert len(sink_a.tokens) > 0, "first request produced nothing before staggered admission"

    sink_b = _Sink()
    _create_request(engine, _PROMPT_B, max_new, sink_b, sinks)
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
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(32)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks = {request: sink}
    prompt = np.asarray(_PROMPT_LONG, dtype=np.int32)

    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(3)
    assert request.begin_turn(prompt, turn_options) == 1
    _run(engine, sinks, close_completed=False)

    assert sink.tokens == predicted_tokens(_PROMPT_LONG, 3)
    assert sink.finish_reason == og.FinishReason.MAX_GENERATED_TOKENS
    assert sink.usage.prompt_tokens == len(prompt)
    assert sink.usage.generated_tokens == 3

    continuation = np.asarray([12], dtype=np.int32)
    turn_options.set_max_generated_tokens(2)
    assert request.begin_turn(continuation, turn_options) == 2
    _run(engine, sinks, close_completed=False)

    assert len(sink.tokens) == 5
    request.close()


def test_request_total_limit_and_cancel_metadata(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_PROMPT_LONG) + 2)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks = {request: sink}

    assert request.begin_turn(np.asarray(_PROMPT_LONG, dtype=np.int32)) == 1
    _run(engine, sinks, close_completed=False)
    assert len(sink.tokens) == 2
    assert sink.finish_reason == og.FinishReason.MAX_SESSION_TOKENS

    request.close()

    canceled = engine.create_request(options=request_options)
    turn_id = canceled.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32))
    assert canceled.cancel_turn(turn_id)
    assert not canceled.cancel_turn(turn_id)
    event = _next_event(engine)
    assert event.request is canceled
    assert event.finish_reason == og.FinishReason.CANCELLED
    canceled.close()


def test_zero_turn_budget_uses_default_limit(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(16)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks = {request: sink}
    prompt = np.asarray(_PROMPT_A, dtype=np.int32)

    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(0)
    assert request.begin_turn(prompt, turn_options) == 1
    _run(engine, sinks, close_completed=False)

    expected_generated_tokens = 16 - len(prompt)
    assert sink.tokens == predicted_tokens(_PROMPT_A, expected_generated_tokens)
    assert sink.finish_reason == og.FinishReason.MAX_SESSION_TOKENS
    assert sink.usage.generated_tokens == expected_generated_tokens
    request.close()


def test_completion_isolation(model):
    short_new, long_new = 5, 20
    long_isolated = _generate_isolated(model, _PROMPT_LONG, long_new)

    engine = og.Engine(model)
    short_sink, long_sink = _Sink(), _Sink()
    sinks = {}
    _create_request(engine, _PROMPT_A, short_new, short_sink, sinks)
    _create_request(engine, _PROMPT_LONG, long_new, long_sink, sinks)
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
    reference = _create_request(reference_engine, _PROMPT_A, short_max_new, reference_sink, reference_sinks)
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
    short = _create_request(engine, _PROMPT_A, short_max_new, short_sink, sinks)
    _create_request(engine, _PROMPT_LONG, long_max_new, long_sink, sinks)

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
    first = _create_request(engine, [5, 8, 57], 40, _Sink(), sinks)
    second = _create_request(engine, [6, 8, 56], 40, _Sink(), sinks)

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
    request = _create_request(engine, _PROMPT_A, 8, _Sink(), sinks)

    with pytest.raises(RuntimeError, match="new request or after the current turn is complete"):
        request.begin_turn(np.asarray([12], dtype=np.int32))

    request.close()


@pytest.mark.parametrize(
    "tokens",
    [
        np.asarray([5, 99, 9, 99, 13], dtype=np.int32)[::2],
        np.asarray(_PROMPT_A, dtype=np.int32)[::-1],
    ],
    ids=["positive-stride", "negative-stride"],
)
def test_begin_turn_copies_non_contiguous_token_views(model, tokens):
    logical_tokens = tokens.tolist()
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(logical_tokens) + 3)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks = {request: sink}

    request.begin_turn(tokens)
    _run(engine, sinks)

    assert sink.tokens == predicted_tokens(logical_tokens, 3)


def test_begin_turn_accepts_read_only_tokens_and_rejects_multiple_dimensions(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_PROMPT_A) + 1)

    read_only_tokens = np.asarray(_PROMPT_A, dtype=np.int32)
    read_only_tokens.setflags(write=False)
    request = engine.create_request(options=request_options)
    request.begin_turn(read_only_tokens)
    assert _next_event(engine).request is request
    request.close()

    invalid_request = engine.create_request(options=request_options)
    with pytest.raises(ValueError, match="one-dimensional"):
        invalid_request.begin_turn(np.asarray([_PROMPT_A, _PROMPT_A], dtype=np.int32))
    invalid_request.close()


def test_request_lifecycle_operations(model):
    engine = og.Engine(model)
    sink = _Sink()
    sinks = {}
    request = _create_request(engine, _PROMPT_A, 61, sink, sinks)

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


def test_request_options_are_snapshotted(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_PROMPT_LONG) + 4)
    request = engine.create_request(options=request_options)
    sink = _Sink()
    sinks = {request: sink}

    request_options.set_max_session_tokens(len(_PROMPT_LONG) + 12)
    request.begin_turn(np.asarray(_PROMPT_LONG, dtype=np.int32))
    _run(engine, sinks)

    assert sink.tokens == predicted_tokens(_PROMPT_LONG, 4)


@pytest.mark.parametrize("state", ["created", "active", "turn-complete"])
def test_close_is_valid_and_idempotent_from_every_state(model, state):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_PROMPT_LONG) + 8)
    request = engine.create_request(options=request_options)
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
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(64)
    request = engine.create_request(options=request_options)

    def run_turn(input_tokens, expected_turn_id):
        assert request.begin_turn(input_tokens) == expected_turn_id
        tokens = []
        finished = False
        while not finished:
            event = _next_event(engine)
            assert event.request is request
            assert event.turn_id == expected_turn_id
            if event.token is not None:
                tokens.append(event.token)
            finished = bool(event.flags & og.EngineEventFlags.TURN_FINISHED)
        return tokens

    first_turn_tokens = run_turn(np.asarray(_PROMPT_A, dtype=np.int32), 1)
    second_turn_tokens = run_turn(follow_up, 2)

    assert first_turn_tokens
    assert second_turn_tokens
    request.close()


def test_last_handle_release_reclaims_retained_capacity(model):
    engine = og.Engine(model)
    sinks = [_Sink() for _ in range(8)]
    sinks_by_request = {}
    requests = [_create_request(engine, [5 + index, 9, 13], 1, sinks[index], sinks_by_request) for index in range(8)]

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
    _create_request(engine, _PROMPT_A, 4, replacement_sink, replacement_sinks)
    _run(engine, replacement_sinks)

    assert replacement_sink.tokens == predicted_tokens(_PROMPT_A, 4)


def test_close_request_freezes_output(model):
    max_new = 40
    sibling_new = 16
    sibling_expected = predicted_tokens(_PROMPT_B, sibling_new)

    engine = og.Engine(model)
    sink_a, sink_b = _Sink(), _Sink()
    sinks = {}
    request_a = _create_request(engine, _PROMPT_A, max_new, sink_a, sinks)
    _create_request(engine, _PROMPT_B, sibling_new, sink_b, sinks)

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
    _create_request(first, _PROMPT_A, max_new, sink1, first_sinks)
    _run(first, first_sinks)
    assert sink1.tokens == expected
    del first
    gc.collect()

    second = og.Engine(model)
    assert not second.has_pending_requests()
    sink2 = _Sink()
    second_sinks = {}
    _create_request(second, _PROMPT_A, max_new, sink2, second_sinks)
    _run(second, second_sinks)
    assert sink2.tokens == expected


# The synthetic decoder's greedy formula (see predicted_tokens) is deterministic, so this prompt is
# chosen to make the first two generated tokens hit vocabulary entries 5 ("ST") and 6 ("OP") in the
# checked-in tokenizer (test/models/engine/synthetic-paged/tokenizer.json), completing the stop
# string "STOP" through real inference rather than scripted logits.
_STOP_MATCH_PROMPT = [61, 2, 5]


def test_stop_string_matches_across_two_tokens_via_real_inference(model):
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_STOP_MATCH_PROMPT) + 8)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_stop_strings(["STOP"])
    request.begin_turn(np.asarray(_STOP_MATCH_PROMPT, dtype=np.int32), turn_options)

    tokens = []
    finish_reason = og.FinishReason.NONE
    matched_index = None
    finished = False
    while not finished:
        event = _next_event(engine)
        if event.flags & og.EngineEventFlags.TOKEN:
            tokens.append(event.token)
        if event.flags & og.EngineEventFlags.TURN_FINISHED:
            finish_reason = event.finish_reason
            matched_index = event.matched_stop_string_index
            finished = True
    request.close()

    assert tokens == [5, 6]
    assert finish_reason == og.FinishReason.STOP_STRING
    assert matched_index == 0


def test_stop_strings_empty_list_disables_and_ordinary_generation_completes(model):
    # Setting a real configuration and then an empty list must clear/disable it, leaving ordinary
    # (non-stop) generation exactly as if stop strings had never been configured.
    expected = predicted_tokens(_STOP_MATCH_PROMPT, 8)
    engine = og.Engine(model)
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(_STOP_MATCH_PROMPT) + 8)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_stop_strings(["STOP"])
    turn_options.set_stop_strings([])
    request.begin_turn(np.asarray(_STOP_MATCH_PROMPT, dtype=np.int32), turn_options)

    sink = _Sink()
    sinks = {request: sink}
    _run(engine, sinks)

    assert sink.tokens == expected
    assert sink.finish_reason != og.FinishReason.STOP_STRING


def test_stop_strings_validation_and_embedded_nul_rejection(model):
    engine = og.Engine(model)
    request = engine.create_request()
    turn_options = og.TurnOptions(request)

    with pytest.raises(RuntimeError):
        turn_options.set_stop_strings([""])  # empty entry is invalid

    with pytest.raises(RuntimeError):
        turn_options.set_stop_strings([f"s{i}" for i in range(17)])  # exceeds the 16-entry bound

    with pytest.raises(ValueError):
        turn_options.set_stop_strings(["a\0b"])  # embedded NUL cannot round-trip through OgaStringArray

    # A valid configuration still works normally after the rejected attempts above.
    turn_options.set_stop_strings(["STOP"])
    request.close()


def _run_turn(engine, request, tokens, turn_options):
    request.begin_turn(np.asarray(tokens, dtype=np.int32), turn_options)
    generated = []
    event_buffer = engine.create_event_buffer(8)
    for _ in range(64):
        finished = False
        for event in engine.run(event_buffer):
            if event.flags & og.EngineEventFlags.TOKEN:
                generated.append(event.token)
            finished = finished or bool(event.flags & og.EngineEventFlags.TURN_FINISHED)
        if finished:
            break
    return generated


def test_turn_options_resolve_policy_per_turn(model):
    engine = og.Engine(model)

    def sampled_request(seed):
        request_options = og.RequestOptions()
        request_options.set_max_session_tokens(64)
        request = engine.create_request(options=request_options)
        turn_options = og.TurnOptions(request)
        turn_options.set_do_sample(True)
        turn_options.set_temperature(0.8)
        turn_options.set_top_p(0.9)
        turn_options.set_top_k(4)
        turn_options.set_repetition_penalty(1.1)
        turn_options.set_no_repeat_ngram_size(0)
        turn_options.set_min_generated_tokens(2)
        turn_options.set_max_generated_tokens(4)
        turn_options.set_seed(seed)
        return request, turn_options

    # Zero is a valid deterministic seed, and the same seed on the same prompt reproduces exactly.
    first, first_options = sampled_request(0)
    reference = _run_turn(engine, first, _PROMPT_A, first_options)
    assert 2 <= len(reference) <= 4

    second, second_options = sampled_request(0)
    assert _run_turn(engine, second, _PROMPT_A, second_options) == reference

    # reset() removes every option, so the following turn is plain model-default generation.
    third, third_options = sampled_request(0)
    third_options.reset()
    third_options.set_max_generated_tokens(4)
    assert _run_turn(engine, third, _PROMPT_A, third_options) == predicted_tokens(_PROMPT_A, 4)

    for request in (first, second, third):
        request.close()


def test_turn_options_reject_contradictory_sampling_scalars_before_mutation(model):
    engine = og.Engine(model)
    request = engine.create_request()
    turn_options = og.TurnOptions(request)
    turn_options.set_do_sample(False)
    turn_options.set_top_k(40)

    # A 40-candidate distribution contradicts top-logit selection.
    with pytest.raises(RuntimeError, match="contradict it: top_k"):
        request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options)

    # The rejected attempt left the request untouched, so a corrected policy still admits.
    turn_options.reset()
    turn_options.set_max_generated_tokens(1)
    assert request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options) == 1
    request.close()


def test_turn_options_accept_scalars_consistent_with_greedy_selection(model):
    """top_k == 1, temperature == 0, and unrestrictive bounds all agree with the top logit."""
    engine = og.Engine(model)
    consistent_policies = (
        {"set_top_k": 1},
        {"set_temperature": 0.0},
        {"set_do_sample": False, "set_top_k": 1},
        {"set_do_sample": False, "set_temperature": 0.0},
        {"set_do_sample": False, "set_top_p": 1.0, "set_top_k": 0},
        # Temperature 1 rescales nothing, so it is neutral rather than a request to sample.
        {"set_do_sample": False, "set_temperature": 1.0},
        {"set_do_sample": True, "set_top_k": 1, "set_temperature": 0.0},
    )

    for policy in consistent_policies:
        request = engine.create_request()
        turn_options = og.TurnOptions(request)
        for setter, value in policy.items():
            getattr(turn_options, setter)(value)
        turn_options.set_max_generated_tokens(4)

        assert _run_turn(engine, request, _PROMPT_A, turn_options) == predicted_tokens(_PROMPT_A, 4)
        request.close()


@pytest.mark.parametrize(
    ("search_overlay", "named_default"),
    (
        ('{"search": {"top_k": 1}}', "search.top_k = 1"),
        ('{"search": {"temperature": 0.0}}', "search.temperature = 0"),
    ),
)
def test_turn_options_reject_do_sample_under_model_greedy_defaults(device, search_overlay, named_default):
    """A model default the caller cannot see must not silently override an explicit do_sample."""
    config = og.Config(str(_MODEL_DIR))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    config.overlay(search_overlay)
    engine = og.Engine(og.Model(config))

    request = engine.create_request()
    turn_options = og.TurnOptions(request)
    turn_options.set_do_sample(True)
    turn_options.set_max_generated_tokens(2)

    with pytest.raises(RuntimeError, match=named_default):
        request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options)

    # The rejection mutated nothing, and overriding the named field is what actually asks to sample.
    turn_options.set_top_k(4)
    turn_options.set_temperature(0.8)
    assert request.begin_turn(np.asarray(_PROMPT_A, dtype=np.int32), turn_options) == 1
    request.close()


def test_create_request_rejects_positional_generator_params(model):
    engine = og.Engine(model)
    params = og.GeneratorParams(model)

    # The Engine no longer accepts caller generation parameters; the old positional call must fail
    # loudly rather than silently ignoring them.
    with pytest.raises(TypeError):
        engine.create_request(params)
