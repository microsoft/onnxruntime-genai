# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from onnx import TensorProto, helper


def _package(root: Path, filename: str = "graphs/arbitrary-name.onnx") -> Path:
    model_path = root / filename
    model_path.parent.mkdir(parents=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "tiny_component",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 13
    onnx.save(model, model_path)
    (root / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic",
                "provenance": {"test": True},
                "components": {
                    "unusual.component": {
                        "role": "test",
                        "filename": filename,
                        "inputs": {"x": "input"},
                    }
                },
            }
        )
    )
    return root


def _write_component_genai_config(root: Path, filename: str, threads: int) -> None:
    (root / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "component",
                    "pad_token_id": 0,
                    "eos_token_id": 0,
                    "vocab_size": 1,
                    "context_length": 8,
                    "decoder": {
                        "filename": filename,
                        "session_options": {
                            "intra_op_num_threads": threads,
                            "inter_op_num_threads": 1,
                            "session.intra_op.allow_spinning": "0",
                            "session.inter_op.allow_spinning": "0",
                        },
                    },
                }
            }
        )
    )


def _cuda_graph_package(
    root: Path, component: str = "backbone"
) -> Path:
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [2, 2],
        [1.0, 0.5, -0.25, 2.0],
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "captured_component",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 13
    onnx.save(model, root / "backbone.onnx")
    (root / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic-cuda-graph",
                "components": {
                    component: {"filename": "backbone.onnx"},
                },
            }
        )
    )
    return root


def test_cuda_graph_capture_replays_and_falls_back_for_new_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("ORT_GENAI_KEV_CUDA_GRAPH", "1")
    try:
        session = og.ComponentSession(str(_cuda_graph_package(tmp_path)), "backbone", ["cuda"])
    except RuntimeError as error:
        if "Cuda interface not available" in str(error):
            pytest.skip(str(error))
        raise

    first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(
        session.run({"input": first})["output"],
        first @ np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        session.run({"input": second})["output"],
        second @ np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32),
    )
    changed_shape = second[:1]
    np.testing.assert_allclose(
        session.run({"input": changed_shape})["output"],
        changed_shape @ np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32),
    )


def test_clm_cuda_graph_capture_replays_multiple_shapes(tmp_path, monkeypatch):
    package = _cuda_graph_package(tmp_path, "fused_state_ranking")
    (package / "component_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "components": {
                    "fused_state_ranking": {
                        "cuda_graph_max_signatures": 2
                    }
                },
            }
        )
    )
    try:
        session = og.ComponentSession(
            str(package), "fused_state_ranking", ["cuda"]
        )
    except RuntimeError as error:
        if "Cuda interface not available" in str(error):
            pytest.skip(str(error))
        raise

    weight = np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
    batches = [
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
        np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
    ]
    for value in batches:
        np.testing.assert_allclose(
            session.run({"input": value})["output"],
            value @ weight,
        )


def test_invalid_component_runtime_policy_is_rejected(tmp_path):
    package = _package(tmp_path)
    (package / "component_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "components": {
                    "unusual.component": {
                        "cuda_graph_max_signatures": -1
                    }
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="non-negative integer"):
        og.ComponentSession(str(package), "unusual.component", ["cpu"])


def test_manifest_mapping_arbitrary_filename_and_native_run(tmp_path):
    session = og.ComponentSession(str(_package(tmp_path)), "unusual.component", ["cpu"])
    value = np.asarray([[1.5, -2.0]], dtype=np.float32)
    result = session.run({"input": value}, ["output"])
    np.testing.assert_array_equal(result["output"], value)
    assert session.input_names == ["input"]
    assert session.input_info["input"]["shape"] == [-1, 2]


def test_component_session_applies_genai_config_session_options(tmp_path):
    filename = "graphs/arbitrary-name.onnx"
    package = _package(tmp_path, filename)
    _write_component_genai_config(package, filename, 1)
    session = og.ComponentSession(str(package), "unusual.component", ["cpu"])
    value = np.asarray([[1.5, -2.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        session.run({"input": value}, ["output"])["output"], value
    )


def test_component_session_rejects_malformed_genai_config(tmp_path):
    filename = "graphs/arbitrary-name.onnx"
    package = _package(tmp_path, filename)
    (package / "genai_config.json").write_text("{")
    with pytest.raises(RuntimeError):
        og.ComponentSession(str(package), "unusual.component", ["cpu"])


def test_zero_element_component_output_is_supported(tmp_path):
    session = og.ComponentSession(str(_package(tmp_path)), "unusual.component", ["cpu"])
    value = np.empty((0, 2), dtype=np.float32)
    result = session.run({"input": value}, ["output"])
    assert result["output"].shape == (0, 2)
    assert result["output"].size == 0


def test_legacy_partial_component_layout_is_rejected(tmp_path):
    for relative in ("encoder/model.onnx", "state_head/model.onnx"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    with pytest.raises(RuntimeError, match="not a recognized CLM/KEV layout"):
        og.ComponentSession(str(tmp_path), "encoder", ["cpu"])


def test_legacy_component_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside.onnx"
    outside.write_bytes(b"fixture")
    for relative in (
        "encoder/model.onnx",
        "state_head/model.onnx",
        "action_head/model.onnx",
        "scorer/model.onnx",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "encoder/model.onnx":
            path.symlink_to(outside)
        else:
            path.write_bytes(b"synthetic")
    with pytest.raises(RuntimeError, match="resolves outside the package"):
        og.ComponentSession(str(tmp_path), "encoder", ["cpu"])


def test_manifest_traversal_is_rejected(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic",
                "components": {"bad": {"filename": "../outside.onnx"}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="traversal"):
        og.ComponentSession(str(tmp_path), "bad")


def test_manifest_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside.onnx"
    outside.write_bytes(b"fixture")
    link = tmp_path / "linked.onnx"
    link.symlink_to(outside)
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic",
                "components": {"bad": {"filename": "linked.onnx"}},
            }
        )
    )
    with pytest.raises(RuntimeError, match="outside the package"):
        og.ComponentSession(str(tmp_path), "bad")


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        og.ComponentSession(str(_package(tmp_path)), "unusual.component", ["not_a_provider"])


def test_bfloat16_uses_uint16_storage_with_onnx_type_metadata(tmp_path):
    model_path = tmp_path / "bf16.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "bf16_component",
        [helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [2])],
        [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 13
    onnx.save(model, model_path)
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic",
                "components": {"bf16": {"filename": "bf16.onnx"}},
            }
        )
    )
    session = og.ComponentSession(str(tmp_path), "bf16", ["cpu"])
    assert session.input_info["input"]["dtype"] == "uint16"
    assert session.input_info["input"]["onnx_type"] == TensorProto.BFLOAT16
    bits = np.asarray([0x3F80, 0xC000], dtype=np.uint16)
    result = session.run({"input": bits}, ["output"])
    np.testing.assert_array_equal(result["output"], bits)


def test_combined_clm_high_level_session(tmp_path):
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", tmp_path / "tokenizer_config.json")

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, None, 4])
    backbone = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["input_ids", "axes"], ["one"]),
                helper.make_node("Cast", ["one"], ["float_one"], to=TensorProto.FLOAT),
                helper.make_node("Concat", ["float_one"] * 4, ["hidden_states"], axis=2),
            ],
            "backbone",
            [ids, mask],
            [hidden],
            [
                helper.make_tensor("axes", TensorProto.INT64, [1], [2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    backbone.ir_version = 13
    onnx.save(backbone, tmp_path / "backbone.onnx")

    state = helper.make_tensor_value_info("state_hidden_states", TensorProto.FLOAT, [None, 4])
    action = helper.make_tensor_value_info("action_hidden_states", TensorProto.FLOAT, [None, 4])
    state_out = helper.make_tensor_value_info("state_embedding", TensorProto.FLOAT, [None, 4])
    action_out = helper.make_tensor_value_info("action_embedding", TensorProto.FLOAT, [None, 4])
    scale_out = helper.make_tensor_value_info("effective_logit_scale", TensorProto.FLOAT, [])
    heads = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["state_hidden_states"], ["state_embedding"]),
                helper.make_node("Identity", ["action_hidden_states"], ["action_embedding"]),
                helper.make_node("Identity", ["scale"], ["effective_logit_scale"]),
            ],
            "heads",
            [state, action],
            [state_out, action_out, scale_out],
            [
                helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    heads.ir_version = 13
    onnx.save(heads, tmp_path / "clm_heads.onnx")
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "clm-v0.1-8b",
                "components": {
                    "backbone": {"filename": "backbone.onnx"},
                    "clm_heads": {"filename": "clm_heads.onnx"},
                },
            }
        )
    )
    session = og.RankingSession(tmp_path, providers=["cpu"])
    answer = session.rank({"state": "1", "questions": {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}})
    assert answer["q"]["type"] == "choice"
    assert set(answer["q"]["probabilities"]) == {"a", "b"}


def test_flat_kev_high_level_session(tmp_path):
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", tmp_path / "tokenizer_config.json")
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, None, 4])
    backbone = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["input_ids", "axis2"], ["one"]),
                helper.make_node("Cast", ["one"], ["float_one"], to=TensorProto.FLOAT),
                helper.make_node("Concat", ["float_one"] * 4, ["hidden_states"], axis=2),
            ],
            "backbone",
            [ids, mask],
            [hidden],
            [
                helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    backbone.ir_version = 13
    onnx.save(backbone, tmp_path / "backbone.onnx")
    head_inputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, 4]),
        helper.make_tensor_value_info("decide_indices", TensorProto.INT64, [None]),
        helper.make_tensor_value_info("option_indices", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("option_mask", TensorProto.BOOL, [None, None]),
    ]
    probabilities = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, None])
    head = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["option_mask"], ["values"], to=TensorProto.FLOAT),
                helper.make_node("ReduceSum", ["values", "axis1"], ["total"], keepdims=1),
                helper.make_node("Div", ["values", "total"], ["probabilities"]),
            ],
            "head",
            head_inputs,
            [probabilities],
            [
                helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    head.ir_version = 13
    onnx.save(head, tmp_path / "kev_head.onnx")
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "jaredpalmer/kev-4b",
                "components": {
                    "backbone": {"filename": "backbone.onnx"},
                    "kev_head": {"filename": "kev_head.onnx"},
                },
            }
        )
    )
    answer = og.DecisionSession(tmp_path, providers=["cpu"]).decide(
        {
            "state": "1",
            "questions": {"q": {"type": "noul", "criteria": {}}},
        }
    )
    assert answer == {"q": {"type": "noul", "noul": 0.5}}


def _stateful_kev_package(root: Path) -> Path:
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", root / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", root / "tokenizer_config.json")
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total_sequence"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, [3, "batch", "sequence"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch", 1, "past_sequence", 1]),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, ["batch", 1, "past_sequence", 1]),
        helper.make_tensor_value_info("past_key_values.1.conv_state", TensorProto.FLOAT, ["batch", 1, 1]),
        helper.make_tensor_value_info("past_key_values.1.recurrent_state", TensorProto.FLOAT, ["batch", 1, 1, 1]),
    ]
    outputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", "sequence", 4]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch", 1, "total_sequence", 1]),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, ["batch", 1, "total_sequence", 1]),
        helper.make_tensor_value_info("present.1.conv_state", TensorProto.FLOAT, ["batch", 1, 1]),
        helper.make_tensor_value_info("present.1.recurrent_state", TensorProto.FLOAT, ["batch", 1, 1, 1]),
    ]
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("CumSum", ["ids_f", "axis1"], ["cumulative"]),
        helper.make_node("Squeeze", ["past_key_values.1.recurrent_state", "axes123"], ["past_scalar"]),
        helper.make_node("Unsqueeze", ["past_scalar", "axes12"], ["past_for_tokens"]),
        helper.make_node("Unsqueeze", ["cumulative", "axis2"], ["cumulative_3d"]),
        helper.make_node("Add", ["cumulative_3d", "past_for_tokens"], ["hidden_one"]),
        helper.make_node("Concat", ["hidden_one"] * 4, ["hidden_states"], axis=2),
        helper.make_node("Unsqueeze", ["ids_f", "axes13"], ["new_kv"]),
        helper.make_node("Concat", ["past_key_values.0.key", "new_kv"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "new_kv"], ["present.0.value"], axis=2),
        helper.make_node("ReduceSum", ["ids_f", "axis1"], ["token_sum"], keepdims=1),
        helper.make_node("Unsqueeze", ["token_sum", "axis2"], ["token_sum_3d"]),
        helper.make_node("Add", ["past_key_values.1.conv_state", "token_sum_3d"], ["present.1.conv_state"]),
        helper.make_node("Unsqueeze", ["token_sum_3d", "axis3"], ["token_sum_4d"]),
        helper.make_node("Add", ["past_key_values.1.recurrent_state", "token_sum_4d"], ["present.1.recurrent_state"]),
    ]
    initializers = [
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("axis3", TensorProto.INT64, [1], [3]),
        helper.make_tensor("axes12", TensorProto.INT64, [2], [1, 2]),
        helper.make_tensor("axes13", TensorProto.INT64, [2], [1, 3]),
        helper.make_tensor("axes123", TensorProto.INT64, [3], [1, 2, 3]),
    ]
    backbone = helper.make_model(
        helper.make_graph(nodes, "stateful_backbone", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    backbone.ir_version = 13
    onnx.save(backbone, root / "backbone.onnx")

    head_inputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, None, 4]),
        helper.make_tensor_value_info("decide_indices", TensorProto.INT64, [None]),
        helper.make_tensor_value_info("option_indices", TensorProto.INT64, [None, None]),
        helper.make_tensor_value_info("option_mask", TensorProto.BOOL, [None, None]),
    ]
    head = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["option_indices", "axis2"], ["grouped_indices"]),
                helper.make_node("Shape", ["option_indices"], ["option_shape"]),
                helper.make_node("Concat", ["option_shape", "hidden_width"], ["gather_shape"], axis=0),
                helper.make_node("Expand", ["grouped_indices", "gather_shape"], ["expanded_indices"]),
                helper.make_node("GatherElements", ["hidden_states", "expanded_indices"], ["selected"], axis=1),
                helper.make_node("ReduceMean", ["selected", "axis2"], ["scores"], keepdims=0),
                helper.make_node("Where", ["option_mask", "scores", "masked_score"], ["masked_scores"]),
                helper.make_node("Softmax", ["masked_scores"], ["probabilities"], axis=1),
            ],
            "stateful_head",
            head_inputs,
            [
                helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, None]),
            ],
            [
                helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
                helper.make_tensor("hidden_width", TensorProto.INT64, [1], [4]),
                helper.make_tensor("masked_score", TensorProto.FLOAT, [], [-1.0e9]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    head.ir_version = 13
    onnx.save(head, root / "kev_head.onnx")
    (root / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "synthetic-stateful-kev",
                "components": {
                    "backbone": {"filename": "backbone.onnx"},
                    "kev_head": {"filename": "kev_head.onnx"},
                },
            }
        )
    )
    return root


def test_stateful_kev_prefix_reuse_parity_cache_and_isolation(tmp_path):
    package = _stateful_kev_package(tmp_path)
    request = {
        "state": {"shared": "prefix"},
        "questions": {
            "one": {"type": "choice", "criteria": {"a": "alpha", "b": "beta"}},
            "two": {"type": "choice", "criteria": {"x": "xray", "y": "yankee"}},
        },
    }
    optimized = og.DecisionSession(package, providers=["cpu"], prefix_reuse=True)
    full = og.DecisionSession(package, providers=["cpu"], prefix_reuse=False)
    optimized_result = optimized.decide(request)
    assert optimized_result == full.decide(request)
    assert optimized.prefix_reuse_status == "compatible explicit state I/O"
    assert optimized.prefix_cache_stats["prefix_runs"] == 1
    assert optimized.prefix_cache_stats["branch_runs"] == 1
    assert optimized.decide(request) == optimized_result
    assert optimized.prefix_cache_stats["hits"] == 1
    assert optimized.prefix_cache_stats["prefix_runs"] == 1

    isolated = og.DecisionSession(package, providers=["cpu"]).decide(
        {"state": request["state"], "questions": {"two": request["questions"]["two"]}}
    )
    assert isolated["two"] == optimized_result["two"]
    optimized.invalidate_cache()
    optimized.decide(request)
    assert optimized.prefix_cache_stats["prefix_runs"] == 2
    optimized.set_prefix_cache_capacity(1, 1)
    assert optimized.prefix_cache_stats["entries"] == 0
    assert optimized.prefix_cache_stats["evictions"] >= 1


def test_malformed_clm_hidden_shape_returns_runtime_error(tmp_path):
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", tmp_path / "tokenizer_config.json")

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, None])
    backbone = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["input_ids"], ["hidden_states"], to=TensorProto.FLOAT),
            ],
            "bad_backbone",
            [ids, mask],
            [hidden],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    backbone.ir_version = 13
    onnx.save(backbone, tmp_path / "backbone.onnx")

    state = helper.make_tensor_value_info("state_hidden_states", TensorProto.FLOAT, [None, 4])
    action = helper.make_tensor_value_info("action_hidden_states", TensorProto.FLOAT, [None, 4])
    heads = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["state_hidden_states"], ["state_embedding"]),
                helper.make_node("Identity", ["action_hidden_states"], ["action_embedding"]),
                helper.make_node("Identity", ["scale"], ["effective_logit_scale"]),
            ],
            "heads",
            [state, action],
            [
                helper.make_tensor_value_info("state_embedding", TensorProto.FLOAT, [None, 4]),
                helper.make_tensor_value_info("action_embedding", TensorProto.FLOAT, [None, 4]),
                helper.make_tensor_value_info("effective_logit_scale", TensorProto.FLOAT, []),
            ],
            [helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    heads.ir_version = 13
    onnx.save(heads, tmp_path / "clm_heads.onnx")
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "clm-v0.1-8b",
                "components": {
                    "backbone": {"filename": "backbone.onnx"},
                    "clm_heads": {"filename": "clm_heads.onnx"},
                },
            }
        )
    )

    session = og.RankingSession(tmp_path, providers=["cpu"])
    with pytest.raises(RuntimeError, match="encoder hidden states must have rank 3"):
        session.rank({"state": "1", "questions": {"q": {"type": "choice", "criteria": {"a": "1", "b": "2"}}}})


def test_malformed_kev_probability_shape_returns_runtime_error(tmp_path):
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", tmp_path / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", tmp_path / "tokenizer_config.json")

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    mask_input = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    hidden = helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, None, 4])
    backbone = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["input_ids", "axis2"], ["one"]),
                helper.make_node("Cast", ["one"], ["float_one"], to=TensorProto.FLOAT),
                helper.make_node("Concat", ["float_one"] * 4, ["hidden_states"], axis=2),
            ],
            "backbone",
            [ids, mask_input],
            [hidden],
            [
                helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    backbone.ir_version = 13
    onnx.save(backbone, tmp_path / "backbone.onnx")

    option_mask = helper.make_tensor_value_info("option_mask", TensorProto.BOOL, [None, None])
    head = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", ["option_mask"], ["values"], to=TensorProto.FLOAT),
                helper.make_node("Reshape", ["values", "flat_shape"], ["probabilities"]),
            ],
            "bad_head",
            [
                helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [None, 4]),
                helper.make_tensor_value_info("decide_indices", TensorProto.INT64, [None]),
                helper.make_tensor_value_info("option_indices", TensorProto.INT64, [None, None]),
                option_mask,
            ],
            [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None])],
            [
                helper.make_tensor("flat_shape", TensorProto.INT64, [1], [-1]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    head.ir_version = 13
    onnx.save(head, tmp_path / "kev_head.onnx")
    (tmp_path / "component_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "jaredpalmer/kev-4b",
                "components": {
                    "backbone": {"filename": "backbone.onnx"},
                    "kev_head": {"filename": "kev_head.onnx"},
                },
            }
        )
    )

    session = og.DecisionSession(tmp_path, providers=["cpu"])
    with pytest.raises(RuntimeError, match="KEV probability matrix must have rank 2"):
        session.decide(
            {
                "state": "1",
                "questions": {"q": {"type": "noul", "criteria": {}}},
            }
        )
