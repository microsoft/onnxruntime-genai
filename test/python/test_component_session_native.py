# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import shutil
from pathlib import Path

import onnx
import onnxruntime_genai as og
import pytest
from onnx import TensorProto, helper


def _write_component_genai_config(root: Path, filename: str, threads: int) -> None:
    (root / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "component",
                    "pad_token_id": 0,
                    "eos_token_id": 0,
                    "vocab_size": 1,
                    "context_length": 8192,
                    "decoder": {
                        "filename": filename,
                        "head_size": 1,
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


def test_combined_clm_high_level_session(tmp_path):
    tmp_path = tmp_path / "模型"
    tmp_path.mkdir()
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
    ranked = session.rank(
        {
            "state": "1",
            "instructions": "choose",
            "candidates": {"a": "1", "b": "2"},
        }
    )
    assert ranked["model"] == "clm"
    assert [item["rank"] for item in ranked["ranked"]] == [1, 2]
    assert {item["key"] for item in ranked["ranked"]} == {"a", "b"}
    numeric = session.rank(
        {
            "state": {
                "whole": 1.0,
                "fraction": 0.1,
                "fixed_boundary": 0.0001,
                "large_fixed": 100000.0,
            },
            "instructions": "choose",
            "candidates": {"a": "1", "b": "2"},
        }
    )
    textual = session.rank(
        {
            "state": {
                "whole": "1.0",
                "fraction": "0.1",
                "fixed_boundary": "0.0001",
                "large_fixed": "100000.0",
            },
            "instructions": "choose",
            "candidates": {"a": "1", "b": "2"},
        }
    )
    assert numeric == textual
    with pytest.raises(RuntimeError, match="embedded NUL"):
        session.rank(
            {
                "state": "left\0right",
                "instructions": "choose",
                "candidates": {"a": "1", "b": "2"},
            }
        )


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
    session = og.DecisionSession(tmp_path, providers=["cpu"])
    answer = session.decide(
        {
            "state": "1",
            "questions": {"q": {"type": "noul", "criteria": {}}},
        }
    )
    assert answer == {
        "model": "kev",
        "answers": {"q": {"type": "noul", "noul": 0.5}},
    }
    score = session.decide(
        {
            "state": "1",
            "questions": {
                "q": {
                    "type": "score",
                    "criteria": ["low", "medium", "high"],
                }
            },
        }
    )
    assert score["answers"]["q"]["confidence"] == 0
    with pytest.raises(RuntimeError, match="temperature is supported only by ranking sessions"):
        session.decide(
            {
                "state": "1",
                "temperature": 0.5,
                "questions": {"q": {"type": "noul", "criteria": {}}},
            }
        )


def _stateful_kev_package(root: Path) -> Path:
    fixture = Path(__file__).parents[1] / "models/multimodal-decoder-no-input-ids"
    shutil.copy(fixture / "tokenizer.json", root / "tokenizer.json")
    shutil.copy(fixture / "tokenizer_config.json", root / "tokenizer_config.json")
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    tokenizer_config.pop("pad_token_id", None)
    tokenizer_config.pop("pad_token", None)
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total_sequence"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, [3, "batch", "sequence"]),
        helper.make_tensor_value_info("past.0.key", TensorProto.FLOAT, ["batch", 1, "past_sequence", "kv_cache_dim"]),
        helper.make_tensor_value_info("past.0.value", TensorProto.FLOAT, ["batch", 1, "past_sequence", "kv_cache_dim"]),
        helper.make_tensor_value_info("past.0.conv", TensorProto.FLOAT, ["batch", 1, "kv_cache_dim"]),
        helper.make_tensor_value_info("past.0.recurrent", TensorProto.FLOAT, ["batch", 1, 1, "kv_cache_dim"]),
    ]
    outputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", "sequence", 4]),
        helper.make_tensor_value_info(
            "present.0.key", TensorProto.FLOAT, ["batch", 1, "total_sequence", "kv_cache_dim"]
        ),
        helper.make_tensor_value_info(
            "present.0.value", TensorProto.FLOAT, ["batch", 1, "total_sequence", "kv_cache_dim"]
        ),
        helper.make_tensor_value_info("present.0.conv", TensorProto.FLOAT, ["batch", 1, "kv_cache_dim"]),
        helper.make_tensor_value_info("present.0.recurrent", TensorProto.FLOAT, ["batch", 1, 1, "kv_cache_dim"]),
    ]
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("CumSum", ["ids_f", "axis1"], ["cumulative"]),
        helper.make_node("Squeeze", ["past.0.recurrent", "axes123"], ["past_scalar"]),
        helper.make_node("Unsqueeze", ["past_scalar", "axes12"], ["past_for_tokens"]),
        helper.make_node("Unsqueeze", ["cumulative", "axis2"], ["cumulative_3d"]),
        helper.make_node("Add", ["cumulative_3d", "past_for_tokens"], ["hidden_one"]),
        helper.make_node("Concat", ["hidden_one"] * 4, ["hidden_states"], axis=2),
        helper.make_node("Unsqueeze", ["ids_f", "axes13"], ["new_kv"]),
        helper.make_node("Concat", ["past.0.key", "new_kv"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past.0.value", "new_kv"], ["present.0.value"], axis=2),
        helper.make_node("ReduceSum", ["ids_f", "axis1"], ["token_sum"], keepdims=1),
        helper.make_node("Unsqueeze", ["token_sum", "axis2"], ["token_sum_3d"]),
        helper.make_node("Add", ["past.0.conv", "token_sum_3d"], ["present.0.conv"]),
        helper.make_node("Unsqueeze", ["token_sum_3d", "axis3"], ["token_sum_4d"]),
        helper.make_node("Add", ["past.0.recurrent", "token_sum_4d"], ["present.0.recurrent"]),
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
    _write_component_genai_config(root, "backbone.onnx", 1)
    config = json.loads((root / "genai_config.json").read_text())
    config["model"]["decoder"]["hidden_size"] = 1
    (root / "genai_config.json").write_text(json.dumps(config))
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


def test_stateful_kev_repeated_and_isolated_requests(tmp_path):
    package = _stateful_kev_package(tmp_path)
    request = {
        "state": {"shared": "prefix"},
        "questions": {
            "one": {"type": "choice", "criteria": {"a": "alpha", "b": "beta"}},
            "two": {"type": "choice", "criteria": {"x": "xray", "y": "yankee"}},
        },
    }
    session = og.DecisionSession(package, providers=["cpu"])
    result = session.decide(request)
    assert session.decide(request) == result

    isolated_session = og.DecisionSession(package, providers=["cpu"])
    isolated = isolated_session.decide({"state": request["state"], "questions": {"two": request["questions"]["two"]}})
    assert isolated["answers"]["two"] == result["answers"]["two"]


def test_kev_enforces_configured_context_length(tmp_path):
    package = _stateful_kev_package(tmp_path)
    config = json.loads((package / "genai_config.json").read_text())
    config["model"]["context_length"] = 16
    (package / "genai_config.json").write_text(json.dumps(config))

    session = og.DecisionSession(package, providers=["cpu"])
    with pytest.raises(
        RuntimeError, match="exceeds configured context length 16"
    ):
        session.decide(
            {
                "state": "state " * 64,
                "questions": {
                    "q": {
                        "type": "choice",
                        "criteria": {"a": "alpha", "b": "beta"},
                    }
                },
            }
        )


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
        session.rank(
            {
                "state": "1",
                "instructions": "choose",
                "candidates": {"a": "1", "b": "2"},
            }
        )


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
