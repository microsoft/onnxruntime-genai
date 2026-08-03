# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


def _write_config(directory: Path, config: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "genai_config.json", "w") as file:
        json.dump(config, file, indent=2)
    return str(directory)


def _make_tiny_eagle_model(
    directory: Path,
    acceptance: str = "full",
    *,
    vocab_size: int = 16,
    hidden_size: int = 2,
    eos_token_id: int = 15,
) -> tuple[str, str]:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    directory.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "hidden_states_before_layer_2",
        "hidden_states_before_layer_14",
        "hidden_states_before_layer_25",
    ]

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, ["batch", "sequence"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "total_sequence"])
    attention_bias = helper.make_tensor_value_info(
        "attention_bias", TensorProto.FLOAT,
        [1, 1, "sequence", "total_sequence"])
    position_ids = helper.make_tensor_value_info(
        "position_ids", TensorProto.INT64, ["batch", "sequence"])
    past_key = helper.make_tensor_value_info(
        "past_key_values.0.key", TensorProto.FLOAT,
        ["batch", 1, "past_sequence", hidden_size])
    past_value = helper.make_tensor_value_info(
        "past_key_values.0.value", TensorProto.FLOAT,
        ["batch", 1, "past_sequence", hidden_size])
    target_outputs = [
        helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, ["batch", "sequence", vocab_size])
    ]
    target_outputs.extend(
        helper.make_tensor_value_info(
            name, TensorProto.FLOAT, ["batch", "sequence", hidden_size])
        for name in feature_names
    )
    target_outputs.extend(
        [
            helper.make_tensor_value_info(
                "present.0.key", TensorProto.FLOAT,
                ["batch", 1, "total_sequence", hidden_size]),
            helper.make_tensor_value_info(
                "present.0.value", TensorProto.FLOAT,
                ["batch", 1, "total_sequence", hidden_size]),
        ]
    )

    logits_table = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for token in range(vocab_size):
        logits_table[token, (token + 1) % vocab_size] = 10.0
    target_initializers = [
        numpy_helper.from_array(logits_table, "target_logits_table"),
        numpy_helper.from_array(
            np.array([1, 3], dtype=np.int64), "target_cache_axes"),
    ]
    target_nodes = [
        helper.make_node(
            "Gather", ["target_logits_table", "input_ids"], ["logits"], axis=0),
        helper.make_node(
            "Cast", ["input_ids"], ["target_cache_tokens"],
            to=TensorProto.FLOAT),
        helper.make_node(
            "Unsqueeze",
            ["target_cache_tokens", "target_cache_axes"],
            ["target_cache_rows"],
        ),
        helper.make_node(
            "Concat",
            ["target_cache_rows"] * hidden_size,
            ["target_new_cache"],
            axis=3,
        ),
        helper.make_node(
            "Concat",
            ["past_key_values.0.key", "target_new_cache"],
            ["present.0.key"],
            axis=2,
        ),
        helper.make_node(
            "Concat",
            ["past_key_values.0.value", "target_new_cache"],
            ["present.0.value"],
            axis=2,
        ),
    ]
    for index, name in enumerate(feature_names):
        table = np.arange(
            vocab_size * hidden_size, dtype=np.float32
        ).reshape(vocab_size, hidden_size) + index * 100
        table_name = f"feature_table_{index}"
        target_initializers.append(numpy_helper.from_array(table, table_name))
        target_nodes.append(
            helper.make_node("Gather", [table_name, "input_ids"], [name], axis=0)
        )

    target_graph = helper.make_graph(
        target_nodes,
        "tiny_eagle_target",
        [
            input_ids,
            attention_mask,
            attention_bias,
            position_ids,
            past_key,
            past_value,
        ],
        target_outputs,
        target_initializers,
    )
    target_model = helper.make_model(
        target_graph, opset_imports=[helper.make_opsetid("", 20)])
    target_model.ir_version = 10
    onnx.save(target_model, directory / "target.onnx")

    eagle_inputs = [
        helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info(
            "target_hidden_states", TensorProto.FLOAT,
            ["batch", "sequence", hidden_size * 3]),
        helper.make_tensor_value_info(
            "recurrent_hidden_states", TensorProto.FLOAT,
            ["batch", "sequence", hidden_size]),
        helper.make_tensor_value_info(
            "use_target_hidden_states", TensorProto.BOOL, []),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch", "total_sequence"]),
        helper.make_tensor_value_info(
            "attention_bias", TensorProto.FLOAT,
            [1, 1, "sequence", "total_sequence"]),
        helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info(
            "past_key", TensorProto.FLOAT,
            ["batch", 1, "past_sequence", hidden_size]),
        helper.make_tensor_value_info(
            "past_value", TensorProto.FLOAT,
            ["batch", 1, "past_sequence", hidden_size]),
    ]
    eagle_outputs = [
        helper.make_tensor_value_info(
            "draft_hidden_states", TensorProto.FLOAT,
            ["batch", "sequence", hidden_size]),
        helper.make_tensor_value_info(
            "draft_logits", TensorProto.FLOAT,
            ["batch", "sequence", vocab_size]),
        helper.make_tensor_value_info(
            "draft_token_id", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info(
            "mapped_token_id", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info(
            "draft_topk_ids", TensorProto.INT64,
            ["batch", "sequence", 10]),
        helper.make_tensor_value_info(
            "draft_topk_log_scores", TensorProto.FLOAT,
            ["batch", "sequence", 10]),
        helper.make_tensor_value_info(
            "mapped_topk_ids", TensorProto.INT64,
            ["batch", "sequence", 10]),
        helper.make_tensor_value_info(
            "present_key", TensorProto.FLOAT,
            ["batch", 1, "total_sequence", hidden_size]),
        helper.make_tensor_value_info(
            "present_value", TensorProto.FLOAT,
            ["batch", 1, "total_sequence", hidden_size]),
    ]

    if acceptance == "full":
        target_offset = recurrent_offset = 1
    elif acceptance == "partial":
        target_offset, recurrent_offset = 1, 2
    elif acceptance == "zero":
        target_offset = recurrent_offset = 2
    else:
        raise ValueError(f"Unknown acceptance mode: {acceptance}")

    eagle_initializers = [
        numpy_helper.from_array(
            np.array(target_offset, dtype=np.int64), "target_offset"),
        numpy_helper.from_array(
            np.array(recurrent_offset, dtype=np.int64), "recurrent_offset"),
        numpy_helper.from_array(
            np.array(vocab_size, dtype=np.int64), "vocab_size"),
        numpy_helper.from_array(
            np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
            "rank_offsets",
        ),
        numpy_helper.from_array(
            np.arange(-1, -101, -10, dtype=np.float32).reshape(1, 1, 10),
            "topk_scores",
        ),
        numpy_helper.from_array(
            np.array([hidden_size], dtype=np.int64), "hidden_dimension"),
        numpy_helper.from_array(
            np.array([vocab_size], dtype=np.int64), "vocab_dimension"),
        numpy_helper.from_array(
            np.array([10], dtype=np.int64), "topk_dimension"),
        numpy_helper.from_array(
            np.array([1], dtype=np.int64), "sequence_axis"),
        numpy_helper.from_array(
            np.array([2], dtype=np.int64), "unsqueeze_axis"),
        numpy_helper.from_array(
            np.array(0, dtype=np.int64), "first_candidate"),
        numpy_helper.from_array(
            np.array([1, 1], dtype=np.int64), "batch_and_heads"),
    ]
    eagle_nodes = [
        helper.make_node(
            "Where",
            ["use_target_hidden_states", "target_offset", "recurrent_offset"],
            ["selected_offset"],
        ),
        helper.make_node(
            "Add", ["rank_offsets", "selected_offset"], ["candidate_offsets"]),
        helper.make_node(
            "Unsqueeze", ["input_ids", "unsqueeze_axis"], ["expanded_input_ids"]),
        helper.make_node(
            "Add",
            ["expanded_input_ids", "candidate_offsets"],
            ["unmapped_topk_ids"],
        ),
        helper.make_node(
            "Mod", ["unmapped_topk_ids", "vocab_size"], ["mapped_topk_ids"]),
        helper.make_node(
            "Identity", ["mapped_topk_ids"], ["draft_topk_ids"]),
        helper.make_node(
            "Gather",
            ["mapped_topk_ids", "first_candidate"],
            ["mapped_token_id"],
            axis=2,
        ),
        helper.make_node("Identity", ["mapped_token_id"], ["draft_token_id"]),
        helper.make_node("Shape", ["input_ids"], ["token_shape"]),
        helper.make_node(
            "Concat", ["token_shape", "hidden_dimension"], ["hidden_shape"],
            axis=0),
        helper.make_node(
            "ConstantOfShape", ["hidden_shape"], ["draft_hidden_states"]),
        helper.make_node(
            "Concat", ["token_shape", "vocab_dimension"], ["logits_shape"],
            axis=0),
        helper.make_node(
            "ConstantOfShape", ["logits_shape"], ["draft_logits"]),
        helper.make_node(
            "Concat", ["token_shape", "topk_dimension"], ["topk_shape"],
            axis=0),
        helper.make_node(
            "Expand", ["topk_scores", "topk_shape"], ["draft_topk_log_scores"]),
        helper.make_node(
            "Gather", ["token_shape", "sequence_axis"], ["sequence_dimension"],
            axis=0),
        helper.make_node(
            "Concat",
            ["batch_and_heads", "sequence_dimension", "hidden_dimension"],
            ["new_cache_shape"],
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape", ["new_cache_shape"], ["new_key"]),
        helper.make_node(
            "ConstantOfShape", ["new_cache_shape"], ["new_value"]),
        helper.make_node(
            "Concat", ["past_key", "new_key"], ["present_key"], axis=2),
        helper.make_node(
            "Concat", ["past_value", "new_value"], ["present_value"], axis=2),
    ]
    eagle_graph = helper.make_graph(
        eagle_nodes, "tiny_eagle_drafter", eagle_inputs, eagle_outputs,
        eagle_initializers)
    eagle_model = helper.make_model(
        eagle_graph, opset_imports=[helper.make_opsetid("", 20)])
    eagle_model.ir_version = 10
    onnx.save(eagle_model, directory / "eagle.onnx")

    decoder = {
        "filename": "target.onnx",
        "session_options": {"provider_options": []},
        "hidden_size": hidden_size,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "head_size": hidden_size,
        "inputs": {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "attention_bias": "attention_bias",
            "position_ids": "position_ids",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        },
        "outputs": {
            "logits": "logits",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
    }
    eagle = {
        "filename": "eagle.onnx",
        "session_options": {"provider_options": []},
        "hidden_size": hidden_size,
        "draft_vocab_size": vocab_size,
        "num_key_value_heads": 1,
        "head_size": hidden_size,
        "total_tokens": 60,
        "depth": 7,
        "top_k": 10,
        "target_hidden_state_names": feature_names,
    }
    model_config = {
        "type": "decoder",
        "vocab_size": vocab_size,
        "context_length": 128,
        "bos_token_id": 0,
        "eos_token_id": [eos_token_id],
        "pad_token_id": 0,
        "decoder": decoder,
        "eagle": eagle,
    }
    eagle_config = {
        "model": model_config,
        "search": {"max_length": 128},
        "speculative": {"max_draft_tokens": 8},
    }
    eagle_path = _write_config(directory, eagle_config)

    target_only_directory = directory / "target-only"
    target_only_directory.mkdir()
    (target_only_directory / "target.onnx").write_bytes(
        (directory / "target.onnx").read_bytes())
    target_only_config = copy.deepcopy(eagle_config)
    del target_only_config["model"]["eagle"]
    target_path = _write_config(target_only_directory, target_only_config)
    return eagle_path, target_path


def _generate(model_path: str, prompt: list[int], max_length: int):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)
    params.set_speculative_options(max_draft_tokens=8)
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([prompt], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    return (
        [int(token) for token in generator.get_sequence(0)],
        generator.get_speculative_stats(),
    )


def _generate_steps(model_path: str, prompt: list[int], steps: int):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=128)
    params.set_speculative_options(max_draft_tokens=8)
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([prompt], dtype=np.int32))
    for _ in range(steps):
        generator.generate_next_token()
    return (
        [int(token) for token in generator.get_sequence(0)],
        generator.get_speculative_stats(),
    )


def _make_generator(model_path: str, prompt: list[int], max_length: int):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)
    params.set_speculative_options(max_draft_tokens=8)
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([prompt], dtype=np.int32))
    return generator


@pytest.mark.parametrize(
    ("acceptance", "accepted"),
    [("zero", 0), ("partial", 1), ("full", 8)],
)
def test_eagle_acceptance_branches_match_target(
    tmp_path, acceptance, accepted
):
    eagle_path, target_path = _make_tiny_eagle_model(
        tmp_path / acceptance, acceptance)
    prompt = [1, 2]

    steps = accepted + 2  # tree root, accepted drafts, then correction/bonus
    eagle_tokens, stats = _generate_steps(eagle_path, prompt, steps)
    target_tokens, _ = _generate_steps(target_path, prompt, steps)

    assert eagle_tokens == target_tokens
    assert stats["draft_tokens_proposed"] == 59
    assert stats["draft_tokens_accepted"] == accepted
    assert stats["target_verify_forward_passes"] == 1


def test_eagle_multiple_tree_rounds_match_target(tmp_path):
    eagle_path, target_path = _make_tiny_eagle_model(tmp_path / "multi-round")

    eagle_tokens, stats = _generate(eagle_path, [1, 2], 14)
    target_tokens, _ = _generate(target_path, [1, 2], 14)

    assert eagle_tokens == target_tokens
    assert stats["rounds"] == 2
    assert stats["draft_tokens_proposed"] == 118
    assert stats["target_verify_forward_passes"] == 2


def test_eagle_get_logits_mid_round_is_side_effect_free(tmp_path):
    model_path, target_path = _make_tiny_eagle_model(tmp_path / "get-logits")
    generator = _make_generator(model_path, [1, 2], 12)
    generator.generate_next_token()

    first = generator.get_logits().copy()
    second = generator.get_logits().copy()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 1, 16)
    assert int(np.argmax(first)) == 4

    while not generator.is_done():
        generator.generate_next_token()
    target_tokens, _ = _generate(target_path, [1, 2], 12)
    assert [int(token) for token in generator.get_sequence(0)] == target_tokens


def test_eagle_set_logits_mid_round_reconditions_tree_root(tmp_path):
    model_path, target_path = _make_tiny_eagle_model(tmp_path / "set-logits")
    generator = _make_generator(model_path, [1, 2], 10)
    generator.generate_next_token()

    forced_token = 9
    forced_logits = np.full((1, 1, 16), -1e9, dtype=np.float32)
    forced_logits[0, 0, forced_token] = 1e9
    generator.set_logits(forced_logits)
    generator.generate_next_token()
    assert int(generator.get_sequence(0)[-1]) == forced_token

    while not generator.is_done():
        generator.generate_next_token()
    clean, _ = _generate(target_path, [1, 2, 3, forced_token], 10)
    assert [int(token) for token in generator.get_sequence(0)] == clean


def test_eagle_stops_on_eos_inside_buffered_round(tmp_path):
    eagle_path, target_path = _make_tiny_eagle_model(
        tmp_path / "eos", "full", eos_token_id=4)
    prompt = [1, 2]

    eagle_tokens, _ = _generate(eagle_path, prompt, 10)
    target_tokens, _ = _generate(target_path, prompt, 10)

    assert eagle_tokens == target_tokens == [1, 2, 3]


def test_eagle_clamps_buffered_round_to_max_length(tmp_path):
    eagle_path, target_path = _make_tiny_eagle_model(tmp_path / "max-length")
    prompt = [1, 2]

    eagle_tokens, _ = _generate(eagle_path, prompt, 5)
    target_tokens, _ = _generate(target_path, prompt, 5)

    assert eagle_tokens == target_tokens
    assert len(eagle_tokens) == 5


def test_eagle_rewind_to_zero_replays_cleanly(tmp_path):
    eagle_path, _ = _make_tiny_eagle_model(tmp_path / "rewind-zero")
    prompt = [1, 2]
    generator = _make_generator(eagle_path, prompt, 8)
    while not generator.is_done():
        generator.generate_next_token()
    first = [int(token) for token in generator.get_sequence(0)]

    generator.rewind_to(0)
    generator.append_tokens(np.array([prompt], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    replay = [int(token) for token in generator.get_sequence(0)]

    assert replay == first


def test_eagle_nonzero_rewind_matches_clean_prefix(tmp_path):
    eagle_path, target_path = _make_tiny_eagle_model(tmp_path / "rewind-prefix")
    prompt = [1, 2]
    generator = _make_generator(eagle_path, prompt, 8)
    while not generator.is_done():
        generator.generate_next_token()

    generator.rewind_to(4)
    while not generator.is_done():
        generator.generate_next_token()
    rewound = [int(token) for token in generator.get_sequence(0)]
    clean, _ = _generate(target_path, [1, 2, 3, 4], 8)

    assert rewound == clean


def test_eagle_mid_round_append_matches_clean_prefix(tmp_path):
    eagle_path, target_path = _make_tiny_eagle_model(tmp_path / "append")
    generator = _make_generator(eagle_path, [1, 2], 8)
    generator.generate_next_token()  # bootstrap root
    generator.generate_next_token()  # first token from a still-buffered full-accept round
    assert [int(token) for token in generator.get_sequence(0)] == [1, 2, 3, 4]

    generator.append_tokens(np.array([[6]], dtype=np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    continued = [int(token) for token in generator.get_sequence(0)]
    clean, _ = _generate(target_path, [1, 2, 3, 4, 6], 8)

    assert continued == clean


@pytest.mark.parametrize(
    "path",
    [
        ("inputs", "input_ids"),
        ("inputs", "target_hidden_states"),
        ("inputs", "recurrent_hidden_states"),
        ("inputs", "use_target_hidden_states"),
        ("inputs", "attention_mask"),
        ("inputs", "attention_bias"),
        ("inputs", "position_ids"),
        ("inputs", "past_key"),
        ("inputs", "past_value"),
        ("outputs", "draft_hidden_states"),
        ("outputs", "draft_logits"),
        ("outputs", "draft_token_id"),
        ("outputs", "mapped_token_id"),
        ("outputs", "draft_topk_ids"),
        ("outputs", "draft_topk_log_scores"),
        ("outputs", "mapped_topk_ids"),
        ("outputs", "present_key"),
        ("outputs", "present_value"),
    ],
)
def test_eagle_rejects_wrong_graph_io_name(tmp_path, path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "bad-io")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    section, name = path
    config["model"]["eagle"].setdefault(section, {})[name] = f"missing_{name}"
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="was not found"):
        og.Model(model_path)


def test_eagle_rejects_wrong_target_feature_name(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "bad-feature")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["eagle"]["target_hidden_state_names"][1] = "missing_feature"
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="target hidden-state output"):
        og.Model(model_path)


def test_eagle_requires_three_target_features(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "feature-count")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["eagle"]["target_hidden_state_names"].pop()
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="exactly three"):
        og.Model(model_path)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("total_tokens", 59),
        ("depth", 6),
        ("top_k", 9),
    ],
)
def test_eagle_rejects_noncanonical_tree_topology(tmp_path, name, value):
    model_path, _ = _make_tiny_eagle_model(tmp_path / f"bad-tree-{name}")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["eagle"][name] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="total_tokens=60, depth=7, and top_k=10"):
        og.Model(model_path)


def test_eagle_requires_target_attention_bias(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "missing-target-bias")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    del config["model"]["decoder"]["inputs"]["attention_bias"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="decoder.inputs.attention_bias"):
        og.Model(model_path)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("hidden_size", 3, "target hidden-state output"),
        ("draft_vocab_size", 7, "draft_logits"),
        ("num_key_value_heads", 2, "past_key"),
        ("head_size", 3, "past_key"),
    ],
)
def test_eagle_rejects_dimension_mismatch(tmp_path, name, value, message):
    model_path, _ = _make_tiny_eagle_model(tmp_path / f"bad-{name}")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["eagle"][name] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match=message):
        og.Model(model_path)


def test_eagle_rejects_draft_model_combination(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "draft-and-eagle")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["draft"] = copy.deepcopy(config["model"]["decoder"])
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="cannot both"):
        og.Model(model_path)


def test_eagle_accepts_explicit_cpu_provider_options(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "explicit-cpu")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    for section in ("decoder", "eagle"):
        config["model"][section]["session_options"]["provider_options"] = [
            {"cpu": {}}
        ]
    config_path.write_text(json.dumps(config))

    og.Model(model_path)


def test_eagle_rejects_cross_provider_configuration(tmp_path):
    model_path, _ = _make_tiny_eagle_model(tmp_path / "cross-provider")
    config_path = Path(model_path) / "genai_config.json"
    config = json.loads(config_path.read_text())
    config["model"]["eagle"]["session_options"]["provider_options"] = [
        {"cuda": {}}
    ]
    config_path.write_text(json.dumps(config))

    with pytest.raises(Exception, match="same execution provider"):
        og.Model(model_path)


@pytest.mark.parametrize(
    ("search_options", "speculative_options", "message"),
    [
        ({"do_sample": True}, {}, "greedy"),
        ({"repetition_penalty": 1.1}, {}, "repetition_penalty"),
        ({"min_length": 2}, {}, "min_length"),
        ({}, {"max_draft_tokens": 3}, "max_draft_tokens=8"),
        ({}, {"adaptive_k_bool": True}, "adaptive K"),
    ],
)
def test_eagle_rejects_unsupported_v0_options(
    tmp_path, search_options, speculative_options, message
):
    model_path, _ = _make_tiny_eagle_model(tmp_path / f"unsupported-{message}")
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=8, **search_options)
    options = {"max_draft_tokens": 8}
    options.update(speculative_options)
    params.set_speculative_options(**options)

    with pytest.raises(Exception, match=message):
        og.Generator(model, params)
