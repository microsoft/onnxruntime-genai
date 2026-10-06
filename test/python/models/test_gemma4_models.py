# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for Gemma4 multimodal model.
Tests cover model loading, text-only processing, and image understanding.

This file can be used in two ways:
1. As a pytest module: pytest test_gemma4_models.py --test_models=/path/to/models
2. As a standalone runner: python test_gemma4_models.py --cwd test/python --test_models test/models
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("gemma4-tests")

GEMMA4_MODEL_NAME = "gemma4"
GEMMA4_IMAGE_TOKEN = "<|image|>"
GEMMA4_IMAGE_TOKEN_ID = 256001


def _get_gemma4_model_path(test_data_path):
    """Return the Gemma4 model path, skipping if it doesn't exist."""
    model_path = os.path.join(test_data_path, GEMMA4_MODEL_NAME)
    if not os.path.exists(model_path):
        pytest.skip(f"Gemma4 test model not found at {model_path}")
    return model_path


def _get_onnx_path(test_data_path, filename):
    """Return a path to a dummy ONNX file under the Gemma4 model dir, skipping if missing."""
    path = os.path.join(test_data_path, GEMMA4_MODEL_NAME, filename)
    if not os.path.exists(path):
        pytest.skip(f"Gemma4 ONNX file not found at {path}")
    return path


def _load_model_and_processor(test_data_path):
    """Load the Gemma4 model and create its multimodal processor."""
    model_path = _get_gemma4_model_path(test_data_path)
    model = og.Model(model_path)
    return model, model.create_multimodal_processor()


def _to_numpy(tensor):
    """Convert an onnxruntime-genai tensor to a numpy array."""
    if hasattr(tensor, "as_numpy"):
        return tensor.as_numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.array(tensor)


def _get_test_media_path(test_data_path, relative_path):
    return Path(test_data_path).parent / relative_path


def _register_gemma4_image_token(model_path):
    """Register the processor's image soft token in the compact test tokenizer."""
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"].append(
        {
            "id": GEMMA4_IMAGE_TOKEN_ID,
            "content": GEMMA4_IMAGE_TOKEN,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    tokenizer["model"]["vocab"][GEMMA4_IMAGE_TOKEN] = GEMMA4_IMAGE_TOKEN_ID
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")


def _create_static_batch_vision_model(onnx, output_path, num_patches="num_patches"):
    """Create a static-B=1 vision model that removes padding and pools 3x3 patches."""
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    pixel_values = helper.make_tensor_value_info("pixel_values", tensor_proto.FLOAT, [1, num_patches, 768])
    position_ids = helper.make_tensor_value_info("pixel_position_ids", tensor_proto.INT64, [1, num_patches, 2])
    image_features = helper.make_tensor_value_info("image_features", tensor_proto.FLOAT, ["num_soft_tokens", 2048])

    nodes = [
        helper.make_node("Gather", ["pixel_position_ids", "x_axis"], ["x_positions"], axis=2),
        helper.make_node("Greater", ["x_positions", "negative_one"], ["valid_mask"]),
        helper.make_node("NonZero", ["valid_mask"], ["valid_indices_transposed"]),
        helper.make_node("Transpose", ["valid_indices_transposed"], ["valid_indices"], perm=[1, 0]),
        helper.make_node("GatherND", ["pixel_values", "valid_indices"], ["valid_patches"]),
        helper.make_node(
            "Slice", ["valid_patches", "slice_start", "slice_end", "slice_axis", "slice_step"], ["pooled_patches"]
        ),
        helper.make_node("Pad", ["pooled_patches", "feature_padding", "zero"], ["image_features"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.array(0, dtype=np.int64), "x_axis"),
        onnx.numpy_helper.from_array(np.array(-1, dtype=np.int64), "negative_one"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_start"),
        onnx.numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), "slice_end"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_axis"),
        onnx.numpy_helper.from_array(np.array([9], dtype=np.int64), "slice_step"),
        onnx.numpy_helper.from_array(np.array([0, 0, 0, 1280], dtype=np.int64), "feature_padding"),
        onnx.numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
    ]
    graph = helper.make_graph(
        nodes, "gemma4_static_batch_vision", [pixel_values, position_ids], [image_features], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)
    onnx.save(model, output_path)


def _create_static_batch_vision_pipeline(onnx, encoder_path, projector_path, num_patches="num_patches"):
    """Split the static-B=1 vision fixture at a vision_features boundary."""
    helper = onnx.helper
    tensor_proto = onnx.TensorProto

    pixel_values = helper.make_tensor_value_info("pixel_values", tensor_proto.FLOAT, [1, num_patches, 768])
    position_ids = helper.make_tensor_value_info("pixel_position_ids", tensor_proto.INT64, [1, num_patches, 2])
    vision_features = helper.make_tensor_value_info("vision_features", tensor_proto.FLOAT, [1, num_patches, 768])
    encoder_graph = helper.make_graph(
        [helper.make_node("Identity", ["pixel_values"], ["vision_features"])],
        "gemma4_vision_encoder",
        [pixel_values, position_ids],
        [vision_features],
    )
    encoder = helper.make_model(encoder_graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)
    onnx.save(encoder, encoder_path)

    image_features = helper.make_tensor_value_info("image_features", tensor_proto.FLOAT, ["num_soft_tokens", 2048])
    projector_nodes = [
        helper.make_node("Gather", ["pixel_position_ids", "x_axis"], ["x_positions"], axis=2),
        helper.make_node("Greater", ["x_positions", "negative_one"], ["valid_mask"]),
        helper.make_node("NonZero", ["valid_mask"], ["valid_indices_transposed"]),
        helper.make_node("Transpose", ["valid_indices_transposed"], ["valid_indices"], perm=[1, 0]),
        helper.make_node("GatherND", ["vision_features", "valid_indices"], ["valid_patches"]),
        helper.make_node(
            "Slice", ["valid_patches", "slice_start", "slice_end", "slice_axis", "slice_step"], ["pooled_patches"]
        ),
        helper.make_node("Pad", ["pooled_patches", "feature_padding", "zero"], ["image_features"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.array(0, dtype=np.int64), "x_axis"),
        onnx.numpy_helper.from_array(np.array(-1, dtype=np.int64), "negative_one"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_start"),
        onnx.numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), "slice_end"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_axis"),
        onnx.numpy_helper.from_array(np.array([9], dtype=np.int64), "slice_step"),
        onnx.numpy_helper.from_array(np.array([0, 0, 0, 1280], dtype=np.int64), "feature_padding"),
        onnx.numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
    ]
    projector_graph = helper.make_graph(
        projector_nodes,
        "gemma4_vision_projector",
        [vision_features, position_ids],
        [image_features],
        initializers,
    )
    projector = helper.make_model(projector_graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)
    onnx.save(projector, projector_path)


def _create_dynamic_embedding_model(
    onnx,
    output_path,
    image_token_id,
    *,
    feature_type=None,
    feature_rank=2,
    consume_features=True,
):
    """Create an embedding fixture that places image features at image-token positions."""
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    feature_type = feature_type or tensor_proto.FLOAT
    input_ids = helper.make_tensor_value_info("input_ids", tensor_proto.INT32, ["batch_size", "sequence_length"])
    feature_shape = ["num_image_tokens", "hidden_size"]
    if feature_rank == 3:
        feature_shape.insert(0, 1)
    image_features = helper.make_tensor_value_info("image_features", feature_type, feature_shape)
    inputs_embeds = helper.make_tensor_value_info(
        "inputs_embeds", tensor_proto.FLOAT, ["batch_size", "sequence_length", 2048]
    )
    nodes = [
        helper.make_node("Shape", ["input_ids"], ["input_shape"]),
        helper.make_node("Concat", ["input_shape", "hidden_size"], ["output_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["output_shape"],
            ["inputs_embeds"],
            value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32)),
        ),
    ]
    graph_inputs = [input_ids]
    initializers = [
        onnx.numpy_helper.from_array(np.array([2048], dtype=np.int64), "hidden_size"),
        onnx.numpy_helper.from_array(np.array(image_token_id, dtype=np.int32), "image_token_id"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "squeeze_axis"),
    ]
    if consume_features:
        graph_inputs.append(image_features)
        nodes.extend(
            [
                helper.make_node("Equal", ["input_ids", "image_token_id"], ["image_mask"]),
                helper.make_node("NonZero", ["image_mask"], ["image_indices_transposed"]),
                helper.make_node("Transpose", ["image_indices_transposed"], ["image_indices"], perm=[1, 0]),
            ]
        )
        features_for_scatter = "image_features"
        if feature_rank == 3:
            nodes.append(helper.make_node("Squeeze", [features_for_scatter, "squeeze_axis"], ["rank2_image_features"]))
            features_for_scatter = "rank2_image_features"
        if feature_type != tensor_proto.FLOAT:
            nodes.append(
                helper.make_node("Cast", [features_for_scatter], ["float_image_features"], to=tensor_proto.FLOAT)
            )
            features_for_scatter = "float_image_features"
        nodes.append(
            helper.make_node(
                "ScatterND", ["inputs_embeds_empty", "image_indices", features_for_scatter], ["inputs_embeds"]
            )
        )
        nodes[2].output[0] = "inputs_embeds_empty"

    graph = helper.make_graph(nodes, "gemma4_dynamic_embedding", graph_inputs, [inputs_embeds], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)
    onnx.save(model, output_path)


def _create_dynamic_decoder_model(onnx, output_path, *, with_kv_cache=True):
    """Create a compact decoder whose sampled token reveals whether features were merged."""
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    inputs = [
        helper.make_tensor_value_info("inputs_embeds", tensor_proto.FLOAT, ["batch_size", "sequence_length", 2048]),
        helper.make_tensor_value_info("attention_mask", tensor_proto.INT64, ["batch_size", "total_sequence_length"]),
        helper.make_tensor_value_info("position_ids", tensor_proto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info(
            "past_key_values.0.key", tensor_proto.FLOAT, ["batch_size", 4, "past_sequence_length", 256]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.value", tensor_proto.FLOAT, ["batch_size", 4, "past_sequence_length", 256]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", tensor_proto.FLOAT, ["batch_size", "sequence_length", 8]),
        helper.make_tensor_value_info(
            "present.0.key", tensor_proto.FLOAT, ["batch_size", 4, "total_sequence_length", 256]
        ),
        helper.make_tensor_value_info(
            "present.0.value", tensor_proto.FLOAT, ["batch_size", 4, "total_sequence_length", 256]
        ),
    ]
    nodes = [
        helper.make_node("Shape", ["inputs_embeds"], ["embeds_shape"]),
        helper.make_node("Slice", ["embeds_shape", "zero_index", "two_index"], ["batch_sequence_shape"]),
        helper.make_node("Mul", ["inputs_embeds", "inputs_embeds"], ["squared_embeds"]),
        helper.make_node("ReduceSum", ["squared_embeds", "reduce_axes"], ["feature_score"], keepdims=1),
        helper.make_node("Concat", ["batch_sequence_shape", "one_index"], ["score_shape"], axis=0),
        helper.make_node("Expand", ["feature_score", "score_shape"], ["expanded_feature_score"]),
        helper.make_node("Concat", ["batch_sequence_shape", "two_index"], ["prefix_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["prefix_shape"],
            ["logits_prefix"],
            value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32)),
        ),
        helper.make_node("Concat", ["batch_sequence_shape", "five_index"], ["suffix_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape",
            ["suffix_shape"],
            ["logits_suffix"],
            value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32)),
        ),
        helper.make_node("Concat", ["logits_prefix", "expanded_feature_score", "logits_suffix"], ["logits"], axis=2),
        helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
        helper.make_node("Slice", ["mask_shape", "zero_index", "one_index"], ["batch_dim"]),
        helper.make_node("Slice", ["mask_shape", "one_index", "two_index"], ["total_sequence_dim"]),
        helper.make_node(
            "Concat", ["batch_dim", "num_heads", "total_sequence_dim", "head_size"], ["present_shape"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["present_shape"],
            ["present.0.key"],
            value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32)),
        ),
        helper.make_node("Identity", ["present.0.key"], ["present.0.value"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_index"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "one_index"),
        onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "two_index"),
        onnx.numpy_helper.from_array(np.array([5], dtype=np.int64), "five_index"),
        onnx.numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "reduce_axes"),
        onnx.numpy_helper.from_array(np.array([4], dtype=np.int64), "num_heads"),
        onnx.numpy_helper.from_array(np.array([256], dtype=np.int64), "head_size"),
    ]
    if not with_kv_cache:
        inputs = inputs[:3]
        outputs = outputs[:1]
        nodes = nodes[:-6]
        initializers = initializers[:-2]
    graph = helper.make_graph(nodes, "gemma4_dynamic_decoder", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7)
    onnx.save(model, output_path)


def test_gemma4_model_load(test_data_path):
    """Test that the Gemma4 model loads successfully."""
    model_path = _get_gemma4_model_path(test_data_path)
    model = og.Model(model_path)
    assert model is not None


def test_gemma4_text_only(test_data_path):
    """Test text-only processing (no images)."""
    _, processor = _load_model_and_processor(test_data_path)

    inputs = processor("What is the capital of France?", images=None)

    assert inputs is not None
    assert "input_ids" in inputs

    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2, f"input_ids should be 2D, got shape {ids.shape}"
    assert ids.shape[0] == 1, f"input_ids batch dim should be 1, got {ids.shape[0]}"
    assert ids.shape[1] >= 5, f"input_ids too short for prompt, got length {ids.shape[1]}"


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_vision_basic(test_data_path, relative_image_path):
    """Test basic image processing with Gemma4."""
    _, processor = _load_model_and_processor(test_data_path)

    image_path = os.fspath(_get_test_media_path(test_data_path, relative_image_path))
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    images = og.Images.open(image_path)

    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}Describe this image", images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "input_ids" in inputs

    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2, f"input_ids should be 2D, got shape {ids.shape}"
    assert ids.shape[0] == 1, f"input_ids batch dim should be 1, got {ids.shape[0]}"
    assert ids.shape[1] > 0, "input_ids should not be empty"


@pytest.mark.parametrize("relative_image_path", [Path("images") / "landscape.jpg"])
def test_gemma4_vision_load_from_bytes(test_data_path, relative_image_path):
    """Test loading images from bytes for Gemma4."""
    _, processor = _load_model_and_processor(test_data_path)

    image_path = os.fspath(_get_test_media_path(test_data_path, relative_image_path))
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    with open(image_path, "rb") as f:
        images = og.Images.open_bytes(f.read())

    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}What is shown in this image?", images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "landscape.jpg"]],
)
def test_gemma4_vision_multiple_images(test_data_path, relative_image_paths):
    """Test that Gemma4 preserves per-image metadata in input image order."""
    _, processor = _load_model_and_processor(test_data_path)

    image_paths = [os.fspath(_get_test_media_path(test_data_path, path)) for path in relative_image_paths]
    for p in image_paths:
        if not os.path.exists(p):
            pytest.skip(f"Test image not found at {p}")
    images = og.Images.open(*image_paths)

    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}{GEMMA4_IMAGE_TOKEN}Compare these images", images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "pixel_position_ids" in inputs
    assert "num_image_tokens" in inputs
    assert "input_ids" in inputs

    pixel_values = _to_numpy(inputs["pixel_values"])
    pixel_position_ids = _to_numpy(inputs["pixel_position_ids"])
    image_token_counts = _to_numpy(inputs["num_image_tokens"])

    assert pixel_values.ndim == 3
    assert pixel_values.shape[0] == len(image_paths)
    assert pixel_position_ids.shape[:2] == pixel_values.shape[:2]
    assert image_token_counts.shape == (len(image_paths),)
    assert np.all(image_token_counts > 0)
    assert pixel_values.shape[1] == int(image_token_counts.max()) * 9

    valid_patch_counts = np.count_nonzero(np.any(pixel_position_ids != -1, axis=-1), axis=1)
    assert np.all(valid_patch_counts % 9 == 0)
    expected_token_counts = valid_patch_counts // 9
    np.testing.assert_array_equal(image_token_counts, expected_token_counts)


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "sheet.png"]],
)
def test_gemma4_static_batch_vision_executes_multiple_images(test_data_path, tmp_path, relative_image_paths):
    """Test that the static-B=1 vision model runs once per differently sized image."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4"
    shutil.copytree(source_model_path, model_path)
    _register_gemma4_image_token(model_path)

    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["speech"] = {"filename": "", "config_filename": ""}
    config["model"]["vocab_size"] = 8
    config["model"]["eos_token_id"] = [1]
    config["model"]["image_token_id"] = GEMMA4_IMAGE_TOKEN_ID
    config["search"]["past_present_share_buffer"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _create_static_batch_vision_model(onnx, model_path / "dummy_vision.onnx")
    _create_dynamic_embedding_model(onnx, model_path / "dummy_embedding.onnx", config["model"]["image_token_id"])
    _create_dynamic_decoder_model(onnx, model_path / "dummy_text.onnx")

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    image_paths = [os.fspath(_get_test_media_path(test_data_path, path)) for path in relative_image_paths]
    images = og.Images.open(*image_paths)
    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}{GEMMA4_IMAGE_TOKEN}Compare these images", images=images)
    image_token_counts = _to_numpy(inputs["num_image_tokens"])
    assert image_token_counts[0] != image_token_counts[1]

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)


def test_gemma4_split_vision_requires_decoder_pipeline(test_data_path, tmp_path):
    """A split vision export cannot use the flat-decoder multimodal model."""
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4"
    shutil.copytree(source_model_path, model_path)

    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["vision"]["pipeline"] = {
        "encoder": {
            "filename": "dummy_vision_encoder.onnx",
        },
        "projector": {
            "filename": "dummy_vision_projector.onnx",
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"split vision requires decoder\.pipeline"):
        og.Model(os.fspath(model_path))


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "sheet.png"]],
)
def test_gemma4_fixed_patch_vision_pads_multiple_images(test_data_path, tmp_path, relative_image_paths):
    """Test that processor output is padded to the vision model's fixed patch capacity."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    image_paths = [os.fspath(_get_test_media_path(test_data_path, path)) for path in relative_image_paths]
    images = og.Images.open(*image_paths)

    source_model = og.Model(os.fspath(source_model_path))
    source_processor = source_model.create_multimodal_processor()
    source_inputs = source_processor("<|image|><|image|>Compare these images", images=images)
    image_token_counts = _to_numpy(source_inputs["num_image_tokens"])
    fixed_num_patches = (int(image_token_counts.max()) + 1) * 9

    model_path = tmp_path / "gemma4"
    shutil.copytree(source_model_path, model_path)
    _register_gemma4_image_token(model_path)
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["speech"] = {"filename": "", "config_filename": ""}
    config["model"]["vocab_size"] = 8
    config["model"]["eos_token_id"] = [1]
    config["model"]["image_token_id"] = GEMMA4_IMAGE_TOKEN_ID
    config["search"]["past_present_share_buffer"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _create_static_batch_vision_model(onnx, model_path / "dummy_vision.onnx", fixed_num_patches)
    _create_dynamic_embedding_model(onnx, model_path / "dummy_embedding.onnx", config["model"]["image_token_id"])
    _create_dynamic_decoder_model(onnx, model_path / "dummy_text.onnx")

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    inputs = processor("<|image|><|image|>Compare these images", images=images)
    pixel_values = _to_numpy(inputs["pixel_values"])
    pixel_position_ids = _to_numpy(inputs["pixel_position_ids"])

    assert pixel_values.shape == (len(image_paths), fixed_num_patches, 768)
    assert pixel_position_ids.shape == (len(image_paths), fixed_num_patches, 2)
    assert np.all(pixel_values[:, -9:, :] == 0)
    assert np.all(pixel_position_ids[:, -9:, :] == -1)
    np.testing.assert_array_equal(_to_numpy(inputs["num_image_tokens"]), image_token_counts)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    generator.generate_next_token()


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_fixed_patch_vision_rejects_too_many_patches(test_data_path, tmp_path, relative_image_path):
    """Test that valid image patches are never truncated to fit a fixed vision input."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    image_path = os.fspath(_get_test_media_path(test_data_path, relative_image_path))
    images = og.Images.open(image_path)

    source_model = og.Model(os.fspath(source_model_path))
    source_processor = source_model.create_multimodal_processor()
    source_inputs = source_processor("<|image|>Describe this image", images=images)
    actual_num_patches = int(_to_numpy(source_inputs["num_image_tokens"])[0]) * 9

    model_path = tmp_path / "gemma4"
    shutil.copytree(source_model_path, model_path)
    _create_static_batch_vision_model(onnx, model_path / "dummy_vision.onnx", actual_num_patches - 9)

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    with pytest.raises(
        RuntimeError, match="vision model accepts .* patches, but preprocessing produced .* valid patches"
    ):
        processor("<|image|>Describe this image", images=images)


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_processor_creates_token_type_ids(test_data_path, relative_image_path):
    """Test that Gemma4 processor creates token_type_ids for image prompts."""
    _, processor = _load_model_and_processor(test_data_path)

    image_path = os.fspath(_get_test_media_path(test_data_path, relative_image_path))
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    images = og.Images.open(image_path)

    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}Describe this image", images=images)

    assert inputs is not None
    assert "token_type_ids" in inputs


def test_gemma4_vision_model_io(test_data_path):
    """Validate the vision ONNX model has expected inputs and outputs."""
    onnx = pytest.importorskip("onnx")

    model = onnx.load(_get_onnx_path(test_data_path, "dummy_vision.onnx"))
    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "pixel_values" in input_names
    assert "pixel_position_ids" in input_names
    assert "image_features" in output_names

    pv_input = next(i for i in model.graph.input if i.name == "pixel_values")
    assert pv_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "pixel_values must be float32"

    dim0 = pv_input.type.tensor_type.shape.dim[0]
    assert dim0.dim_param != "", f"pixel_values dim-0 should be dynamic, got static dim_value={dim0.dim_value}"


def test_gemma4_embedding_model_io(test_data_path):
    """Validate the embedding ONNX model has expected inputs and outputs."""
    onnx = pytest.importorskip("onnx")

    model = onnx.load(_get_onnx_path(test_data_path, "dummy_embedding.onnx"))
    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "input_ids" in input_names
    assert "image_features" in input_names
    assert "inputs_embeds" in output_names


def test_gemma4_text_model_io(test_data_path):
    """Validate the text/decoder ONNX model has expected inputs and outputs."""
    onnx = pytest.importorskip("onnx")

    model = onnx.load(_get_onnx_path(test_data_path, "dummy_text.onnx"))
    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "inputs_embeds" in input_names, "Decoder must accept inputs_embeds"
    assert "attention_mask" in input_names
    assert "position_ids" in input_names
    assert "past_key_values.0.key" in input_names, "Decoder must have KV cache inputs"
    assert "past_key_values.0.value" in input_names

    assert "logits" in output_names
    assert "present.0.key" in output_names, "Decoder must have KV cache outputs"
    assert "present.0.value" in output_names


def test_gemma4_speech_model_io(test_data_path):
    """Validate the speech encoder ONNX model has expected inputs and outputs."""
    onnx = pytest.importorskip("onnx")

    model = onnx.load(_get_onnx_path(test_data_path, "dummy_speech.onnx"))
    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "audio_embeds" in input_names, "Speech model must have audio_embeds input"
    assert "audio_sizes" in input_names, "Speech model must have audio_sizes input"
    assert "audio_features" in output_names, "Speech model must have audio_features output"

    # audio_embeds should be float32 with shape (batch, num_frames, 128)
    ae_input = next(i for i in model.graph.input if i.name == "audio_embeds")
    assert ae_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "audio_embeds must be float32"
    assert ae_input.type.tensor_type.shape.dim[2].dim_value == 128, "audio_embeds feature dim should be 128"

    # audio_sizes should be int64
    as_input = next(i for i in model.graph.input if i.name == "audio_sizes")
    assert as_input.type.tensor_type.elem_type == onnx.TensorProto.INT64, "audio_sizes must be int64"


@pytest.mark.parametrize("relative_audio_path", [Path("audios") / "jfk.flac"])
def test_gemma4_audio_preprocessing(test_data_path, relative_audio_path):
    """Test audio preprocessing with Gemma4 (Gemma4LogMel feature extraction)."""
    _, processor = _load_model_and_processor(test_data_path)

    audio_path = os.fspath(Path(test_data_path) / relative_audio_path)
    if not os.path.exists(audio_path):
        pytest.skip(f"Test audio file not found at {audio_path}")

    audios = og.Audios.open(audio_path)
    prompt = "<|audio|>Transcribe this audio"
    inputs = processor(prompt, audios=audios)

    assert inputs is not None
    assert "input_ids" in inputs

    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2, f"input_ids should be 2D, got shape {ids.shape}"
    assert ids.shape[0] == 1, f"input_ids batch dim should be 1, got {ids.shape[0]}"
    # Audio prompt expands <|audio|> tokens based on audio duration,
    # so sequence length should be significantly longer than just the text tokens.
    # "Transcribe this audio" = 4 text tokens + BOS + expanded audio tokens
    assert ids.shape[1] > 5, f"input_ids should contain expanded audio tokens, got length {ids.shape[1]}"

    # Audio preprocessing should produce audio_embeds and attention mask
    assert "audio_embeds" in inputs, "Processor should output audio_embeds"
    assert "audio_attention_mask" in inputs, "Processor should output audio attention mask"

    # Validate audio_embeds structure: should be float with 128-dim features
    audio_embeds = _to_numpy(inputs["audio_embeds"])
    assert len(audio_embeds.shape) >= 2, f"audio_embeds should be at least 2D, got {audio_embeds.shape}"
    assert audio_embeds.shape[-1] == 128, f"audio_embeds feature dim should be 128, got {audio_embeds.shape[-1]}"
    assert audio_embeds.dtype == np.float32, f"audio_embeds should be float32, got {audio_embeds.dtype}"

    # Validate audio_sizes is present and positive
    assert "audio_sizes" in inputs, "Processor should output audio_sizes"
    audio_sizes = _to_numpy(inputs["audio_sizes"])
    assert audio_sizes[0] > 0, f"audio_sizes should be positive, got {audio_sizes[0]}"


def _write_pipelined_gemma4(
    onnx,
    source_model_path,
    model_path,
    *,
    feature_type=None,
    feature_rank=2,
    consume_features=True,
    with_kv_cache=True,
    **config_overrides,
):
    """Copy the Gemma4 fixture and rewrite it as a pipelined decoder.

    A pipelined decoder routes the config to Qwen2_5_VL_PipelineModel, which supports
    either a single vision session or a split encoder/projector.
    """
    shutil.copytree(source_model_path, model_path)
    _register_gemma4_image_token(model_path)
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["speech"] = {"filename": "", "config_filename": ""}
    config["model"]["vocab_size"] = 8
    config["model"]["eos_token_id"] = [1]
    config["model"]["image_token_id"] = GEMMA4_IMAGE_TOKEN_ID
    config["search"]["past_present_share_buffer"] = False
    vision = config["model"]["vision"]
    vision["filename"] = "dummy_vision.onnx"
    vision.pop("pipeline", None)
    vision["inputs"] = {"pixel_values": "pixel_values", "pixel_position_ids": "pixel_position_ids"}
    vision["outputs"] = {"image_features": "image_features"}
    _create_static_batch_vision_model(onnx, model_path / "dummy_vision.onnx")
    _create_dynamic_embedding_model(
        onnx,
        model_path / "dummy_embedding.onnx",
        config["model"]["image_token_id"],
        feature_type=feature_type,
        feature_rank=feature_rank,
        consume_features=consume_features,
    )
    _create_dynamic_decoder_model(onnx, model_path / "dummy_text.onnx", with_kv_cache=with_kv_cache)

    decoder = config["model"]["decoder"]
    decoder.pop("filename", None)
    decoder["pipeline"] = [
        {
            "embedding": {
                "filename": "dummy_embedding.onnx",
                "inputs": ["input_ids", *(["image_features"] if consume_features else [])],
                "outputs": ["inputs_embeds"],
            },
            "text": {
                "filename": "dummy_text.onnx",
                "inputs": [
                    "inputs_embeds",
                    "attention_mask",
                    "position_ids",
                    "past_key_values.0.key",
                    "past_key_values.0.value",
                ],
                "outputs": ["logits", "present.0.key", "present.0.value"],
            },
        }
    ]
    if not with_kv_cache:
        text_stage = decoder["pipeline"][0]["text"]
        text_stage["inputs"] = text_stage["inputs"][:3]
        text_stage["outputs"] = ["logits"]

    for key, value in config_overrides.items():
        config["model"][key] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_pipelined_gemma4_fixture_selects_dummy_vision_graph(test_data_path, tmp_path):
    """Inherited vision graph settings must not change what the pipeline fixture executes."""
    onnx = pytest.importorskip("onnx")
    source_model_path = tmp_path / "source"
    shutil.copytree(_get_gemma4_model_path(test_data_path), source_model_path)
    source_config_path = source_model_path / "genai_config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_config["model"]["vision"].update(
        filename="other_vision.onnx",
        pipeline=[{"other": {"filename": "other_vision.onnx"}}],
        inputs={"pixel_values": "other_pixels"},
        outputs={"image_features": "other_features"},
    )
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")

    config_path = _write_pipelined_gemma4(onnx, source_model_path, tmp_path / "pipeline")
    vision = json.loads(config_path.read_text(encoding="utf-8"))["model"]["vision"]
    assert vision["filename"] == "dummy_vision.onnx"
    assert not vision.get("pipeline")
    assert vision["inputs"] == {"pixel_values": "pixel_values", "pixel_position_ids": "pixel_position_ids"}
    assert vision["outputs"] == {"image_features": "image_features"}


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "sheet.png"]],
)
@pytest.mark.parametrize("feature_rank,feature_dtype", [(2, "float32"), (3, "float16")])
@pytest.mark.parametrize("window_size", [None, 64])
def test_gemma4_pipelined_decoder_runs_single_session_vision(
    test_data_path, tmp_path, relative_image_paths, feature_rank, feature_dtype, window_size
):
    """A pipelined Gemma4 decoder must still encode every image and bind the features.

    Covers Qwen2_5_VL_PipelineState::RunSingleSessionVision: the vision graph has a static
    batch of 1, so two differently sized images have to be encoded one at a time and their
    features concatenated before the embedding stage consumes them.
    """
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-pipeline"
    config_path = _write_pipelined_gemma4(
        onnx,
        source_model_path,
        model_path,
        feature_type=onnx.TensorProto.FLOAT16 if feature_dtype == "float16" else onnx.TensorProto.FLOAT,
        feature_rank=feature_rank,
        with_kv_cache=window_size is None,
    )
    if window_size is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["decoder"]["sliding_window"] = {
            "window_size": window_size,
            "slide_key_value_cache": False,
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    image_paths = [os.fspath(_get_test_media_path(test_data_path, path)) for path in relative_image_paths]
    for p in image_paths:
        if not os.path.exists(p):
            pytest.skip(f"Test image not found at {p}")
    images = og.Images.open(*image_paths)
    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}{GEMMA4_IMAGE_TOKEN}Compare these images", images=images)

    image_token_counts = _to_numpy(inputs["num_image_tokens"])
    assert image_token_counts[0] != image_token_counts[1], "fixture images must differ in token count"

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    pixel_values = _to_numpy(inputs["pixel_values"])
    positions = _to_numpy(inputs["pixel_position_ids"])
    expected_features = []
    for pixels, image_positions in zip(pixel_values, positions, strict=True):
        valid = image_positions[:, 0] > -1
        expected_features.append(np.pad(pixels[valid][::9], ((0, 0), (0, 1280))))
    expected_features = np.concatenate(expected_features).astype(feature_dtype).astype(np.float32)
    ids = _to_numpy(inputs["input_ids"])
    image_mask = ids == GEMMA4_IMAGE_TOKEN_ID
    assert np.count_nonzero(image_mask) == int(image_token_counts.sum())
    expected_embeds = np.zeros((*ids.shape, 2048), dtype=np.float32)
    expected_embeds[image_mask] = expected_features
    if window_size is not None:
        padding = (-ids.shape[1]) % window_size
        expected_embeds = np.pad(expected_embeds, ((0, 0), (padding, 0), (0, 0)))[:, -window_size:]
    np.testing.assert_allclose(
        generator.get_output("inputs_embeds"),
        expected_embeds,
        rtol=1e-3 if feature_dtype == "float16" else 0,
        atol=0,
    )
    generator.generate_next_token()
    assert generator.get_next_tokens() == [2]
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]
    np.testing.assert_array_equal(generator.get_output("inputs_embeds"), np.zeros((1, 1, 2048)))


def test_pipelined_mistral3_is_rejected_before_using_qwen_vision(test_data_path, tmp_path):
    """Pixtral's rank-four, per-image cropping contract must not enter the Qwen pipeline."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "mistral3-pipeline"
    _write_pipelined_gemma4(onnx, source_model_path, model_path, type="mistral3")
    with pytest.raises(RuntimeError, match="Pipelined decoder is not supported for model type 'mistral3'"):
        og.Model(os.fspath(model_path))


def test_text_only_pipelined_decoder_requires_append_tokens(test_data_path, tmp_path):
    """Text pipelines must not load a vision session or accept multimodal SetInputs."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    inputs = og.Model(os.fspath(source_model_path)).create_multimodal_processor()("Hello")
    model_path = tmp_path / "text-pipeline"
    _write_pipelined_gemma4(
        onnx,
        source_model_path,
        model_path,
        consume_features=False,
        type="decoder-pipeline",
        vision={"filename": "nonexistent_vision.onnx"},
    )

    model = og.Model(os.fspath(model_path))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=32)
    generator = og.Generator(model, params)
    with pytest.raises(RuntimeError, match="Please use generator.AppendTokens for decoder-pipeline"):
        generator.set_inputs(inputs)
    generator.append_tokens(np.array([0, 2], dtype=np.int32))
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]


def test_gemma4_pipelined_text_only_uses_rank3_float16_empty_features(test_data_path, tmp_path):
    """Text decode must receive a correctly ranked and typed empty feature tensor."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-text-only"
    _write_pipelined_gemma4(
        onnx,
        source_model_path,
        model_path,
        feature_type=onnx.TensorProto.FLOAT16,
        feature_rank=3,
    )

    model = og.Model(os.fspath(model_path))
    inputs = model.create_multimodal_processor()("Hello")
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=32)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    generator.generate_next_token()
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_pipelined_decoder_injects_features_when_embedding_has_no_feature_input(
    test_data_path, tmp_path, relative_image_path
):
    """Exports without image_features must retain post-embedding feature injection."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-injection"
    _write_pipelined_gemma4(onnx, source_model_path, model_path, consume_features=False)

    image_path = os.fspath(_get_test_media_path(test_data_path, relative_image_path))
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    model = og.Model(os.fspath(model_path))
    inputs = model.create_multimodal_processor()(
        f"{GEMMA4_IMAGE_TOKEN}Describe this image", images=og.Images.open(image_path)
    )
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    generator.generate_next_token()
    assert generator.get_next_tokens() == [2]
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]


@pytest.mark.parametrize("fixed_patches", [False, True])
def test_gemma4_pipelined_decoder_runs_split_vision(test_data_path, tmp_path, fixed_patches):
    """A decoder pipeline must also run both vision stages for every image."""
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-split-pipeline"
    _write_pipelined_gemma4(onnx, source_model_path, model_path)
    _create_static_batch_vision_pipeline(
        onnx,
        model_path / "dummy_vision_encoder.onnx",
        model_path / "dummy_vision_projector.onnx",
        2520 if fixed_patches else "num_patches",
    )
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision = config["model"]["vision"]
    vision.pop("filename", None)
    vision["pipeline"] = [
        {
            "encoder": {
                "filename": "dummy_vision_encoder.onnx",
                "inputs": ["pixel_values", "pixel_position_ids"],
                "outputs": ["vision_features"],
                "session_options": {"provider_options": []},
            },
            "projector": {
                "filename": "dummy_vision_projector.onnx",
                "inputs": ["vision_features", "pixel_position_ids"],
                "outputs": ["image_features"],
                "run_on_cpu": True,
            },
        }
    ]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    image_paths = [
        os.fspath(_get_test_media_path(test_data_path, Path("images") / name))
        for name in ("australia.jpg", "sheet.png")
    ]
    if not all(os.path.exists(path) for path in image_paths):
        pytest.skip("Gemma4 test images not available")
    inputs = processor("<|image|><|image|>Compare these images", images=og.Images.open(*image_paths))
    if fixed_patches:
        assert _to_numpy(inputs["pixel_values"]).shape[1] == 2520
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    generator.generate_next_token()
    assert len(generator.get_next_tokens()) == 1


def test_gemma4_pipelined_decoder_rejects_missing_image_features_output(test_data_path, tmp_path):
    """A configured image_features output that the vision graph lacks must fail loudly.

    Silently falling back to output 0 would feed whatever that output happens to be into the
    embedding stage as if it were image features.
    """
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-bad-features"
    vision = {
        "filename": "dummy_vision.onnx",
        "config_filename": "processor_config.json",
        "inputs": {"pixel_values": "pixel_values", "pixel_position_ids": "pixel_position_ids"},
        "outputs": {"image_features": "not_an_output_of_this_graph"},
        "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
    }
    _write_pipelined_gemma4(onnx, source_model_path, model_path, vision=vision)

    image_path = os.fspath(_get_test_media_path(test_data_path, Path("images") / "australia.jpg"))
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    inputs = processor(f"{GEMMA4_IMAGE_TOKEN}Describe this image", images=og.Images.open(image_path))

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    with pytest.raises(Exception, match="not_an_output_of_this_graph"):
        generator.set_inputs(inputs)


@pytest.mark.parametrize(
    "speech",
    [
        {"filename": "dummy_speech.onnx", "config_filename": "audio_feature_extraction.json"},
        {"filename": "dummy_speech.onnx", "config_filename": ""},
        {"filename": "", "config_filename": "audio_feature_extraction.json"},
    ],
)
def test_gemma4_pipelined_decoder_disables_declared_speech_encoder(test_data_path, tmp_path, speech):
    """A pipelined decoder has no speech session, so a declared one is disabled, not rejected.

    Gemma 4 exports ship an audio encoder beside the vision encoder, so this config shape is
    the normal one. Loading must succeed and serve text and image prompts: the unused audio
    encoder changes no result, because an unfilled modality is bound an empty [0, hidden]
    tensor and the in-graph merge is a no-op. Audio itself stays unavailable, since clearing
    the speech config stops the multimodal processor resolving audio inputs that no session
    provides.
    """
    onnx = pytest.importorskip("onnx")
    source_model_path = Path(_get_gemma4_model_path(test_data_path))
    model_path = tmp_path / "gemma4-pipeline-speech"
    _write_pipelined_gemma4(onnx, source_model_path, model_path, speech=speech)

    model = og.Model(os.fspath(model_path))
    processor = model.create_multimodal_processor()
    assert processor is not None
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=32)
    generator = og.Generator(model, params)
    generator.set_inputs(processor("Hello"))
    generator.generate_next_token()
    generator.generate_next_token()
    assert generator.get_next_tokens() == [0]


# Standalone runner functionality
def run_gemma4_vision_tests(
    cwd: str | bytes | os.PathLike,
    log: logging.Logger,
    test_models: str | bytes | os.PathLike,
):
    """Run the vision model tests using pytest."""
    log.debug("Running: Gemma4 Vision Model Tests")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        os.path.abspath(__file__),
        "--test_models",
        test_models,
    ]
    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def parse_arguments():
    """Parse command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Test runner for Gemma4 vision models")
    parser.add_argument(
        "--cwd",
        help="Path to the current working directory",
        default=Path(__file__).parent.resolve().absolute(),
    )
    parser.add_argument(
        "--test_models",
        help="Path to the 'models' directory",
        default=Path(__file__).parent.parent.resolve().absolute() / "models",
    )
    return parser.parse_args()


def main():
    """Main entry point for standalone execution."""
    args = parse_arguments()

    log.info("Running Gemma4 vision model tests")
    log.info(f"Test models path: {args.test_models}")
    log.info(f"Working directory: {args.cwd}")

    run_gemma4_vision_tests(os.path.abspath(args.cwd), log, os.path.abspath(args.test_models))

    log.info("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
