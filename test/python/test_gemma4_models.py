# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for Gemma4 multimodal model.
Tests cover model loading, text-only processing, and image understanding.

Usage:
  pytest test_gemma4_models.py --test_models=test/test_models
"""

import logging
import os
from pathlib import Path

import onnxruntime_genai as og
import pytest

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("gemma4-tests")

GEMMA4_MODEL_PATH = Path("gemma4-vision-preprocessing")


def test_gemma4_model_load(test_data_path):
    """Test that the Gemma4 model loads successfully."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)
    assert model is not None


def test_gemma4_text_only(test_data_path):
    """Test text-only processing (no images)."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()
    prompt = "What is the capital of France?"
    inputs = processor(prompt, images=None)

    assert inputs is not None
    assert "input_ids" in inputs


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_vision_basic(test_data_path, relative_image_path):
    """Test basic image processing with Gemma4."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<start_of_image>Describe this image"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "input_ids" in inputs


@pytest.mark.parametrize("relative_image_path", [Path("images") / "landscape.jpg"])
def test_gemma4_vision_load_from_bytes(test_data_path, relative_image_path):
    """Test loading images from bytes for Gemma4."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    images = og.Images.open_bytes(image_bytes)

    prompt = "<start_of_image>What is shown in this image?"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "landscape.jpg"]],
)
def test_gemma4_vision_multiple_images(test_data_path, relative_image_paths):
    """Test processing multiple images with Gemma4."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_paths = [
        os.fspath(Path(test_data_path) / p) for p in relative_image_paths
    ]
    images = og.Images.open(*image_paths)

    prompt = "<start_of_image><start_of_image>Compare these images"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "input_ids" in inputs


def test_gemma4_processor_creates_token_type_ids(test_data_path):
    """Test that Gemma4 processor creates token_type_ids for image prompts."""
    model_path = os.fspath(Path(test_data_path) / GEMMA4_MODEL_PATH)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / "images" / "australia.jpg")
    images = og.Images.open(image_path)

    prompt = "<start_of_image>Describe this image"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "token_type_ids" in inputs


def test_gemma4_vision_model_io(test_data_path):
    """Validate the vision ONNX model has expected inputs and outputs.

    Gemma4 vision model requires:
    - pixel_values (float32, dynamic num_patches)
    - pixel_position_ids (int64, dynamic num_patches)
    - image_features output (float32, dynamic num_soft_tokens)
    """
    onnx = pytest.importorskip("onnx")

    vision_path = os.path.join(
        test_data_path, "gemma4-vision-preprocessing", "dummy_vision.onnx"
    )
    model = onnx.load(vision_path)

    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "pixel_values" in input_names, "Vision model must have pixel_values input"
    assert "pixel_position_ids" in input_names, "Vision model must have pixel_position_ids input"
    assert "image_features" in output_names, "Vision model must have image_features output"

    # pixel_values should be float32
    pv_input = next(i for i in model.graph.input if i.name == "pixel_values")
    assert pv_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, \
        "pixel_values must be float32"

    # pixel_values dim-0 should be dynamic (num_patches)
    dim0 = pv_input.type.tensor_type.shape.dim[0]
    assert dim0.dim_param != "", \
        f"pixel_values dim-0 should be dynamic, got static dim_value={dim0.dim_value}"


def test_gemma4_embedding_model_io(test_data_path):
    """Validate the embedding ONNX model has expected inputs and outputs.

    Gemma4 embedding model requires input_ids, image_features, and token_type_ids.
    """
    onnx = pytest.importorskip("onnx")

    emb_path = os.path.join(
        test_data_path, "gemma4-vision-preprocessing", "dummy_embedding.onnx"
    )
    model = onnx.load(emb_path)

    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}

    assert "input_ids" in input_names
    assert "image_features" in input_names
    assert "token_type_ids" in input_names, "Gemma4 embedding requires token_type_ids"
    assert "inputs_embeds" in output_names


def test_gemma4_text_model_io(test_data_path):
    """Validate the text/decoder ONNX model has expected inputs and outputs.

    Gemma4 decoder uses inputs_embeds (not input_ids), attention_mask,
    position_ids, and KV cache inputs/outputs.
    """
    onnx = pytest.importorskip("onnx")

    text_path = os.path.join(
        test_data_path, "gemma4-vision-preprocessing", "dummy_text.onnx"
    )
    model = onnx.load(text_path)

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
