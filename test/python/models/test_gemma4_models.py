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
import logging
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("gemma4-tests")

GEMMA4_MODEL_NAME = "gemma4"


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

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    images = og.Images.open(image_path)

    inputs = processor("<|image|>Describe this image", images=images)

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

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    with open(image_path, "rb") as f:
        images = og.Images.open_bytes(f.read())

    inputs = processor("<|image|>What is shown in this image?", images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "landscape.jpg"]],
)
def test_gemma4_vision_multiple_images(test_data_path, relative_image_paths):
    """Test processing multiple images with Gemma4."""
    _, processor = _load_model_and_processor(test_data_path)

    image_paths = [os.fspath(Path(test_data_path) / p) for p in relative_image_paths]
    for p in image_paths:
        if not os.path.exists(p):
            pytest.skip(f"Test image not found at {p}")
    images = og.Images.open(*image_paths)

    inputs = processor("<|image|><|image|>Compare these images", images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "input_ids" in inputs

    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2, f"input_ids should be 2D, got shape {ids.shape}"
    assert ids.shape[0] == 1, f"input_ids batch dim should be 1, got {ids.shape[0]}"
    assert ids.shape[1] > 0, "input_ids should not be empty"


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_gemma4_processor_creates_token_type_ids(test_data_path, relative_image_path):
    """Test that Gemma4 processor creates token_type_ids for image prompts."""
    _, processor = _load_model_and_processor(test_data_path)

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
    images = og.Images.open(image_path)

    inputs = processor("<|image|>Describe this image", images=images)

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
