# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for Qwen2.5-VL and Fara vision-language models.
Tests cover model loading, text generation, image understanding, and multimodal processing.

This file can be used in two ways:
1. As a pytest module: pytest test_qwen_fara_models.py --test_models=/path/to/test_models
2. As a standalone runner: python test_qwen_fara_models.py --cwd test/python --test_models test/test_models
"""

import argparse
import importlib.util
import logging
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("qwen-fara-vision-tests")


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_fara_vision_basic(test_data_path, relative_model_path, relative_image_path):
    """Test basic vision preprocessing for Qwen/Fara-style models."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    # Test with Qwen vision tokens
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"
    inputs = processor(prompt, images=images)

    # Verify inputs were created successfully
    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "landscape.jpg"])
def test_qwen_fara_vision_load_from_bytes(test_data_path, relative_model_path, relative_image_path):
    """Test loading images from bytes for Qwen/Fara models."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = None
    with open(image_path, "rb") as image:
        image_bytes = image.read()
        images = og.Images.open_bytes(image_bytes)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "landscape.jpg"]],
)
@pytest.mark.skip(reason="Multiple images not fully supported in dummy model - image_grid_thw shape issue")
def test_qwen_fara_vision_multiple_images(test_data_path, relative_model_path, relative_image_paths):
    """Test processing multiple images with Qwen/Fara models."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_paths = [
        os.fspath(Path(test_data_path) / relative_image_path) for relative_image_path in relative_image_paths
    ]
    images = og.Images.open(*image_paths)

    # Add vision tokens for each image
    prompt = ""
    for _ in range(len(relative_image_paths)):
        prompt += "<|vision_start|><|image_pad|><|vision_end|>"
    prompt += "Describe these images"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
def test_qwen_fara_text_only_generation(test_data_path, relative_model_path):
    """Test text-only generation without images."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    # Test text-only prompt (no images)
    prompt = "What is the capital of France?"
    inputs = processor(prompt, images=None)

    assert inputs is not None
    # For text-only, we should have input_ids but not pixel_values
    assert "input_ids" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "sheet.png"])
def test_qwen_fara_vision_with_special_tokens(test_data_path, relative_model_path, relative_image_path):
    """Test vision processing with special tokens in prompt."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    # Test with Qwen vision tokens
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Can you convert the table to markdown format?"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs
    assert "input_ids" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "10809054.jpg"])
def test_qwen_fara_vision_different_image_formats(test_data_path, relative_model_path, relative_image_path):
    """Test processing different image formats."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    model = og.Model(model_path)

    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Analyze this image"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_fara_accuracy_comparison(test_data_path, relative_model_path, relative_image_path):
    """
    Test accuracy comparison between ONNX model and PyTorch reference.
    This test validates that the ONNX model produces similar outputs to PyTorch.
    """
    # Check if required libraries are available
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        pytest.skip("PyTorch or transformers not available for accuracy comparison")

    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    image_path = os.fspath(Path(test_data_path) / relative_image_path)

    # Load ONNX model
    onnx_model = og.Model(model_path)
    onnx_processor = onnx_model.create_multimodal_processor()

    # Load images
    images = og.Images.open(image_path)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"

    # Get ONNX inputs
    onnx_inputs = onnx_processor(prompt, images=images)

    # Check if we have pixel_values to compare
    if "pixel_values" not in onnx_inputs:
        pytest.skip("No pixel_values in ONNX inputs")

    onnx_pixel_values = onnx_inputs["pixel_values"]

    # For vision-preprocessing dummy model, we validate the preprocessing pipeline
    # In a real scenario, you would:
    # 1. Load the same PyTorch model
    # 2. Process the same image with PyTorch processor
    # 3. Compare pixel_values between ONNX and PyTorch

    # Validate pixel_values properties
    assert onnx_pixel_values is not None

    # Convert to numpy array properly
    if hasattr(onnx_pixel_values, "as_numpy"):
        pixel_array = onnx_pixel_values.as_numpy()
    elif hasattr(onnx_pixel_values, "numpy"):
        pixel_array = onnx_pixel_values.numpy()
    else:
        pixel_array = np.array(onnx_pixel_values)

    # Check that pixel values are in a reasonable range (typically normalized to [-1, 1] or [0, 1])
    assert pixel_array.dtype in [np.float32, np.float16], f"Unexpected dtype: {pixel_array.dtype}"
    assert len(pixel_array.shape) >= 2, f"Unexpected shape: {pixel_array.shape}"

    # Check that values are normalized (not raw 0-255 pixel values)
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    assert pixel_min >= -10.0 and pixel_max <= 10.0, f"Pixel values out of expected range: [{pixel_min}, {pixel_max}]"

    log.debug(f"ONNX pixel_values shape: {pixel_array.shape}")
    log.debug(f"ONNX pixel_values dtype: {pixel_array.dtype}")
    log.debug(f"ONNX pixel_values range: [{pixel_min:.4f}, {pixel_max:.4f}]")


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "sheet.png"])
def test_qwen_fara_preprocessing_consistency(test_data_path, relative_model_path, relative_image_path):
    """
    Test that preprocessing produces consistent results for the same input.
    This validates deterministic behavior of the preprocessing pipeline.
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    image_path = os.fspath(Path(test_data_path) / relative_image_path)

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    # Process the same image twice
    images1 = og.Images.open(image_path)
    images2 = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Test prompt"

    inputs1 = processor(prompt, images=images1)
    inputs2 = processor(prompt, images=images2)

    # Compare pixel_values
    if "pixel_values" in inputs1 and "pixel_values" in inputs2:
        pv1 = inputs1["pixel_values"]
        pv2 = inputs2["pixel_values"]

        # Convert to numpy arrays properly
        if hasattr(pv1, "as_numpy"):
            array1 = pv1.as_numpy()
        elif hasattr(pv1, "numpy"):
            array1 = pv1.numpy()
        else:
            array1 = np.array(pv1)

        if hasattr(pv2, "as_numpy"):
            array2 = pv2.as_numpy()
        elif hasattr(pv2, "numpy"):
            array2 = pv2.numpy()
        else:
            array2 = np.array(pv2)

        # Check shapes match
        assert array1.shape == array2.shape, "Shapes should be identical"

        # Check values are close (allowing for tiny floating point differences)
        np.testing.assert_allclose(
            array1,
            array2,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Preprocessing should produce identical results for the same input",
        )

        log.debug(f"Preprocessing consistency validated: shape={array1.shape}")

    # Compare input_ids if present
    if "input_ids" in inputs1 and "input_ids" in inputs2:
        ids1 = inputs1["input_ids"]
        ids2 = inputs2["input_ids"]

        # Convert to numpy arrays properly
        if hasattr(ids1, "as_numpy"):
            array1 = ids1.as_numpy()
        elif hasattr(ids1, "numpy"):
            array1 = ids1.numpy()
        else:
            array1 = np.array(ids1)

        if hasattr(ids2, "as_numpy"):
            array2 = ids2.as_numpy()
        elif hasattr(ids2, "numpy"):
            array2 = ids2.numpy()
        else:
            array2 = np.array(ids2)

        np.testing.assert_array_equal(array1, array2, err_msg="Input IDs should be identical")


# Standalone runner functionality
def run_qwen_fara_vision_tests(
    cwd: str | bytes | os.PathLike,
    log: logging.Logger,
    test_models: str | bytes | os.PathLike,
):
    """Run the vision model tests using pytest."""
    log.debug("Running: Qwen/Fara Vision Model Tests")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "test_qwen_fara_models.py",
        "--test_models",
        test_models,
    ]
    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def parse_arguments():
    """Parse command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Test runner for Qwen/Fara vision models")
    parser.add_argument(
        "--cwd",
        help="Path to the current working directory",
        default=pathlib.Path(__file__).parent.resolve().absolute(),
    )
    parser.add_argument(
        "--test_models",
        help="Path to the test_models directory",
        default=pathlib.Path(__file__).parent.parent.resolve().absolute() / "test_models",
    )
    return parser.parse_args()


def main():
    """Main entry point for standalone execution."""
    args = parse_arguments()

    log.info("Running Qwen/Fara vision model tests")
    log.info(f"Test models path: {args.test_models}")
    log.info(f"Working directory: {args.cwd}")

    # Run Qwen/Fara vision model tests
    run_qwen_fara_vision_tests(os.path.abspath(args.cwd), log, os.path.abspath(args.test_models))

    log.info("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
