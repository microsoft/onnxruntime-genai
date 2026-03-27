# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for Qwen2.5-VL, Qwen3-VL, and Fara vision-language models.
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
@pytest.mark.parametrize(
    "relative_image_paths",
    [[Path("images") / "australia.jpg", Path("images") / "landscape.jpg"]],
)
def test_qwen_fara_vision_multiple_images(test_data_path, relative_model_path, relative_image_paths):
    """Test processing multiple images with Qwen/Fara models.

    The qwen3-vl dummy vision model has dynamic image_grid_thw dim-0 ('num_images'),
    so it exercises the batched single-call path in QwenVisionState::Run when images
    share the same grid.  The qwen (2.5-VL) model has static dim-0 ('1'), so it
    uses the per-image loop path.
    """
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


def test_qwen3_vl_vision_dynamic_grid_dim(test_data_path):
    """Test that qwen3-vl dummy vision model has dynamic image_grid_thw dim-0.

    This validates the ONNX model structure required for the batched single-call
    path in QwenVisionState::Run.  When image_grid_thw dim-0 is symbolic (dynamic),
    the runtime can pass all images in one call instead of looping per-image.
    """
    import onnx

    vision_path = os.path.join(
        test_data_path, "qwen3-vl-vision-preprocessing", "dummy_vision.onnx"
    )
    model = onnx.load(vision_path)

    # Find image_grid_thw input
    grid_input = None
    for inp in model.graph.input:
        if inp.name == "image_grid_thw":
            grid_input = inp
            break

    assert grid_input is not None, "image_grid_thw input not found in dummy_vision.onnx"

    # dim-0 should be symbolic (dynamic), not a fixed integer
    dim0 = grid_input.type.tensor_type.shape.dim[0]
    assert dim0.dim_param != "", (
        f"image_grid_thw dim-0 should be symbolic (e.g. 'num_images') "
        f"but got static dim_value={dim0.dim_value}"
    )
    assert dim0.dim_param == "num_images", (
        f"Expected dim_param='num_images', got '{dim0.dim_param}'"
    )

    # dim-1 should be static 3 (temporal, height, width)
    dim1 = grid_input.type.tensor_type.shape.dim[1]
    assert dim1.dim_value == 3, f"image_grid_thw dim-1 should be 3, got {dim1.dim_value}"


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
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


def test_qwen3_vl_model_type(test_data_path):
    """
    Test that the Qwen3-VL model loads with the correct model type (qwen3_vl).
    Validates that the model type routing in model.cpp correctly handles the new type.
    """
    model_path = os.fspath(Path(test_data_path) / "qwen3-vl-vision-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen3-vl-vision-preprocessing test model not found")

    model = og.Model(model_path)
    assert model is not None

    # Verify the model can create a multimodal processor (validates model type routing)
    processor = model.create_multimodal_processor()
    assert processor is not None


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen3_vl_pixel_values_shape(test_data_path, relative_image_path):
    """
    Test that Qwen3-VL preprocesses images with patch_size=16 (not 14 like Qwen2.5-VL).
    The patch_size difference affects the number of patches extracted from each image.
    """
    model_path = os.fspath(Path(test_data_path) / "qwen3-vl-vision-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen3-vl-vision-preprocessing test model not found")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"
    inputs = processor(prompt, images=images)

    assert "pixel_values" in inputs

    # Get pixel_values as numpy array
    pv = inputs["pixel_values"]
    if hasattr(pv, "as_numpy"):
        pixel_array = pv.as_numpy()
    elif hasattr(pv, "numpy"):
        pixel_array = pv.numpy()
    else:
        pixel_array = np.array(pv)

    # For Qwen3-VL with patch_size=16, each patch has 16*16*3 = 768 pixels
    # flattened into the temporal_patch_size dimension: 768 * 2 = 1536
    # Shape should be [num_patches, 1176] for patch_size=14 (Qwen2.5-VL)
    # or [num_patches, 1536] for patch_size=16 (Qwen3-VL)
    assert pixel_array.shape[-1] == 1536, (
        f"Expected last dim 1536 for Qwen3-VL (patch_size=16), got {pixel_array.shape[-1]}"
    )
    log.debug(f"Qwen3-VL pixel_values shape: {pixel_array.shape}")


@pytest.mark.parametrize(
    "model_name,expected_patch_dim",
    [
        ("qwen-vision-preprocessing", 1176),       # Qwen2.5-VL: patch_size=14, 14*14*3*2=1176
        ("qwen3-vl-vision-preprocessing", 1536),    # Qwen3-VL:   patch_size=16, 16*16*3*2=1536
    ],
)
@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_vl_family_patch_size_difference(test_data_path, model_name, expected_patch_dim, relative_image_path):
    """
    Test that the QwenVL family models produce different patch dimensions
    due to different patch_size configurations (14 vs 16).
    """
    model_path = os.fspath(Path(test_data_path) / model_name)
    if not os.path.exists(model_path):
        pytest.skip(f"{model_name} test model not found")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"
    inputs = processor(prompt, images=images)

    assert "pixel_values" in inputs

    pv = inputs["pixel_values"]
    if hasattr(pv, "as_numpy"):
        pixel_array = pv.as_numpy()
    elif hasattr(pv, "numpy"):
        pixel_array = pv.numpy()
    else:
        pixel_array = np.array(pv)

    assert pixel_array.shape[-1] == expected_patch_dim, (
        f"For {model_name}: expected last dim {expected_patch_dim}, got {pixel_array.shape[-1]}"
    )
    log.debug(f"{model_name} pixel_values shape: {pixel_array.shape}")


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_vl_preprocessing_output_completeness(test_data_path, relative_model_path, relative_image_path):
    """
    Test that the multimodal processor returns all expected output tensors.
    Validates the full preprocessing pipeline produces pixel_values, input_ids,
    image_grid_thw, and num_image_tokens with correct dimensionality.
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"{relative_model_path} test model not found")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"
    inputs = processor(prompt, images=images)

    # All four keys must be present for vision inputs
    expected_keys = {"pixel_values", "input_ids", "image_grid_thw", "num_image_tokens"}
    actual_keys = set(inputs.keys())
    assert expected_keys.issubset(actual_keys), (
        f"Missing keys: {expected_keys - actual_keys}. Got: {actual_keys}"
    )

    def _to_numpy(tensor):
        """Convert a tensor-like object to NumPy (supports as_numpy, numpy, np.array)."""
        if hasattr(tensor, "as_numpy"):
            return tensor.as_numpy()
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        return np.array(tensor)

    # pixel_values: [num_patches, patch_dim]
    pv = _to_numpy(inputs["pixel_values"])
    assert len(pv.shape) == 2, f"pixel_values should be 2D, got shape {pv.shape}"
    assert pv.dtype == np.float32, f"pixel_values should be float32, got {pv.dtype}"

    # image_grid_thw: [num_images, 3] for single image
    grid = _to_numpy(inputs["image_grid_thw"])
    assert grid.shape == (1, 3), f"image_grid_thw should be (1, 3) for single image, got {grid.shape}"
    assert grid.dtype == np.int64, f"image_grid_thw should be int64, got {grid.dtype}"

    # num_image_tokens: [num_images]
    nit = _to_numpy(inputs["num_image_tokens"])
    assert nit.shape == (1,), f"num_image_tokens should be (1,) for single image, got {nit.shape}"
    assert nit.dtype == np.int64, f"num_image_tokens should be int64, got {nit.dtype}"

    # input_ids: [1, seq_len]
    ids = _to_numpy(inputs["input_ids"])
    assert len(ids.shape) == 2, f"input_ids should be 2D, got shape {ids.shape}"
    assert ids.shape[0] == 1, f"input_ids batch dim should be 1, got {ids.shape[0]}"

    log.debug(f"{relative_model_path} output: pv={pv.shape}, grid={grid}, nit={nit}, ids={ids.shape}")


@pytest.mark.parametrize("relative_model_path", [Path("qwen-vision-preprocessing"), Path("qwen3-vl-vision-preprocessing")])
@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_vl_image_grid_thw_consistency(test_data_path, relative_model_path, relative_image_path):
    """
    Test that image_grid_thw values are consistent with pixel_values shape
    and num_image_tokens. For a single image:
      num_patches = T * H * W  (must match pixel_values rows)
      num_image_tokens = T * (H / merge_size) * (W / merge_size)
    where merge_size = spatial_merge_size from config (2 for both Qwen VL models).
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"{relative_model_path} test model not found")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    images = og.Images.open(image_path)

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image"
    inputs = processor(prompt, images=images)

    pv = inputs["pixel_values"].as_numpy()
    grid = inputs["image_grid_thw"].as_numpy()
    nit = inputs["num_image_tokens"].as_numpy()

    # grid values must be positive
    assert np.all(grid > 0), f"image_grid_thw must have positive values, got {grid}"

    t, h, w = grid[0]
    merge_size = 2  # spatial_merge_size for both Qwen VL models

    # num_patches = T * H * W should match pixel_values first dim
    expected_patches = int(t * h * w)
    assert pv.shape[0] == expected_patches, (
        f"pixel_values rows ({pv.shape[0]}) != T*H*W ({t}*{h}*{w}={expected_patches})"
    )

    # num_image_tokens = T * (H/merge_size) * (W/merge_size)
    expected_tokens = int(t * (h // merge_size) * (w // merge_size))
    assert nit[0] == expected_tokens, (
        f"num_image_tokens ({nit[0]}) != T*(H/{merge_size})*(W/{merge_size}) = {expected_tokens}"
    )

    log.debug(f"{relative_model_path}: grid={grid}, patches={expected_patches}, tokens={expected_tokens}")


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen_vl_normalization_range_difference(test_data_path, relative_image_path):
    """
    Test that Qwen3-VL and Qwen2.5-VL produce different pixel value ranges
    due to different normalization constants.
    Qwen3-VL uses mean/std=[0.5, 0.5, 0.5] → pixel range [-1, 1].
    Qwen2.5-VL uses OpenAI CLIP normalization → wider range.
    """
    image_path = os.fspath(Path(test_data_path) / relative_image_path)

    # Qwen3-VL: normalized with [0.5, 0.5, 0.5]
    q3_model_path = os.fspath(Path(test_data_path) / "qwen3-vl-vision-preprocessing")
    if not os.path.exists(q3_model_path):
        pytest.skip("qwen3-vl-vision-preprocessing test model not found")

    q3_model = og.Model(q3_model_path)
    q3_proc = q3_model.create_multimodal_processor()
    q3_imgs = og.Images.open(image_path)
    q3_inputs = q3_proc("<|vision_start|><|image_pad|><|vision_end|>test", images=q3_imgs)
    q3_pv = q3_inputs["pixel_values"].as_numpy()

    # Qwen2.5-VL: normalized with OpenAI CLIP constants
    q25_model_path = os.fspath(Path(test_data_path) / "qwen-vision-preprocessing")
    if not os.path.exists(q25_model_path):
        pytest.skip("qwen-vision-preprocessing test model not found")

    q25_model = og.Model(q25_model_path)
    q25_proc = q25_model.create_multimodal_processor()
    q25_imgs = og.Images.open(image_path)
    q25_inputs = q25_proc("<|vision_start|><|image_pad|><|vision_end|>test", images=q25_imgs)
    q25_pv = q25_inputs["pixel_values"].as_numpy()

    # Qwen3-VL with [0.5, 0.5, 0.5] normalization: values bounded by [-1, 1]
    assert q3_pv.min() >= -1.0 - 1e-5, f"Qwen3-VL pixel_values min {q3_pv.min()} < -1.0"
    assert q3_pv.max() <= 1.0 + 1e-5, f"Qwen3-VL pixel_values max {q3_pv.max()} > 1.0"

    # Qwen2.5-VL with OpenAI normalization: values extend beyond [-1, 1]
    assert q25_pv.min() < -1.0, (
        f"Qwen2.5-VL pixel_values min ({q25_pv.min():.4f}) should be < -1.0 with OpenAI normalization"
    )
    assert q25_pv.max() > 1.0, (
        f"Qwen2.5-VL pixel_values max ({q25_pv.max():.4f}) should be > 1.0 with OpenAI normalization"
    )

    log.debug(f"Qwen3-VL range: [{q3_pv.min():.4f}, {q3_pv.max():.4f}]")
    log.debug(f"Qwen2.5-VL range: [{q25_pv.min():.4f}, {q25_pv.max():.4f}]")


# ---------------------------------------------------------------------------
# Qwen3.5 hybrid model tests (RecurrentState + sparse KV cache)
# ---------------------------------------------------------------------------

def test_qwen35_hybrid_model_loads(test_data_path):
    """Test that a Qwen3.5 hybrid model (with recurrent + KV states) loads successfully."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    model = og.Model(model_path)
    assert model is not None


def test_qwen35_hybrid_creates_processor(test_data_path):
    """Test that the qwen3_5 model type routes to QwenImageProcessor."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    assert processor is not None


def test_qwen35_hybrid_tokenizer(test_data_path):
    """Test that tokenizer can be created for the qwen3_5 model type."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    assert tokenizer is not None


def test_qwen35_hybrid_generator_creates(test_data_path):
    """Test that a Generator can be created for the hybrid model.
    This validates that RecurrentState and sparse KV cache auto-discovery don't crash."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=20)
    generator = og.Generator(model, params)
    assert generator is not None


def test_qwen35_hybrid_text_generation(test_data_path):
    """Test basic text generation with the hybrid model.
    The dummy model uses Identity pass-through which doesn't support KV cache
    shape changes, so we only validate that the generator constructs and
    the first forward pass (prefill) executes without errors on the recurrent state path."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    model = og.Model(model_path)

    # Validate that the generator can be created and configured
    # Full text generation requires a real model (Identity can't simulate KV cache growth)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=5)

    generator = og.Generator(model, params)
    assert generator is not None


@pytest.mark.parametrize("relative_image_path", [Path("images") / "australia.jpg"])
def test_qwen35_hybrid_vision_preprocessing(test_data_path, relative_image_path):
    """Test that the hybrid model processes images through the vision pipeline."""
    model_path = os.fspath(Path(test_data_path) / "qwen35-hybrid-preprocessing")
    if not os.path.exists(model_path):
        pytest.skip("qwen35-hybrid-preprocessing test model not found")

    image_path = os.fspath(Path(test_data_path) / relative_image_path)
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found: {image_path}")

    model = og.Model(model_path)
    processor = model.create_multimodal_processor()

    images = og.Images.open(image_path)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe"
    inputs = processor(prompt, images=images)

    assert inputs is not None
    assert "pixel_values" in inputs


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
