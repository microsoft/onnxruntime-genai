# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
LFM2-VL image token accounting tests.

Three pieces have to agree on how many decoder tokens an image is worth, or the vision features and
the `<image>` placeholders in the prompt will not line up:

  * the Hugging Face processor (`Lfm2VlProcessor._compute_tokens_for_image`), which decides how many
    placeholders the reference implementation writes,
  * the smart resize in onnxruntime-extensions, which decides what resolution the image arrives at,
  * `Lfm2VlImageProcessor` in C++, which counts tokens from that resolution.

These tests are pure arithmetic reimplementations of all three, so they run without a model. The
integration test at the bottom exercises the real C++ processor when an LFM2-VL model is available.

Run with:
    python -m pytest test/python/models/test_lfm2_vl_tokens.py -v
"""

import math
import os

import pytest

# Defaults from LiquidAI/LFM2.5-VL-1.6B's config.json and processor_config.json.
ENCODER_PATCH_SIZE = 16
DOWNSAMPLE_FACTOR = 2
MIN_IMAGE_TOKENS = 64
MAX_IMAGE_TOKENS = 256
TOTAL_FACTOR = ENCODER_PATCH_SIZE * DOWNSAMPLE_FACTOR
MIN_PIXELS = MIN_IMAGE_TOKENS * ENCODER_PATCH_SIZE**2 * DOWNSAMPLE_FACTOR**2
MAX_PIXELS = MAX_IMAGE_TOKENS * ENCODER_PATCH_SIZE**2 * DOWNSAMPLE_FACTOR**2
# max_image_tokens * downsample_factor^2: the fixed length every image is padded to.
MAX_NUM_PATCHES = MAX_IMAGE_TOKENS * DOWNSAMPLE_FACTOR**2


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def hf_smart_resize(height: int, width: int) -> tuple[int, int]:
    """Python reference for `Lfm2VlImageProcessor.smart_resize`, returning (height, width)."""
    h_bar = max(TOTAL_FACTOR, round_by_factor(height, TOTAL_FACTOR))
    w_bar = max(TOTAL_FACTOR, round_by_factor(width, TOTAL_FACTOR))
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(TOTAL_FACTOR, math.floor(height / beta / TOTAL_FACTOR) * TOTAL_FACTOR)
        w_bar = max(TOTAL_FACTOR, math.floor(width / beta / TOTAL_FACTOR) * TOTAL_FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / TOTAL_FACTOR) * TOTAL_FACTOR
        w_bar = math.ceil(width * beta / TOTAL_FACTOR) * TOTAL_FACTOR
    return h_bar, w_bar


def hf_tokens_for_image(height: int, width: int) -> int:
    """Python reference for `Lfm2VlProcessor._compute_tokens_for_image`."""
    patches_h = math.ceil((height // ENCODER_PATCH_SIZE) / DOWNSAMPLE_FACTOR)
    patches_w = math.ceil((width // ENCODER_PATCH_SIZE) / DOWNSAMPLE_FACTOR)
    return patches_h * patches_w


def genai_image_geometry(height: int, width: int) -> tuple[int, int, int, int]:
    """Python reference for C++ `ComputeLfm2VlImageGeometry`.

    Returns (patch_rows, patch_cols, num_patches, num_tokens).
    """
    assert height % ENCODER_PATCH_SIZE == 0 and width % ENCODER_PATCH_SIZE == 0
    patch_rows = height // ENCODER_PATCH_SIZE
    patch_cols = width // ENCODER_PATCH_SIZE
    tokens = math.ceil(patch_rows / DOWNSAMPLE_FACTOR) * math.ceil(patch_cols / DOWNSAMPLE_FACTOR)
    return patch_rows, patch_cols, patch_rows * patch_cols, tokens


def build_image_placeholder(num_tokens: int) -> str:
    """Python reference for C++ `BuildLfm2VlImagePlaceholder`."""
    return "<|image_start|>" + "<image>" * num_tokens + "<|image_end|>"


IMAGE_SIZES = [
    (1, 1),
    (64, 64),
    (224, 224),
    (256, 384),
    (512, 512),
    (640, 480),
    (1024, 768),
    (1920, 1080),
    (4000, 3000),
    (100, 2000),
]


class TestImageGeometry:
    """The patch grid and token count the C++ processor derives from a resized image."""

    def test_tile_sized_image_is_worth_max_tokens(self):
        assert genai_image_geometry(512, 512) == (32, 32, 1024, MAX_IMAGE_TOKENS)

    def test_non_square_image_keeps_per_axis_grid(self):
        assert genai_image_geometry(256, 384) == (16, 24, 384, 8 * 12)

    def test_smallest_image_is_one_token(self):
        assert genai_image_geometry(32, 32) == (2, 2, 4, 1)

    def test_odd_patch_grid_rounds_tokens_up(self):
        assert genai_image_geometry(16 * 5, 16 * 3)[3] == 3 * 2

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_matches_hugging_face_token_count(self, height, width):
        """After smart resize, both implementations must agree on the token count."""
        resized_h, resized_w = hf_smart_resize(height, width)
        assert genai_image_geometry(resized_h, resized_w)[3] == hf_tokens_for_image(resized_h, resized_w)

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_resized_image_fits_the_padded_patch_budget(self, height, width):
        """max_num_patches has to cover every resolution smart resize can produce."""
        resized_h, resized_w = hf_smart_resize(height, width)
        num_patches = genai_image_geometry(resized_h, resized_w)[2]
        assert num_patches <= MAX_NUM_PATCHES

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_resized_image_is_a_whole_number_of_patches(self, height, width):
        """The C++ processor rejects sizes that are not, so smart resize must never produce one."""
        resized_h, resized_w = hf_smart_resize(height, width)
        assert resized_h % ENCODER_PATCH_SIZE == 0
        assert resized_w % ENCODER_PATCH_SIZE == 0

    @pytest.mark.parametrize(("height", "width"), IMAGE_SIZES)
    def test_token_count_stays_within_the_configured_range(self, height, width):
        resized_h, resized_w = hf_smart_resize(height, width)
        tokens = genai_image_geometry(resized_h, resized_w)[3]
        assert 1 <= tokens <= MAX_IMAGE_TOKENS


class TestImagePlaceholder:
    """The prompt rewriting that pairs each image with its vision features."""

    def test_placeholder_wraps_image_tokens_in_start_and_end_markers(self):
        assert build_image_placeholder(3) == "<|image_start|><image><image><image><|image_end|>"

    def test_placeholder_token_count_matches_the_image(self):
        tokens = genai_image_geometry(*hf_smart_resize(640, 480))[3]
        assert build_image_placeholder(tokens).count("<image>") == tokens


def test_genai_processor_token_counts():
    """End-to-end check against the C++ processor when an LFM2-VL model is available."""
    og = pytest.importorskip("onnxruntime_genai")

    model_path = os.environ.get("LFM2_VL_MODEL_PATH")
    if not model_path or not os.path.exists(os.path.join(model_path, "genai_config.json")):
        pytest.skip("Set LFM2_VL_MODEL_PATH to an LFM2-VL ONNX model directory to run this test")
    image_path = os.environ.get("LFM2_VL_IMAGE_PATH")
    if not image_path or not os.path.exists(image_path):
        pytest.skip("Set LFM2_VL_IMAGE_PATH to an image file to run this test")

    config = og.Config(model_path)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    images = og.Images.open(image_path)
    inputs = processor(prompt="<image>describe this", images=images)

    pixel_values = inputs["pixel_values"].as_numpy()
    # The processor keys its outputs by the nominal names; they are mapped to the vision graph's
    # input names (spatial_shapes, pixel_attention_mask) only when handed to the generator.
    spatial_shapes = inputs["image_sizes"].as_numpy()
    pixel_attention_mask = inputs["image_attention_mask"].as_numpy()
    num_image_tokens = int(inputs["num_image_tokens"].as_numpy().sum())

    assert pixel_values.ndim == 3
    assert pixel_values.shape[2] == ENCODER_PATCH_SIZE * ENCODER_PATCH_SIZE * 3
    assert pixel_attention_mask.shape == pixel_values.shape[:2]

    patch_rows, patch_cols = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
    # Only the real patches are unmasked, and the token count follows the same grid.
    assert int(pixel_attention_mask[0].sum()) == patch_rows * patch_cols
    assert num_image_tokens == math.ceil(patch_rows / DOWNSAMPLE_FACTOR) * math.ceil(patch_cols / DOWNSAMPLE_FACTOR)
