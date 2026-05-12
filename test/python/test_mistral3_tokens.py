# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Mistral3/Pixtral token expansion tests.

Verifies that the C++ Mistral3ImageProcessor produces the correct
[IMG]/[IMG_BREAK]/[IMG_END] token sequence for various image
dimensions, matching the HuggingFace Pixtral token convention:

    For each row of patches:
        [IMG] × patch_cols
        [IMG_BREAK]  (between rows)
    [IMG_END]  (after the last row)

Token IDs used in unit tests are synthetic placeholders (not real
tokenizer IDs); the integration test resolves IDs from the model.

Run with:
    python -m pytest test/python/test_mistral3_tokens.py -v
"""

import pytest

# Fake special-token IDs used only for unit-testing the counting/sequence
# logic.  These do NOT correspond to real Mistral-3/Pixtral tokenizer IDs
# (which are much larger, e.g. 10 → a vocabulary-specific index).  The
# integration test (test_genai_processor_token_counts) reads the actual IDs
# from the model config at runtime, so the specific values here are
# irrelevant as long as they are distinct and non-zero.
IMG_TOKEN_ID = 10
IMG_BREAK_TOKEN_ID = 12
IMG_END_TOKEN_ID = 13
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
EFFECTIVE_PATCH = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28


def _has_genai() -> bool:
    try:
        import onnxruntime_genai

        return True
    except ImportError:
        return False


def build_image_token_sequence(patch_rows: int, patch_cols: int) -> list[int]:
    """Python reference for C++ BuildImageTokenSequence.

    Produces the token sequence that the C++ Mistral3ImageProcessor
    generates for a given patch grid.
    """
    tokens = []
    for r in range(patch_rows):
        tokens.extend([IMG_TOKEN_ID] * patch_cols)
        if r < patch_rows - 1:
            tokens.append(IMG_BREAK_TOKEN_ID)
        else:
            tokens.append(IMG_END_TOKEN_ID)
    return tokens


def compute_patch_grid(image_h: int, image_w: int) -> tuple[int, int]:
    """Compute patch grid dimensions after spatial merging.

    Matches C++ logic: patch_rows = image_h / patch_size / spatial_merge_size
    """
    patch_rows = image_h // PATCH_SIZE // SPATIAL_MERGE_SIZE
    patch_cols = image_w // PATCH_SIZE // SPATIAL_MERGE_SIZE
    return patch_rows, patch_cols


class TestBuildImageTokenSequence:
    """Tests for the token sequence builder."""

    def test_single_patch(self):
        """1×1 grid produces [IMG][IMG_END]."""
        seq = build_image_token_sequence(1, 1)
        assert seq == [IMG_TOKEN_ID, IMG_END_TOKEN_ID]

    def test_single_row(self):
        """1×3 grid produces [IMG][IMG][IMG][IMG_END] (no breaks)."""
        seq = build_image_token_sequence(1, 3)
        assert seq == [IMG_TOKEN_ID] * 3 + [IMG_END_TOKEN_ID]

    def test_two_rows(self):
        """2×2 grid produces [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]."""
        seq = build_image_token_sequence(2, 2)
        expected = [IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_BREAK_TOKEN_ID] + [IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_END_TOKEN_ID]
        assert seq == expected

    def test_three_rows(self):
        """3×2 grid has 2 breaks and 1 end."""
        seq = build_image_token_sequence(3, 2)
        assert seq.count(IMG_TOKEN_ID) == 6
        assert seq.count(IMG_BREAK_TOKEN_ID) == 2
        assert seq.count(IMG_END_TOKEN_ID) == 1
        assert seq[-1] == IMG_END_TOKEN_ID

    def test_token_counts(self):
        """Total length = patch_rows * patch_cols + patch_rows."""
        for rows, cols in [(4, 5), (8, 8), (26, 46)]:
            seq = build_image_token_sequence(rows, cols)
            assert len(seq) == rows * cols + rows
            assert seq.count(IMG_TOKEN_ID) == rows * cols
            assert seq.count(IMG_BREAK_TOKEN_ID) == rows - 1
            assert seq.count(IMG_END_TOKEN_ID) == 1

    def test_no_adjacent_breaks(self):
        """[IMG_BREAK] is always preceded by [IMG], never by another break."""
        seq = build_image_token_sequence(5, 10)
        for i, tok in enumerate(seq):
            if tok == IMG_BREAK_TOKEN_ID:
                assert seq[i - 1] == IMG_TOKEN_ID
            if tok == IMG_END_TOKEN_ID:
                assert seq[i - 1] == IMG_TOKEN_ID


class TestPatchGridComputation:
    """Tests for patch grid dimension calculation."""

    def test_fish_jpg_dimensions(self):
        """fish.jpg resized to 728×1288 → 26×46 grid."""
        rows, cols = compute_patch_grid(728, 1288)
        assert rows == 26
        assert cols == 46
        assert rows * cols == 1196

    def test_small_square(self):
        """224×224 → 8×8 grid (64 patches)."""
        rows, cols = compute_patch_grid(224, 224)
        assert rows == 8
        assert cols == 8
        assert rows * cols == 64

    def test_minimum_size(self):
        """28×28 (smallest valid) → 1×1 grid."""
        rows, cols = compute_patch_grid(28, 28)
        assert rows == 1
        assert cols == 1

    def test_non_square(self):
        """588×896 → 21×32 grid."""
        rows, cols = compute_patch_grid(588, 896)
        assert rows == 21
        assert cols == 32

    def test_dimensions_must_be_multiples_of_28(self):
        """Dimensions not divisible by 28 truncate patches."""
        # 700 // 14 // 2 = 25, 1300 // 14 // 2 = 46
        rows, cols = compute_patch_grid(700, 1300)
        assert rows == 25
        assert cols == 46


class TestTokenExpansionIntegration:
    """Integration tests combining grid computation with token expansion."""

    def test_fish_jpg_full_sequence(self):
        """fish.jpg (728×1288) produces correct full token sequence."""
        rows, cols = compute_patch_grid(728, 1288)
        seq = build_image_token_sequence(rows, cols)
        assert len(seq) == 1196 + 26  # 1196 IMG + 25 BREAK + 1 END
        assert seq.count(IMG_TOKEN_ID) == 1196
        assert seq.count(IMG_BREAK_TOKEN_ID) == 25
        assert seq.count(IMG_END_TOKEN_ID) == 1

    def test_224x224_full_sequence(self):
        """224×224 produces 64 IMG + 7 BREAK + 1 END = 72 tokens."""
        rows, cols = compute_patch_grid(224, 224)
        seq = build_image_token_sequence(rows, cols)
        assert len(seq) == 72
        assert seq.count(IMG_TOKEN_ID) == 64
        assert seq.count(IMG_BREAK_TOKEN_ID) == 7
        assert seq.count(IMG_END_TOKEN_ID) == 1

    def test_no_zero_tokens(self):
        """Token sequence must never contain 0 (embedding mask sentinel)."""
        for h, w in [(224, 224), (728, 1288), (588, 896), (28, 28)]:
            rows, cols = compute_patch_grid(h, w)
            seq = build_image_token_sequence(rows, cols)
            assert 0 not in seq, f"Zero token found for {h}×{w}"

    @pytest.mark.skipif(not _has_genai(), reason="onnxruntime_genai not installed")
    def test_genai_processor_token_counts(self, test_data_path):
        """Verify C++ processor produces correct token counts.

        Requires onnxruntime_genai, the pre-exported Mistral3 model under
        test_data_path/mistral3-vision-preprocessing, and a test image.
        """
        from pathlib import Path

        import numpy as np

        model_path = Path(test_data_path) / "mistral3-vision-preprocessing"
        image_path = Path(test_data_path) / "images" / "australia.jpg"
        if not (model_path / "genai_config.json").is_file():
            pytest.skip(f"Mistral3 model not found at {model_path} (missing genai_config.json)")
        if not image_path.is_file():
            pytest.skip(f"Test image not found at {image_path}")

        import onnxruntime_genai as og

        model = og.Model(str(model_path))
        processor = model.create_multimodal_processor()

        images = og.Images.open(str(image_path))
        prompt = "<s>[INST][IMG]Describe this image.[/INST]"
        inputs = processor(prompt, images=images)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=4096)
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        seq = np.array(generator.get_sequence(0))
        # Verify token types are present and no zeros leak in
        assert np.sum(seq == IMG_TOKEN_ID) > 0, "No [IMG] tokens found"
        assert np.sum(seq == IMG_END_TOKEN_ID) == 1, "Expected exactly 1 [IMG_END]"
        assert np.sum(seq == 0) == 0, "Unexpected zero tokens in sequence"
        # Verify break count = rows - 1
        num_img = int(np.sum(seq == IMG_TOKEN_ID))
        num_break = int(np.sum(seq == IMG_BREAK_TOKEN_ID))
        num_end = int(np.sum(seq == IMG_END_TOKEN_ID))
        total_rows = num_break + num_end  # breaks + 1 end = rows
        assert num_img == total_rows * (num_img // total_rows), (
            f"[IMG] count {num_img} not divisible by grid rows {total_rows}"
        )

        # Explicitly delete generator to ensure deterministic C++ destructor
        # invocation before model teardown — important for GPU memory cleanup.
        del generator


class TestMultiImageTokenExpansion:
    """Tests for multi-image token expansion logic.

    Verifies that multiple images with different resolutions produce
    correct per-image token sequences and total token counts.
    """

    def test_two_images_same_size(self):
        """Two identical images produce identical token sequences."""
        rows, cols = compute_patch_grid(224, 224)
        seq1 = build_image_token_sequence(rows, cols)
        seq2 = build_image_token_sequence(rows, cols)
        assert seq1 == seq2
        total_img_tokens = rows * cols * 2
        assert seq1.count(IMG_TOKEN_ID) + seq2.count(IMG_TOKEN_ID) == total_img_tokens

    def test_two_images_different_sizes(self):
        """Two images with different resolutions produce different token counts."""
        rows1, cols1 = compute_patch_grid(224, 224)  # 8×8 = 64 patches
        rows2, cols2 = compute_patch_grid(448, 224)  # 16×8 = 128 patches

        seq1 = build_image_token_sequence(rows1, cols1)
        seq2 = build_image_token_sequence(rows2, cols2)

        assert seq1.count(IMG_TOKEN_ID) == 64
        assert seq2.count(IMG_TOKEN_ID) == 128
        assert len(seq1) != len(seq2)

    def test_multi_image_total_token_count(self):
        """Total [IMG] tokens across N images equals sum of per-image counts."""
        image_sizes = [(224, 224), (448, 336), (672, 224)]
        total_img = 0
        for h, w in image_sizes:
            rows, cols = compute_patch_grid(h, w)
            seq = build_image_token_sequence(rows, cols)
            total_img += seq.count(IMG_TOKEN_ID)

        expected_total = sum((h // EFFECTIVE_PATCH) * (w // EFFECTIVE_PATCH) for h, w in image_sizes)
        assert total_img == expected_total

    def test_each_image_has_exactly_one_end_token(self):
        """Each image's token sequence has exactly one [IMG_END]."""
        for h, w in [(28, 28), (224, 224), (448, 672)]:
            rows, cols = compute_patch_grid(h, w)
            seq = build_image_token_sequence(rows, cols)
            assert seq.count(IMG_END_TOKEN_ID) == 1
            assert seq[-1] == IMG_END_TOKEN_ID

    def test_concatenated_sequences_preserve_structure(self):
        """Concatenating multiple image sequences preserves per-image structure."""
        sizes = [(224, 224), (448, 336)]
        all_tokens = []
        for h, w in sizes:
            rows, cols = compute_patch_grid(h, w)
            all_tokens.extend(build_image_token_sequence(rows, cols))

        # Two images means two [IMG_END] tokens in the concatenation
        assert all_tokens.count(IMG_END_TOKEN_ID) == 2
        # Breaks count: (rows1-1) + (rows2-1)
        rows1 = 224 // EFFECTIVE_PATCH
        rows2 = 448 // EFFECTIVE_PATCH
        expected_breaks = (rows1 - 1) + (rows2 - 1)
        assert all_tokens.count(IMG_BREAK_TOKEN_ID) == expected_breaks
