# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for the DecoderState input_ids injection fix (PR #2148).

Bug: DecoderState constructor checked *combined* session_info_ (decoder + vision +
embedding) for HasInput('input_ids').  The embedding session always declares
input_ids, so the check incorrectly injected input_ids into the decoder for models
like Mistral3 whose decoder has no input_ids input — causing an ORT error
"Invalid Feed Input Name: input_ids".

Fix: use a decoder-only SessionInfo for the HasInput('input_ids') check.

Two test model variants (in test/models/):
  - multimodal-decoder-no-input-ids/   Mistral3-like: embedding has input_ids,
                                        decoder does NOT.
  - multimodal-decoder-with-input-ids/ Gemma4-like:   both embedding and decoder
                                        declare input_ids.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


def _run_text_generation(
    model_path: str, prompt: np.ndarray | None = None, chunk_size: int | None = None
) -> list[np.ndarray]:
    """Load the model and run one round of greedy text generation (text-only, no image).

    Appends a seed prompt then generates up to max_length tokens.
    Raises if the underlying ORT sessions receive an unexpected input feed.
    """
    if prompt is None:
        prompt = np.array([[2]], dtype=np.int32)

    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    options = {"do_sample": False, "max_length": 8, "batch_size": prompt.shape[0]}
    if chunk_size is not None:
        options["chunk_size"] = chunk_size
    params.set_search_options(**options)

    generator = og.Generator(model, params)
    generator.append_tokens(prompt)

    while not generator.is_done():
        generator.generate_next_token()

    return [generator.get_sequence(i).copy() for i in range(prompt.shape[0])]


@pytest.mark.parametrize("relative_model_path", [Path("multimodal-decoder-no-input-ids")])
def test_decoder_no_input_ids_does_not_inject_input_ids(test_data_path, relative_model_path):
    """Mistral3-like model: decoder declares no input_ids input.

    With the fix, DecoderState uses decoder-only SessionInfo and does NOT inject
    input_ids into decoder feeds.  Generation must succeed.

    Without the fix, DecoderState would use combined session_info_ (which includes
    the embedding session that always has input_ids) and incorrectly inject input_ids
    into the decoder, causing ORT to raise "Invalid Feed Input Name: input_ids".
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # Should not raise — decoder receives only the inputs it declared
    _run_text_generation(model_path)


@pytest.mark.parametrize("relative_model_path", [Path("multimodal-decoder-with-input-ids")])
def test_decoder_with_input_ids_receives_input_ids(test_data_path, relative_model_path):
    """Gemma4-like model: decoder declares input_ids as one of its inputs.

    With the fix, DecoderState uses decoder-only SessionInfo and correctly injects
    input_ids into decoder feeds because the decoder session declares it.
    Generation must succeed.
    """
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # Should not raise — decoder receives input_ids because it declared it
    _run_text_generation(model_path)


@pytest.mark.parametrize(
    "relative_model_path",
    [
        Path("multimodal-decoder-no-input-ids"),
        Path("multimodal-decoder-with-input-ids"),
    ],
)
def test_multimodal_prefill_chunking_matches_unchunked_generation(test_data_path, relative_model_path):
    """Chunking must preserve generation for both decoder input_ids contracts."""
    model_path = os.fspath(Path(test_data_path) / relative_model_path)
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    # A five-token prompt with chunk_size=2 exercises two full chunks, a final
    # partial chunk, and the decode step after the embedding view is restored.
    prompt = np.array([[2, 3, 4, 5, 6]], dtype=np.int32)
    unchunked = _run_text_generation(model_path, prompt)
    chunked = _run_text_generation(model_path, prompt, chunk_size=2)

    assert len(chunked) == len(unchunked)
    for actual, expected in zip(chunked, unchunked, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_multimodal_prefill_chunking_falls_back_for_batched_prompts(test_data_path):
    """A batch of embeddings is non-contiguous per sequence and must not be sliced."""
    model_path = os.fspath(Path(test_data_path) / "multimodal-decoder-with-input-ids")
    if not os.path.exists(model_path):
        pytest.skip(f"Test model not found: {model_path}")

    prompt = np.array([[2, 3, 4, 5], [6, 5, 4, 3]], dtype=np.int32)
    unchunked = _run_text_generation(model_path, prompt)
    with_chunk_size = _run_text_generation(model_path, prompt, chunk_size=2)

    assert len(with_chunk_size) == len(unchunked)
    for actual, expected in zip(with_chunk_size, unchunked, strict=True):
        np.testing.assert_array_equal(actual, expected)
