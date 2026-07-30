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


def _run_text_generation(model_path: str) -> None:
    """Load the model and run one round of greedy text generation (text-only, no image).

    Appends a single seed token then generates up to max_length tokens.
    Raises if the underlying ORT sessions receive an unexpected input feed.
    """
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=5)

    generator = og.Generator(model, params)
    # Feed one seed token (token id=2, within vocab_size=10)
    generator.append_tokens(np.array([[2]], dtype=np.int32))

    while not generator.is_done():
        generator.generate_next_token()


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
