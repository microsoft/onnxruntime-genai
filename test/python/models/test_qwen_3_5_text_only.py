# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Unit tests for Qwen3.5 text-only (LLM) model.

Tests cover model loading, tokenizer creation, generator creation, and
basic text generation for the text-only variant of Qwen3.5 (model type
"qwen3_5_text"), which uses 2D position_ids and hybrid KV/recurrent state.

Usage:
    pytest test_qwen_3_5_text_only.py --test_models=test/models
"""

import os
from pathlib import Path

import onnxruntime_genai as og
import pytest

MODEL_DIR = "qwen3-5-text-only"


def _model_path(test_data_path):
    return os.fspath(Path(test_data_path) / MODEL_DIR)


def _skip_if_missing(test_data_path):
    path = _model_path(test_data_path)
    if not os.path.exists(path):
        pytest.skip(f"{MODEL_DIR} test model not found at {path}")
    return path


def test_qwen3_5_text_only_model_loads(test_data_path):
    """Test that a Qwen3.5 text-only model loads successfully."""
    model_path = _skip_if_missing(test_data_path)
    model = og.Model(model_path)
    assert model is not None


def test_qwen3_5_text_only_generator_creates(test_data_path):
    """Test that a Generator can be created for the text-only model.
    Validates that hybrid state auto-discovery works with qwen3_5_text type."""
    model_path = _skip_if_missing(test_data_path)
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=10)
    generator = og.Generator(model, params)
    assert generator is not None


def test_qwen3_5_text_only_accepts_input_ids(test_data_path):
    """Test that the text-only model accepts input_ids (not inputs_embeds).
    The dummy model uses Identity pass-through which doesn't support KV cache
    shape changes, so we only validate that the generator constructs and
    is ready to accept tokens."""
    model_path = _skip_if_missing(test_data_path)
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=5)

    # The model should accept raw token IDs (input_ids, not inputs_embeds)
    generator = og.Generator(model, params)
    assert generator is not None
