# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Text-to-text integration test.

Loads the model with the requested execution provider, generates a short
deterministic continuation, and asserts non-empty bounded output. The prompt
is a classic completion ("The capital of France is") chosen because it
reliably triggers a continuation across base/instruct/reasoning models
without chat templating. A soft semantic check warns (but does not fail)
when the expected token is absent.

The model under test is selected via ``--model <id>`` (a single id per
pytest run); the integration pipeline fans the model set out across one
ADO job per model.
"""

from __future__ import annotations

import sys
import warnings

import onnxruntime_genai as og
import pytest

from . import ep_support

_PROMPT = "The capital of France is"
_EXPECTED_SUBSTRING = "paris"
_MAX_NEW_TOKENS = 64


def test_generates_text(device, model, model_path):
    if not ep_support.ep_available(device):
        pytest.skip(f"Execution provider '{device}' is not available in this build.")
    if (sys.platform, device, model) in ep_support.VRAM_CONSTRAINED_SKIPS:
        pytest.skip(
            f"Model '{model}' on device '{device}' ({sys.platform}) "
            "is skipped pending more VRAM on the test agent."
        )

    config = og.Config(str(model_path))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)

    og_model = og.Model(config)
    tokenizer = og.Tokenizer(og_model)
    input_tokens = tokenizer.encode(_PROMPT)

    params = og.GeneratorParams(og_model)
    params.set_search_options(
        max_length=len(input_tokens) + _MAX_NEW_TOKENS,
        do_sample=False,
    )
    generator = og.Generator(og_model, params)
    generator.append_tokens(input_tokens)
    while not generator.is_done():
        generator.generate_next_token()

    new_tokens = generator.get_sequence(0)[len(input_tokens):]
    assert len(new_tokens) > 0, "generator produced no new tokens"
    assert len(new_tokens) <= _MAX_NEW_TOKENS

    text = tokenizer.decode(new_tokens)
    assert isinstance(text, str) and text.strip(), "decoded text was empty"

    if _EXPECTED_SUBSTRING not in text.lower():
        warnings.warn(
            f"[{model}/{device}] expected '{_EXPECTED_SUBSTRING}' in completion of "
            f"{_PROMPT!r}; got {text!r}",
            stacklevel=2,
        )
