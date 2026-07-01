# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-speculative decoding integration test.

Speculative decoding is exact for greedy sampling: the token committed each
round is always the target model's argmax, so speculative-greedy output must
equal plain-greedy output of the target alone, token for token. This test wraps
each model under test as a *self-speculative* model (draft == target) and
asserts that equality on a real execution provider -- the decisive end-to-end
correctness check (any KV-cache misalignment, device/dtype mishandling in the
verify read, or EP-selection bug would diverge).

Following ``test_integration_text.py``'s style, the on-disk model directory is
treated as immutable: the real ``genai_config.json`` is loaded with ``og.Config``
and only the speculative wiring is layered on with ``Config.overlay`` --
``model.type`` becomes ``speculative`` and a ``draft`` block that copies the real
decoder (draft == target) is added. Nothing is written to disk and no config
field is hand-reconstructed, so every model-specific setting (input names, RoPE,
quantization, ``past_present_share_buffer``, ...) is preserved verbatim. The
execution provider is applied through the same public API the text test uses:
``clear_providers`` / ``append_provider`` for the target, plus a matching
``provider_options`` on the draft in the overlay (ORT GenAI derives the draft's
providers list from it at load, so target and draft share an EP).

It reuses the suite's ``model`` / ``device`` / ``model_path`` fixtures, so the
integration pipeline runs it once per (model, ep) job with no pipeline change,
inheriting the CPU / CUDA / WebGPU coverage the suite already provides.
"""

from __future__ import annotations

import copy
import gc
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
import pytest

from . import ep_support

_PROMPT = "The capital of France is"
_MAX_NEW_TOKENS = 24
_MAX_DRAFT_TOKENS = 4

# og.Model raises one of these when the model isn't a valid speculative
# target/draft (sliding-window KV, LFM2, combined-KV, prune_lm_head, Phi-3
# long-RoPE, ...). Those are out of scope for speculative decoding, so skip
# rather than fail.
_SPEC_UNSUPPORTED_MARKERS = (
    "Speculative decoding does not support",
    "Speculative decoding requires",
    "Speculative decoding runtime",
    "Target and draft",
    "per-position logits",
    "prune_lm_head",
)

# ORT allocator failures surface as generic runtime errors. Self-speculative
# loads two copies of the model, so on the VRAM-constrained GPU agents it can
# OOM at load where plain single-model decoding still fits. Treat these as an
# environment skip, not a correctness failure.
_OOM_MARKERS = (
    "Failed to allocate memory",
    "out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "bad_alloc",
    "bad allocation",
)


def _provider_options_for(device: str) -> list[dict]:
    """provider_options entry for a config block. Empty selects the default CPU EP."""
    return [] if device == "cpu" else [{device: {}}]


def _self_speculative_config(model_dir: Path, device: str) -> og.Config:
    """Load ``model_dir`` as a self-speculative ``og.Config`` on ``device``.

    The real config is loaded untouched; an overlay adds ``type: speculative``
    and a ``draft`` block that copies the real decoder (draft == target). The
    draft's EP comes from its own ``provider_options`` (ORT GenAI derives the
    draft's providers from them at load); the target's EP is applied with the
    public ``clear_providers`` / ``append_provider`` API, exactly like the text
    test.
    """
    real = json.loads((model_dir / "genai_config.json").read_text())
    draft = copy.deepcopy(real["model"]["decoder"])
    draft.setdefault("session_options", {})["provider_options"] = _provider_options_for(device)

    overlay = {
        "model": {"type": "speculative", "draft": draft},
        "speculative": {"max_draft_tokens": _MAX_DRAFT_TOKENS},
    }

    config = og.Config(str(model_dir))
    config.overlay(json.dumps(overlay))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    return config


def _standard_greedy(model_dir: Path, device: str):
    """Greedy-decode the model alone. Returns ``(prompt_ids, full_sequence)``.

    The tokenizer lives here (the speculative run reuses these ids; target and
    draft share a vocabulary, so the ids are valid for both).
    """
    config = og.Config(str(model_dir))
    config.clear_providers()
    if device != "cpu":
        config.append_provider(device)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    prompt_ids = tokenizer.encode(_PROMPT)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt_ids) + _MAX_NEW_TOKENS, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(prompt_ids)
    while not generator.is_done():
        generator.generate_next_token()
    sequence = [int(t) for t in generator.get_sequence(0)]

    # Free the standalone model before loading the (2x) speculative model so the
    # peak footprint is two copies, not three, on the constrained GPU agents.
    del generator, tokenizer, model, config
    gc.collect()
    return [int(t) for t in prompt_ids], sequence


def _speculative_greedy(config: og.Config, prompt_ids: list[int]) -> list[int]:
    """Greedy-decode the speculative model on ``prompt_ids`` (already encoded)."""
    model = og.Model(config)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt_ids) + _MAX_NEW_TOKENS, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(prompt_ids)
    while not generator.is_done():
        generator.generate_next_token()
    return [int(t) for t in generator.get_sequence(0)]


def test_self_speculative_matches_standard(device, model, model_path):
    """Self-speculative (draft == target) greedy must equal the model's own
    greedy output, token for token, on the requested execution provider."""
    if not ep_support.ep_available(device):
        pytest.skip(f"Execution provider '{device}' is not available in this build.")

    if (sys.platform, device, model) in ep_support.VRAM_CONSTRAINED_SKIPS:
        pytest.skip(
            f"[{model}/{device}] self-speculative loads two model copies; skipped on this "
            "VRAM-constrained agent (see ep_support.VRAM_CONSTRAINED_SKIPS)."
        )

    model_dir = Path(model_path)
    prompt_ids, ref = _standard_greedy(model_dir, device)

    try:
        config = _self_speculative_config(model_dir, device)
        spec = _speculative_greedy(config, prompt_ids)
    except Exception as e:  # narrowed to the known skip cases below
        message = str(e)
        if any(marker in message for marker in _SPEC_UNSUPPORTED_MARKERS):
            pytest.skip(f"[{model}] not a supported speculative model: {message}")
        if any(marker in message for marker in _OOM_MARKERS):
            pytest.skip(
                f"[{model}/{device}] self-speculative ran out of memory on this agent: {message}"
            )
        raise

    assert len(spec) > len(prompt_ids), "speculative run produced no new tokens"
    assert spec == ref, (
        f"[{model}/{device}] self-speculative greedy diverged from standard greedy:\n"
        f"  standard:    {ref}\n  speculative: {spec}"
    )
