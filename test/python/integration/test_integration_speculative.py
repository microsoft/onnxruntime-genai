# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-speculative decoding integration test.

Speculative decoding is exact for greedy sampling: the token committed each
round is always the target model's argmax, so speculative greedy output must
equal plain greedy output of the target alone, token for token. This test wraps
each fanned-out model as a *self-speculative* model (draft == target) and asserts
that equality on a real execution provider -- the decisive end-to-end correctness
check (any KV-cache misalignment, device/dtype mishandling in the verify read, or
EP-selection bug would diverge).

It reuses the suite's ``model`` / ``device`` / ``model_path`` fixtures, so the
integration pipeline runs it once per (model, ep) job with no pipeline change,
inheriting the CPU / CUDA / WebGPU coverage the suite already provides.

The EP is applied by writing ``provider_options`` into BOTH the decoder and the
draft config blocks; speculative decoding requires target and draft to share an
EP, and the draft's providers are derived from its provider_options at load.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
from pathlib import Path

import onnxruntime_genai as og
import pytest

_PROMPT = "The capital of France is"
_MAX_NEW_TOKENS = 24
_MAX_DRAFT_TOKENS = 4

# Self-speculative loads two full copies of the model (target + draft), doubling
# the memory footprint versus plain decoding. These (platform, device, model)
# combinations don't fit on the VRAM-constrained accelerator agents; CPU (ample
# system RAM) still exercises every size. Mirrors test_integration_text.py's set
# and may need to grow as the doubled footprint surfaces new OOMs.
_VRAM_CONSTRAINED_SKIPS: set[tuple[str, str, str]] = {
    ("win32", "cuda", "ministral-3-3b-Instruct-2512"),
    ("win32", "cuda", "Phi-4-mini-instruct"),
    ("linux", "cuda", "ministral-3-3b-Instruct-2512"),
    ("linux", "cuda", "Phi-4-mini-instruct"),
}

# og.Model raises one of these when the model isn't a valid speculative target/
# draft (sliding-window KV, LFM2, combined-KV, prune_lm_head, Phi-3 long-RoPE,
# ...). Those are out of scope for speculative decoding, so skip rather than fail.
_SPEC_UNSUPPORTED_MARKERS = (
    "Speculative decoding does not support",
    "Speculative decoding requires",
    "Speculative decoding runtime",
    "Target and draft",
    "per-position logits",
    "prune_lm_head",
)


# --------------------------------------------------------------------------
# Execution provider availability (mirrors test_integration_text.py)
# --------------------------------------------------------------------------

def _register_webgpu_plugin_once() -> bool:
    if getattr(_register_webgpu_plugin_once, "_done", False):
        return True
    try:
        import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415
    except ImportError:
        return False
    og.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())
    _register_webgpu_plugin_once._done = True
    return True


def _ep_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return og.is_cuda_available()
    if device == "webgpu":
        return _register_webgpu_plugin_once()
    return False


# --------------------------------------------------------------------------
# Config synthesis
# --------------------------------------------------------------------------

def _provider_options_for(device: str) -> list[dict]:
    """provider_options entry for a config block. Empty = default CPU EP."""
    return [] if device == "cpu" else [{device: {}}]


def _decoder_block_from(source_dir: Path, dest_dir: Path):
    """Copy ``source_dir``'s decoder block, rewriting the ONNX filename to a path
    relative to ``dest_dir`` so external weights load from the original folder.
    Returns ``(decoder_block, full_source_config)``."""
    with open(source_dir / "genai_config.json") as f:
        src = json.load(f)
    decoder = copy.deepcopy(src["model"]["decoder"])
    model_onnx = (source_dir / decoder["filename"]).resolve()
    decoder["filename"] = os.path.relpath(model_onnx, dest_dir.resolve())
    return decoder, src


def _write_self_spec_config(dest_dir: Path, model_dir: Path, device: str) -> Path:
    """Compose a self-speculative genai_config.json (draft == target)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    decoder, src = _decoder_block_from(model_dir, dest_dir)
    draft = copy.deepcopy(decoder)

    ep = _provider_options_for(device)
    decoder.setdefault("session_options", {})["provider_options"] = copy.deepcopy(ep)
    draft.setdefault("session_options", {})["provider_options"] = copy.deepcopy(ep)

    model = src["model"]
    cfg = {
        "model": {
            "type": "speculative",
            "vocab_size": model["vocab_size"],
            "context_length": model.get("context_length", 2048),
            "bos_token_id": model.get("bos_token_id", 0),
            "eos_token_id": model.get("eos_token_id", 0),
            "pad_token_id": model.get("pad_token_id", 0),
            "decoder": decoder,
            "draft": draft,
        },
        "search": {"max_length": model.get("context_length", 2048)},
        "speculative": {"max_draft_tokens": _MAX_DRAFT_TOKENS},
    }
    with open(dest_dir / "genai_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return dest_dir


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _standard_greedy(model_dir: Path, device: str):
    """Greedy-decode the model alone. Returns ``(prompt_ids, full_sequence)``.

    The tokenizer lives here (the synthesized speculative config dir has no
    tokenizer), so this also yields the prompt ids the speculative run reuses.
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
    sequence = list(int(t) for t in generator.get_sequence(0))

    # Free the standalone model before loading the (2x) speculative model.
    del generator, tokenizer, model, config
    gc.collect()
    return list(int(t) for t in prompt_ids), sequence


def _speculative_greedy(spec_dir: Path, prompt_ids: list[int]) -> list[int]:
    """Greedy-decode the speculative model on the same prompt ids (no tokenizer
    needed; ids are shared because target and draft share a vocabulary)."""
    model = og.Model(str(spec_dir))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt_ids) + _MAX_NEW_TOKENS, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(prompt_ids)
    while not generator.is_done():
        generator.generate_next_token()
    return list(int(t) for t in generator.get_sequence(0))


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------

def test_self_speculative_matches_standard(device, model, model_path, tmp_path):
    """Self-speculative (draft == target) greedy must equal the model's own
    greedy output, token for token, on the requested execution provider."""
    if not _ep_available(device):
        pytest.skip(f"Execution provider '{device}' is not available in this build.")

    if (sys.platform, device, model) in _VRAM_CONSTRAINED_SKIPS:
        pytest.skip(
            f"[{model}/{device}] self-speculative loads two model copies; skipped on this "
            "VRAM-constrained agent (see _VRAM_CONSTRAINED_SKIPS)."
        )

    model_dir = Path(model_path)
    prompt_ids, ref = _standard_greedy(model_dir, device)

    spec_dir = _write_self_spec_config(tmp_path / "self_spec", model_dir, device)
    try:
        spec = _speculative_greedy(spec_dir, prompt_ids)
    except Exception as e:  # noqa: BLE001 - narrowed to speculative guards below
        message = str(e)
        if any(marker in message for marker in _SPEC_UNSUPPORTED_MARKERS):
            pytest.skip(f"[{model}] not a supported speculative model: {message}")
        raise

    assert len(spec) > len(prompt_ids), "speculative run produced no new tokens"
    assert spec == ref, (
        f"[{model}/{device}] self-speculative greedy diverged from standard greedy:\n"
        f"  standard:    {ref}\n  speculative: {spec}"
    )
