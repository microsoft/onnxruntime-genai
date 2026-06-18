# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""End-to-end tests for CPU speculative decoding (target verifies draft proposals).

Two tiers:
  * Config guards (TestSpeculativeConfigGuards) construct a Model and assert the
    feature guards fire. These run WITHOUT model weights because every guard fires
    during model/state construction, before/at session setup, so they are CI-robust.
  * Generation tests (TestSpeculativeGeneration) need a real decoder-only model and
    skip when none is available under --test_models. They use a "self-speculative"
    wrapper (draft == target) so the result is deterministically comparable to plain
    greedy decoding of the same model.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _decoder_block(filename: str = "model.onnx", **overrides) -> dict:
    """A minimal, parseable decoder-only (separate key/value KV cache) config block."""
    block = {
        "session_options": {"provider_options": []},
        "filename": filename,
        "head_size": 8,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "inputs": {
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        },
        "outputs": {
            "logits": "logits",
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        },
    }
    block.update(overrides)
    return block


_DEFAULT_DRAFT = object()  # sentinel: use a copy of the default decoder block


def _spec_config(decoder: dict | None = None, draft=_DEFAULT_DRAFT,
                 speculative: dict | None = None, **model_overrides) -> dict:
    decoder = decoder if decoder is not None else _decoder_block()
    model = {
        "type": "speculative",
        "vocab_size": 1000,
        "context_length": 512,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "decoder": decoder,
    }
    if draft is _DEFAULT_DRAFT:
        model["draft"] = copy.deepcopy(decoder)
    elif draft is not None:
        model["draft"] = draft
    model.update(model_overrides)
    return {
        "model": model,
        "search": {"max_length": 512},
        "speculative": speculative if speculative is not None else {"max_draft_tokens": 4},
    }


def _write_config(directory: Path, config: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "genai_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return os.fspath(directory)


# ---------------------------------------------------------------------------
# Config guards (no model weights required)
# ---------------------------------------------------------------------------

class TestSpeculativeConfigGuards:
    def test_combined_kv_committed_gpt2_fixture(self, test_data_path):
        """The committed gpt2 speculative fixture uses the legacy combined-KV graph,
        which DecoderOnly_Model cannot bind. Must be rejected cleanly (not crash)."""
        path = os.path.join(test_data_path, "hf-internal-testing", "tiny-random-gpt2-speculative")
        if not os.path.exists(os.path.join(path, "genai_config.json")):
            pytest.skip("tiny-random-gpt2-speculative fixture not found")
        with pytest.raises(Exception, match="separate key/value KV-cache format"):
            og.Model(path)

    def test_combined_kv_inline(self, tmp_path):
        decoder = _decoder_block()
        decoder["inputs"]["past_names"] = "past_%d"
        decoder["outputs"]["present_names"] = "present_%d"
        path = _write_config(tmp_path / "combined_kv", _spec_config(decoder=decoder))
        with pytest.raises(Exception, match="separate key/value KV-cache format"):
            og.Model(path)

    def test_missing_draft(self, tmp_path):
        path = _write_config(tmp_path / "no_draft", _spec_config(draft=None))
        with pytest.raises(Exception, match="draft.filename is not set"):
            og.Model(path)

    def test_sliding_window_kv_cache(self, tmp_path):
        decoder = _decoder_block(sliding_window={"window_size": 16, "slide_key_value_cache": True})
        path = _write_config(tmp_path / "sliding", _spec_config(decoder=decoder))
        with pytest.raises(Exception, match="sliding-window"):
            og.Model(path)

    def test_lfm2_hybrid_layer_types(self, tmp_path):
        decoder = _decoder_block(layer_types=["full_attention", "conv"])
        path = _write_config(tmp_path / "lfm2", _spec_config(decoder=decoder))
        with pytest.raises(Exception, match="LFM2"):
            og.Model(path)

    def test_multimodal_vision_rejected(self, tmp_path):
        cfg = _spec_config()
        cfg["model"]["vision"] = {"filename": "vision.onnx"}
        path = _write_config(tmp_path / "vision", cfg)
        with pytest.raises(Exception, match="multimodal"):
            og.Model(path)

    def test_max_draft_tokens_too_large(self, tmp_path):
        path = _write_config(tmp_path / "k_big", _spec_config(speculative={"max_draft_tokens": 99}))
        with pytest.raises(Exception, match="max_draft_tokens"):
            og.Model(path)

    def test_max_draft_tokens_zero(self, tmp_path):
        path = _write_config(tmp_path / "k_zero", _spec_config(speculative={"max_draft_tokens": 0}))
        with pytest.raises(Exception, match="max_draft_tokens"):
            og.Model(path)


# ---------------------------------------------------------------------------
# Generation tests (require a real decoder-only model under --test_models)
# ---------------------------------------------------------------------------

# Candidate decoder-only models to wrap as a self-speculative (draft == target) model.
_SELF_SPEC_CANDIDATES = [
    ("qwen3-speculative", "qwen3-0.6b"),
    ("qwen3-speculative", "qwen3-1.7b"),
]

# A short, arbitrary, in-vocabulary prompt (Qwen3 token ids).
_PROMPT = [785, 3838, 374, 279, 6722, 315, 9625, 30]


def _find_decoder_only_model(test_data_path: str) -> str | None:
    for parts in _SELF_SPEC_CANDIDATES:
        candidate = os.path.join(test_data_path, *parts)
        if os.path.exists(os.path.join(candidate, "genai_config.json")) and \
                os.path.exists(os.path.join(candidate, "model.onnx")):
            return candidate
    return None


def _build_self_spec(source_dir: str, dest_dir: Path, max_draft_tokens: int = 4) -> str:
    """Wrap a decoder-only model as a speculative model whose draft == target."""
    source_dir = os.path.abspath(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(source_dir, "genai_config.json")) as f:
        src = json.load(f)
    decoder = copy.deepcopy(src["model"]["decoder"])
    # onnxruntime-genai resolves the ONNX filename relative to the config directory, so
    # express it as a path from the wrapper dir to the real model.onnx. External weights
    # (model.onnx.data) then load from the source dir alongside the resolved model.onnx.
    model_abs = os.path.join(source_dir, decoder["filename"])
    decoder["filename"] = os.path.relpath(model_abs, os.path.abspath(dest_dir))
    model = {
        "type": "speculative",
        "vocab_size": src["model"]["vocab_size"],
        "context_length": src["model"].get("context_length", 2048),
        "bos_token_id": src["model"].get("bos_token_id", 0),
        "eos_token_id": src["model"].get("eos_token_id", 0),
        "pad_token_id": src["model"].get("pad_token_id", 0),
        "decoder": decoder,
        "draft": copy.deepcopy(decoder),
    }
    cfg = {
        "model": model,
        "search": {"max_length": model["context_length"]},
        "speculative": {"max_draft_tokens": max_draft_tokens},
    }
    return _write_config(dest_dir, cfg)


@pytest.fixture
def decoder_only_model_path(test_data_path):
    path = _find_decoder_only_model(test_data_path)
    if path is None:
        pytest.skip("No decoder-only model available under --test_models for speculative tests")
    return path


def _greedy(model_path: str, prompt, max_length: int, k: int | None = None):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)
    if k is not None:
        params.set_speculative_options(max_draft_tokens=k)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array([prompt], dtype=np.int32))
    while not gen.is_done():
        gen.generate_next_token()
    seq = list(int(t) for t in gen.get_sequence(0))
    stats = gen.get_speculative_stats() if k is not None else None
    return seq, stats


class TestSpeculativeGeneration:
    def test_k1_matches_standalone_greedy(self, decoder_only_model_path, tmp_path):
        """With K=1 every verify is a single-token pass, so self-speculative greedy is
        bitwise-identical to plain greedy of the same model."""
        max_length = len(_PROMPT) + 12
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "selfspec_k1", 1)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=1)
        assert spec == ref
        assert stats["draft_tokens_proposed"] > 0
        assert stats["acceptance_rate"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_self_spec_high_acceptance(self, decoder_only_model_path, tmp_path, k):
        """draft == target, so essentially every proposal is accepted regardless of K."""
        max_length = len(_PROMPT) + 12
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"selfspec_{k}", k)
        _, stats = _greedy(spec_path, _PROMPT, max_length, k=k)
        assert stats["rounds"] > 0
        assert stats["acceptance_rate"] >= 0.9

    def test_stats_are_consistent(self, decoder_only_model_path, tmp_path):
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "selfspec_stats", 4)
        _, s = _greedy(spec_path, _PROMPT, max_length, k=4)
        assert s["rounds"] > 0
        assert s["draft_tokens_accepted"] <= s["draft_tokens_proposed"]
        # Every round commits exactly one correction or bonus token.
        assert s["correction_tokens"] + s["bonus_tokens"] == s["rounds"]
        assert 0.0 <= s["acceptance_rate"] <= 1.0

    def test_draft_token_count_clamped_to_range(self, decoder_only_model_path, tmp_path):
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "selfspec_clamp", 4)
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=16)
        with pytest.raises(Exception, match="max_draft_tokens"):
            params.set_speculative_options(max_draft_tokens=0)


class TestSpeculativeStateGuards:
    """Guards that fire when the generator/state is created (model must load first)."""

    def _params(self, model, **search):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=16, **search)
        params.set_speculative_options(max_draft_tokens=4)
        return params

    def test_batch_size_gt_1(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_batch", 4))
        with pytest.raises(Exception, match="batch_size"):
            og.Generator(model, self._params(model, do_sample=False, batch_size=2))

    def test_num_beams_gt_1(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_beams", 4))
        with pytest.raises(Exception, match="num_beams"):
            og.Generator(model, self._params(model, do_sample=False, num_beams=2))

    def test_repetition_penalty(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_rep", 4))
        with pytest.raises(Exception, match="repetition_penalty"):
            og.Generator(model, self._params(model, do_sample=False, repetition_penalty=1.2))

    def test_min_length(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_min", 4))
        with pytest.raises(Exception, match="min_length"):
            og.Generator(model, self._params(model, do_sample=False, min_length=8))

    def test_guidance(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_guid", 4))
        params = self._params(model, do_sample=False)
        params.set_guidance("regex", r"[0-9]+")
        with pytest.raises(Exception, match="guidance|constrained"):
            og.Generator(model, params)


class TestSpeculativeRewind:
    def test_rewind_then_generate_is_rejected(self, decoder_only_model_path, tmp_path):
        """Rewind clears the cross-round anchor; generating without a fresh prefill would
        silently produce wrong output, so it must raise."""
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "rw_bad", 4))
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=24)
        params.set_speculative_options(max_draft_tokens=4)
        gen = og.Generator(model, params)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen.generate_next_token()
        gen.rewind_to(len(_PROMPT))
        with pytest.raises(Exception, match="without fresh logits|RewindToLength"):
            gen.generate_next_token()

    def test_rewind_then_append_then_generate_works(self, decoder_only_model_path, tmp_path):
        """The supported continuous-decoding flow: re-prefill after rewind, then generate."""
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "rw_ok", 4))
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=24)
        params.set_speculative_options(max_draft_tokens=4)
        gen = og.Generator(model, params)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen.generate_next_token()
        gen.rewind_to(len(_PROMPT))
        gen.append_tokens(np.array([[785, 3838]], dtype=np.int32))
        gen.generate_next_token()
        assert gen.get_sequence(0) is not None
