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
import shutil
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


def _make_fp16_logits_model(source_dir: str, dest_dir: Path) -> str:
    """Copy a decoder-only model but make its `logits` graph output FLOAT16 while leaving all
    internal compute in fp32 (one Cast node on the output). Running the result yields fp16 logits,
    which forces the speculative verify read down its Cast(fp16 -> fp32) branch -- the path GPU/NPU
    EPs hit but a plain fp32 CPU model never exercises. Returns the dest model dir."""
    import onnx
    from onnx import TensorProto, helper

    source_dir = os.path.abspath(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(os.path.join(source_dir, "model.onnx"))
    graph = model.graph

    # Rewire whatever node currently produces `logits` to an internal fp32 name, then append a
    # Cast back to `logits` as FLOAT16 so the model's external output dtype is fp16.
    rewired = False
    for node in graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == "logits":
                node.output[i] = "logits_fp32_internal"
                rewired = True
    if not rewired:
        raise RuntimeError("expected a node producing a `logits` output")
    graph.node.append(
        helper.make_node("Cast", ["logits_fp32_internal"], ["logits"],
                         to=TensorProto.FLOAT16, name="logits_to_fp16"))
    for output in graph.output:
        if output.name == "logits":
            output.type.tensor_type.elem_type = TensorProto.FLOAT16

    onnx.save(model, os.path.join(dest_dir, "model.onnx"))
    shutil.copyfile(os.path.join(source_dir, "genai_config.json"),
                    os.path.join(dest_dir, "genai_config.json"))
    return os.fspath(dest_dir)


@pytest.fixture(scope="module")
def fp16_decoder_only_model_path(request, tmp_path_factory):
    """An fp16-logits-output copy of an available decoder-only model, built once per module.
    Reads --test_models directly (not the function-scoped test_data_path fixture) so it can be
    module-scoped and pay the one-time model-rewrite cost only once."""
    test_data = request.config.getoption("--test_models")
    source = _find_decoder_only_model(test_data) if test_data else None
    if source is None:
        pytest.skip("No decoder-only model available under --test_models for fp16 speculative test")
    try:
        import onnx  # noqa: F401
    except ImportError:
        pytest.skip("onnx not available to synthesize an fp16-logits model")
    dest = tmp_path_factory.mktemp("fp16_decoder_only")
    return _make_fp16_logits_model(source, dest)


def _find_two_distinct_models(test_data_path: str):
    """Return (target_dir, draft_dir) for two distinct, vocab-compatible decoder-only models,
    or None if fewer than two candidates are available."""
    found = []
    for parts in _SELF_SPEC_CANDIDATES:
        candidate = os.path.join(test_data_path, *parts)
        if os.path.exists(os.path.join(candidate, "genai_config.json")) and \
                os.path.exists(os.path.join(candidate, "model.onnx")):
            found.append(candidate)
    if len(found) < 2:
        return None
    # Larger model as target, smaller as draft (order here is arbitrary but distinct).
    return found[-1], found[0]


def _decoder_from(source_dir: str, dest_dir: Path) -> dict:
    with open(os.path.join(source_dir, "genai_config.json")) as f:
        src = json.load(f)
    decoder = copy.deepcopy(src["model"]["decoder"])
    model_abs = os.path.join(os.path.abspath(source_dir), decoder["filename"])
    decoder["filename"] = os.path.relpath(model_abs, os.path.abspath(dest_dir))
    return decoder, src


def _build_spec(target_dir: str, draft_dir: str, dest_dir: Path, max_draft_tokens: int = 4) -> str:
    """Wrap a distinct target + draft model pair as a speculative model."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    decoder, src = _decoder_from(target_dir, dest_dir)
    draft, _ = _decoder_from(draft_dir, dest_dir)
    model = {
        "type": "speculative",
        "vocab_size": src["model"]["vocab_size"],
        "context_length": src["model"].get("context_length", 2048),
        "bos_token_id": src["model"].get("bos_token_id", 0),
        "eos_token_id": src["model"].get("eos_token_id", 0),
        "pad_token_id": src["model"].get("pad_token_id", 0),
        "decoder": decoder,
        "draft": draft,
    }
    cfg = {
        "model": model,
        "search": {"max_length": model["context_length"]},
        "speculative": {"max_draft_tokens": max_draft_tokens},
    }
    return _write_config(dest_dir, cfg)


def _greedy(model_path: str, prompt, max_length: int, k: int | None = None,
            repetition_penalty: float | None = None, min_length: int | None = None):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    opts = dict(do_sample=False, max_length=max_length)
    if repetition_penalty is not None:
        opts["repetition_penalty"] = repetition_penalty
    if min_length is not None:
        opts["min_length"] = min_length
    params.set_search_options(**opts)
    if k is not None:
        params.set_speculative_options(max_draft_tokens=k)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array([prompt], dtype=np.int32))
    while not gen.is_done():
        gen.generate_next_token()
    seq = list(int(t) for t in gen.get_sequence(0))
    stats = gen.get_speculative_stats() if k is not None else None
    return seq, stats


def _eos_ids(model_dir: str) -> set:
    """The end-of-stream token id(s) declared in a model's genai_config."""
    with open(os.path.join(model_dir, "genai_config.json")) as f:
        cfg = json.load(f)
    eos = cfg["model"].get("eos_token_id", [])
    if isinstance(eos, int):
        eos = [eos]
    return set(int(e) for e in eos)


def _vocab_size(model_dir: str) -> int:
    with open(os.path.join(model_dir, "genai_config.json")) as f:
        cfg = json.load(f)
    return int(cfg["model"]["vocab_size"])


def _sample(model_path: str, prompt, max_length: int, seed: int, k: int | None = None,
            top_k: int = 0, top_p: float = 0.0, temperature: float = 1.0):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    opts = dict(do_sample=True, max_length=max_length, random_seed=seed, temperature=temperature)
    if top_k:
        opts["top_k"] = top_k
    if top_p:
        opts["top_p"] = top_p
    params.set_search_options(**opts)
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


class TestSpeculativeFp16Verify:
    """Target models commonly emit fp16/bf16 logits (the norm on GPU/NPU EPs). The device-agnostic
    verify read casts them to fp32 with the same Cast/WrapTensor path regular decoding uses. Drive
    that Cast branch with an fp16-logits-output model and require speculative greedy to still match
    standalone greedy of the same model token-for-token (fp16 rounding is applied identically on
    both paths, so any mismatch means the cast/read is wrong)."""

    @pytest.mark.parametrize("k", [1, 4])
    def test_fp16_self_spec_matches_standalone_greedy(self, fp16_decoder_only_model_path, tmp_path, k):
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(fp16_decoder_only_model_path, tmp_path / f"fp16_selfspec_{k}", k)
        ref, _ = _greedy(fp16_decoder_only_model_path, _PROMPT, max_length)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=k)
        assert spec == ref
        assert stats["draft_tokens_proposed"] > 0


class TestSpeculativeKVReanchor:
    """KV cache re-anchor invariants. After each round the target's KV cache must be rewound to
    the committed prefix and advanced on the committed token, so the next round's target step sees
    exactly the streamed prefix and no stale draft influence. The decisive end-to-end check is that
    speculative greedy output equals plain greedy output token-for-token: any cache misalignment
    would diverge."""

    @pytest.mark.parametrize("k", [2, 4, 8, 16])
    def test_self_spec_greedy_matches_standalone_greedy(self, decoder_only_model_path, tmp_path, k):
        """draft == target greedy accepts every proposal, so each round commits an all-accepted
        window plus a bonus token (the all-accepted + bonus path). Exercises rewind/advance/fold
        across multi-token rounds for every K; output must match standalone greedy exactly."""
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"reanchor_{k}", k)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=k)
        assert spec == ref
        # Self-spec greedy never rejects: every round ends on the bonus path, never a correction.
        assert stats["correction_tokens"] == 0
        assert stats["bonus_tokens"] == stats["rounds"]

    def test_reanchor_is_stable_across_repeated_runs(self, decoder_only_model_path, tmp_path):
        """Re-anchoring must be deterministic: the same model + prompt + K produces identical
        output on every run (no leftover cache state between rounds)."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "reanchor_stable", 4)
        first, _ = _greedy(spec_path, _PROMPT, max_length, k=4)
        second, _ = _greedy(spec_path, _PROMPT, max_length, k=4)
        assert first == second

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_draft_neq_target_matches_target_only_greedy(self, test_data_path, tmp_path, k):
        """With a real draft != target pair, the draft genuinely mispredicts, so rounds reject at
        various positions (reject-early, reject-late) and re-anchor on a correction token. The
        target's cache must rewind to the accepted prefix and advance on the correction, with no
        stale draft influence; speculative greedy must still equal target-only greedy exactly."""
        pair = _find_two_distinct_models(test_data_path)
        if pair is None:
            pytest.skip("Need two distinct decoder-only models under --test_models for reject-path test")
        target_dir, draft_dir = pair
        max_length = len(_PROMPT) + 20
        spec_path = _build_spec(target_dir, draft_dir, tmp_path / f"spec_{k}", k)
        ref, _ = _greedy(target_dir, _PROMPT, max_length)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=k)
        assert spec == ref
        # A genuine draft/target mismatch must reject at least once (correction path exercised).
        assert stats["correction_tokens"] > 0


class TestSpeculativeStatsContract:
    """Pin the documented stats formulas (see speculative_stats.h) on a deterministic scenario."""

    def test_self_spec_greedy_stats_formulas(self, decoder_only_model_path, tmp_path):
        k = 4
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "stats_contract", k)
        _, s = _greedy(spec_path, _PROMPT, max_length, k=k)

        rounds = s["rounds"]
        proposed = s["draft_tokens_proposed"]
        accepted = s["draft_tokens_accepted"]
        corrections = s["correction_tokens"]
        bonuses = s["bonus_tokens"]
        assert rounds > 0 and proposed > 0

        # Round contract: each round commits exactly one correction or bonus token.
        assert corrections + bonuses == rounds
        # Self-spec greedy accepts everything: no corrections, one bonus per round, all accepted.
        assert corrections == 0
        assert bonuses == rounds
        assert accepted == proposed
        # acceptance_rate = accepted / proposed.
        assert s["acceptance_rate"] == pytest.approx(accepted / proposed, abs=1e-6)
        assert s["acceptance_rate"] == pytest.approx(1.0, abs=1e-6)
        # mean_accepted_tokens = committed / rounds, committed = accepted + corrections + bonuses.
        committed = accepted + corrections + bonuses
        assert s["mean_accepted_tokens"] == pytest.approx(committed / rounds, abs=1e-4)
        # Timing averages are per proposed token and non-negative.
        assert s["avg_draft_ms_per_token"] >= 0.0
        assert s["avg_target_ms_per_token"] >= 0.0


class TestSpeculativePenalties:
    """repetition_penalty and min_length must produce EXACTLY the tokens that standard decoding
    of the target produces. Both penalties are applied to every target verify row (and, so the
    draft keeps approximating the *penalized* target, to every draft row) through the same
    helpers Search_Cpu uses. Oracle: speculative greedy == standalone greedy of the same model."""

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_self_spec_repetition_penalty_matches_standalone(self, decoder_only_model_path, tmp_path, k):
        """repetition_penalty bites at every already-seen token, so this exercises the penalty at
        many positions. Self-spec (draft == target) penalizes both sides identically, so every
        proposal is still accepted (correction == 0, bonus == rounds)."""
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"rep_{k}", k)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length, repetition_penalty=1.3)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=k, repetition_penalty=1.3)
        assert spec == ref
        assert stats["correction_tokens"] == 0
        assert stats["bonus_tokens"] == stats["rounds"]

    @pytest.mark.parametrize("k", [2, 4])
    def test_self_spec_min_length_matches_standalone(self, decoder_only_model_path, tmp_path, k):
        """min_length masks EOS for several generated positions; output must still match regular."""
        min_length = len(_PROMPT) + 12
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"minlen_{k}", k)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length, min_length=min_length)
        spec, _ = _greedy(spec_path, _PROMPT, max_length, k=k, min_length=min_length)
        assert spec == ref

    def test_min_length_blocks_early_eos(self, decoder_only_model_path, tmp_path):
        """No EOS token may be committed before min_length is reached."""
        min_length = len(_PROMPT) + 10
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "minlen_block", 4)
        seq, _ = _greedy(spec_path, _PROMPT, max_length, k=4, min_length=min_length)
        eos_ids = _eos_ids(decoder_only_model_path)
        for pos in range(min(min_length, len(seq))):
            assert seq[pos] not in eos_ids

    def test_combined_penalties_match_standalone(self, decoder_only_model_path, tmp_path):
        """min_length + repetition_penalty applied together must still equal regular decoding."""
        min_length = len(_PROMPT) + 8
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "combo", 4)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length,
                         repetition_penalty=1.3, min_length=min_length)
        spec, _ = _greedy(spec_path, _PROMPT, max_length, k=4,
                          repetition_penalty=1.3, min_length=min_length)
        assert spec == ref

    def test_draft_neq_target_repetition_penalty(self, test_data_path, tmp_path):
        """Real draft != target: proposals get rejected, so the correction path runs under the
        penalty. Speculative greedy with repetition_penalty must still equal target-only greedy."""
        pair = _find_two_distinct_models(test_data_path)
        if pair is None:
            pytest.skip("Need two distinct decoder-only models under --test_models")
        target_dir, draft_dir = pair
        max_length = len(_PROMPT) + 16
        spec_path = _build_spec(target_dir, draft_dir, tmp_path / "rep_neq", 4)
        ref, _ = _greedy(target_dir, _PROMPT, max_length, repetition_penalty=1.3)
        spec, stats = _greedy(spec_path, _PROMPT, max_length, k=4, repetition_penalty=1.3)
        assert spec == ref


class TestSpeculativeCommitToken:
    """Item #1: committed tokens go straight to the search (Search::CommitToken) instead of a
    per-token one-hot logits row + argmax. This is behavior-invariant -- the equivalence tests are
    the real guard -- so these pin CommitToken's own EOS / max-length bookkeeping explicitly."""

    @pytest.mark.parametrize("k", [1, 4, 8])
    def test_stops_at_or_before_max_length(self, decoder_only_model_path, tmp_path, k):
        """CommitToken must mark the search done at max_length: the sequence never overruns the
        cap, and (absent an EOS) lands exactly on it -- even though a round commits several tokens
        at once and could otherwise sail past the boundary."""
        max_length = len(_PROMPT) + 7
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"commit_maxlen_{k}", k)
        seq, _ = _greedy(spec_path, _PROMPT, max_length, k=k)
        assert len(seq) <= max_length
        eos_ids = _eos_ids(decoder_only_model_path)
        if not (set(seq[len(_PROMPT):]) & eos_ids):
            assert len(seq) == max_length

    def test_commit_matches_standalone_greedy(self, decoder_only_model_path, tmp_path):
        """Pin item #1 specifically: the tokens CommitToken appends equal plain greedy of the
        same model, across a multi-token round window."""
        max_length = len(_PROMPT) + 18
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "commit_eq", 8)
        ref, _ = _greedy(decoder_only_model_path, _PROMPT, max_length)
        spec, _ = _greedy(spec_path, _PROMPT, max_length, k=8)
        assert spec == ref


class TestSpeculativeSampling:
    """Item #2 keeps the sampling path behavior-identical: target rows are stored as truncated
    categoricals and densified on demand only for the one correction/bonus row, so the RNG draws
    match the old dense path. These tests exercise that path (greedy tests never do)."""

    @pytest.mark.parametrize("k", [2, 4])
    def test_sampling_is_deterministic_with_fixed_seed(self, decoder_only_model_path, tmp_path, k):
        """Same seed -> identical output: guards against uninitialized/nondeterministic state in
        the sparse row path."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"samp_det_{k}", k)
        a, _ = _sample(spec_path, _PROMPT, max_length, seed=1234, k=k, top_k=40, top_p=0.95, temperature=0.8)
        b, _ = _sample(spec_path, _PROMPT, max_length, seed=1234, k=k, top_k=40, top_p=0.95, temperature=0.8)
        assert a == b

    @pytest.mark.parametrize("k", [2, 4])
    def test_self_spec_sampling_accepts_everything(self, decoder_only_model_path, tmp_path, k):
        """Self-spec sampling: the target and draft distributions are identical, so the accept
        probability min(1, p_t/p_d) is exactly 1 and every proposal is accepted (correction == 0,
        bonus == rounds). This directly validates item #2's sparse target-prob lookup -- a wrong
        p_t would break p_t == p_d and produce corrections."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"samp_accept_{k}", k)
        seq, stats = _sample(spec_path, _PROMPT, max_length, seed=1234, k=k, top_k=40, top_p=0.95, temperature=0.8)
        assert len(seq) > len(_PROMPT)
        assert stats["correction_tokens"] == 0
        assert stats["bonus_tokens"] == stats["rounds"]

    def test_sampling_produces_valid_tokens(self, decoder_only_model_path, tmp_path):
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "samp_valid", 4)
        vocab = _vocab_size(decoder_only_model_path)
        seq, stats = _sample(spec_path, _PROMPT, max_length, seed=7, k=4, top_k=40, top_p=0.95, temperature=0.8)
        assert len(seq) > len(_PROMPT)
        assert all(0 <= t < vocab for t in seq)
        assert stats["draft_tokens_proposed"] > 0

    def test_sampling_draft_neq_target_exercises_correction(self, test_data_path, tmp_path):
        """Draft != target under sampling: proposals get rejected, driving the on-demand densify of
        the correction row. Generation must still run and produce valid output."""
        pair = _find_two_distinct_models(test_data_path)
        if pair is None:
            pytest.skip("Need two distinct decoder-only models under --test_models")
        target_dir, draft_dir = pair
        max_length = len(_PROMPT) + 16
        spec_path = _build_spec(target_dir, draft_dir, tmp_path / "samp_neq", 4)
        vocab = _vocab_size(target_dir)
        seq, stats = _sample(spec_path, _PROMPT, max_length, seed=99, k=4, top_k=40, top_p=0.95, temperature=0.8)
        assert len(seq) > len(_PROMPT)
        assert all(0 <= t < vocab for t in seq)
        assert stats["rounds"] > 0


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

    def test_guidance(self, decoder_only_model_path, tmp_path):
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "g_guid", 4))
        params = self._params(model, do_sample=False)
        params.set_guidance("regex", r"[0-9]+")
        with pytest.raises(Exception, match="guidance|constrained"):
            og.Generator(model, params)

    # Note: repetition_penalty and min_length are now SUPPORTED (see TestSpeculativePenalties),
    # so they are intentionally not in the fail-fast list below.
    @pytest.mark.parametrize("bad", [
        {"batch_size": 2},
        {"num_beams": 2},
    ])
    def test_unsupported_config_fails_fast_without_corrupting_model(
            self, decoder_only_model_path, tmp_path, bad):
        """Each unsupported config must throw at Generator construction (before any token is
        generated) and must not leave the model in a bad state: a valid generator built on the
        same model afterward still works and produces output."""
        model = og.Model(_build_self_spec(decoder_only_model_path, tmp_path / "no_corrupt", 4))

        # The guard fires during Generator construction, before generation proceeds.
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(_PROMPT) + 8, do_sample=False, **bad)
        params.set_speculative_options(max_draft_tokens=4)
        with pytest.raises(Exception):
            og.Generator(model, params)

        # The model is unharmed: a valid generator still builds and generates normally.
        good = og.GeneratorParams(model)
        good.set_search_options(max_length=len(_PROMPT) + 8, do_sample=False)
        good.set_speculative_options(max_draft_tokens=4)
        gen = og.Generator(model, good)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        assert len(list(gen.get_sequence(0))) > len(_PROMPT)


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
