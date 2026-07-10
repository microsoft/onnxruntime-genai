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
import re
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
        with pytest.raises(Exception, match="sliding-window KV cache"):
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


def _make_pruned_logits_model(source_dir: str, dest_dir: Path) -> str:
    """Make the graph return only its final logits row, exercising speculative's sequential
    verification fallback for targets that do not expose every verified token's logits."""
    import onnx
    from onnx import helper, numpy_helper

    source_dir = os.path.abspath(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(os.path.join(source_dir, "model.onnx"))

    rewired = False
    for node in model.graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == "logits":
                node.output[i] = "logits_all_tokens"
                rewired = True
    if not rewired:
        raise RuntimeError("expected a node producing a `logits` output")

    model.graph.initializer.append(
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_logits_index"))
    model.graph.node.append(
        helper.make_node(
            "Gather", ["logits_all_tokens", "last_logits_index"], ["logits"],
            axis=1, name="keep_last_logits"))
    onnx.save(model, os.path.join(dest_dir, "model.onnx"))
    shutil.copyfile(
        os.path.join(source_dir, "genai_config.json"),
        os.path.join(dest_dir, "genai_config.json"))
    return os.fspath(dest_dir)


def _make_control_input_model(source_dir: str, dest_dir: Path) -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source_dir = os.path.abspath(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(os.path.join(source_dir, "model.onnx"))
    vocab_size = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value

    for node in model.graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == "logits":
                node.output[i] = "logits_base"

    zeros = np.zeros((vocab_size,), dtype=np.float32)
    for name in ("adapter_bias", "user_bias"):
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [vocab_size]))
    model.graph.initializer.append(numpy_helper.from_array(zeros, "adapter_bias"))

    model.graph.node.extend([
        helper.make_node("Add", ["adapter_bias", "user_bias"], ["combined_bias"]),
        helper.make_node("Add", ["logits_base", "combined_bias"], ["logits"]),
    ])
    onnx.save(model, os.path.join(dest_dir, "model.onnx"))
    shutil.copyfile(
        os.path.join(source_dir, "genai_config.json"),
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

    # Note: guidance now supports greedy, sampling, AND repetition_penalty / min_length
    # (see TestSpeculativeGuidance), so none of those combinations fail fast anymore.
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
    """Continuous decoding: RewindToLength followed directly by GenerateNextToken must resume
    generation. The post-rewind first step recomputes the boundary logits the same way
    StandardDecodingStrategy does -- replay the boundary token through the model -- which for
    speculative refreshes both the target's pos0 row and the draft's pending row (via
    SpeculativeDecodingState::Run).

    Reproduction guarantees match what the mechanism can offer:
      * K == 1 verifies one token per pass (like regular decoding), so a rewind resumes bit-exactly
        and the output matches the uninterrupted run token-for-token (the regular-path
        RewindGptFp32CAPI contract).
      * K > 1: a rewind turns one K-wide batched verify into a single-token forward at the seam, so
        near-tie greedy argmaxes may flip -- exactly as they do for the already-supported
        rewind -> re-prefill flow. The contract there is that the direct rewind -> generate path is
        byte-identical to that supported rewind -> append(boundary) -> generate path.
    """

    def _spec_gen(self, spec_path, k, max_length):
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        return og.Generator(model, params)

    def test_k1_rewind_then_generate_matches_uninterrupted(self, decoder_only_model_path, tmp_path):
        """K=1 verifies one token per pass (like regular decoding), so rewind -> generate resumes
        from the boundary token and reproduces the uninterrupted greedy output token-for-token."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "rw_k1", 1)
        ref, _ = _greedy(spec_path, _PROMPT, max_length, k=1)

        gen = self._spec_gen(spec_path, 1, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen.generate_next_token()
        gen.rewind_to(len(_PROMPT))
        while not gen.is_done():
            gen.generate_next_token()
        got = [int(t) for t in gen.get_sequence(0)]
        assert got == ref

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_rewind_then_generate_matches_rewind_then_append(self, decoder_only_model_path, tmp_path, k):
        """Core continuous-decoding contract for every K: the direct post-rewind flow
        (rewind -> generate) must be byte-identical to the already-supported re-prefill flow
        (rewind -> append(boundary) -> generate). Both recompute the same boundary logits, so
        resuming is equivalent whichever way the caller does it -- including the K>1 seam case where
        the resumed output legitimately differs from the uninterrupted run."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"rw_equiv_{k}", k)

        # Flow A: rewind then generate directly (recomputes the boundary logits in the strategy).
        gen_a = self._spec_gen(spec_path, k, max_length)
        gen_a.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen_a.generate_next_token()
        boundary = int(gen_a.get_sequence(0)[len(_PROMPT)])
        gen_a.rewind_to(len(_PROMPT))
        while not gen_a.is_done():
            gen_a.generate_next_token()
        seq_a = [int(t) for t in gen_a.get_sequence(0)]

        # Flow B: rewind then re-append the boundary token (re-prefill), then generate.
        gen_b = self._spec_gen(spec_path, k, max_length)
        gen_b.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen_b.generate_next_token()
        gen_b.rewind_to(len(_PROMPT))
        gen_b.append_tokens(np.array([[boundary]], dtype=np.int32))
        while not gen_b.is_done():
            gen_b.generate_next_token()
        seq_b = [int(t) for t in gen_b.get_sequence(0)]

        assert seq_a == seq_b

    @pytest.mark.parametrize("k", [1, 4])
    def test_rewind_to_zero_then_reprefill_matches_clean_run(self, decoder_only_model_path, tmp_path, k):
        """Full rewind (to length 0) then a fresh whole-prompt prefill + generate reproduces a clean
        run exactly (a full prefill has no partial-round seam), confirming both inner caches and the
        draft carry-over reset cleanly."""
        max_length = len(_PROMPT) + 12
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"rw_zero_{k}", k)
        ref, _ = _greedy(spec_path, _PROMPT, max_length, k=k)

        gen = self._spec_gen(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            gen.generate_next_token()
        gen.rewind_to(0)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        got = [int(t) for t in gen.get_sequence(0)]
        assert got == ref

    def test_generate_without_prior_state_raises(self, decoder_only_model_path, tmp_path):
        """No AppendTokens/prefill at all: there is no boundary token to recompute from, so a first
        GenerateNextToken must still raise (the same guard StandardDecodingStrategy applies)."""
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "rw_nostate", 4)
        gen = self._spec_gen(spec_path, 4, len(_PROMPT) + 8)
        with pytest.raises(Exception, match="no prior state"):
            gen.generate_next_token()


class TestSpeculativeAppendContinuous:
    """The repo's other form of continuous decoding: calling AppendTokens again mid-generation
    (e.g. a chat turn). Regular decoding allows this at any point because it never buffers. The
    speculative loop buffers a whole round and defers the KV re-anchor, so a mid-round append used
    to leave the two inner caches out of sync with the committed sequence (crash) or leave stale
    round bookkeeping (crash on the next generate). PrepareForAppend now reconciles the inner caches
    to the committed length before the append runs, so append works at ANY point -- like regular.

    Guaranteed invariants (independent of speculative round boundaries):
      * no crash for any number of generated tokens before the append,
      * the already-committed tokens plus the freshly appended tokens are preserved exactly (the
        reconcile only rebuilds KV, it never rewrites emitted tokens),
      * generation is deterministic.
    """

    def _spec_gen(self, spec_path, k, max_length):
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        return og.Generator(model, params)

    def _append_after(self, spec_path, k, max_length, n_generate, extra):
        """Prompt, generate n tokens (possibly stopping mid-round), append `extra`, generate to
        done. Returns (committed_before_append, full_sequence)."""
        gen = self._spec_gen(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(n_generate):
            if gen.is_done():
                break
            gen.generate_next_token()
        committed_before = [int(t) for t in gen.get_sequence(0)]
        gen.append_tokens(np.array([extra], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        return committed_before, [int(t) for t in gen.get_sequence(0)]

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_mid_round_append_preserves_prefix_and_is_deterministic(
            self, decoder_only_model_path, tmp_path, k, n):
        """Appending after any number of generated tokens (including mid-round) must not crash,
        must keep the committed tokens + appended tokens as an exact prefix, and must be
        deterministic. Covers every K and both round-boundary and mid-round stop points."""
        extra = [785, 6722]
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"ap_{k}_{n}", k)

        committed_before, full = self._append_after(spec_path, k, max_length, n, extra)
        # Prefix preserved: the reconcile rebuilds only KV, never the emitted tokens.
        assert full[:len(committed_before)] == committed_before
        assert full[len(committed_before):len(committed_before) + len(extra)] == extra
        # Generation actually continued past the append (unless it legitimately hit max_length).
        assert len(full) >= len(committed_before) + len(extra)
        # Deterministic: an identical run yields an identical sequence.
        _, full2 = self._append_after(spec_path, k, max_length, n, extra)
        assert full == full2

    @pytest.mark.parametrize("k", [2, 4])
    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_mid_round_append_matches_full_reprefill(
            self, decoder_only_model_path, tmp_path, k, n):
        """A mid-round append yields the same continuation as building the identical committed
        sequence from scratch (fresh prefill of committed_prefix + appended tokens) and generating.
        This is the strong correctness check: the reconcile produces a state equivalent to a clean
        rebuild of 'committed[0:L] + appended', so speculative append-based continuous decoding is
        consistent with the model's own decoding of that sequence."""
        extra = [785, 6722]
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"apr_{k}_{n}", k)

        committed_before, mid_round = self._append_after(spec_path, k, max_length, n, extra)

        # Oracle: feed the exact committed prefix to a fresh generator, then the same appended
        # tokens, and generate. Same logical sequence, built via a clean prefill.
        ref_gen = self._spec_gen(spec_path, k, max_length)
        ref_gen.append_tokens(np.array([committed_before], dtype=np.int32))
        ref_gen.append_tokens(np.array([extra], dtype=np.int32))
        while not ref_gen.is_done():
            ref_gen.generate_next_token()
        ref = [int(t) for t in ref_gen.get_sequence(0)]

        assert mid_round == ref


class TestSpeculativeExternalApis:
    def _generator(self, spec_path, k, max_length):
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        return og.Generator(model, params)

    def _finish(self, generator):
        while not generator.is_done():
            generator.generate_next_token()
        return [int(t) for t in generator.get_sequence(0)]

    def test_named_io_forwards_to_target_and_missing_names_throw(
            self, decoder_only_model_path, tmp_path):
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "external_named_io", 4)
        gen = self._generator(spec_path, 4, len(_PROMPT) + 12)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))

        assert gen.get_input("input_ids").size > 0
        assert gen.get_output("logits").size >= _vocab_size(decoder_only_model_path)
        with pytest.raises(Exception, match="input 'missing_input' was not found"):
            gen.get_input("missing_input")
        with pytest.raises(Exception, match="output 'missing_output' was not found"):
            gen.get_output("missing_output")

    def test_named_input_falls_back_to_draft(self, decoder_only_model_path, tmp_path):
        controlled_draft = _make_control_input_model(
            decoder_only_model_path, tmp_path / "draft_named_input")
        spec_path = _build_spec(
            decoder_only_model_path, controlled_draft, tmp_path / "draft_named_spec", 4)
        gen = self._generator(spec_path, 4, len(_PROMPT) + 8)
        bias = np.zeros((_vocab_size(decoder_only_model_path),), dtype=np.float32)
        gen.set_model_input("user_bias", bias)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))

        np.testing.assert_array_equal(gen.get_input("user_bias"), bias)

    def test_unknown_extra_input_throws(self, decoder_only_model_path, tmp_path):
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "external_extra", 4)
        gen = self._generator(spec_path, 4, len(_PROMPT) + 12)
        value = np.ones((1,), dtype=np.float32)
        gen.set_model_input("missing_extra_input", value)
        with pytest.raises(Exception, match="was not found in the target or draft model"):
            gen.append_tokens(np.array([_PROMPT], dtype=np.int32))

    def test_termination_reaches_and_releases_child_sessions(
            self, decoder_only_model_path, tmp_path):
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "external_terminate", 4)
        gen = self._generator(spec_path, 4, len(_PROMPT) + 12)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        gen.set_runtime_option("terminate_session", "1")
        with pytest.raises(Exception, match="Terminated state"):
            gen.generate_next_token()
        gen.set_runtime_option("terminate_session", "0")
        gen.generate_next_token()

    def test_runtime_profiling_reaches_both_child_sessions(
            self, decoder_only_model_path, tmp_path):
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "external_profile", 4)
        gen = self._generator(spec_path, 4, len(_PROMPT) + 8)
        profile_prefix = os.fspath(tmp_path / "child_profile")
        try:
            gen.set_runtime_option("enable_profiling", profile_prefix)
        except Exception as exc:
            if "requires ONNX Runtime 1.25 or later" in str(exc):
                pytest.skip(str(exc))
            raise

        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        gen.generate_next_token()

        # A speculative prefill executes both child states. The unused composite State would emit no
        # profile, so at least two files proves the option reached the actual target and draft runs.
        assert len(list(tmp_path.glob("child_profile*.json"))) >= 2

    def test_extra_input_reaches_both_children_and_adapter_reaches_target(
            self, decoder_only_model_path, tmp_path):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime is required to create an adapter fixture")

        controlled = _make_control_input_model(
            decoder_only_model_path, tmp_path / "controlled_model")
        vocab_size = _vocab_size(decoder_only_model_path)
        forced_token = _PROMPT[0]
        bias = np.zeros((vocab_size,), dtype=np.float32)
        bias[forced_token] = 1e4

        extra_spec = _build_self_spec(controlled, tmp_path / "extra_spec", 4)
        extra_gen = self._generator(extra_spec, 4, len(_PROMPT) + 4)
        extra_gen.set_model_input("user_bias", bias)
        extra_gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        extra_gen.generate_next_token()
        assert int(extra_gen.get_sequence(0)[-1]) == forced_token

        adapter_spec = _build_spec(
            controlled, decoder_only_model_path, tmp_path / "adapter_spec", 4)
        model = og.Model(adapter_spec)
        adapters = og.Adapters(model)
        adapter_path = os.fspath(tmp_path / "bias.onnx_adapter")
        adapter = onnxruntime.AdapterFormat()
        adapter.set_adapter_version(1)
        adapter.set_model_version(1)
        adapter.set_parameters({
            "adapter_bias": onnxruntime.OrtValue.ortvalue_from_numpy(bias)
        })
        adapter.export_adapter(adapter_path)
        adapters.load(adapter_path, "bias")

        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(_PROMPT) + 4)
        params.set_speculative_options(max_draft_tokens=4)
        adapter_gen = og.Generator(model, params)
        adapter_gen.set_active_adapter(adapters, "bias")
        zero_bias = np.zeros_like(bias)
        adapter_gen.set_model_input("user_bias", zero_bias)
        adapter_gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        adapter_gen.generate_next_token()
        assert int(adapter_gen.get_sequence(0)[-1]) == forced_token

    @pytest.mark.parametrize("k,n_generate", [(2, 1), (4, 2), (8, 3)])
    def test_get_logits_mid_round_matches_clean_rebuild(
            self, decoder_only_model_path, tmp_path, k, n_generate):
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"external_get_{k}", k)
        gen = self._generator(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(n_generate):
            gen.generate_next_token()

        committed = [int(t) for t in gen.get_sequence(0)]
        logits = gen.get_logits()
        assert logits.shape == (1, 1, _vocab_size(decoder_only_model_path))
        actual = self._finish(gen)

        ref = self._generator(spec_path, k, max_length)
        ref.append_tokens(np.array([committed], dtype=np.int32))
        assert actual == self._finish(ref)

    @pytest.mark.parametrize("n_generate", [1, 2, 4, 5])
    def test_get_logits_does_not_change_sampled_generation(
            self, decoder_only_model_path, tmp_path, n_generate):
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(
            decoder_only_model_path, tmp_path / f"external_sample_get_{n_generate}", 4)

        def run(inspect_logits):
            model = og.Model(spec_path)
            params = og.GeneratorParams(model)
            params.set_search_options(
                do_sample=True, max_length=max_length, random_seed=1234,
                top_k=40, top_p=0.95, temperature=0.8)
            params.set_speculative_options(max_draft_tokens=4)
            gen = og.Generator(model, params)
            gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
            for _ in range(n_generate):
                gen.generate_next_token()
            if inspect_logits:
                first = gen.get_logits().copy()
                second = gen.get_logits().copy()
                np.testing.assert_array_equal(first, second)
            sequence = self._finish(gen)
            stats = gen.get_speculative_stats()
            counters = {
                key: stats[key]
                for key in (
                    "rounds", "draft_tokens_proposed", "draft_tokens_accepted",
                    "correction_tokens", "bonus_tokens", "target_forward_passes")
            }
            return sequence, counters

        baseline = run(False)
        inspected = run(True)
        assert inspected == baseline

    def test_get_logits_mid_round_fp16_is_side_effect_free(
            self, fp16_decoder_only_model_path, tmp_path):
        max_length = len(_PROMPT) + 12
        spec_path = _build_self_spec(
            fp16_decoder_only_model_path, tmp_path / "external_fp16_get", 4)

        baseline = self._generator(spec_path, 4, max_length)
        baseline.append_tokens(np.array([_PROMPT], dtype=np.int32))
        baseline.generate_next_token()
        baseline_sequence = self._finish(baseline)

        inspected = self._generator(spec_path, 4, max_length)
        inspected.append_tokens(np.array([_PROMPT], dtype=np.int32))
        inspected.generate_next_token()
        assert inspected.get_logits().shape == (
            1, 1, _vocab_size(fp16_decoder_only_model_path))
        assert self._finish(inspected) == baseline_sequence

    def test_get_logits_mid_round_pruned_target_is_side_effect_free(
            self, decoder_only_model_path, tmp_path):
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx is required to create a pruned-logits fixture")

        pruned = _make_pruned_logits_model(
            decoder_only_model_path, tmp_path / "pruned_logits_model")
        max_length = len(_PROMPT) + 12
        spec_path = _build_self_spec(pruned, tmp_path / "external_pruned_get", 4)

        baseline = self._generator(spec_path, 4, max_length)
        baseline.append_tokens(np.array([_PROMPT], dtype=np.int32))
        baseline.generate_next_token()
        baseline_sequence = self._finish(baseline)

        inspected = self._generator(spec_path, 4, max_length)
        inspected.append_tokens(np.array([_PROMPT], dtype=np.int32))
        inspected.generate_next_token()
        assert inspected.get_logits().shape == (1, 1, _vocab_size(pruned))
        assert self._finish(inspected) == baseline_sequence

    @pytest.mark.parametrize("k,n_generate", [(2, 1), (4, 2), (8, 3)])
    def test_set_logits_mid_round_forces_token_and_matches_clean_rebuild(
            self, decoder_only_model_path, tmp_path, k, n_generate):
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"external_set_{k}", k)
        gen = self._generator(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(n_generate):
            gen.generate_next_token()

        forced_token = _PROMPT[0]
        forced_logits = np.full(
            (1, 1, _vocab_size(decoder_only_model_path)), -1e9, dtype=np.float32)
        forced_logits[0, 0, forced_token] = 1e9
        gen.set_logits(forced_logits)
        gen.generate_next_token()
        assert int(gen.get_sequence(0)[-1]) == forced_token

        committed = [int(t) for t in gen.get_sequence(0)]
        actual = self._finish(gen)
        ref = self._generator(spec_path, k, max_length)
        ref.append_tokens(np.array([committed], dtype=np.int32))
        assert actual == self._finish(ref)

    def test_set_logits_mid_round_sampling_is_deterministic(
            self, decoder_only_model_path, tmp_path):
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(
            decoder_only_model_path, tmp_path / "external_sample_set", 4)
        forced_token = _PROMPT[0]
        forced_logits = np.full(
            (1, 1, _vocab_size(decoder_only_model_path)), -1e9, dtype=np.float32)
        forced_logits[0, 0, forced_token] = 1e9

        def run():
            model = og.Model(spec_path)
            params = og.GeneratorParams(model)
            params.set_search_options(
                do_sample=True, max_length=max_length, random_seed=4321,
                top_k=40, top_p=0.95, temperature=0.8)
            params.set_speculative_options(max_draft_tokens=4)
            gen = og.Generator(model, params)
            gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
            gen.generate_next_token()
            gen.set_logits(forced_logits)
            gen.generate_next_token()
            assert int(gen.get_sequence(0)[-1]) == forced_token
            return self._finish(gen)

        assert run() == run()

    def test_get_logits_after_rewind_matches_reappend(
            self, decoder_only_model_path, tmp_path):
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "external_rewind", 4)
        gen = self._generator(spec_path, 4, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(4):
            gen.generate_next_token()

        rewind_length = len(_PROMPT) + 1
        boundary = int(gen.get_sequence(0)[rewind_length])
        gen.rewind_to(rewind_length)
        assert gen.get_logits().shape == (1, 1, _vocab_size(decoder_only_model_path))
        committed = [int(t) for t in gen.get_sequence(0)]
        assert len(committed) == rewind_length + 1
        assert committed[-1] == boundary

        actual = self._finish(gen)
        ref = self._generator(spec_path, 4, max_length)
        ref.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(4):
            ref.generate_next_token()
        ref.rewind_to(rewind_length)
        ref.append_tokens(np.array([[boundary]], dtype=np.int32))
        assert actual == self._finish(ref)


# ---------------------------------------------------------------------------
# Guidance (constrained decoding) parity. Requires a USE_GUIDANCE build (llguidance) and a model
# with tokenizer.json. Speculative greedy + guidance must reproduce regular greedy + guidance.
# ---------------------------------------------------------------------------

def _build_self_spec_guided(source_dir, dest_dir, k):
    """Self-spec wrapper that also carries the tokenizer files guidance (llguidance) needs."""
    spec_path = _build_self_spec(source_dir, dest_dir, k)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(source_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(spec_path, fn))
    return spec_path


def _build_spec_guided(target_dir, draft_dir, dest_dir, k):
    """Draft != target wrapper that also carries the tokenizer files guidance needs (guidance uses
    the target tokenizer)."""
    spec_path = _build_spec(target_dir, draft_dir, dest_dir, k)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(target_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(spec_path, fn))
    return spec_path


_GUIDANCE_SUPPORTED_CACHE = {}


def _guidance_supported(model_path):
    """True if this build actually constrains output (USE_GUIDANCE=ON and tokenizer.json present).
    On a guidance-off build CreateGuidanceLogitsProcessor returns nullptr and masking is skipped."""
    if model_path in _GUIDANCE_SUPPORTED_CACHE:
        return _GUIDANCE_SUPPORTED_CACHE[model_path]
    ok = False
    try:
        model = og.Model(model_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(_PROMPT) + 4)
        params.set_guidance("regex", r"[0-9]")  # forces the next token to be a single digit
        gen = og.Generator(model, params)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        gen.generate_next_token()
        first = int(gen.get_sequence(0)[len(_PROMPT)])
        ok = og.Tokenizer(model).decode([first]).strip().isdigit()
    except Exception:
        ok = False
    _GUIDANCE_SUPPORTED_CACHE[model_path] = ok
    return ok


@pytest.fixture
def guidance_model_path(decoder_only_model_path):
    if not _guidance_supported(decoder_only_model_path):
        pytest.skip("Build lacks guidance (USE_GUIDANCE=OFF) or model has no tokenizer.json")
    return decoder_only_model_path


def _guided_tokens(model_path, gtype, gdata, max_length, k=None, want_stats=False,
                   repetition_penalty=None, min_length=None):
    """Greedy + guidance generation. k=None -> regular; k set -> speculative. Returns the generated
    tail token ids (after the prompt); with want_stats=True also returns the speculative stats dict."""
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
    params.set_guidance(gtype, gdata)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
    while not gen.is_done():
        gen.generate_next_token()
    tail = [int(t) for t in gen.get_sequence(0)][len(_PROMPT):]
    if want_stats:
        return tail, gen.get_speculative_stats()
    return tail


def _sampled_guided_tokens(model_path, gtype, gdata, max_length, seed, k=None,
                           temperature=1.0, top_p=1.0, top_k=0):
    """Sampling (do_sample=True) + guidance. k=None -> regular; k set -> speculative. Returns the
    generated tail token ids. A fixed seed makes a single run reproducible."""
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    opts = dict(do_sample=True, max_length=max_length, temperature=temperature, random_seed=seed)
    if top_p:
        opts["top_p"] = top_p
    if top_k:
        opts["top_k"] = top_k
    params.set_search_options(**opts)
    if k is not None:
        params.set_speculative_options(max_draft_tokens=k)
    params.set_guidance(gtype, gdata)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
    while not gen.is_done():
        gen.generate_next_token()
    return [int(t) for t in gen.get_sequence(0)][len(_PROMPT):]


_GUIDANCE_CASES = [
    ("regex", r"[0-9]{3}-[0-9]{3}"),
    ("regex", r"(yes|no) [a-z]{2,5}"),
    ("json_schema", json.dumps({"type": "object",
                                "properties": {"a": {"type": "integer"}},
                                "required": ["a"]})),
    ("lark_grammar", 'start: "answer:" /[0-9]{1,6}/'),
]


class TestSpeculativeGuidance:
    """Speculative greedy + guidance. Contract mirrors the rest of speculative decoding:
      - K=1 reproduces regular greedy + guidance token-for-token (no batched verify -> bit-exact).
      - K>1 stays grammar-valid and deterministic; it may differ from regular only where the batched
        verify's logits differ from per-token logits at fp near-ties in UNCONSTRAINED free regions
        (the same batched-verify seam the non-guidance path has).
    Draft proposals are grammar-masked so the masked target accepts them -- that is what preserves
    the speculative speedup under guidance. The strategy reuses the regular constrained-decoding
    ops (ProcessLogits mask + CommitTokens + fast-forward) and commits accepted draft tokens by their
    per-token value, exactly like the non-guidance accept loop."""

    @pytest.mark.parametrize("gtype,gdata", _GUIDANCE_CASES)
    def test_k1_matches_regular_guidance(self, guidance_model_path, tmp_path, gtype, gdata):
        """K=1: bit-exact with regular greedy + guidance (no batching, so the guarantee is exact)."""
        max_length = len(_PROMPT) + 24
        ref = _guided_tokens(guidance_model_path, gtype, gdata, max_length)
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "guid_k1", 1)
        got = _guided_tokens(spec_path, gtype, gdata, max_length, k=1)
        assert got == ref

    @pytest.mark.parametrize("k", [2, 4, 8])
    @pytest.mark.parametrize("gtype,gdata", _GUIDANCE_CASES)
    def test_kgt1_is_deterministic(self, guidance_model_path, tmp_path, k, gtype, gdata):
        """K>1: identical output run-to-run (the batched-verify seam must stay deterministic)."""
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_det_{k}", k)
        a = _guided_tokens(spec_path, gtype, gdata, max_length, k=k)
        b = _guided_tokens(spec_path, gtype, gdata, max_length, k=k)
        assert a == b

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_regex_output_satisfies_grammar(self, guidance_model_path, tmp_path, k):
        """The decoded speculative output actually matches the constraining regex at every K (a
        bounded regex completes before max_length, so the whole output is checkable)."""
        pattern = r"[0-9]{3}-[0-9]{3}"
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_sat_{k}", k)
        got = _guided_tokens(spec_path, "regex", pattern, max_length, k=k)
        eos = _eos_ids(guidance_model_path)
        text = og.Tokenizer(og.Model(spec_path)).decode([t for t in got if t not in eos])
        assert re.fullmatch(pattern, text), f"output {text!r} does not match {pattern!r}"

    @pytest.mark.parametrize("k", [4, 8])
    def test_draft_masking_keeps_acceptance_high(self, guidance_model_path, tmp_path, k):
        """Draft proposals are grammar-masked, so under self-spec (draft == target) the masked target
        accepts essentially all of them. Without masking the draft proposes grammar-invalid tokens
        and acceptance collapses (~0.1 at K=8), which is what would kill the speedup under guidance."""
        pattern = r"[0-9]{3}-[0-9]{3}"
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_acc_{k}", k)
        _, stats = _guided_tokens(spec_path, "regex", pattern, max_length, k=k, want_stats=True)
        assert stats["acceptance_rate"] >= 0.6

    def test_forced_span_rides_one_verify_pass(self, guidance_model_path, tmp_path):
        """FF-in-batch optimization (the whole point -- fewer target passes): a grammar-forced token
        span is packed into the target's batched verify, so many forced tokens are confirmed in ONE
        target forward pass. For a fully-forced string at K=8 the target runs far fewer passes than it
        emits tokens. The old design ended the round at the first forced span and replayed each forced
        token per-token (~1 token per target pass), so it emitted as many passes as tokens and could
        not satisfy generated > passes."""
        grammar = 'start: "The capital of France is Paris."'  # every output token is grammar-forced
        max_length = len(_PROMPT) + 24
        k = 8
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "guid_ff", k)
        tail, stats = _guided_tokens(spec_path, "lark_grammar", grammar, max_length, k=k,
                                     want_stats=True)
        eos = _eos_ids(guidance_model_path)
        generated = len([t for t in tail if t not in eos])
        passes = stats["target_forward_passes"]
        assert generated >= 4, f"expected a multi-token forced span, got {generated} tokens"
        assert passes >= 1
        # Forced tokens ride the batched verify -> strictly fewer target passes than emitted tokens
        # (the old per-token-replay design did ~1 token per pass and cannot satisfy this).
        assert generated > passes, f"{generated} tokens took {passes} target passes (no FF batching)"
        assert generated / passes >= 2.0, f"only {generated / passes:.2f} tokens per target pass"

    @pytest.mark.parametrize("gtype,gdata", _GUIDANCE_CASES)
    def test_draft_neq_target_k1_matches_regular_guidance(self, test_data_path, tmp_path, gtype, gdata):
        """A real draft != target pair under guidance: the draft genuinely mispredicts, so rounds
        reject and re-anchor on the target's masked correction token. K=1 speculative + guidance
        must equal regular + guidance of the target exactly (the guaranteed greedy contract)."""
        pair = _find_two_distinct_models(test_data_path)
        if pair is None:
            pytest.skip("Need two distinct decoder-only models under --test_models")
        target_dir, draft_dir = pair
        if not _guidance_supported(target_dir):
            pytest.skip("Build lacks guidance (USE_GUIDANCE=OFF) or model has no tokenizer.json")
        max_length = len(_PROMPT) + 24
        ref = _guided_tokens(target_dir, gtype, gdata, max_length)
        spec_path = _build_spec_guided(target_dir, draft_dir, tmp_path / "guid_pair", 1)
        got = _guided_tokens(spec_path, gtype, gdata, max_length, k=1)
        assert got == ref

    # --- Sampling + guidance -------------------------------------------------------------------
    # Speculative sampling over the grammar-masked distributions (accept min(1, p_t/p_d) else
    # resample the residual) is unbiased, so the output is a valid sample from the masked target
    # distribution. We assert grammar-validity and per-seed determinism (matching regular sampling
    # bit-for-bit is not expected: the speculative path consumes the RNG differently, exactly like
    # non-guidance sampling).

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_sampling_output_satisfies_grammar(self, guidance_model_path, tmp_path, k):
        """Sampling + guidance output matches the constraining regex at every K."""
        pattern = r"[0-9]{3}-[0-9]{3}"
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_samp_{k}", k)
        eos = _eos_ids(guidance_model_path)
        tok = og.Tokenizer(og.Model(spec_path))
        for seed in range(4):
            got = _sampled_guided_tokens(spec_path, "regex", pattern, max_length, seed, k=k)
            text = tok.decode([t for t in got if t not in eos])
            assert re.fullmatch(pattern, text), f"seed {seed}: {text!r} !~ {pattern!r}"

    @pytest.mark.parametrize("k", [1, 4])
    def test_sampling_deterministic_with_seed(self, guidance_model_path, tmp_path, k):
        """Same seed -> identical sampled output (the batched-verify seam stays deterministic)."""
        max_length = len(_PROMPT) + 20
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_sampdet_{k}", k)
        a = _sampled_guided_tokens(spec_path, "regex", r"[0-9]{3}-[0-9]{3}", max_length, 1234, k=k)
        b = _sampled_guided_tokens(spec_path, "regex", r"[0-9]{3}-[0-9]{3}", max_length, 1234, k=k)
        assert a == b

    # --- Penalty + guidance --------------------------------------------------------------------
    # repetition_penalty / min_length combine with guidance by applying the penalty AFTER the grammar
    # mask (matching the regular ProcessLogits -> ApplyMinLength -> ApplyRepetitionPenalty order).

    @pytest.mark.parametrize("gtype,gdata", _GUIDANCE_CASES)
    def test_k1_penalty_matches_regular_guidance(self, guidance_model_path, tmp_path, gtype, gdata):
        """K=1 greedy + guidance + repetition_penalty is bit-exact with regular + guidance + penalty."""
        max_length = len(_PROMPT) + 24
        ref = _guided_tokens(guidance_model_path, gtype, gdata, max_length, repetition_penalty=1.3)
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "guid_pen_k1", 1)
        got = _guided_tokens(spec_path, gtype, gdata, max_length, k=1, repetition_penalty=1.3)
        assert got == ref

    def test_min_length_with_guidance_blocks_early_eos(self, guidance_model_path, tmp_path):
        """min_length + guidance: the constrained output must reach at least min_length tokens (EOS is
        masked below it, applied after the grammar mask)."""
        min_length = len(_PROMPT) + 12
        max_length = len(_PROMPT) + 24
        # A grammar that CAN stop early (0+ digits), so only min_length forces continuation.
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "guid_minlen", 4)
        got = _guided_tokens(spec_path, "regex", r"[0-9]*", max_length, k=4, min_length=min_length)
        eos = _eos_ids(guidance_model_path)
        non_eos = [t for t in got if t not in eos]
        assert len(non_eos) >= (min_length - len(_PROMPT))

    # --- Continuous decoding + guidance ---------------------------------------------------------
    # rewind-to-0 + a fresh whole-prompt reprefill reuses the generator for a clean constrained run
    # (grammar Reset to initial, sequence emptied, draft carry-over reset). Rewinding to a
    # mid-generation length with a grammar active is NOT supported -- the llguidance grammar cannot
    # roll back to an arbitrary position, a limitation shared with the regular guidance path -- so
    # only rewind-to-0 is exercised.

    @pytest.mark.parametrize("k", [1, 4])
    def test_rewind_to_zero_reprefill_with_guidance(self, guidance_model_path, tmp_path, k):
        """rewind_to(0) + reprefill + guidance reproduces a fresh guided run for every K (both inner
        caches, the draft carry-over, and the grammar all reset cleanly)."""
        pattern = r"[0-9]{2}-[0-9]{2}"
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"guid_rw0_{k}", k)
        ref = _guided_tokens(spec_path, "regex", pattern, max_length, k=k)

        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        params.set_guidance("regex", pattern)
        gen = og.Generator(model, params)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            if gen.is_done():
                break
            gen.generate_next_token()
        gen.rewind_to(0)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        got = [int(t) for t in gen.get_sequence(0)][len(_PROMPT):]
        assert got == ref


class TestSpeculativeGuidanceProduction:
    """Deeper guidance (constrained decoding) coverage for production readiness. Focuses on the
    fast-forward (ff_carry_) machinery that batches grammar-forced spans across rounds, on clean
    termination when the grammar stops, on truncation at max_length, and on state hygiene across a
    rewind. These are the paths most likely to leak state or desync the two inner caches under a
    live grammar."""

    # A lark grammar whose whole output is a forced literal long enough to span several K rounds, so
    # the fast-forward carry (ff_carry_) is exercised round-to-round and then forces a clean stop.
    _FORCED = 'start: "The capital of France is Paris and Berlin is in Germany"'

    def _spec_gen(self, spec_path, k, max_length, gtype, gdata):
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        params.set_guidance(gtype, gdata)
        return og.Generator(model, params)

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_long_forced_span_exact_and_deterministic(self, guidance_model_path, tmp_path, k):
        """A long grammar-forced literal must be emitted exactly (decodes to the literal) and be
        identical run-to-run for every K -- the ff_carry_ span batching must not drop, duplicate, or
        reorder forced tokens."""
        max_length = len(_PROMPT) + 40
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"gp_forced_{k}", k)
        eos = _eos_ids(guidance_model_path)
        tok = og.Tokenizer(og.Model(spec_path))
        a = _guided_tokens(spec_path, "lark_grammar", self._FORCED, max_length, k=k)
        b = _guided_tokens(spec_path, "lark_grammar", self._FORCED, max_length, k=k)
        assert a == b, "forced-span output must be deterministic across runs"
        text = tok.decode([t for t in a if t not in eos]).strip()
        assert text == "The capital of France is Paris and Berlin is in Germany", \
            f"forced span decoded to {text!r}"

    def test_grammar_forced_stop_terminates_cleanly(self, guidance_model_path, tmp_path):
        """A bounded grammar reaches an accepting state and forces EOS. Generation must terminate on
        its own (is_done) well before max_length, with the output decoding to exactly the grammar's
        string -- no hang, no overrun, no extra tokens past the forced stop."""
        max_length = len(_PROMPT) + 40
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "gp_stop", 4)
        eos = _eos_ids(guidance_model_path)
        tok = og.Tokenizer(og.Model(spec_path))
        gen = self._spec_gen(spec_path, 4, max_length, "lark_grammar", 'start: "yes it is"')
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        steps = 0
        while not gen.is_done() and steps < max_length:
            gen.generate_next_token()
            steps += 1
        assert gen.is_done(), "grammar-forced stop must terminate generation"
        tail = [int(t) for t in gen.get_sequence(0)][len(_PROMPT):]
        assert len(gen.get_sequence(0)) < max_length, "must stop before max_length, not by truncation"
        assert tok.decode([t for t in tail if t not in eos]).strip() == "yes it is"

    def test_forced_span_truncates_at_max_length_without_crash(self, guidance_model_path, tmp_path):
        """A forced literal longer than the budget: max_length caps generation mid-span. Must not
        crash or overrun, and what was emitted must be a valid prefix of the forced string."""
        full = "The capital of France is Paris and Berlin is in Germany"
        max_new = 6
        max_length = len(_PROMPT) + max_new
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "gp_trunc", 8)
        eos = _eos_ids(guidance_model_path)
        tok = og.Tokenizer(og.Model(spec_path))
        got = _guided_tokens(spec_path, "lark_grammar", f'start: "{full}"', max_length, k=8)
        assert len(got) <= max_new, f"emitted {len(got)} tokens, cap was {max_new}"
        text = tok.decode([t for t in got if t not in eos])
        assert full.startswith(text.strip()) or text.strip() in full, \
            f"truncated output {text!r} is not a prefix of {full!r}"

    @pytest.mark.parametrize("k", [1, 4])
    def test_ff_carry_cleared_on_rewind_to_zero(self, guidance_model_path, tmp_path, k):
        """Rewind hygiene under a forced-span grammar: stop mid forced span (so ff_carry_ holds
        pending forced tokens), rewind_to(0), reprefill and finish. Must reproduce a clean guided run
        exactly -- proving the fast-forward carry is dropped on rewind and does not leak into the new
        generation."""
        max_length = len(_PROMPT) + 40
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"gp_rwcarry_{k}", k)
        ref = _guided_tokens(spec_path, "lark_grammar", self._FORCED, max_length, k=k)

        gen = self._spec_gen(spec_path, k, max_length, "lark_grammar", self._FORCED)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        # Emit a few tokens so a forced span is in flight (ff_carry_ non-empty for K>1).
        for _ in range(3):
            if gen.is_done():
                break
            gen.generate_next_token()
        gen.rewind_to(0)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        got = [int(t) for t in gen.get_sequence(0)][len(_PROMPT):]
        assert got == ref

    def test_guidance_stable_across_many_runs(self, guidance_model_path, tmp_path):
        """Five independent guided greedy runs of the same grammar must be byte-identical -- no
        cross-run state leakage in the grammar processor, the draft carry-over, or the caches."""
        pattern = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / "gp_stable", 4)
        runs = [_guided_tokens(spec_path, "regex", pattern, max_length, k=4) for _ in range(5)]
        for r in runs[1:]:
            assert r == runs[0], "guided output must be identical across repeated runs"

    @pytest.mark.parametrize("k", [2, 4])
    def test_draft_neq_target_forced_span_grammar_valid(self, test_data_path, tmp_path, k):
        """Draft != target under a forced-span grammar at K>1: the draft mispredicts, rounds reject
        and re-anchor, and forced spans still batch through the verify. The decoded output must still
        satisfy the grammar exactly."""
        pair = _find_two_distinct_models(test_data_path)
        if pair is None:
            pytest.skip("Need two distinct decoder-only models under --test_models")
        target_dir, draft_dir = pair
        if not _guidance_supported(target_dir):
            pytest.skip("Build lacks guidance (USE_GUIDANCE=OFF) or model has no tokenizer.json")
        max_length = len(_PROMPT) + 40
        spec_path = _build_spec_guided(target_dir, draft_dir, tmp_path / f"gp_pair_{k}", k)
        eos = _eos_ids(target_dir)
        tok = og.Tokenizer(og.Model(spec_path))
        got = _guided_tokens(spec_path, "lark_grammar", self._FORCED, max_length, k=k)
        text = tok.decode([t for t in got if t not in eos]).strip()
        assert text == "The capital of France is Paris and Berlin is in Germany", \
            f"draft!=target forced span decoded to {text!r}"

    @pytest.mark.parametrize("k", [1, 4])
    def test_mid_generation_append_with_guidance_is_safe(self, guidance_model_path, tmp_path, k):
        """Appending tokens mid-generation while a grammar is active (e.g. a chat continuation) is an
        unusual operation -- the injected tokens bypass the grammar -- but it must be production-safe:
        no crash, the caches stay consistent, and generation still terminates (is_done or max_length)
        without overrunning. This guards the guidance + PrepareForAppend intersection from regressing
        into a crash or a runaway loop."""
        max_length = len(_PROMPT) + 30
        spec_path = _build_self_spec_guided(guidance_model_path, tmp_path / f"gp_append_{k}", k)
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length)
        params.set_speculative_options(max_draft_tokens=k)
        params.set_guidance("regex", r"[0-9]{3}-[0-9]{3}")
        gen = og.Generator(model, params)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(2):
            if not gen.is_done():
                gen.generate_next_token()
        before = [int(t) for t in gen.get_sequence(0)]
        gen.append_tokens(np.array([[785, 6722]], dtype=np.int32))
        # Appended tokens preserved; generation resumes under the grammar and terminates.
        assert [int(t) for t in gen.get_sequence(0)][:len(before)] == before
        steps = 0
        while not gen.is_done() and steps < max_length:
            gen.generate_next_token()
            steps += 1
        assert len(gen.get_sequence(0)) <= max_length, "must not overrun max_length"


class TestSpeculativeContinuousProduction:
    """Deeper continuous-decoding coverage for production readiness: repeated/interleaved
    AppendTokens and rewind, the max_length boundary, and the shared-buffer path. The speculative
    loop buffers a whole round and defers the KV re-anchor, so these sequences are where the two
    inner caches most easily desync from the committed sequence. Greedy is used throughout so the
    output is deterministic and checkable against a clean rebuild."""

    def _spec_gen(self, spec_path, k, max_length, share_buffer=False):
        model = og.Model(spec_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=max_length,
                                  past_present_share_buffer=share_buffer)
        params.set_speculative_options(max_draft_tokens=k)
        return og.Generator(model, params)

    @pytest.mark.parametrize("k", [2, 4])
    def test_multiple_sequential_appends_match_rebuild(self, decoder_only_model_path, tmp_path, k):
        """Three appends interleaved with generation. Each appended chunk must be preserved exactly,
        and the final continuation after the last append must equal a clean rebuild of the identical
        committed sequence -- i.e. repeated reconciliation stays consistent with the model's own
        decoding of that sequence."""
        e1, e2 = [785, 6722], [279, 9625]
        max_length = len(_PROMPT) + 40
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"cp_multi_{k}", k)

        gen = self._spec_gen(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(2):
            if not gen.is_done():
                gen.generate_next_token()
        gen.append_tokens(np.array([e1], dtype=np.int32))
        for _ in range(2):
            if not gen.is_done():
                gen.generate_next_token()
        committed_before = [int(t) for t in gen.get_sequence(0)]
        gen.append_tokens(np.array([e2], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        full = [int(t) for t in gen.get_sequence(0)]

        # Appended chunks preserved exactly, in order.
        assert full[:len(committed_before)] == committed_before
        assert full[len(committed_before):len(committed_before) + len(e2)] == e2

        # Oracle: clean rebuild of committed_before + e2, then generate.
        ref = self._spec_gen(spec_path, k, max_length)
        ref.append_tokens(np.array([committed_before], dtype=np.int32))
        ref.append_tokens(np.array([e2], dtype=np.int32))
        while not ref.is_done():
            ref.generate_next_token()
        assert full == [int(t) for t in ref.get_sequence(0)]

    @pytest.mark.parametrize("k", [2, 4])
    def test_rewind_append_interleave_deterministic(self, decoder_only_model_path, tmp_path, k):
        """Interleave rewind and append in one session (generate, rewind to a nonzero length,
        generate, append, generate). Must not crash and must be deterministic run-to-run."""
        extra = [785, 6722]
        max_length = len(_PROMPT) + 40

        def run():
            spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"cp_inter_{k}", k)
            gen = self._spec_gen(spec_path, k, max_length)
            gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
            for _ in range(4):
                if not gen.is_done():
                    gen.generate_next_token()
            gen.rewind_to(len(_PROMPT) + 2)
            for _ in range(2):
                if not gen.is_done():
                    gen.generate_next_token()
            gen.append_tokens(np.array([extra], dtype=np.int32))
            while not gen.is_done():
                gen.generate_next_token()
            return [int(t) for t in gen.get_sequence(0)]

        assert run() == run()

    @pytest.mark.parametrize("k", [1, 4])
    def test_append_reaching_max_length_terminates(self, decoder_only_model_path, tmp_path, k):
        """Append tokens that bring the sequence to the max_length boundary. Generation must stop at
        exactly max_length without overrunning or crashing."""
        max_length = len(_PROMPT) + 8
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"cp_maxlen_{k}", k)
        gen = self._spec_gen(spec_path, k, max_length)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(2):
            if not gen.is_done():
                gen.generate_next_token()
        # Append right up to (max_length - 1) so at most one more token can be produced.
        remaining = max_length - len(gen.get_sequence(0))
        if remaining > 1:
            filler = [785] * (remaining - 1)
            gen.append_tokens(np.array([filler], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        assert len(gen.get_sequence(0)) <= max_length, "must not overrun max_length"

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    @pytest.mark.parametrize("before", [1, 2, 5])
    def test_rewind_to_nonzero_points_match_reappend(self, decoder_only_model_path, tmp_path, k, before):
        """rewind_to(nonzero) at several stop points and every K: the direct post-rewind flow must
        match the re-prefill flow (rewind -> append(boundary) -> generate), the established
        continuous-decoding equivalence, at each rewind point."""
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"cp_rwpt_{k}_{before}", k)

        gen_a = self._spec_gen(spec_path, k, max_length)
        gen_a.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(before + 2):
            if not gen_a.is_done():
                gen_a.generate_next_token()
        if len(gen_a.get_sequence(0)) <= len(_PROMPT) + before:
            pytest.skip("model stopped before the rewind point")
        boundary = int(gen_a.get_sequence(0)[len(_PROMPT) + before])
        gen_a.rewind_to(len(_PROMPT) + before)
        while not gen_a.is_done():
            gen_a.generate_next_token()
        seq_a = [int(t) for t in gen_a.get_sequence(0)]

        gen_b = self._spec_gen(spec_path, k, max_length)
        gen_b.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(before + 2):
            if not gen_b.is_done():
                gen_b.generate_next_token()
        gen_b.rewind_to(len(_PROMPT) + before)
        gen_b.append_tokens(np.array([[boundary]], dtype=np.int32))
        while not gen_b.is_done():
            gen_b.generate_next_token()
        assert seq_a == [int(t) for t in gen_b.get_sequence(0)]

    def test_repeated_rewind_zero_reprefill_stable(self, decoder_only_model_path, tmp_path):
        """Reuse one generator for three rewind_to(0) + reprefill cycles. Every cycle must reproduce
        the same output as the first -- no state (caches, draft carry-over, round buffers) leaks
        across a full rewind."""
        max_length = len(_PROMPT) + 16
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / "cp_rw0_stable", 4)
        gen = self._spec_gen(spec_path, 4, max_length)
        results = []
        for _ in range(3):
            gen.rewind_to(0)
            gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
            while not gen.is_done():
                gen.generate_next_token()
            results.append([int(t) for t in gen.get_sequence(0)])
        assert results[1] == results[0]
        assert results[2] == results[0]

    @pytest.mark.parametrize("k", [2, 4])
    def test_shared_buffer_mid_round_append_matches_rebuild(self, decoder_only_model_path, tmp_path, k):
        """Mid-generation append with the shared KV buffer (past_present_share_buffer=True), the
        configuration every graph-capture EP uses. The reconcile (PrepareForAppend) runs on the
        RewindTo-no-op cache, so this proves continuous decoding stays consistent with a clean
        rebuild on the GPU-style memory layout too."""
        extra = [785, 6722]
        max_length = len(_PROMPT) + 24
        spec_path = _build_self_spec(decoder_only_model_path, tmp_path / f"cp_sb_{k}", k)

        gen = self._spec_gen(spec_path, k, max_length, share_buffer=True)
        gen.append_tokens(np.array([_PROMPT], dtype=np.int32))
        for _ in range(3):
            if not gen.is_done():
                gen.generate_next_token()
        committed_before = [int(t) for t in gen.get_sequence(0)]
        gen.append_tokens(np.array([extra], dtype=np.int32))
        while not gen.is_done():
            gen.generate_next_token()
        full = [int(t) for t in gen.get_sequence(0)]
        assert full[:len(committed_before)] == committed_before

        ref = self._spec_gen(spec_path, k, max_length, share_buffer=True)
        ref.append_tokens(np.array([committed_before], dtype=np.int32))
        ref.append_tokens(np.array([extra], dtype=np.int32))
        while not ref.is_done():
            ref.generate_next_token()
        assert full == [int(t) for t in ref.get_sequence(0)]
