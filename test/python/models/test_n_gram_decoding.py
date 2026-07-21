# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""End-to-end tests for model-free n-gram speculative decoding."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


_PROMPT = [785, 3838, 374, 279, 6722, 315, 9625, 30]
_REPETITIVE_PROMPT = [785, 3838, 374, 785, 3838]


@pytest.fixture(scope="module")
def qwen3_model_path(request):
    test_models = request.config.getoption("--test_models")
    candidate = os.path.join(test_models, "qwen3-speculative", "qwen3-0.6b")
    if not os.path.exists(os.path.join(candidate, "genai_config.json")):
        pytest.skip("qwen3-0.6b test model is unavailable")
    return candidate


def _generator(model_path, prompt, max_length, *, ngram_size=0,
               max_draft_tokens=4, **search_options):
    model = og.Model(model_path)
    params = og.GeneratorParams(model)
    options = {"do_sample": False, "max_length": max_length}
    options.update(search_options)
    params.set_search_options(**options)
    if ngram_size:
        params.set_speculative_options(
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
        )
    generator = og.Generator(model, params)
    generator.append_tokens(np.array([prompt], dtype=np.int32))
    return generator


def _finish(generator):
    while not generator.is_done():
        generator.generate_next_token()
    return [int(token) for token in generator.get_sequence(0)]


def _generate(model_path, prompt, max_length, *, ngram_size=0,
              max_draft_tokens=4, **search_options):
    generator = _generator(
        model_path,
        prompt,
        max_length,
        ngram_size=ngram_size,
        max_draft_tokens=max_draft_tokens,
        **search_options,
    )
    sequence = _finish(generator)
    return sequence, generator.get_speculative_stats()


def _build_self_speculative_model(source_dir, dest_dir):
    source_dir = os.path.abspath(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(source_dir, "genai_config.json")) as config_file:
        config = json.load(config_file)

    decoder = copy.deepcopy(config["model"]["decoder"])
    model_path = os.path.join(source_dir, decoder["filename"])
    decoder["filename"] = os.path.relpath(model_path, os.fspath(dest_dir))
    config["model"]["type"] = "speculative"
    config["model"]["decoder"] = decoder
    config["model"]["draft"] = copy.deepcopy(decoder)
    with open(dest_dir / "genai_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    return os.fspath(dest_dir)


def _make_tiny_ngram_model(directory, logits_table, *, prune_logits=False,
                           decoder_overrides=None):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logits_table = np.asarray(logits_table, dtype=np.float32)
    vocab_size = logits_table.shape[0]
    assert logits_table.shape == (vocab_size, vocab_size)

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, ["batch", "sequence"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, ["batch", "total_sequence"])
    output_sequence = 1 if prune_logits else "sequence"
    logits = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["batch", output_sequence, vocab_size])
    nodes = [
        helper.make_node(
            "Gather", ["logits_table", "input_ids"], ["all_logits"], axis=0)
    ]
    initializers = [numpy_helper.from_array(logits_table, "logits_table")]
    if prune_logits:
        initializers.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_index"))
        nodes.append(helper.make_node(
            "Gather", ["all_logits", "last_index"], ["logits"], axis=1))
    else:
        nodes.append(helper.make_node("Identity", ["all_logits"], ["logits"]))

    graph = helper.make_graph(
        nodes, "tiny_ngram", [input_ids, attention_mask], [logits], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, directory / "model.onnx")

    decoder = {
        "session_options": {"provider_options": []},
        "filename": "model.onnx",
        "head_size": 1,
        "hidden_size": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_hidden_layers": 0,
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
    if decoder_overrides:
        decoder.update(decoder_overrides)
    config = {
        "model": {
            "type": "llama",
            "vocab_size": vocab_size,
            "context_length": 64,
            "bos_token_id": 0,
            "eos_token_id": [vocab_size - 1],
            "pad_token_id": 0,
            "decoder": decoder,
        },
        "search": {"max_length": 64},
    }
    with open(directory / "genai_config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)
    return os.fspath(directory)


def _transition_logits(transitions, vocab_size=10):
    table = np.full((vocab_size, vocab_size), -20.0, dtype=np.float32)
    for token in range(vocab_size):
        table[token, (token + 1) % (vocab_size - 1)] = 20.0
    for source, target in transitions.items():
        table[source, :] = -20.0
        table[source, target] = 20.0
    return table


def _assert_stats_consistent(stats):
    assert stats["rounds"] == (
        stats["completed_rounds"] +
        stats["interrupted_rounds"] +
        stats["active_rounds"]
    )
    assert stats["draft_tokens_accepted"] <= stats["draft_tokens_evaluated"]
    assert stats["draft_tokens_evaluated"] <= stats["draft_tokens_proposed"]
    assert stats["tokens_queued"] == (
        stats["tokens_emitted"] +
        stats["tokens_discarded"] +
        stats["tokens_buffered"]
    )
    assert 0.0 <= stats["acceptance_rate"] <= 1.0


def _invalid_guard_params(model, guard):
    params = og.GeneratorParams(model)
    search_options = {
        "do_sample": False,
        "max_length": len(_PROMPT) + 4,
    }
    if guard == "batch_size":
        search_options["batch_size"] = 2
    elif guard == "num_beams":
        search_options["num_beams"] = 2
    elif guard == "num_return_sequences":
        search_options["num_return_sequences"] = 2
    elif guard == "guidance":
        params.set_guidance("regex", r"[0-9]")
    else:
        raise ValueError(f"Unknown n-gram guard: {guard}")
    params.set_search_options(**search_options)
    params.set_speculative_options(ngram_size=3)
    return params


class TestNGramControlledSemantics:
    def test_all_proposals_accept_then_bonus_is_buffered(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "all_accept", _transition_logits({1: 2, 2: 1}))
        prompt = [1, 2, 1, 2]
        generator = _generator(
            model_path, prompt, len(prompt) + 3,
            ngram_size=3, max_draft_tokens=2)

        for expected_token in [1, 2, 1]:
            old_length = len(generator.get_sequence(0))
            generator.generate_next_token()
            assert len(generator.get_sequence(0)) == old_length + 1
            assert int(generator.get_sequence(0)[-1]) == expected_token

        stats = generator.get_speculative_stats()
        assert stats["rounds"] == 1
        assert stats["completed_rounds"] == 1
        assert stats["draft_tokens_proposed"] == 2
        assert stats["draft_tokens_evaluated"] == 2
        assert stats["draft_tokens_accepted"] == 2
        assert stats["bonus_tokens"] == 1
        assert stats["correction_tokens"] == 0
        assert stats["tokens_queued"] == 3
        assert stats["tokens_emitted"] == 3
        assert stats["target_forward_passes"] == 1
        assert stats["draft_forward_passes"] == 0
        _assert_stats_consistent(stats)

    def test_first_proposal_rejection_stops_verification(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "first_reject", _transition_logits({2: 5}))
        prompt = [1, 2, 3, 4, 1, 2]
        generator = _generator(
            model_path, prompt, len(prompt) + 4,
            ngram_size=3, max_draft_tokens=2)
        generator.generate_next_token()
        result = [int(token) for token in generator.get_sequence(0)]
        stats = generator.get_speculative_stats()

        assert result == prompt + [5]
        assert stats["draft_tokens_proposed"] == 2
        assert stats["draft_tokens_evaluated"] == 1
        assert stats["draft_tokens_accepted"] == 0
        assert stats["correction_tokens"] == 1
        assert stats["bonus_tokens"] == 0
        _assert_stats_consistent(stats)

    def test_partial_acceptance_commits_prefix_then_correction(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "partial_accept", _transition_logits({2: 3, 3: 5}))
        prompt = [1, 2, 3, 4, 1, 2]
        result, stats = _generate(
            model_path, prompt, len(prompt) + 2,
            ngram_size=3, max_draft_tokens=2)

        assert result == prompt + [3, 5]
        assert stats["draft_tokens_proposed"] == 2
        assert stats["draft_tokens_evaluated"] == 2
        assert stats["draft_tokens_accepted"] == 1
        assert stats["correction_tokens"] == 1
        assert stats["bonus_tokens"] == 0
        _assert_stats_consistent(stats)

    def test_max_length_discards_precomputed_bonus(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "max_length", _transition_logits({1: 2, 2: 1}))
        prompt = [1, 2, 1, 2]
        result, stats = _generate(
            model_path, prompt, len(prompt) + 1,
            ngram_size=3, max_draft_tokens=4)

        assert result == prompt + [1]
        assert stats["draft_tokens_proposed"] == 1
        assert stats["draft_tokens_accepted"] == 1
        assert stats["bonus_tokens"] == 1
        assert stats["tokens_queued"] == 2
        assert stats["tokens_emitted"] == 1
        assert stats["tokens_discarded"] == 1
        assert stats["interrupted_rounds"] == 1
        _assert_stats_consistent(stats)

    def test_eos_in_accepted_proposal_discards_remaining_buffer(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "accepted_eos", _transition_logits({2: 9, 9: 5}))
        prompt = [1, 2, 9, 4, 1, 2]
        result, stats = _generate(
            model_path, prompt, len(prompt) + 4,
            ngram_size=3, max_draft_tokens=2)

        assert result == prompt
        assert stats["draft_tokens_accepted"] == 1
        assert stats["correction_tokens"] == 1
        assert stats["tokens_queued"] == 2
        assert stats["tokens_emitted"] == 0
        assert stats["tokens_discarded"] == 2
        assert stats["interrupted_rounds"] == 1
        _assert_stats_consistent(stats)

    def test_k1_and_lookup_miss_match_standard_greedy(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "k1_and_miss", _transition_logits({1: 2, 2: 1}))
        repetitive = [1, 2, 1, 2]
        unique = [1, 3, 5]

        standard, _ = _generate(
            model_path, repetitive, len(repetitive) + 8)
        k1, k1_stats = _generate(
            model_path, repetitive, len(repetitive) + 8,
            ngram_size=3, max_draft_tokens=1)
        miss, miss_stats = _generate(
            model_path, unique, len(unique) + 1,
            ngram_size=4, max_draft_tokens=4)

        assert k1 == standard
        assert k1_stats["draft_tokens_proposed"] > 0
        assert miss == unique + [6]
        assert miss_stats["rounds"] == 0
        assert miss_stats["draft_forward_passes"] == 0

    def test_sampling_matches_standard_across_mixed_hits_and_misses(self, tmp_path):
        table = np.full((10, 10), -20.0, dtype=np.float32)
        table[:, 1:5] = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
        model_path = _make_tiny_ngram_model(tmp_path / "sampling", table)
        prompt = [1, 2, 1, 2]
        outputs = set()

        for seed in range(24):
            options = {
                "do_sample": True,
                "top_k": 4,
                "temperature": 0.8,
                "random_seed": seed,
            }
            standard, _ = _generate(
                model_path, prompt, len(prompt) + 10, **options)
            ngram, stats = _generate(
                model_path, prompt, len(prompt) + 10,
                ngram_size=3, max_draft_tokens=4, **options)
            assert ngram == standard
            assert stats["draft_forward_passes"] == 0
            outputs.add(tuple(ngram[len(prompt):]))

        assert len(outputs) > 1

    def test_mid_round_append_discards_pending_tokens_and_rebuilds(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "mid_round_append", _transition_logits({1: 2, 2: 1}))
        prompt = [1, 2, 1, 2]
        max_length = len(prompt) + 12
        generator = _generator(
            model_path, prompt, max_length,
            ngram_size=3, max_draft_tokens=4)
        generator.generate_next_token()
        assert generator.get_speculative_stats()["active_rounds"] == 1

        committed = [int(token) for token in generator.get_sequence(0)]
        appended = [7, 1, 2]
        generator.append_tokens(np.array([appended], dtype=np.int32))
        actual = _finish(generator)

        expected, _ = _generate(
            model_path, committed + appended, max_length,
            ngram_size=3, max_draft_tokens=4)
        stats = generator.get_speculative_stats()
        assert actual == expected
        assert stats["interrupted_rounds"] >= 1
        assert stats["active_rounds"] == 0
        assert stats["tokens_buffered"] == 0
        _assert_stats_consistent(stats)

    def test_mid_round_append_discards_unobserved_sampling_draws(self, tmp_path):
        table = _transition_logits({1: 2, 2: 1})
        table[7, :] = -20.0
        table[7, 3:5] = 0.0
        model_path = _make_tiny_ngram_model(
            tmp_path / "sampling_append", table)
        prompt = [1, 2, 1, 2]
        outputs = set()

        def run(seed, ngram_size):
            generator = _generator(
                model_path, prompt, len(prompt) + 8,
                ngram_size=ngram_size, max_draft_tokens=4,
                do_sample=True, top_k=2, random_seed=seed)
            generator.generate_next_token()
            assert int(generator.get_sequence(0)[-1]) == 1
            generator.append_tokens(np.array([[7]], dtype=np.int32))
            return _finish(generator)

        for seed in range(16):
            expected = run(seed, 0)
            actual = run(seed, 3)
            assert actual == expected
            outputs.add(tuple(actual[len(prompt) + 2:]))

        assert len(outputs) > 1

    @pytest.mark.parametrize("ngram_size", [0, 3])
    def test_interleaved_sampling_generators_have_independent_rng_streams(
            self, tmp_path, ngram_size):
        table = np.full((10, 10), -20.0, dtype=np.float32)
        table[:, 1:5] = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
        model_path = _make_tiny_ngram_model(
            tmp_path / f"interleaved_{ngram_size}", table)
        prompt = [1, 2, 1, 2]
        max_length = len(prompt) + 12
        options = {
            "do_sample": True,
            "top_k": 4,
            "temperature": 0.8,
            "random_seed": 2468,
        }
        expected, _ = _generate(
            model_path, prompt, max_length, ngram_size=ngram_size,
            **options)

        first = _generator(
            model_path, prompt, max_length, ngram_size=ngram_size,
            **options)
        second = _generator(
            model_path, prompt, max_length, ngram_size=ngram_size,
            **options)
        while not first.is_done() or not second.is_done():
            if not first.is_done():
                first.generate_next_token()
            if not second.is_done():
                second.generate_next_token()

        assert [int(token) for token in first.get_sequence(0)] == expected
        assert [int(token) for token in second.get_sequence(0)] == expected

    def test_get_logits_mid_round_is_side_effect_free(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "get_logits", _transition_logits({1: 2, 2: 1}))
        prompt = [1, 2, 1, 2]
        max_length = len(prompt) + 10

        baseline = _generator(
            model_path, prompt, max_length,
            ngram_size=3, max_draft_tokens=4)
        baseline.generate_next_token()
        expected = _finish(baseline)

        inspected = _generator(
            model_path, prompt, max_length,
            ngram_size=3, max_draft_tokens=4)
        inspected.generate_next_token()
        first = inspected.get_logits().copy()
        second = inspected.get_logits().copy()
        np.testing.assert_array_equal(first, second)
        assert _finish(inspected) == expected

    def test_set_logits_mid_round_forces_token_and_reanchors(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "set_logits", _transition_logits({1: 2, 2: 1}))
        prompt = [1, 2, 1, 2]
        max_length = len(prompt) + 12
        generator = _generator(
            model_path, prompt, max_length,
            ngram_size=3, max_draft_tokens=4)
        generator.generate_next_token()
        assert generator.get_speculative_stats()["active_rounds"] == 1

        forced_token = 7
        forced_logits = np.full_like(generator.get_logits(), -1e9)
        forced_logits[0, 0, forced_token] = 1e9
        generator.set_logits(forced_logits)
        generator.generate_next_token()
        assert int(generator.get_sequence(0)[-1]) == forced_token

        committed = [int(token) for token in generator.get_sequence(0)]
        actual = _finish(generator)
        expected, _ = _generate(
            model_path, committed, max_length,
            ngram_size=3, max_draft_tokens=4)
        assert actual == expected


class TestNGramDecoding:
    def test_options_round_trip_and_validation(self, qwen3_model_path):
        model = og.Model(qwen3_model_path)
        params = og.GeneratorParams(model)
        params.set_speculative_options(ngram_size=4, max_draft_tokens=8)
        options = params.get_speculative_options()
        assert options["ngram_size"] == 4
        assert options["max_draft_tokens"] == 8

        with pytest.raises(Exception, match="ngram_size"):
            params.set_speculative_options(ngram_size=1)
        with pytest.raises(Exception, match="ngram_size"):
            params.set_speculative_options(ngram_size=17)
        with pytest.raises(Exception, match="max_draft_tokens"):
            params.set_speculative_options(max_draft_tokens=0)
        with pytest.raises(Exception, match="max_draft_tokens"):
            params.set_speculative_options(max_draft_tokens=17)

    def test_repetitive_prompt_matches_standard_greedy(self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 12
        expected, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length)
        actual, stats = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
        )

        assert actual == expected
        assert stats["rounds"] > 0
        assert stats["draft_tokens_proposed"] > 0
        assert stats["draft_forward_passes"] == 0
        assert stats["draft_tokens_accepted"] <= stats["draft_tokens_evaluated"]
        assert stats["draft_tokens_evaluated"] <= stats["draft_tokens_proposed"]

    def test_lookup_miss_uses_standard_greedy_path(self, qwen3_model_path):
        max_length = len(_PROMPT) + 4
        expected, _ = _generate(qwen3_model_path, _PROMPT, max_length)
        actual, stats = _generate(
            qwen3_model_path,
            _PROMPT,
            max_length,
            ngram_size=16,
        )

        assert actual == expected
        assert stats["rounds"] == 0
        assert stats["draft_forward_passes"] == 0

    @pytest.mark.parametrize(
        "search_options",
        [
            {"repetition_penalty": 1.1},
            {"min_length": len(_REPETITIVE_PROMPT) + 8},
        ],
    )
    def test_logits_penalties_match_standard_greedy(
            self, qwen3_model_path, search_options):
        max_length = len(_REPETITIVE_PROMPT) + 10
        expected, _ = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            **search_options,
        )
        actual, _ = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            **search_options,
        )
        assert actual == expected

    def test_rewind_and_resume_matches_clean_run(self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 10
        expected, _ = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
        )

        generator = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
        )
        for _ in range(3):
            generator.generate_next_token()
        generator.rewind_to(0)
        generator.append_tokens(
            np.array([_REPETITIVE_PROMPT], dtype=np.int32))

        assert _finish(generator) == expected

    def test_k1_nonzero_rewind_matches_uninterrupted_generation(
            self, qwen3_model_path):
        """K=1 resumes the same sequential greedy trajectory after a nonzero rewind."""
        max_length = len(_REPETITIVE_PROMPT) + 18
        expected, _ = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=1,
        )
        generator = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=1,
        )
        for _ in range(5):
            generator.generate_next_token()

        generator.rewind_to(len(_REPETITIVE_PROMPT) + 2)

        assert _finish(generator) == expected
        assert generator.get_speculative_stats()["draft_forward_passes"] == 0

    @pytest.mark.parametrize("max_draft_tokens", [1, 2, 4, 8])
    @pytest.mark.parametrize("rewind_offset", [0, 1, 3])
    def test_nonzero_direct_rewind_matches_boundary_reappend(
            self, qwen3_model_path, max_draft_tokens, rewind_offset):
        """Direct nonzero rewind must match explicitly re-appending its boundary token."""
        max_length = len(_REPETITIVE_PROMPT) + 18
        rewind_length = len(_REPETITIVE_PROMPT) + rewind_offset
        generated_tokens = rewind_offset + 3

        direct = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
        )
        for _ in range(generated_tokens):
            direct.generate_next_token()
        boundary_token = int(direct.get_sequence(0)[rewind_length])
        direct.rewind_to(rewind_length)
        direct_sequence = _finish(direct)

        reappended = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
        )
        for _ in range(generated_tokens):
            reappended.generate_next_token()
        reappended.rewind_to(rewind_length)
        reappended.append_tokens(
            np.array([[boundary_token]], dtype=np.int32))
        reappended_sequence = _finish(reappended)

        assert direct_sequence == reappended_sequence
        assert direct.get_speculative_stats()["draft_forward_passes"] == 0
        assert reappended.get_speculative_stats()["draft_forward_passes"] == 0

    @pytest.mark.parametrize("max_draft_tokens", [2, 4])
    def test_sampling_nonzero_rewind_matches_boundary_reappend(
            self, qwen3_model_path, max_draft_tokens):
        """Sampling preserves its canonical RNG stream across equivalent rewind flows."""
        max_length = len(_REPETITIVE_PROMPT) + 18
        rewind_length = len(_REPETITIVE_PROMPT) + 2
        search_options = {
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.7,
            "random_seed": 2468,
        }

        direct = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
            **search_options,
        )
        for _ in range(5):
            direct.generate_next_token()
        boundary_token = int(direct.get_sequence(0)[rewind_length])
        direct.rewind_to(rewind_length)
        direct_sequence = _finish(direct)

        reappended = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
            **search_options,
        )
        for _ in range(5):
            reappended.generate_next_token()
        reappended.rewind_to(rewind_length)
        reappended.append_tokens(
            np.array([[boundary_token]], dtype=np.int32))
        reappended_sequence = _finish(reappended)

        assert direct_sequence == reappended_sequence

    @pytest.mark.parametrize("max_draft_tokens", [1, 4])
    def test_get_logits_after_nonzero_rewind_matches_boundary_reappend(
            self, qwen3_model_path, max_draft_tokens):
        """Post-rewind logits and subsequent generation match explicit boundary replay."""
        max_length = len(_REPETITIVE_PROMPT) + 18
        rewind_length = len(_REPETITIVE_PROMPT) + 1

        direct = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
        )
        for _ in range(4):
            direct.generate_next_token()
        boundary_token = int(direct.get_sequence(0)[rewind_length])
        direct.rewind_to(rewind_length)
        direct_logits = direct.get_logits().copy()

        reappended = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=max_draft_tokens,
        )
        for _ in range(4):
            reappended.generate_next_token()
        reappended.rewind_to(rewind_length)
        reappended.append_tokens(
            np.array([[boundary_token]], dtype=np.int32))
        reappended_logits = reappended.get_logits().copy()

        np.testing.assert_array_equal(direct_logits, reappended_logits)
        assert _finish(direct) == _finish(reappended)

    def test_nonzero_rewind_discards_mid_round_buffer_and_resumes(
            self, tmp_path):
        """Rewind invalidates buffered proposal tokens and rebuilds lookup state."""
        model_path = _make_tiny_ngram_model(
            tmp_path / "rewind_mid_round",
            _transition_logits({1: 2, 2: 1}),
        )
        prompt = [1, 2, 1, 2]
        max_length = len(prompt) + 10
        generator = _generator(
            model_path,
            prompt,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
        )
        generator.generate_next_token()
        before = generator.get_speculative_stats()
        assert before["active_rounds"] == 1
        assert before["tokens_buffered"] > 0

        generator.rewind_to(len(prompt) - 1)
        after = generator.get_speculative_stats()

        assert after["active_rounds"] == 0
        assert after["interrupted_rounds"] == before["interrupted_rounds"] + 1
        assert after["tokens_buffered"] == 0
        assert after["tokens_discarded"] == (
            before["tokens_discarded"] + before["tokens_buffered"])

        expected, _ = _generate(
            model_path,
            prompt,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
        )
        assert _finish(generator) == expected
        _assert_stats_consistent(generator.get_speculative_stats())

    def test_nonzero_rewind_with_shared_kv_buffer_matches_reappend(
            self, qwen3_model_path):
        """The shared KV-buffer path satisfies the same nonzero rewind contract."""
        max_length = len(_REPETITIVE_PROMPT) + 18
        rewind_length = len(_REPETITIVE_PROMPT) + 2
        search_options = {"past_present_share_buffer": True}

        direct = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
            **search_options,
        )
        for _ in range(5):
            direct.generate_next_token()
        boundary_token = int(direct.get_sequence(0)[rewind_length])
        direct.rewind_to(rewind_length)
        direct_sequence = _finish(direct)

        reappended = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
            **search_options,
        )
        for _ in range(5):
            reappended.generate_next_token()
        reappended.rewind_to(rewind_length)
        reappended.append_tokens(
            np.array([[boundary_token]], dtype=np.int32))

        assert direct_sequence == _finish(reappended)

    @pytest.mark.parametrize("max_draft_tokens", [2, 4])
    def test_repeated_rewind_append_cycles_are_deterministic(
            self, qwen3_model_path, max_draft_tokens):
        """Repeated nonzero rewind/resume and append cycles must be deterministic."""
        max_length = len(_REPETITIVE_PROMPT) + 32
        first_append = [785, 6722]
        second_append = [3838, 374]

        def run():
            generator = _generator(
                qwen3_model_path,
                _REPETITIVE_PROMPT,
                max_length,
                ngram_size=3,
                max_draft_tokens=max_draft_tokens,
            )
            for _ in range(5):
                generator.generate_next_token()

            generator.rewind_to(len(_REPETITIVE_PROMPT) + 2)
            for _ in range(2):
                generator.generate_next_token()
            first_append_start = len(generator.get_sequence(0))
            generator.append_tokens(
                np.array([first_append], dtype=np.int32))
            assert [
                int(token)
                for token in generator.get_sequence(0)[first_append_start:]
            ] == first_append

            for _ in range(3):
                generator.generate_next_token()
            generator.rewind_to(len(generator.get_sequence(0)) - 2)
            for _ in range(2):
                generator.generate_next_token()
            second_append_start = len(generator.get_sequence(0))
            generator.append_tokens(
                np.array([second_append], dtype=np.int32))
            assert [
                int(token)
                for token in generator.get_sequence(0)[second_append_start:]
            ] == second_append

            actual = _finish(generator)
            stats = generator.get_speculative_stats()

            assert stats["rounds"] > 0
            assert stats["draft_forward_passes"] == 0
            _assert_stats_consistent(stats)
            return actual

        assert run() == run()

    def test_repeated_full_rewind_and_reprefill_is_stable(
            self, qwen3_model_path):
        """Full reset and reprefill can reuse one n-gram generator without state leakage."""
        max_length = len(_REPETITIVE_PROMPT) + 16
        generator = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
        )
        results = [_finish(generator)]

        for _ in range(2):
            generator.rewind_to(0)
            generator.append_tokens(
                np.array([_REPETITIVE_PROMPT], dtype=np.int32))
            results.append(_finish(generator))

        assert results[1] == results[0]
        assert results[2] == results[0]
        assert generator.get_speculative_stats()["draft_forward_passes"] == 0

    def test_mid_generation_append_preserves_committed_prefix(
            self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 14
        generator = _generator(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
        )
        generator.generate_next_token()
        committed = [int(token) for token in generator.get_sequence(0)]
        appended = [785, 3838]
        generator.append_tokens(np.array([appended], dtype=np.int32))
        result = _finish(generator)

        assert result[:len(committed)] == committed
        assert result[len(committed):len(committed) + len(appended)] == appended

    def test_sampling_lookup_miss_uses_canonical_rng(self, qwen3_model_path):
        max_length = len(_PROMPT) + 4
        search_options = {
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.8,
            "temperature": 0.7,
            "random_seed": 42,
        }
        expected, _ = _generate(
            qwen3_model_path, _PROMPT, max_length, **search_options)
        actual, stats = _generate(
            qwen3_model_path,
            _PROMPT,
            max_length,
            ngram_size=16,
            **search_options,
        )

        assert actual == expected
        assert stats["rounds"] == 0
        assert stats["draft_forward_passes"] == 0

    @pytest.mark.parametrize(
        "search_options",
        [
            {"top_k": 0, "top_p": 0.0, "temperature": 1.0},
            {"top_k": 20, "temperature": 0.6},
            {"top_k": 0, "top_p": 0.8, "temperature": 0.9},
            {"top_k": 20, "top_p": 0.8, "temperature": 0.7,
             "repetition_penalty": 1.1},
            {"top_k": 20, "temperature": 0.7,
             "min_length": len(_REPETITIVE_PROMPT) + 8},
        ],
    )
    def test_sampling_matches_standard_with_fixed_seed(
            self, qwen3_model_path, search_options):
        max_length = len(_REPETITIVE_PROMPT) + 10
        options = {
            "do_sample": True,
            "random_seed": 1234,
            **search_options,
        }
        expected, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length, **options)
        actual, stats = _generate(
            qwen3_model_path,
            _REPETITIVE_PROMPT,
            max_length,
            ngram_size=3,
            max_draft_tokens=4,
            **options,
        )

        assert actual == expected
        assert stats["rounds"] > 0
        assert stats["draft_tokens_proposed"] > 0
        assert stats["draft_forward_passes"] == 0

    @pytest.mark.parametrize(
        "search_options",
        [
            {"do_sample": True, "top_k": 1, "random_seed": 9},
            {"do_sample": True, "top_k": 20, "temperature": 0.0,
             "random_seed": 9},
        ],
    )
    def test_degenerate_sampling_settings_use_greedy_semantics(
            self, qwen3_model_path, search_options):
        max_length = len(_REPETITIVE_PROMPT) + 10
        expected, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            **search_options)
        actual, stats = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, max_draft_tokens=4, **search_options)

        assert actual == expected
        assert stats["rounds"] > 0
        assert stats["draft_forward_passes"] == 0

    def test_sampling_is_deterministic_for_fixed_seed(self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 10
        options = {
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.7,
            "random_seed": 5678,
        }

        first, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, **options)
        second, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, **options)

        assert first == second

    @pytest.mark.parametrize(
        ("ngram_size", "max_draft_tokens", "prompt"),
        [
            (2, 1, _REPETITIVE_PROMPT),
            (16, 16, [785, 3838] * 16),
        ],
    )
    def test_supported_option_boundaries_match_standard(
            self, qwen3_model_path, ngram_size, max_draft_tokens, prompt):
        max_length = len(prompt) + 8
        expected, _ = _generate(qwen3_model_path, prompt, max_length)
        actual, stats = _generate(
            qwen3_model_path, prompt, max_length,
            ngram_size=ngram_size, max_draft_tokens=max_draft_tokens)

        assert actual == expected
        assert stats["rounds"] > 0
        assert stats["draft_forward_passes"] == 0
        _assert_stats_consistent(stats)

    @pytest.mark.parametrize(
        "search_options",
        [
            {"past_present_share_buffer": True},
            {"chunk_size": 2},
            {"past_present_share_buffer": True, "chunk_size": 2},
        ],
    )
    def test_kv_buffer_and_chunked_prefill_match_standard(
            self, qwen3_model_path, search_options):
        max_length = len(_REPETITIVE_PROMPT) + 12
        expected, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            **search_options)
        actual, stats = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, max_draft_tokens=4, **search_options)

        assert actual == expected
        assert stats["rounds"] > 0
        assert stats["draft_forward_passes"] == 0
        _assert_stats_consistent(stats)

    def test_long_run_has_one_visible_token_per_call_and_consistent_stats(
            self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 40
        generator = _generator(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, max_draft_tokens=8)

        while not generator.is_done():
            old_length = len(generator.get_sequence(0))
            generator.generate_next_token()
            new_length = len(generator.get_sequence(0))
            assert new_length - old_length in (0, 1)
            if new_length == old_length:
                assert generator.is_done()

        actual = [int(token) for token in generator.get_sequence(0)]
        stats = generator.get_speculative_stats()
        assert actual[:len(_REPETITIVE_PROMPT)] == _REPETITIVE_PROMPT
        assert len(_REPETITIVE_PROMPT) < len(actual) <= max_length
        assert stats["rounds"] > 1
        assert stats["active_rounds"] == 0
        assert stats["draft_forward_passes"] == 0
        assert stats["correction_tokens"] + stats["bonus_tokens"] == stats["rounds"]
        _assert_stats_consistent(stats)

    def test_get_logits_does_not_change_sampling_or_logical_stats(
            self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 16
        options = {
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.7,
            "random_seed": 4321,
        }

        def run(inspect_logits):
            generator = _generator(
                qwen3_model_path, _REPETITIVE_PROMPT, max_length,
                ngram_size=3, max_draft_tokens=4, **options)
            generator.generate_next_token()
            if inspect_logits:
                first = generator.get_logits().copy()
                second = generator.get_logits().copy()
                np.testing.assert_array_equal(first, second)
            sequence = _finish(generator)
            stats = generator.get_speculative_stats()
            counters = {
                key: stats[key]
                for key in (
                    "rounds", "completed_rounds", "interrupted_rounds",
                    "active_rounds", "draft_tokens_proposed",
                    "draft_tokens_evaluated", "draft_tokens_accepted",
                    "correction_tokens", "bonus_tokens", "tokens_queued",
                    "tokens_emitted", "tokens_discarded", "tokens_buffered",
                    "draft_forward_passes",
                )
            }
            return sequence, counters

        assert run(True) == run(False)

    def test_set_logits_reanchors_before_continuing(self, qwen3_model_path):
        max_length = len(_REPETITIVE_PROMPT) + 16
        generator = _generator(
            qwen3_model_path, _REPETITIVE_PROMPT, max_length,
            ngram_size=3, max_draft_tokens=4)
        generator.generate_next_token()
        forced_token = _REPETITIVE_PROMPT[0]
        forced_logits = np.full_like(generator.get_logits(), -1e9)
        forced_logits[0, 0, forced_token] = 1e9
        generator.set_logits(forced_logits)
        generator.generate_next_token()
        assert int(generator.get_sequence(0)[-1]) == forced_token

        committed = [int(token) for token in generator.get_sequence(0)]
        actual = _finish(generator)
        expected, _ = _generate(
            qwen3_model_path, committed, max_length,
            ngram_size=3, max_draft_tokens=4)
        assert actual == expected

    def test_interleaved_generators_do_not_share_lookup_state(
            self, qwen3_model_path):
        model = og.Model(qwen3_model_path)

        def make_generator(prompt):
            params = og.GeneratorParams(model)
            params.set_search_options(
                do_sample=False, max_length=len(prompt) + 10)
            params.set_speculative_options(
                ngram_size=3, max_draft_tokens=4)
            generator = og.Generator(model, params)
            generator.append_tokens(np.array([prompt], dtype=np.int32))
            return generator

        first = make_generator(_REPETITIVE_PROMPT)
        second_prompt = [785, 3838, 785, 3838]
        second = make_generator(second_prompt)
        while not first.is_done() or not second.is_done():
            if not first.is_done():
                first.generate_next_token()
            if not second.is_done():
                second.generate_next_token()

        first_expected, _ = _generate(
            qwen3_model_path, _REPETITIVE_PROMPT,
            len(_REPETITIVE_PROMPT) + 10)
        second_expected, _ = _generate(
            qwen3_model_path, second_prompt, len(second_prompt) + 10)
        assert [int(token) for token in first.get_sequence(0)] == first_expected
        assert [int(token) for token in second.get_sequence(0)] == second_expected


class TestNGramCapabilityGuards:
    def test_pruned_logits_model_is_rejected_before_state_creation(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "pruned_logits",
            _transition_logits({1: 2, 2: 1}),
            prune_logits=True)
        model = og.Model(model_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=12)
        params.set_speculative_options(ngram_size=3)

        with pytest.raises(Exception, match="pruned last-token-only logits"):
            og.Generator(model, params)

    def test_sliding_kv_cache_model_is_rejected_before_state_creation(self, tmp_path):
        model_path = _make_tiny_ngram_model(
            tmp_path / "sliding_kv",
            _transition_logits({1: 2, 2: 1}),
            decoder_overrides={
                "sliding_window": {
                    "window_size": 8,
                    "slide_key_value_cache": True,
                },
            })
        model = og.Model(model_path)
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=12)
        params.set_speculative_options(ngram_size=3)

        with pytest.raises(Exception, match="rewindable KV cache"):
            og.Generator(model, params)

    @pytest.mark.parametrize(
        ("guard", "message"),
        [
            ("batch_size", "batch_size"),
            ("num_beams", "beam search"),
            ("num_return_sequences", "num_return_sequences"),
            ("guidance", "guidance"),
        ],
    )
    def test_unsupported_generator_config_fails_fast(
            self, qwen3_model_path, guard, message):
        model = og.Model(qwen3_model_path)
        with pytest.raises(Exception, match=message):
            og.Generator(model, _invalid_guard_params(model, guard))

    @pytest.mark.parametrize(
        "guard",
        ["batch_size", "num_beams", "num_return_sequences", "guidance"],
    )
    def test_guard_failure_does_not_corrupt_model(
            self, qwen3_model_path, guard):
        model = og.Model(qwen3_model_path)
        with pytest.raises(Exception):
            og.Generator(model, _invalid_guard_params(model, guard))

        valid = og.GeneratorParams(model)
        valid.set_search_options(
            do_sample=False,
            max_length=len(_PROMPT) + 4,
        )
        valid.set_speculative_options(ngram_size=3)
        generator = og.Generator(model, valid)
        generator.append_tokens(np.array([_PROMPT], dtype=np.int32))
        assert len(_finish(generator)) > len(_PROMPT)

    def test_draft_model_combination_is_rejected(
            self, qwen3_model_path, tmp_path):
        model_path = _build_self_speculative_model(
            qwen3_model_path, tmp_path / "ngram_with_draft")
        model = og.Model(model_path)
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=len(_PROMPT) + 4,
        )
        params.set_speculative_options(ngram_size=3)

        with pytest.raises(Exception, match="cannot be combined"):
            og.Generator(model, params)
