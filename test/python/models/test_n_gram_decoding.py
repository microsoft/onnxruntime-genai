# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""End-to-end tests for model-free n-gram speculative decoding."""

from __future__ import annotations

import os

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
    params.set_search_options(
        do_sample=False,
        max_length=max_length,
        **search_options,
    )
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

    def test_sampling_is_rejected(self, qwen3_model_path):
        model = og.Model(qwen3_model_path)
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=True,
            top_k=20,
            temperature=0.6,
            max_length=len(_PROMPT) + 4,
        )
        params.set_speculative_options(ngram_size=3)
        with pytest.raises(Exception, match="greedy"):
            og.Generator(model, params)
