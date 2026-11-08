# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import _test_utils
import pytest


@pytest.fixture
def runner(monkeypatch):
    api = SimpleNamespace(Model=Mock(), Tokenizer=Mock(), GeneratorParams=Mock(), Generator=Mock())
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", api)
    monkeypatch.setattr(_test_utils, "register_plugin_providers", Mock())
    path = Path(__file__).with_name("test_onnxruntime_genai_e2e.py")
    spec = importlib.util.spec_from_file_location("e2e_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("longrope", [False, True])
def test_runner_generates_every_prompt_and_preserves_supported_batching(runner, longrope):
    generators = []
    for _ in range(3 if longrope else 1):
        generator = Mock()
        generator.is_done.side_effect = [False, True]
        generators.append(generator)
    attempts = list(generators)
    if longrope:
        attempts.insert(0, RuntimeError("LongRoPE context selection currently requires batch_size=1 and num_beams=1"))
    runner.og.Generator.side_effect = attempts

    runner.run_model("model")

    assert runner.og.GeneratorParams.return_value.set_search_options.call_args_list == (
        [call(batch_size=3, max_length=200)] + [call(batch_size=1, max_length=200)] * 3
        if longrope
        else [call(batch_size=3, max_length=200)]
    )
    prompts = ["def is_prime(n):", "def compute_gcd(x, y):", "def binary_search(arr, x):"]
    tokenizer = runner.og.Tokenizer.return_value
    assert tokenizer.encode_batch.call_args_list == (
        [call([prompt]) for prompt in prompts] if longrope else [call(prompts)]
    )
    for generator in generators:
        generator.append_tokens.assert_called_once_with(tokenizer.encode_batch.return_value)
        generator.generate_next_token.assert_called_once()
        assert generator.get_sequence.call_args_list == [call(i) for i in range(1 if longrope else 3)]


def test_runner_does_not_retry_unrelated_errors(runner):
    runner.og.Generator.side_effect = RuntimeError("invalid input")
    with pytest.raises(RuntimeError, match="invalid input"):
        runner.run_model("model")
    runner.og.Generator.assert_called_once()


@pytest.mark.parametrize("graph_capture", [False, True])
def test_runner_reports_model_failures_after_running_remaining_models(runner, monkeypatch, graph_capture):
    models = ["broken-model", "working-model"]
    monkeypatch.setattr(
        runner,
        "get_args",
        lambda: SimpleNamespace(
            models=[] if graph_capture else models, graph_capture_models=models if graph_capture else []
        ),
    )
    run = Mock(side_effect=[RuntimeError("inference failed"), None])
    monkeypatch.setattr(runner, "run_graph_capture_model" if graph_capture else "run_model", run)
    for name in ("run_whisper", "run_nemotron_speech", "run_parakeet_tdt", "run_tool_calling"):
        monkeypatch.setattr(runner, name, Mock())
    with pytest.raises(RuntimeError, match=r"Failed to run models.*broken-model"):
        runner.main()
    assert run.call_args_list == [call(path) for path in models]
