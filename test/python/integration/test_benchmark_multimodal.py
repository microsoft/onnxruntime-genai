# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from models.test_gemma4_models import _fixed_output_gemma4, _get_test_media_path, _to_numpy


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[3] / "benchmark" / "python" / "benchmark_multimodal.py"
    spec = importlib.util.spec_from_file_location("benchmark_multimodal_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
@pytest.mark.parametrize("image_count", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("template_error", ["Invalid or unsupported chat template", "Empty chat template."])
def test_gemma4_benchmark_prompt_with_unsupported_template(benchmark, model_type, image_count, template_error):
    tokenizer = SimpleNamespace(apply_chat_template=Mock(side_effect=RuntimeError(template_error)))
    prompt = benchmark.build_prompt(model_type, image_count, False, tokenizer)
    content = "<|image|>" * image_count + "Describe these images." if image_count else "What is the meaning of life?"
    assert prompt == f"<bos><|turn>user\n{content}<turn|>\n<|turn>model\n"
    assert prompt.count("<|image|>") == image_count
    assert "<|image_1|>" not in prompt
    tokenizer.apply_chat_template.assert_called_once_with(
        json.dumps([{"role": "user", "content": content}]), add_generation_prompt=True
    )


def test_gemma4_benchmark_uses_model_chat_template(benchmark):
    tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value="model-formatted prompt"))
    assert benchmark.build_prompt("gemma4", 2, False, tokenizer) == "model-formatted prompt"
    tokenizer.apply_chat_template.assert_called_once_with(
        json.dumps([{"role": "user", "content": "<|image|><|image|>Describe these images."}]),
        add_generation_prompt=True,
    )


def test_gemma4_benchmark_does_not_hide_tokenizer_errors(benchmark):
    tokenizer = SimpleNamespace(apply_chat_template=Mock(side_effect=RuntimeError("Tokenizer failure")))
    with pytest.raises(RuntimeError, match="Tokenizer failure"):
        benchmark.build_prompt("gemma4", 1, False, tokenizer)


@pytest.mark.parametrize("model_type", ["phi3v", "phi4mm"])
def test_multimodal_benchmark_preserves_image_and_audio_markers(benchmark, model_type):
    prompt = benchmark.build_prompt(model_type, 2, True)
    if model_type == "phi4mm":
        assert prompt.count("<|endoftext10|>") == 2
        assert "<|endoftext11|>" in prompt
    else:
        assert "<|image_1|>" in prompt and "<|image_2|>" in prompt
        assert "<|audio_1|>" in prompt


@pytest.mark.parametrize("multi_image", [False, True])
def test_multimodal_benchmark_runs_gemma4_and_checks_baseline(benchmark, request, tmp_path, multi_image):
    data = request.config.getoption("--test_models")
    if not data:
        pytest.skip("--test_models is required")
    model_path = tmp_path / "model"
    model, processor = _fixed_output_gemma4(data, model_path)
    names = ["australia.jpg", "sheet.png"] if multi_image else ["australia.jpg"]
    paths = [str(_get_test_media_path(data, Path("images") / name)) for name in names]
    prompt = benchmark.build_prompt("gemma4", len(paths), False, benchmark.og.Tokenizer(model))
    inputs = processor(prompt, images=benchmark.og.Images.open(*paths))
    expected_prompt_length = _to_numpy(inputs["input_ids"]).shape[-1]
    args = SimpleNamespace(
        input_folder=str(model_path),
        image_paths=paths if multi_image else [],
        image_path="" if multi_image else paths[0],
        audio_path="",
        generation_lengths=[1, 3],
        max_lengths=[4096, 4096],
        repetitions=2,
        warmup=1,
        top_k=1,
        top_p=1.0,
        verbose=False,
        print_model_output=False,
        output=str(tmp_path / "baseline.csv"),
        baseline_csv=None,
        chunk_size=None,
    )
    benchmark.main(args)
    baseline = benchmark.pd.read_csv(args.output)
    assert baseline["Prompt Length"].tolist() == [expected_prompt_length] * 2
    assert baseline["Image Count"].tolist() == [len(paths)] * 2
    assert baseline.loc[0, "Tokens Generated"] == 1
    assert benchmark.pd.isna(baseline.loc[0, "Token Generation Latency (ms)"])
    assert (baseline["TTFT Median (ms)"] > 0).all()
    assert (baseline["TTFT P95 (ms)"] >= baseline["TTFT Median (ms)"]).all()
    args.baseline_csv = args.output
    args.output = str(tmp_path / "candidate.csv")
    benchmark.main(args)
    candidate = benchmark.pd.read_csv(args.output)
    assert (candidate["TTFT Median Speedup"] > 0).all()

    baseline["Output Tokens SHA256"] = "different"
    baseline.to_csv(args.baseline_csv, index=False)
    rows = candidate.drop(columns="TTFT Median Speedup").values.tolist()
    with pytest.raises(ValueError, match="generated tokens differ"):
        benchmark.save_results(rows, str(tmp_path / "mismatch.csv"), args.baseline_csv)
