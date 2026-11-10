# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from models.test_gemma4_models import _fixed_output_gemma4, _get_test_media_path


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmark" / "python" / "benchmark_multimodal.py"
    spec = importlib.util.spec_from_file_location("benchmark_multimodal_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("image_count", [1, 2, 4, 8])
def test_gemma4_benchmark_prompt_has_one_marker_per_image(benchmark, image_count):
    prompt = benchmark.build_prompt("gemma4", image_count, False)
    assert prompt.count("<|image|>") == image_count
    assert "<|image_1|>" not in prompt
    assert prompt.endswith("<start_of_turn>model\n")


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
    _fixed_output_gemma4(data, model_path)
    names = ["australia.jpg", "sheet.png"] if multi_image else ["australia.jpg"]
    paths = [str(_get_test_media_path(data, Path("images") / name)) for name in names]
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
    )
    benchmark.main(args)
    baseline = benchmark.pd.read_csv(args.output)
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
