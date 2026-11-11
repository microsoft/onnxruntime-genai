# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

import onnxruntime_genai as og
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "python"
MODEL = Path(__file__).resolve().parents[1] / "models" / "hf-internal-testing" / "tiny-random-gpt2-fp32"


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("common", EXAMPLES / "common.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "common", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [0, -1, -(2**31)])
def test_nonpositive_max_length_before_config(common, tmp_path, value):
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        common.get_config(str(tmp_path / "missing"), "cpu", None, search_options={"max_length": value})


@pytest.mark.parametrize("name", ["batch_size", "num_beams"])
@pytest.mark.parametrize("value", [0, -1, 33, 2**31 - 1])
def test_invalid_dimensions_before_config(common, tmp_path, name, value):
    options = {"batch_size": 2, "num_beams": 2, name: value}
    with pytest.raises(ValueError, match=rf"{name} .*must be in \[1, 32\]"):
        common.get_config(str(tmp_path / "missing"), "cpu", None, search_options=options)


@pytest.mark.parametrize("name", ["batch_size", "num_beams"])
@pytest.mark.parametrize("value", [1, 32])
def test_dimension_boundaries(common, name, value):
    common.get_config(str(MODEL), "cpu", None, search_options={name: value})


@pytest.mark.parametrize("value", [513, 2**31 - 1])
def test_max_length_exceeds_context(common, value):
    with pytest.raises(ValueError, match="cannot be greater than model context_length"):
        common.get_config(str(MODEL), "cpu", None, search_options={"max_length": value})


@pytest.mark.parametrize("flag,limit", [("-l", 16), ("--max_length", 32), (None, 512)])
def test_limits_reach_model_and_generation(common, flag, limit):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    common.get_generator_params_args(parser)
    options = common.get_search_options(parser.parse_args([flag, str(limit)] if flag else []))
    config = common.get_config(str(MODEL), "cpu", None, search_options=options)
    model = og.Model(config)
    params = og.GeneratorParams(model)
    # Check the model's overlay before applying generator-time options.
    assert params.get_search_options()["max_length"] == limit
    params.set_search_options(**options, min_length=limit)
    generator = og.Generator(model, params)
    generator.append_tokens([1, 2, 3, 4])
    while not generator.is_done():
        generator.generate_next_token()
    assert len(generator.get_sequence(0)) == limit


def test_none_options_are_omitted(common):
    assert common.get_search_options(argparse.Namespace(max_length=None, min_length=None)) == {"batch_size": 1}


@pytest.mark.parametrize("script", ["model-qa.py", "model-chat.py", "model-mm.py", "model-generate.py"])
def test_entry_points_validate_before_model(common, tmp_path, script):
    spec = importlib.util.spec_from_file_location("example", EXAMPLES / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        debug=False,
        verbose=False,
        execution_provider="cpu",
        ep_path=None,
        use_winml=False,
        model_path=str(tmp_path / "missing"),
        prompts=["test"],
        max_length=0,
    )
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        module.main(args)


@pytest.mark.parametrize("script", ["model-qa.py", "model-chat.py", "model-mm.py", "model-generate.py"])
def test_cli_preserves_default_limit(common, monkeypatch, script):
    class StopBeforeModelError(Exception):
        pass

    def check_options(*args, **kwargs):
        assert "max_length" not in kwargs["search_options"]
        raise StopBeforeModelError

    monkeypatch.setattr(common, "get_config", check_options)
    argv = [script, "-m", str(MODEL), "-e", "cpu"]
    if script == "model-generate.py":
        argv.append("--non_interactive")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(StopBeforeModelError):
        runpy.run_path(str(EXAMPLES / script), run_name="__main__")
