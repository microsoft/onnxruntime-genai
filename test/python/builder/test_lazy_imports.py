# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("relative_path", "dependency"),
    [
        (("builder.py",), "builders"),
        (("builders", "base.py"), "transformers"),
        (("builders", "mistral.py"), "transformers"),
        (("builders", "qwen.py"), "transformers"),
    ],
)
def test_no_local_model_imports(relative_path, dependency):
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    tree = ast.parse(models_dir.joinpath(*relative_path).read_text(encoding="utf-8"))
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] == dependency for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != dependency


@pytest.fixture
def weight_loader(monkeypatch, request):
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    monkeypatch.syspath_prepend(str(models_dir))
    module_name, builder_name = getattr(request, "param", ("base", "Model"))
    builder_class = getattr(importlib.import_module(f"builders.{module_name}"), builder_name)
    model = builder_class.__new__(builder_class)
    model.model_type = "LlamaForCausalLM"
    model.model_name_or_path = "local-checkpoint"
    model.cache_dir = "cache"
    model.hf_token = False
    model.hf_remote = False
    model.quant_type = None
    model.num_layers = 2
    model.extra_options = {"num_hidden_layers": 2}
    transformers = ModuleType("transformers")
    monkeypatch.setattr(importlib.import_module("builders.base"), "transformers", transformers)
    return model, transformers


@pytest.mark.parametrize(
    ("model_type", "class_name"),
    [
        ("LlamaForCausalLM", "AutoModelForCausalLM"),
        ("Gemma3ForCausalLM", "AutoModelForCausalLM"),
        ("llama", "AutoModelForCausalLM"),
        ("gemma3_vl_text", "Gemma3ForConditionalGeneration"),
        ("lfm2_vl", "Lfm2VlForConditionalGeneration"),
        ("lfm2_vl_text", "Lfm2VlForConditionalGeneration"),
        ("mistral3_text", "Mistral3ForConditionalGeneration"),
        ("Mistral3ForConditionalGeneration", "Mistral3ForConditionalGeneration"),
        ("qwen2_5_vl_text", "Qwen2_5_VLForConditionalGeneration"),
        ("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen3_vl_text", "Qwen3VLForConditionalGeneration"),
        ("Qwen3VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen3_5_text", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("WhisperForConditionalGeneration", "AutoModelForSpeechSeq2Seq"),
    ],
)
def test_load_weights_requires_only_selected_transformers_class(weight_loader, model_type, class_name):
    model, transformers = weight_loader
    model.model_type = model_type
    loader = Mock()
    setattr(transformers, class_name, loader)

    assert model.load_weights("") is loader.from_pretrained.return_value
    loader.from_pretrained.assert_called_once_with(
        model.model_name_or_path,
        cache_dir=model.cache_dir,
        token=model.hf_token,
        trust_remote_code=model.hf_remote,
        num_hidden_layers=model.num_layers,
    )


@pytest.mark.parametrize(
    ("weight_loader", "class_name", "version"),
    [
        (("base", "Model"), "Qwen3_5ForConditionalGeneration", "4.45.0"),
        (("base", "Model"), "Qwen3_5ForConditionalGeneration", None),
        (("mistral", "Mistral3TextModel"), "Mistral3ForConditionalGeneration", "4.45.0"),
        (("qwen", "VideoChatFlashQwenModel"), "Qwen2ForCausalLM", "4.45.0"),
    ],
    indirect=["weight_loader"],
)
def test_load_weights_reports_missing_selected_transformers_class(weight_loader, class_name, version):
    model, transformers = weight_loader
    model.model_type = "qwen3_5_text"
    transformers.AutoModelForCausalLM = Mock()
    if version is not None:
        transformers.__version__ = version

    with pytest.raises(ImportError, match=rf"requires transformers\.{class_name}") as error:
        model.load_weights("")
    assert f"Transformers {version or 'unknown'} does not provide that class" in str(error.value)
    assert "Upgrade Transformers" in str(error.value)
    assert isinstance(error.value.__cause__, AttributeError)
    transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError, RuntimeError])
def test_transformers_resolver_preserves_dependency_errors(weight_loader, error_type):
    model, transformers = weight_loader
    error = error_type("Selected model dependency failed to import")
    transformers.__getattr__ = Mock(side_effect=error)

    with pytest.raises(error_type) as raised:
        model.resolve_transformers_class("AutoModelForCausalLM")
    assert raised.value is error
    transformers.__getattr__.assert_called_once_with("AutoModelForCausalLM")


@pytest.mark.parametrize(
    ("weight_loader", "class_name", "local_checkpoint"),
    [
        (("mistral", "Mistral3TextModel"), "Mistral3ForConditionalGeneration", False),
        (("qwen", "VideoChatFlashQwenModel"), "Qwen2ForCausalLM", False),
        (("qwen", "VideoChatFlashQwenModel"), "Qwen2ForCausalLM", True),
    ],
    indirect=["weight_loader"],
)
def test_custom_weight_loaders_preserve_loading_arguments(weight_loader, class_name, local_checkpoint, tmp_path):
    model, transformers = weight_loader
    if local_checkpoint:
        model.model_name_or_path = str(tmp_path)
    loader = Mock()
    loader.from_pretrained.return_value.named_modules.return_value = []
    setattr(transformers, class_name, loader)
    assert model.load_weights("") is loader.from_pretrained.return_value

    kwargs = {"token": model.hf_token}
    if class_name == "Mistral3ForConditionalGeneration":
        kwargs.update(cache_dir=model.cache_dir, trust_remote_code=model.hf_remote, num_hidden_layers=model.num_layers)
    elif not local_checkpoint:
        kwargs["cache_dir"] = model.cache_dir
    loader.from_pretrained.assert_called_once_with(model.model_name_or_path, **kwargs)


def test_importing_entrypoint_does_not_load_transformers_architectures():
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    script = """
import sys
import builder
loaded = {
    name for name in sys.modules
    if name.startswith("transformers.models.")
    and not name.startswith("transformers.models.auto.")
    and ".modeling_" in name
}
assert not loaded, loaded
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=models_dir, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_collecting_builder_tests_preserves_module_metadata():
    script = f"""
import inspect
import runpy
import sys
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).parents[3] / "src" / "python" / "py" / "models")!r})
import builders
original_builders = builders
test_dir = Path({str(Path(__file__).parent)!r})
for test_module in (
    "test_paged_block_size.py",
    "test_precision.py",
    "test_quantized_kv_cache.py",
    "test_lfm2_vl.py",
    "test_max_draft_tokens.py",
):
    runpy.run_path(str(test_dir / test_module))
    assert sys.modules["builders"] is original_builders, test_module
    assert inspect.getsourcefile(sys.modules["builders"]) == original_builders.__file__, test_module
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
