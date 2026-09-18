# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def test_all_builder_exports_are_lazy(monkeypatch):
    builders_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
    spec = importlib.util.spec_from_file_location("builders", builders_dir / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)

    assert package.__all__
    assert not set(package.__all__) & vars(package).keys(), "Builder exports must not be eagerly loaded"

    for name in package.__all__:
        exported_class = object()
        importer = Mock(return_value=SimpleNamespace(**{name: exported_class}))
        monkeypatch.setattr(package, "import_module", importer)

        assert getattr(package, name) is exported_class
        importer.assert_called_once()
        module_name, package_name = importer.call_args.args
        assert package_name == package.__name__
        assert module_name.startswith(".")
        source_path = builders_dir.joinpath(*module_name[1:].split(".")).with_suffix(".py")
        source = ast.parse(source_path.read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.ClassDef) and node.name == name for node in source.body), name


@pytest.mark.parametrize(
    ("relative_path", "dependency"),
    [
        (("builder.py",), "builders"),
        (("builders", "base.py"), "transformers"),
        (("builders", "mistral.py"), "transformers"),
        (("builders", "qwen.py"), "transformers"),
    ],
)
def test_no_module_level_model_imports(relative_path, dependency):
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    tree = ast.parse(models_dir.joinpath(*relative_path).read_text(encoding="utf-8"))
    pending = list(tree.body)
    violations = []

    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "transformers"
            and all(alias.name in {"AutoTokenizer", "GenerationConfig"} for alias in node.names)
        ):
            continue
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or "", *(alias.name for alias in node.names)]
        else:
            modules = []
        if any(dependency in module.split(".") for module in modules):
            violations.append(node.lineno)
        pending.extend(ast.iter_child_nodes(node))

    assert not violations, f"{dependency} imports must be inside functions, found at lines {sorted(violations)}"


@pytest.fixture
def weight_loader(monkeypatch):
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    monkeypatch.syspath_prepend(str(models_dir))
    base = importlib.import_module("builders.base")
    model = base.Model.__new__(base.Model)
    model.model_type = "LlamaForCausalLM"
    model.model_name_or_path = "local-checkpoint"
    model.cache_dir = "cache"
    model.hf_token = False
    model.hf_remote = False
    model.quant_type = None
    model.num_layers = 2
    model.extra_options = {"num_hidden_layers": 2}
    transformers = ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", transformers)
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


def test_load_weights_reports_missing_selected_transformers_class(weight_loader):
    model, transformers = weight_loader
    model.model_type = "qwen3_5_text"
    transformers.AutoModelForCausalLM = Mock()

    with pytest.raises(AttributeError, match="Qwen3_5ForConditionalGeneration"):
        model.load_weights("")
    transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    ("statement", "expected_modules"),
    [
        ("import builder", set()),
        ("import builders", set()),
        ("from builders import Model", {"base"}),
        ("from builders import LlamaModel", {"base", "llama"}),
        ("from builders import QwenModel", {"base", "mtp", "qwen"}),
        ("from builders.qwen import Qwen35Model, Qwen35MoEModel", {"base", "mtp", "qwen"}),
    ],
)
def test_only_requested_builders_are_imported(statement, expected_modules):
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    script = f"""
import sys
{statement}
loaded = {{
    name.removeprefix("builders.")
    for name in sys.modules
    if name.startswith("builders.") and name.count(".") == 1
}}
assert loaded == {expected_modules!r}, loaded
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=models_dir, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_builder_attribute_raises():
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builders; assert not hasattr(builders, 'UnknownModel'); "
            "assert not hasattr(builders, '__missing__')",
        ],
        cwd=models_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
