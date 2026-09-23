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
        (("builders", "base.py"), "transformers"),
        (("builders", "mistral.py"), "transformers"),
        (("builders", "qwen.py"), "transformers"),
    ],
)
def test_no_module_level_model_class_imports(relative_path, dependency):
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
        if isinstance(node, ast.Import) and all(alias.name == "transformers" for alias in node.names):
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

    assert not violations, f"Eager {dependency} model imports found at lines {sorted(violations)}"


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
    ("weight_loader", "class_name"),
    [
        (("base", "Model"), "Qwen3_5ForConditionalGeneration"),
        (("mistral", "Mistral3TextModel"), "Mistral3ForConditionalGeneration"),
        (("qwen", "VideoChatFlashQwenModel"), "Qwen2ForCausalLM"),
    ],
    indirect=["weight_loader"],
)
@pytest.mark.parametrize("version", ["4.45.0", None])
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
    ("weight_loader", "class_name"),
    [
        (("mistral", "Mistral3TextModel"), "Mistral3ForConditionalGeneration"),
        (("qwen", "VideoChatFlashQwenModel"), "Qwen2ForCausalLM"),
    ],
    indirect=["weight_loader"],
)
@pytest.mark.parametrize("local_checkpoint", [False, True])
def test_custom_weight_loaders_use_shared_resolver(weight_loader, class_name, local_checkpoint, tmp_path):
    model, transformers = weight_loader
    if local_checkpoint:
        model.model_name_or_path = str(tmp_path)
    loader = Mock()
    loader.from_pretrained.return_value.named_modules.return_value = []
    setattr(transformers, class_name, loader)
    resolver = Mock(wraps=model.resolve_transformers_class)
    model.resolve_transformers_class = resolver

    assert model.load_weights("") is loader.from_pretrained.return_value

    resolver.assert_called_once_with(class_name)
    kwargs = {"token": model.hf_token}
    if class_name == "Mistral3ForConditionalGeneration":
        kwargs.update(cache_dir=model.cache_dir, trust_remote_code=model.hf_remote, num_hidden_layers=model.num_layers)
    elif not local_checkpoint:
        kwargs["cache_dir"] = model.cache_dir
    loader.from_pretrained.assert_called_once_with(model.model_name_or_path, **kwargs)


@pytest.mark.parametrize(
    ("statement", "expected_modules"),
    [
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


def test_importing_entrypoint_does_not_load_transformers_architectures():
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    script = """
import sys
import builder
assert builder.LlamaModel.__module__ == "builders.llama"
assert builder.Qwen35MoEModel.__module__ == "builders.qwen"
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


def test_unknown_builder_attribute_raises():
    models_dir = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
    script = """
import builders
assert not hasattr(builders, "UnknownModel")
assert not hasattr(builders, "__missing__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=models_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("test_module", ["test_paged_block_size.py", "test_precision.py", "test_quantized_kv_cache.py"])
def test_collecting_builder_tests_preserves_module_metadata(test_module):
    script = f"""
import inspect
import runpy
import sys
sys.path.insert(0, {str(Path(__file__).parents[3] / "src" / "python" / "py" / "models")!r})
import builders
original_builders = builders
runpy.run_path({str(Path(__file__).with_name(test_module))!r})
assert sys.modules["builders"] is original_builders, "Test collection replaced the builders package"
assert inspect.getsourcefile(sys.modules["builders"]) == original_builders.__file__
from transformers import Qwen2_5_VLForConditionalGeneration
assert Qwen2_5_VLForConditionalGeneration.__name__ == "Qwen2_5_VLForConditionalGeneration"
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
