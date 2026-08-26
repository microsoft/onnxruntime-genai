# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Registration of Qwen-3.8 Flash Next (``Qwen4Exp``) across the builder and the runtime.

A new architecture is only reachable if every registration point agrees: the builders package
must export it, ``builder.py`` must dispatch both the multimodal and the text-only architecture
strings, and ``src/models/model_type.h`` must classify the resulting ``model.type``.  These
checks are static (no checkpoint, no ONNX export) so they run anywhere.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
MODELS_DIR = REPO_ROOT / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))

sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_builder_module("base")
_load_builder_module("quant_config")
qwen_module = _load_builder_module("qwen")
qwen4exp = _load_builder_module("qwen4exp")

BUILDER_SOURCE = (MODELS_DIR / "builder.py").read_text()
BUILDERS_INIT_SOURCE = (BUILDERS_DIR / "__init__.py").read_text()
MODEL_TYPE_SOURCE = (REPO_ROOT / "src" / "models" / "model_type.h").read_text()

MULTIMODAL_ARCH = "Qwen4ExpForConditionalGeneration"
TEXT_ARCH = "Qwen4ExpForCausalLM"


#####################################################################################
# Class hierarchy
#####################################################################################


def test_text_builder_extends_the_qwen35_moe_builder():
    # The hybrid attention, mRoPE and MoE machinery is inherited rather than duplicated.
    assert issubclass(qwen4exp.Qwen4ExpTextModel, qwen_module.Qwen35MoeTextModel)


def test_composite_builder_is_also_the_text_builder():
    assert issubclass(qwen4exp.Qwen4ExpModel, qwen4exp.Qwen4ExpTextModel)


def test_auxiliary_builders_extend_the_common_model_base():
    assert issubclass(qwen4exp.Qwen4ExpVisionModel, base_module.Model)
    assert issubclass(qwen4exp.Qwen4ExpEmbeddingModel, base_module.Model)


@pytest.mark.parametrize(
    "builder, expected_filename",
    [
        (qwen4exp.Qwen4ExpEmbeddingModel, "embedding.onnx"),
        (qwen4exp.Qwen4ExpVisionModel, "vision.onnx"),
    ],
)
def test_auxiliary_builders_declare_their_own_output_filename(builder, expected_filename):
    assert builder.DEFAULT_FILENAME == expected_filename


#####################################################################################
# Model type strings
#####################################################################################


def test_model_type_distinguishes_multimodal_from_text_only():
    multimodal = object.__new__(qwen4exp.Qwen4ExpTextModel)
    multimodal.is_text_only = False
    text_only = object.__new__(qwen4exp.Qwen4ExpTextModel)
    text_only.is_text_only = True

    assert multimodal._get_model_type(None) == MULTIMODAL_ARCH
    assert text_only._get_model_type(None) == "Qwen4Exp_textForCausalLM"


@pytest.mark.parametrize(
    "model_type_string, expected_genai_type",
    [(MULTIMODAL_ARCH, "qwen4exp"), ("Qwen4Exp_textForCausalLM", "qwen4exp_text")],
)
def test_genai_model_type_derivation_matches_the_runtime_registry(model_type_string, expected_genai_type):
    # `Model.make_genai_config` derives `model.type` by truncating at "For".
    assert model_type_string[: model_type_string.find("For")].lower() == expected_genai_type


#####################################################################################
# builders package exports
#####################################################################################


@pytest.mark.parametrize(
    "name",
    ["Qwen4ExpModel", "Qwen4ExpTextModel", "Qwen4ExpEmbeddingModel", "Qwen4ExpVisionModel"],
)
def test_builders_package_exports_every_qwen4exp_builder(name):
    assert f"from .qwen4exp import" in BUILDERS_INIT_SOURCE
    assert f'"{name}",' in BUILDERS_INIT_SOURCE, f"{name} missing from builders/__init__.py __all__"


def test_qwen4exp_exports_are_grouped_with_the_other_qwen_builders():
    tree = ast.parse(BUILDERS_INIT_SOURCE)
    all_names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            all_names = [element.value for element in node.value.elts]
    assert all_names is not None

    qwen4exp_positions = [i for i, name in enumerate(all_names) if name.startswith("Qwen4Exp")]
    assert len(qwen4exp_positions) == 4
    # Contiguous, and slotted between the Qwen3.5 builders and the plain QwenModel entry.
    assert qwen4exp_positions == list(range(qwen4exp_positions[0], qwen4exp_positions[0] + 4))
    assert all_names.index("Qwen35MoeTextModel") < qwen4exp_positions[0]
    assert all_names.index("QwenModel") > qwen4exp_positions[-1]
    assert len(all_names) == len(set(all_names))


#####################################################################################
# builder.py dispatch
#####################################################################################


@pytest.mark.parametrize(
    "architecture, builder_name",
    [(MULTIMODAL_ARCH, "Qwen4ExpModel"), (TEXT_ARCH, "Qwen4ExpTextModel")],
)
def test_create_model_dispatches_both_qwen4exp_architectures(architecture, builder_name):
    pattern = (
        rf'elif config\.architectures\[0\] == "{architecture}":'
        rf'(?:.|\n)*?onnx_model = {builder_name}\(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options\)'
    )
    assert re.search(pattern, BUILDER_SOURCE), f"no dispatch branch for {architecture}"


def test_builder_imports_the_dispatched_classes():
    assert re.search(r"^\s+Qwen4ExpModel,$", BUILDER_SOURCE, re.MULTILINE)
    assert re.search(r"^\s+Qwen4ExpTextModel,$", BUILDER_SOURCE, re.MULTILINE)


def test_text_only_dispatch_keeps_the_embedding_inside_the_decoder():
    # The PLE layers hash raw `input_ids`, so a text-only export must not exclude the embedding.
    branch = BUILDER_SOURCE.split(f'elif config.architectures[0] == "{TEXT_ARCH}":')[1]
    branch = branch.split("elif config.architectures[0]")[0]
    assert 'extra_options["exclude_embeds"] = False' in branch


def test_multimodal_dispatch_does_not_force_text_only_export():
    branch = BUILDER_SOURCE.split(f'elif config.architectures[0] == "{MULTIMODAL_ARCH}":')[1]
    branch = branch.split("elif config.architectures[0]")[0]
    assert "exclude_embeds" not in branch, "the composite builder exports embedding.onnx itself"


#####################################################################################
# Runtime model-type registry
#####################################################################################


def _parse_model_type_array(name):
    match = re.search(
        rf"std::array<std::string_view, (\d+)> {name} = \{{(.*?)\}};",
        MODEL_TYPE_SOURCE,
        re.DOTALL,
    )
    assert match, f"could not find the {name} array in model_type.h"
    declared_size = int(match.group(1))
    entries = re.findall(r'"([^"]+)"', match.group(2))
    return declared_size, entries


@pytest.mark.parametrize(
    "array_name, expected_entry",
    [("LLM", "qwen4exp_text"), ("VLM", "qwen4exp"), ("QwenVL", "qwen4exp")],
)
def test_runtime_registry_contains_the_new_model_types(array_name, expected_entry):
    _, entries = _parse_model_type_array(array_name)
    assert expected_entry in entries


@pytest.mark.parametrize("array_name", ["LLM", "VLM", "QwenVL"])
def test_runtime_registry_sizes_match_their_contents(array_name):
    # These std::array sizes are hard-coded, so adding an entry without bumping the size is a
    # compile error that is easy to miss in a Python-only change.
    declared_size, entries = _parse_model_type_array(array_name)
    assert declared_size == len(entries)


def test_multimodal_type_is_registered_as_a_mrope_qwen_vl_model():
    # Qwen4Exp uses 3D mRoPE position ids, exactly like the other Qwen-VL family models.
    _, vlm = _parse_model_type_array("VLM")
    _, qwen_vl = _parse_model_type_array("QwenVL")
    assert set(qwen_vl).issubset(set(vlm))
    assert "qwen4exp" in qwen_vl


def test_text_only_type_is_not_registered_as_a_vision_model():
    _, vlm = _parse_model_type_array("VLM")
    assert "qwen4exp_text" not in vlm
