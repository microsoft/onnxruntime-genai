# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the `matmul_mixed_precision` extra option in the model builder.

`matmul_mixed_precision` decouples *which* MatMul groups are upgraded (selectors
``last_matmul`` / ``mixed_layers`` / ``linear_attn``) from *what* quant type they use
(``int4`` / ``int8``, extensible to fp8/fp4). It replaces the removed per-target int8
boolean flags while keeping the pre-existing compound ``algo_config`` aliases
(``rtn_last`` / ``k_quant_last`` / ``k_quant_mixed`` / ``k_quant_linear``) working.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
Model = base_module.Model


# ---------------------------------------------------------------------------
# matmul_mixed_precision parsing (via resolve_quant_config -> quant_config)
# ---------------------------------------------------------------------------


def _resolve_mixed_precision(value):
    model = Model.__new__(Model)
    model.resolve_quant_config({"algo_config": "default", "matmul_mixed_precision": value})
    return model.matmul_mixed_precision


def test_normalize_parses_selector_quant_type_string():
    result = _resolve_mixed_precision("last_matmul:int8,mixed_layers:int8,linear_attn:int4")
    assert result == {"last_matmul": "int8", "mixed_layers": "int8", "linear_attn": "int4"}


def test_normalize_tolerates_whitespace_and_empty_entries():
    result = _resolve_mixed_precision(" last_matmul : int8 , , linear_attn:int4 ")
    assert result == {"last_matmul": "int8", "linear_attn": "int4"}


def test_normalize_accepts_dict_passthrough():
    assert _resolve_mixed_precision({"last_matmul": "int8"}) == {"last_matmul": "int8"}


def test_normalize_empty_returns_empty_dict():
    assert _resolve_mixed_precision("") == {}
    assert _resolve_mixed_precision({}) == {}


def test_normalize_rejects_missing_colon():
    with pytest.raises(ValueError, match="must be 'selector:quant_type'"):
        _resolve_mixed_precision("last_matmul")


def test_normalize_rejects_unknown_selector():
    with pytest.raises(ValueError, match="selector must be one of"):
        _resolve_mixed_precision("first_matmul:int8")


def test_normalize_rejects_unknown_quant_type():
    with pytest.raises(ValueError, match="unknown quant dtype"):
        _resolve_mixed_precision("last_matmul:int6")


# ---------------------------------------------------------------------------
# resolve_quant_config
# ---------------------------------------------------------------------------


def test_resolve_base_method_only_has_no_mixed_precision():
    model = Model.__new__(Model)
    model.resolve_quant_config({"algo_config": "k_quant"})
    assert model.quantization_algo == "k_quant"
    assert model.matmul_mixed_precision == {}


def test_resolve_reads_matmul_mixed_precision_string():
    model = Model.__new__(Model)
    model.resolve_quant_config(
        {"algo_config": "default", "matmul_mixed_precision": "last_matmul:int8,linear_attn:int4"}
    )
    assert model.quantization_algo == "default"
    assert model.matmul_mixed_precision == {"last_matmul": "int8", "linear_attn": "int4"}


def test_resolve_expands_legacy_compound_alias():
    model = Model.__new__(Model)
    model.resolve_quant_config({"algo_config": "k_quant_mixed"})
    assert model.quantization_algo == "k_quant"
    assert model.matmul_mixed_precision == {"last_matmul": "int8", "mixed_layers": "int8"}


def test_resolve_explicit_config_overrides_legacy_alias_defaults():
    model = Model.__new__(Model)
    model.resolve_quant_config({"algo_config": "k_quant_last", "matmul_mixed_precision": "last_matmul:int4"})
    assert model.quantization_algo == "k_quant"
    # Explicit last_matmul:int4 overrides the alias-implied last_matmul:int8.
    assert model.matmul_mixed_precision == {"last_matmul": "int4"}


# ---------------------------------------------------------------------------
# make_matmul_mixed_precision -> customized_weight_config
# ---------------------------------------------------------------------------


def test_make_config_empty_placement_yields_empty_config():
    model = Model.__new__(Model)
    model.make_matmul_mixed_precision({})
    assert model.int4_customized_weight_config == {}


@pytest.mark.parametrize("quant_type,expected_bits", [("int8", 8), ("int4", 4)])
def test_make_config_last_matmul_uses_quant_type_bits(quant_type, expected_bits):
    model = Model.__new__(Model)
    model.make_matmul_mixed_precision({"last_matmul": quant_type})
    assert model.int4_customized_weight_config == {"/lm_head/MatMul": {"bits": expected_bits}}


def test_make_config_mixed_layers_upgrades_sensitive_matmuls():
    model = Model.__new__(Model)
    model.num_layers = 8
    model.make_matmul_mixed_precision({"mixed_layers": "int8"})

    cfg = model.int4_customized_weight_config
    assert cfg  # non-empty
    # Every upgraded node targets int8, and only the expected projection types are touched.
    assert all(entry == {"bits": 8} for entry in cfg.values())
    assert all(node.endswith(("/attn/qkv_proj/MatMul", "/attn/v_proj/MatMul", "/mlp/down_proj/MatMul")) for node in cfg)
    # Layer 0 is in the first eighth, so it must be upgraded.
    assert "/model/layers.0/attn/qkv_proj/MatMul" in cfg


def test_make_config_linear_attn_upgrades_linear_layers_only():
    model = Model.__new__(Model)
    model.layer_types = ["linear_attention", "full_attention"]
    model.make_matmul_mixed_precision({"linear_attn": "int8"})

    cfg = model.int4_customized_weight_config
    assert all(entry == {"bits": 8} for entry in cfg.values())
    # Layer 0 (linear) projections + MLP are upgraded; layer 1 (full attention) is untouched.
    assert "/model/layers.0/linear_attn/out_proj/MatMul" in cfg
    assert "/model/layers.0/mlp/down_proj/MatMul" in cfg
    assert not any(node.startswith("/model/layers.1/") for node in cfg)


def test_make_config_supports_distinct_types_per_selector():
    model = Model.__new__(Model)
    model.num_layers = 8
    model.make_matmul_mixed_precision({"last_matmul": "int8", "mixed_layers": "int4"})

    cfg = model.int4_customized_weight_config
    assert cfg["/lm_head/MatMul"] == {"bits": 8}
    # mixed_layers entries use int4 (4 bits), independent of the last_matmul type.
    mixed_layer_nodes = [node for node in cfg if node != "/lm_head/MatMul"]
    assert mixed_layer_nodes
    assert all(cfg[node] == {"bits": 4} for node in mixed_layer_nodes)
