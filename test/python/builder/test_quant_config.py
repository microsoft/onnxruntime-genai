# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the unified ``QuantConfig`` model-builder quantization config.

``QuantConfig`` is a structured (JSON/``dict``) surface for the model builder's
quantization options. These tests exercise it standalone: the dtype vocabulary,
``from_dict`` validation, JSON loading, and the ``from_extra_options`` back-compat
adapter that desugars today's flat ``extra_options`` into the same structure.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]

qc = _load_builder_module("quant_config")
QuantConfig = qc.QuantConfig
WeightsConfig = qc.WeightsConfig
MoEConfig = qc.MoEConfig
RuntimeConfig = qc.RuntimeConfig
Override = qc.Override
resolve_dtype = qc.resolve_dtype


# ---------------------------------------------------------------------------
# Dtype vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,kind,bits",
    [
        ("fp16", "float", 16),
        ("bf16", "float", 16),
        ("int8", "int", 8),
        ("uint8", "int", 8),
        ("int4", "int", 4),
        ("uint4", "int", 4),
        ("mxfp4", "mx", 4),
    ],
)
def test_resolve_dtype_known(name, kind, bits):
    d = resolve_dtype(name)
    assert d.kind == kind
    assert d.bits == bits


def test_resolve_dtype_is_case_insensitive_and_trims():
    assert resolve_dtype(" INT4 ").name == "int4"


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="unknown quant dtype"):
        resolve_dtype("int6")


def test_float_dtype_is_not_quantized():
    assert resolve_dtype("fp16").is_quantized is False
    assert resolve_dtype("int4").is_quantized is True


# ---------------------------------------------------------------------------
# from_dict validation
# ---------------------------------------------------------------------------


def test_from_dict_empty_uses_defaults():
    cfg = QuantConfig.from_dict({})
    assert cfg.io_dtype == "fp16"
    assert cfg.weights.type == "none"
    assert cfg.moe.type == "int4"
    assert cfg.runtime.use_qdq is False


def test_from_dict_accepts_quantization_wrapper():
    cfg = QuantConfig.from_dict({"quantization": {"io_dtype": "bf16", "weights": {"type": "int4"}}})
    assert cfg.io_dtype == "bf16"
    assert cfg.weights.type == "int4"


def test_from_dict_rejects_unknown_top_level_field():
    with pytest.raises(ValueError, match="unknown quantization field"):
        QuantConfig.from_dict({"weight": {}})


def test_from_dict_rejects_bad_io_dtype():
    with pytest.raises(ValueError, match="io_dtype must be one of"):
        QuantConfig.from_dict({"io_dtype": "int4"})


def test_weights_rejects_bad_method():
    with pytest.raises(ValueError, match="weights.method must be one of"):
        WeightsConfig.from_dict({"type": "int4", "method": "gptq"})


def test_weights_per_channel_block_size_string():
    w = WeightsConfig.from_dict({"type": "int4", "block_size": "per_channel"})
    assert w.block_size == 0


def test_weights_mx_dtype_block_size_conflict():
    with pytest.raises(ValueError, match="fixes block_size"):
        WeightsConfig.from_dict({"type": "mxfp4", "block_size": 64})


def test_moe_mxfp4_forces_block_size_32():
    m = MoEConfig.from_dict({"type": "mxfp4", "block_size": 128})
    assert m.block_size == 32


def test_moe_rejects_bad_prepacked():
    with pytest.raises(ValueError, match="weights_prepacked must be"):
        MoEConfig.from_dict({"type": "int4", "weights_prepacked": 2})


def test_runtime_rejects_bad_prepacked():
    with pytest.raises(ValueError, match="matmulnbits_weights_prepacked must be"):
        RuntimeConfig.from_dict({"matmulnbits_weights_prepacked": 3})


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_override_preset_type():
    o = Override.from_dict({"match": {"preset": "last_matmul"}, "type": "int8"})
    assert o.match == {"preset": "last_matmul"}
    assert o.type == "int8"


def test_override_exclude():
    o = Override.from_dict({"match": {"name": "/lm_head/MatMul"}, "exclude": True})
    assert o.exclude is True
    assert o.type is None


def test_override_rejects_unknown_preset():
    with pytest.raises(ValueError, match="preset must be one of"):
        Override.from_dict({"match": {"preset": "first_matmul"}, "type": "int8"})


def test_override_rejects_type_and_exclude():
    with pytest.raises(ValueError, match="cannot set both"):
        Override.from_dict({"match": {"name": "x"}, "type": "int8", "exclude": True})


def test_override_requires_type_or_exclude():
    with pytest.raises(ValueError, match="must set either"):
        Override.from_dict({"match": {"name": "x"}})


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------


def test_from_json_inline_string():
    cfg = QuantConfig.from_json('{"weights": {"type": "int4", "block_size": 128}}')
    assert cfg.weights.type == "int4"
    assert cfg.weights.block_size == 128


def test_from_json_file(tmp_path):
    path = tmp_path / "quant.json"
    path.write_text(json.dumps({"quantization": {"moe": {"type": "int8"}}}))
    cfg = QuantConfig.from_json(str(path))
    assert cfg.moe.type == "int8"


def test_to_dict_is_reloadable():
    cfg = QuantConfig.from_dict(
        {
            "io_dtype": "fp16",
            "weights": {"type": "int4", "method": "rtn", "overrides": [{"match": {"preset": "last_matmul"}, "type": "int8"}]},
            "moe": {"type": "mxfp4"},
        }
    )
    reloaded = QuantConfig.from_dict(cfg.to_dict())
    assert reloaded.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------------------
# from_extra_options back-compat adapter (§9 mapping)
# ---------------------------------------------------------------------------


def test_extra_options_precision_int4_defaults():
    cfg = QuantConfig.from_extra_options({}, precision="int4", execution_provider="cuda")
    assert cfg.io_dtype == "fp16"
    assert cfg.weights.type == "int4"
    assert cfg.weights.method == "default"
    assert cfg.weights.block_size == 32
    assert cfg.weights.overrides == []
    assert cfg.moe.type == "int4"
    # CUDA default QMoE block size is 32 (128 is TRT-RTX only).
    assert cfg.moe.block_size == 32
    assert cfg.moe.weights_prepacked == -1
    assert cfg.runtime.matmulnbits_weights_prepacked == 0


def test_extra_options_precision_float_disables_weight_quant():
    cfg = QuantConfig.from_extra_options({}, precision="bf16", execution_provider="cuda")
    assert cfg.io_dtype == "bf16"
    assert cfg.weights.type == "none"


def test_extra_options_cpu_default_block_size_and_accuracy_level():
    cfg = QuantConfig.from_extra_options({}, precision="int4", execution_provider="cpu")
    assert cfg.moe.block_size == 32
    assert cfg.weights.accuracy_level == 4


def test_extra_options_trt_rtx_default_qmoe_block_size():
    cfg = QuantConfig.from_extra_options({}, precision="int4", execution_provider="trt-rtx")
    assert cfg.moe.block_size == 128


def test_extra_options_legacy_rtn_last_alias():
    cfg = QuantConfig.from_extra_options({"int4_algo_config": "rtn_last"}, precision="int4")
    assert cfg.weights.method == "rtn"
    assert cfg.weights.overrides == [Override(match={"preset": "last_matmul"}, type="int8")]


def test_extra_options_legacy_k_quant_mixed_alias():
    cfg = QuantConfig.from_extra_options({"int4_algo_config": "k_quant_mixed"}, precision="int4")
    assert cfg.weights.method == "k_quant"
    presets = [(o.match["preset"], o.type) for o in cfg.weights.overrides]
    assert presets == [("last_matmul", "int8"), ("mixed_layers", "int8")]


def test_extra_options_matmul_mixed_precision_string():
    cfg = QuantConfig.from_extra_options(
        {"int4_algo_config": "k_quant", "matmul_mixed_precision": "last_matmul:int8,linear_attn:int4"},
        precision="int4",
    )
    assert cfg.weights.method == "k_quant"
    presets = {o.match["preset"]: o.type for o in cfg.weights.overrides}
    assert presets == {"last_matmul": "int8", "linear_attn": "int4"}


def test_extra_options_explicit_mixed_precision_overrides_alias_default():
    cfg = QuantConfig.from_extra_options(
        {"int4_algo_config": "k_quant_last", "matmul_mixed_precision": "last_matmul:int4"},
        precision="int4",
    )
    presets = {o.match["preset"]: o.type for o in cfg.weights.overrides}
    assert presets == {"last_matmul": "int4"}


def test_extra_options_nodes_to_exclude_become_overrides():
    cfg = QuantConfig.from_extra_options(
        {"int4_nodes_to_exclude": ["/model/embed_tokens/Gather"]}, precision="int4"
    )
    excludes = [o for o in cfg.weights.overrides if o.exclude]
    assert excludes == [Override(match={"name": "/model/embed_tokens/Gather"}, exclude=True)]


def test_extra_options_moe_quant_type_and_use_8bits_moe():
    assert QuantConfig.from_extra_options({"moe_quant_type": "mxfp4"}, precision="int4").moe.type == "mxfp4"
    # Deprecated use_8bits_moe maps to int8 when moe_quant_type is absent.
    assert QuantConfig.from_extra_options({"use_8bits_moe": True}, precision="int4").moe.type == "int8"
    # moe_quant_type wins over the deprecated flag.
    cfg = QuantConfig.from_extra_options({"use_8bits_moe": True, "moe_quant_type": "int4"}, precision="int4")
    assert cfg.moe.type == "int4"


def test_extra_options_runtime_and_prepack_knobs():
    cfg = QuantConfig.from_extra_options(
        {
            "use_qdq": True,
            "matmulnbits_weights_prepacked": 2,
            "qmoe_weights_prepacked": 1,
            "qmoe_block_size": 64,
        },
        precision="int4",
        execution_provider="cuda",
    )
    assert cfg.runtime.use_qdq is True
    assert cfg.runtime.matmulnbits_weights_prepacked == 2
    assert cfg.moe.weights_prepacked == 1
    assert cfg.moe.block_size == 64
