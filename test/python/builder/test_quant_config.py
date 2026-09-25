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

QUANTIZATION_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "quantization"


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(
        f"models.quantization.{module_name}", QUANTIZATION_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.quantization.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules.setdefault("models.quantization", types.ModuleType("models.quantization"))

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
    assert cfg.checkpoint_policy == "preserve"
    assert cfg.weights.type == "none"
    assert cfg.moe.type == "int4"
    assert cfg.format.use_qdq is False


@pytest.mark.parametrize("weights_type", ["uint4", "uint8"])
def test_unsigned_weights_default_to_asymmetric(weights_type):
    weights = WeightsConfig.from_dict({"type": weights_type})

    assert weights.symmetric is False
    assert weights.to_dict()["symmetric"] is False


@pytest.mark.parametrize("weights_type", ["uint4", "uint8"])
def test_unsigned_weights_reject_explicit_symmetric_mode(weights_type):
    with pytest.raises(ValueError, match="requires weights.symmetric=false"):
        WeightsConfig.from_dict({"type": weights_type, "symmetric": True})


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


def test_from_dict_rejects_bad_checkpoint_policy():
    with pytest.raises(ValueError, match="checkpoint_policy must be"):
        QuantConfig.from_dict({"checkpoint_policy": "convert"})


def test_from_dict_accepts_runtime_compatibility_alias():
    cfg = QuantConfig.from_dict({"runtime": {"use_qdq": True}})
    assert cfg.format.use_qdq is True
    assert cfg.to_dict()["runtime"]["use_qdq"] is True
    assert "checkpoint_policy" not in cfg.to_dict()


def test_quant_config_positional_arguments_keep_legacy_order():
    weights = WeightsConfig(type="int4")
    moe = MoEConfig(type="none")
    runtime = RuntimeConfig(use_qdq=True)

    cfg = QuantConfig("bf16", weights, moe, runtime)

    assert cfg.io_dtype == "bf16"
    assert cfg.weights is weights
    assert cfg.moe is moe
    assert cfg.runtime is runtime


@pytest.mark.parametrize(
    "factory,data,field",
    [
        (WeightsConfig.from_dict, {"symmetric": "false"}, "weights.symmetric"),
        (WeightsConfig.from_dict, {"symmetric": 0}, "weights.symmetric"),
        (RuntimeConfig.from_dict, {"use_qdq": "false"}, "format.use_qdq"),
        (Override.from_dict, {"match": {"name": "x"}, "exclude": 0}, "override.exclude"),
    ],
)
def test_structured_booleans_require_json_booleans(factory, data, field):
    with pytest.raises(ValueError, match=field):
        factory(data)


def test_from_dict_rejects_conflicting_format_alias():
    with pytest.raises(ValueError, match="format and compatibility alias runtime conflict"):
        QuantConfig.from_dict({"format": {"use_qdq": True}, "runtime": {"use_qdq": False}})


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


def test_moe_mixed_width_round_trip():
    config = MoEConfig.from_dict({"type": "int4", "fc1_type": "int2", "fc2_type": "int4", "block_size": 64})

    assert config.fc1_type == "int2"
    assert config.fc2_type == "int4"
    assert MoEConfig.from_dict(config.to_dict()) == config


def test_moe_mixed_width_rejects_non_integer_type():
    with pytest.raises(ValueError, match="only supported for integer QMoE"):
        MoEConfig.from_dict({"type": "mxfp4", "fc1_type": "int2"})


def test_dense_weights_reject_int2():
    with pytest.raises(ValueError, match="dense integer weights require int4 or int8"):
        QuantConfig.from_dict({"weights": {"type": "int2"}})


def test_extra_options_moe_mixed_width():
    config = QuantConfig.from_extra_options(
        {"qmoe_fc1_type": "int2", "qmoe_fc2_type": "int4", "qmoe_block_size": 64},
        precision="int4",
        execution_provider="cuda",
    )

    assert config.moe.fc1_type == "int2"
    assert config.moe.fc2_type == "int4"
    assert config.moe.block_size == 64


def test_runtime_rejects_bad_prepacked():
    with pytest.raises(ValueError, match="matmulnbits_weights_prepacked must be"):
        RuntimeConfig.from_dict({"matmulnbits_weights_prepacked": 3})


@pytest.mark.parametrize(
    "config_type,field,valid_values",
    [
        (WeightsConfig, "accuracy_level", range(5)),
        (MoEConfig, "weights_prepacked", (-1, 0, 1)),
        (RuntimeConfig, "matmulnbits_weights_prepacked", (0, 1, 2)),
    ],
)
def test_structured_integer_fields_require_in_range_json_integers(config_type, field, valid_values):
    invalid_values = (True, False, 0.0, 0.9, 1.5, 2.7, "1", None, min(valid_values) - 1, max(valid_values) + 1)
    for value in invalid_values:
        with pytest.raises(ValueError, match=field):
            config_type.from_dict({field: value})
        with pytest.raises(ValueError, match=field):
            config_type(**{field: value})
    for value in valid_values:
        config = config_type.from_dict({field: value})
        assert getattr(config, field) == value
        assert config_type.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "op_types",
    [
        "MatMul",
        "MatMul/Gather",
        ("MatMul",),
        {"MatMul": True},
        None,
        [],
        [""],
        ["matmul"],
        ["Add"],
        [1],
        [None],
        ["MatMul", ""],
    ],
)
def test_structured_op_types_rejects_invalid_arrays(op_types):
    with pytest.raises(ValueError, match="weights.op_types"):
        WeightsConfig.from_dict({"op_types": op_types})


@pytest.mark.parametrize("op_types", [["MatMul"], ["Gather"], ["MatMul", "Gather"]])
def test_structured_op_types_accepts_supported_arrays(op_types):
    config = WeightsConfig.from_dict({"op_types": op_types})
    assert config.op_types == tuple(op_types)
    assert WeightsConfig.from_dict(config.to_dict()) == config


def test_extra_options_accepts_legacy_integer_strings():
    config = QuantConfig.from_extra_options(
        {"accuracy_level": "4", "qmoe_weights_prepacked": "-1", "matmulnbits_weights_prepacked": "2"},
        precision="int4",
    )
    assert config.weights.accuracy_level == 4
    assert config.moe.weights_prepacked == -1
    assert config.runtime.matmulnbits_weights_prepacked == 2


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


def test_override_exclusion_requires_exact_name():
    with pytest.raises(ValueError, match="require an exact node name"):
        Override.from_dict({"match": {"name_regex": ".*mlp.*"}, "exclude": True})


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
            "weights": {
                "type": "int4",
                "method": "rtn",
                "overrides": [{"match": {"preset": "last_matmul"}, "type": "int8"}],
            },
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
    assert cfg.moe.type == "none"


def test_extra_options_cpu_default_block_size_and_accuracy_level():
    cfg = QuantConfig.from_extra_options({}, precision="int4", execution_provider="cpu")
    assert cfg.moe.block_size == 32
    assert cfg.weights.accuracy_level == 4


def test_extra_options_trt_rtx_default_qmoe_block_size():
    cfg = QuantConfig.from_extra_options({}, precision="int4", execution_provider="trt-rtx")
    assert cfg.moe.block_size == 128


def test_extra_options_legacy_rtn_last_alias():
    cfg = QuantConfig.from_extra_options({"algo_config": "rtn_last"}, precision="int4")
    assert cfg.weights.method == "rtn"
    assert cfg.weights.overrides == [Override(match={"preset": "last_matmul"}, type="int8")]


def test_extra_options_legacy_k_quant_mixed_alias():
    cfg = QuantConfig.from_extra_options({"algo_config": "k_quant_mixed"}, precision="int4")
    assert cfg.weights.method == "k_quant"
    presets = [(o.match["preset"], o.type) for o in cfg.weights.overrides]
    assert presets == [("last_matmul", "int8"), ("mixed_layers", "int8")]


def test_extra_options_matmul_mixed_precision_string():
    cfg = QuantConfig.from_extra_options(
        {"algo_config": "k_quant", "matmul_mixed_precision": "last_matmul:int8,linear_attn:int4"},
        precision="int4",
    )
    assert cfg.weights.method == "k_quant"
    presets = {o.match["preset"]: o.type for o in cfg.weights.overrides}
    assert presets == {"last_matmul": "int8", "linear_attn": "int4"}


def test_extra_options_explicit_mixed_precision_overrides_alias_default():
    cfg = QuantConfig.from_extra_options(
        {"algo_config": "k_quant_last", "matmul_mixed_precision": "last_matmul:int4"},
        precision="int4",
    )
    presets = {o.match["preset"]: o.type for o in cfg.weights.overrides}
    assert presets == {"last_matmul": "int4"}


def test_extra_options_nodes_to_exclude_become_overrides():
    cfg = QuantConfig.from_extra_options({"nodes_to_exclude": ["/model/embed_tokens/Gather"]}, precision="int4")
    excludes = [o for o in cfg.weights.overrides if o.exclude]
    assert excludes == [Override(match={"name": "/model/embed_tokens/Gather"}, exclude=True)]
    assert cfg.legacy_nodes_to_exclude == frozenset({"/model/embed_tokens/Gather"})


def test_extra_options_nodes_to_exclude_precede_mixed_precision_presets():
    # Legacy nodes_to_exclude is unconditional, so it must win first-match resolution
    # against a preset that selects the same node.
    cfg = QuantConfig.from_extra_options(
        {"matmul_mixed_precision": "last_matmul:int8", "nodes_to_exclude": ["/lm_head/MatMul"]},
        precision="int4",
    )
    assert cfg.weights.overrides[0] == Override(match={"name": "/lm_head/MatMul"}, exclude=True)
    assert cfg.weights.overrides[1].match == {"preset": "last_matmul"}


def test_extra_options_moe_quant_type_and_use_8bits_moe():
    assert QuantConfig.from_extra_options({"moe_quant_type": "mxfp4"}, precision="int4").moe.type == "mxfp4"
    # Deprecated use_8bits_moe maps to int8 when moe_quant_type is absent.
    assert QuantConfig.from_extra_options({"use_8bits_moe": True}, precision="int4").moe.type == "int8"
    # moe_quant_type wins over the deprecated flag.
    cfg = QuantConfig.from_extra_options({"use_8bits_moe": True, "moe_quant_type": "int4"}, precision="int4")
    assert cfg.moe.type == "int4"


def test_extra_options_int8_precision_defaults_moe_to_int8():
    # int8 precision quantizes MoE experts to 8-bit to match the dense weights.
    assert QuantConfig.from_extra_options({}, precision="int8").moe.type == "int8"
    assert QuantConfig.from_extra_options({}, precision="int8").weights.type == "int8"
    # An explicit moe_quant_type still wins for int8 precision.
    assert QuantConfig.from_extra_options({"moe_quant_type": "int4"}, precision="int8").moe.type == "int4"


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
