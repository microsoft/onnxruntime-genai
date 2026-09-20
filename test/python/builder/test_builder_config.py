# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
sys.path.insert(0, str(MODELS_DIR))

from builder_config import (  # noqa: E402
    apply_runtime_config,
    load_json_object,
    normalize_builder_config,
    validate_model_dependent_config,
)


def test_legacy_configuration_preserves_options():
    effective = normalize_builder_config("int4", "cuda", {"block_size": "64"})
    assert effective.version == 1
    assert effective.extra_options == {"block_size": "64"}


def test_structured_fields_select_version_two():
    effective = normalize_builder_config(
        None,
        "cuda",
        target_options={
            "quant_config": {
                "io_dtype": "bf16",
                "weights": {"type": "int4", "block_size": 64},
            }
        },
    )
    assert effective.version == 2
    assert effective.precision == "int4"
    assert effective.execution_provider == "cuda"
    assert effective.extra_options["_target_io_dtype"] == "bf16"
    assert effective.target_options["quant_config"]["moe"]["type"] == "int4"


def test_version_one_rejects_structured_fields():
    with pytest.raises(ValueError, match="version=1 cannot be combined"):
        normalize_builder_config("int4", "cuda", builder_config_version=1, target_options={})


def test_unsupported_version_is_rejected():
    with pytest.raises(ValueError, match="unsupported builder_config_version=3"):
        normalize_builder_config("int4", "cuda", builder_config_version=3)


@pytest.mark.parametrize(
    "provider,accuracy_level,moe_block_size",
    [("cpu", 4, 32), ("webgpu", 4, 32), ("cuda", 0, 32), ("NvTensorRtRtx", 0, 128)],
)
def test_provider_defaults_are_recorded(provider, accuracy_level, moe_block_size):
    effective = normalize_builder_config(
        "int4",
        provider,
        builder_config_version=2,
        target_options={},
    )
    quant = effective.target_options["quant_config"]
    assert quant["weights"]["accuracy_level"] == accuracy_level
    assert quant["moe"]["block_size"] == moe_block_size
    assert quant["format"]["use_qdq"] is (provider == "NvTensorRtRtx")
    assert effective.execution_provider == ("trt-rtx" if provider == "NvTensorRtRtx" else provider)


def test_structured_target_overrides_legacy_alias():
    with pytest.warns(UserWarning, match="weights.block_size overrides"):
        effective = normalize_builder_config(
            "int4",
            "cuda",
            {"block_size": 32},
            target_options={"quant_config": {"weights": {"block_size": 128}}},
        )
    assert effective.target_options["quant_config"]["weights"]["block_size"] == 128


def test_structured_config_normalizes_legacy_quantization_lists():
    effective = normalize_builder_config(
        "int4",
        "cuda",
        {
            "op_types_to_quantize": "MatMul/Gather",
            "nodes_to_exclude": "/model/a/MatMul,/model/b/MatMul",
        },
        target_options={},
    )
    weights = effective.target_options["quant_config"]["weights"]
    assert weights["op_types"] == ["MatMul", "Gather"]
    assert [override["match"]["name"] for override in weights["overrides"]] == [
        "/model/a/MatMul",
        "/model/b/MatMul",
    ]


def make_drafter_checkpoint(tmp_path):
    path = tmp_path / "drafter"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"dflash_config": {"target_layer_ids": [1, 3]}}))
    return str(path)


def test_dflash2_policy_is_independent_from_target(tmp_path):
    effective = normalize_builder_config(
        "int4",
        "cuda",
        target_options={
            "quant_config": {"weights": {"type": "int4", "block_size": 128}},
            "attention": {"implementation": "paged"},
        },
        drafter_options={"drafter_type": "dflash2", "path": make_drafter_checkpoint(tmp_path)},
    )
    drafter_quant = effective.drafter_options["quant_config"]
    assert drafter_quant["io_dtype"] == "bf16"
    assert drafter_quant["weights"]["type"] == "none"
    assert drafter_quant["weights"]["block_size"] == 32
    assert effective.extra_options["aux_hidden_state_layers"] == "2,4"


def test_dflash2_rejects_unsupported_body_dtype(tmp_path):
    with pytest.raises(ValueError, match="body io_dtype must be bf16"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": {"io_dtype": "fp16"},
            },
        )


@pytest.mark.parametrize("field", ["accuracy_level", "op_types", "overrides"])
def test_dflash2_rejects_unconsumed_weight_policy(tmp_path, field):
    values = {
        "accuracy_level": 4,
        "op_types": ["MatMul"],
        "overrides": [{"match": {"name": "/model/a/MatMul"}, "exclude": True}],
    }
    with pytest.raises(ValueError, match=f"weights.{field} is not supported"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": {"weights": {field: values[field]}},
            },
        )


@pytest.mark.parametrize(
    "drafter_options,legacy_options",
    [
        ({"drafter_type": "none"}, {"dflash2_path": "legacy"}),
        ({"drafter_type": "mtp"}, {"dspark_path": "legacy"}),
        ({"drafter_type": "dflash2", "path": "structured"}, {"dspark_path": "legacy"}),
    ],
)
def test_structured_drafter_rejects_conflicting_legacy_selection(tmp_path, drafter_options, legacy_options):
    if drafter_options.get("path"):
        drafter_options["path"] = make_drafter_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="conflicts with legacy drafter selection"):
        normalize_builder_config(
            "int4",
            "cuda",
            legacy_options,
            target_options={"attention": {"implementation": "paged"}},
            drafter_options=drafter_options,
        )


def test_block_drafter_rejects_windowed_kv_cache(tmp_path):
    with pytest.raises(ValueError, match="windowed KV cache is not supported"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "attention": {"kv_cache": {"windowed": True}},
            },
        )


def test_search_alone_is_preserved_in_legacy_configuration():
    effective = normalize_builder_config("int4", "cuda", search={"top_k": 7})
    assert effective.version == 1
    assert effective.runtime_config == {"search": {"top_k": 7}}


def test_unquantized_moe_requires_explicit_expert_policy():
    effective = normalize_builder_config(
        None,
        "cuda",
        target_options={"quant_config": {"io_dtype": "bf16", "weights": {"type": "none"}}},
    )
    model_config = type("Config", (), {"num_experts": 8})()
    with pytest.raises(ValueError, match="moe.type is required"):
        validate_model_dependent_config(effective, model_config)

    explicit = normalize_builder_config(
        None,
        "cuda",
        target_options={
            "quant_config": {
                "io_dtype": "bf16",
                "weights": {"type": "none"},
                "moe": {"type": "none"},
            }
        },
    )
    validate_model_dependent_config(explicit, model_config)


def test_speculative_layers_flatten_in_order():
    effective = normalize_builder_config(
        "int4",
        "cuda",
        target_options={"attention": {"implementation": "paged"}},
        speculative_options={"aux_hidden_state_layers": [6, 20, 34]},
    )
    assert effective.extra_options["aux_hidden_state_layers"] == "6,20,34"


def test_json_loader_rejects_duplicates_and_null(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"search": {}, "search": {}}')
    with pytest.raises(ValueError, match="duplicate JSON key 'search'"):
        load_json_object(duplicate, "runtime_config")
    with pytest.raises(ValueError, match="must not contain null"):
        load_json_object('{"search": {"top_k": null}}', "runtime_config")


def test_runtime_merge_replaces_arrays_and_fixed_allocation():
    generated = {
        "model": {
            "decoder": {
                "filename": "model.onnx",
                "session_options": {"provider_options": [{"CUDA": {"required": "1"}}]},
            }
        },
        "engine": {"dynamic_batching": {"block_size": 256, "gpu_utilization_factor": 0.6}},
        "search": {"top_k": 50},
    }
    original = copy.deepcopy(generated)
    runtime = {
        "model": {"decoder": {"session_options": {"provider_options": [{"CUDA": {"new": "1"}}]}}},
        "engine": {"dynamic_batching": {"num_blocks": 128}},
        "search": {"top_k": 1},
    }
    merged = apply_runtime_config(generated, runtime)
    assert merged["model"]["decoder"]["filename"] == "model.onnx"
    assert merged["model"]["decoder"]["session_options"]["provider_options"] == [{"CUDA": {"new": "1"}}]
    assert merged["engine"]["dynamic_batching"] == {"block_size": 256, "num_blocks": 128}
    assert merged["search"]["top_k"] == 1
    assert generated == original


def test_runtime_rejects_protected_and_absent_components():
    generated = {"model": {"decoder": {}}, "engine": {"dynamic_batching": {"block_size": 256}}}
    with pytest.raises(ValueError, match="unknown runtime_config.engine.dynamic_batching"):
        apply_runtime_config(generated, {"engine": {"dynamic_batching": {"block_size": 16}}})
    with pytest.raises(ValueError, match="absent model component 'dflash2'"):
        apply_runtime_config(generated, {"model": {"dflash2": {"session_options": {}}}})


def test_runtime_rejects_absent_engine_and_speculative_capabilities():
    generated = {"model": {"decoder": {}}, "search": {}}
    with pytest.raises(ValueError, match="absent engine configuration"):
        apply_runtime_config(generated, {"engine": {"dynamic_batching": {"num_blocks": 16}}})
    with pytest.raises(ValueError, match="absent speculative configuration"):
        apply_runtime_config(generated, {"speculative": {"max_draft_tokens": 4}})


@pytest.mark.parametrize("value", [True, 0, 17, 1.5, "4"])
def test_runtime_rejects_invalid_max_draft_tokens(value):
    generated = {"model": {"decoder": {}}, "speculative": {"max_draft_tokens": 4}}
    with pytest.raises(ValueError, match="max_draft_tokens must be an integer between 1 and 16"):
        apply_runtime_config(generated, {"speculative": {"max_draft_tokens": value}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_batch_size", 0),
        ("max_scheduled_tokens", -1),
        ("num_blocks", 0),
        ("gpu_utilization_factor", 0),
        ("gpu_utilization_factor", 1.1),
    ],
)
def test_runtime_rejects_invalid_dynamic_batching_values(field, value):
    generated = {"model": {"decoder": {}}, "engine": {"dynamic_batching": {"block_size": 256}}}
    with pytest.raises(ValueError, match=field):
        apply_runtime_config(generated, {"engine": {"dynamic_batching": {field: value}}})


def test_runtime_rejects_overwriting_required_session_option():
    generated = {
        "model": {
            "dflash2": {
                "filename": "dflash2.onnx",
                "session_options": {"ep.cuda.fpa_intb_gemm": "0"},
            }
        }
    }
    with pytest.raises(ValueError, match="required session option"):
        apply_runtime_config(
            generated,
            {"model": {"dflash2": {"session_options": {"ep.cuda.fpa_intb_gemm": "1"}}}},
        )


def test_runtime_rejects_provider_changes():
    generated = {
        "model": {
            "decoder": {
                "session_options": {"provider_options": [{"CUDA": {}}]},
            }
        }
    }
    with pytest.raises(ValueError, match="cannot change execution providers"):
        apply_runtime_config(
            generated,
            {"model": {"decoder": {"session_options": {"provider_options": [{"CPU": {}}]}}}},
        )
    with pytest.raises(ValueError, match="cannot change execution providers"):
        apply_runtime_config(
            generated,
            {"model": {"decoder": {"session_options": {"provider_options": []}}}},
        )


def test_runtime_rejects_non_session_model_members():
    generated = {"model": {"decoder": {"session_options": {}}, "vocab_size": 32000}}
    with pytest.raises(ValueError, match="not a session-bearing component"):
        apply_runtime_config(generated, {"model": {"vocab_size": {"session_options": {}}}})


def test_runtime_rejects_draft_limit_above_exported_capacity():
    generated = {
        "model": {
            "decoder": {"session_options": {}},
            "dflash2": {"session_options": {}, "num_draft_tokens": 4},
        },
        "speculative": {"max_draft_tokens": 4},
    }
    with pytest.raises(ValueError, match="exceeds the exported drafter/state capacity"):
        apply_runtime_config(generated, {"speculative": {"max_draft_tokens": 5}})
