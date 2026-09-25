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


def test_unsigned_weight_type_defaults_to_asymmetric_quantization():
    effective = normalize_builder_config(
        None,
        "cuda",
        target_options={"quant_config": {"weights": {"type": "uint4"}}},
    )

    assert effective.extra_options["_quant_config"].weights.symmetric is False
    assert effective.extra_options["is_symmetric"] is False


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


@pytest.mark.parametrize(
    "precision,provider,legacy_options,expected_io_dtype",
    [
        ("int4", "cpu", {}, "fp32"),
        ("int8", "cpu", {}, "fp32"),
        ("int4", "cuda", {"use_cuda_bf16": "true"}, "bf16"),
        ("int4", "webgpu", {"use_webgpu_fp32": "true"}, "fp32"),
    ],
)
def test_structured_defaults_preserve_provider_aware_io_dtype(precision, provider, legacy_options, expected_io_dtype):
    effective = normalize_builder_config(
        precision,
        provider,
        legacy_options,
        builder_config_version=2,
        target_options={},
    )

    assert effective.target_options["quant_config"]["io_dtype"] == expected_io_dtype


def test_explicit_target_checkpoint_policy_is_rejected():
    with pytest.raises(ValueError, match="checkpoint_policy is not supported"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"quant_config": {"checkpoint_policy": "preserve"}},
        )


def test_dense_target_rejects_weight_overrides():
    with pytest.raises(ValueError, match="weight overrides are not supported when weights.type=none"):
        normalize_builder_config(
            None,
            "cuda",
            target_options={
                "quant_config": {
                    "io_dtype": "bf16",
                    "weights": {
                        "type": "none",
                        "overrides": [{"match": {"name": "/model/a/MatMul"}, "type": "int8"}],
                    },
                }
            },
        )


def test_structured_target_overrides_legacy_alias():
    with pytest.warns(UserWarning, match="weights.block_size overrides"):
        effective = normalize_builder_config(
            "int4",
            "cuda",
            {"block_size": 32},
            target_options={"quant_config": {"weights": {"block_size": 128}}},
        )
    assert effective.target_options["quant_config"]["weights"]["block_size"] == 128


@pytest.mark.parametrize(
    "legacy_options",
    [
        {"moe_quant_type": "int8"},
        {"use_8bits_moe": True},
    ],
)
def test_structured_dense_weights_preserve_explicit_legacy_moe_policy(legacy_options):
    effective = normalize_builder_config(
        "int4",
        "cuda",
        legacy_options,
        target_options={"quant_config": {"weights": {"type": "int4"}}},
    )
    assert effective.target_options["quant_config"]["moe"]["type"] == "int8"


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
    assert effective.extra_options["_quant_config"].legacy_nodes_to_exclude == frozenset(
        {"/model/a/MatMul", "/model/b/MatMul"}
    )


def test_structured_exclusions_replace_legacy_exclusion_provenance():
    node_name = "/model/a/MatMul"
    effective = normalize_builder_config(
        "int4",
        "cuda",
        {"nodes_to_exclude": node_name},
        target_options={
            "quant_config": {
                "weights": {
                    "overrides": [{"match": {"name": node_name}, "exclude": True}],
                }
            }
        },
    )

    assert effective.extra_options["_quant_config"].legacy_nodes_to_exclude == frozenset()


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


def test_dflash2_rejects_unsupported_checkpoint_policy(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_policy other than 'preserve' is not supported"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": {"checkpoint_policy": "requantize"},
            },
        )


@pytest.mark.parametrize("block_size", [0, "per_channel", 17])
def test_dflash2_rejects_unsupported_body_block_size(tmp_path, block_size):
    with pytest.raises(ValueError, match="integer weights.block_size must be one of"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": {"weights": {"type": "int4", "block_size": block_size}},
            },
        )


def test_dflash2_accepts_prepacked_body_weights_on_cuda(tmp_path):
    effective = normalize_builder_config(
        "int4",
        "cuda",
        target_options={"attention": {"implementation": "paged"}},
        drafter_options={
            "drafter_type": "dflash2",
            "path": make_drafter_checkpoint(tmp_path),
            "quant_config": {
                "weights": {"type": "int4", "block_size": 32},
                "format": {"matmulnbits_weights_prepacked": 1},
            },
        },
    )

    assert effective.drafter_options["quant_config"]["format"]["matmulnbits_weights_prepacked"] == 1


def test_dflash2_rejects_prepacked_body_weights_off_cuda(tmp_path):
    with pytest.raises(ValueError, match="prepacked MatMulNBits weights are supported only on CUDA"):
        normalize_builder_config(
            "int4",
            "cpu",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": {
                    "weights": {"type": "int4", "block_size": 32},
                    "format": {"matmulnbits_weights_prepacked": 1},
                },
            },
        )


@pytest.mark.parametrize(
    "quant_config",
    [
        {"weights": {"type": "none", "block_size": 64}},
        {"weights": {"type": "none", "method": "rtn"}},
        {"weights": {"type": "none", "overrides": [{"match": {"name": "/model/a/MatMul"}, "exclude": True}]}},
        {"format": {"use_qdq": True}},
    ],
)
def test_dspark_rejects_unconsumed_quantization_policy(tmp_path, quant_config):
    with pytest.raises(ValueError, match="DSpark quantization settings"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dspark",
                "path": make_drafter_checkpoint(tmp_path),
                "quant_config": quant_config,
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


def test_structured_mtp_rejects_legacy_exclusion():
    with pytest.raises(ValueError, match="drafter_type=mtp conflicts with legacy extra_options.exclude_mtp"):
        normalize_builder_config(
            "int4",
            "cuda",
            {"exclude_mtp": True},
            drafter_options={"drafter_type": "mtp"},
        )


@pytest.mark.parametrize(
    "precision,provider,legacy_options,expected_io_dtype",
    [
        ("int4", "cuda", {}, "fp16"),
        ("int4", "cuda", {"use_cuda_bf16": "true"}, "bf16"),
        ("fp32", "cpu", {}, "fp32"),
    ],
)
def test_structured_mtp_inherits_target_io_dtype(precision, provider, legacy_options, expected_io_dtype):
    effective = normalize_builder_config(
        precision,
        provider,
        legacy_options,
        drafter_options={"drafter_type": "mtp"},
    )
    assert effective.drafter_options["quant_config"]["io_dtype"] == expected_io_dtype
    assert effective.extra_options["mtp_quant_config"].io_dtype == expected_io_dtype


def test_structured_mtp_rejects_io_dtype_that_differs_from_the_target():
    with pytest.raises(ValueError, match="MTP io_dtype must match the target io_dtype 'fp16'"):
        normalize_builder_config(
            "int4",
            "cuda",
            drafter_options={"drafter_type": "mtp", "quant_config": {"io_dtype": "bf16"}},
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


@pytest.mark.parametrize(
    "model_config,error",
    [
        (
            type("Config", (), {"architectures": ["Qwen3_5ForConditionalGeneration"], "mtp_num_hidden_layers": 0})(),
            "checkpoint with an MTP head",
        ),
        (
            type("Config", (), {"architectures": ["LlamaForCausalLM"], "mtp_num_hidden_layers": 1})(),
            "supported Qwen3.5 architecture",
        ),
    ],
)
def test_explicit_mtp_requires_supported_checkpoint(model_config, error):
    effective = normalize_builder_config(
        "int4",
        "cuda",
        drafter_options={"drafter_type": "mtp"},
    )

    with pytest.raises(ValueError, match=error):
        validate_model_dependent_config(effective, model_config)


def test_explicit_mtp_accepts_supported_checkpoint():
    effective = normalize_builder_config(
        "int4",
        "cuda",
        drafter_options={"drafter_type": "mtp"},
    )
    model_config = type(
        "Config",
        (),
        {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "mtp_num_hidden_layers": 1},
    )()

    validate_model_dependent_config(effective, model_config)


def test_explicit_unquantized_moe_is_not_copied_to_legacy_options():
    with pytest.warns(UserWarning, match="moe.type overrides legacy"):
        effective = normalize_builder_config(
            "int4",
            "cuda",
            {"moe_quant_type": "int8", "use_8bits_moe": True},
            target_options={"quant_config": {"moe": {"type": "none"}}},
        )

    assert effective.extra_options["_quant_config"].moe.type == "none"
    assert "moe_quant_type" not in effective.extra_options
    assert "use_8bits_moe" not in effective.extra_options


def test_checkpoint_moe_policy_updates_implicit_structured_config():
    effective = normalize_builder_config(
        "bf16",
        "cuda",
        target_options={"quant_config": {"weights": {"type": "none"}}},
    )
    effective.extra_options["moe_quant_type"] = "nvfp4"

    validate_model_dependent_config(effective, type("Config", (), {"num_experts": 8})())

    assert effective.extra_options["_quant_config"].moe.type == "nvfp4"
    assert effective.extra_options["_quant_config"].moe.block_size == 16
    assert effective.target_options["quant_config"]["moe"]["type"] == "nvfp4"


def test_checkpoint_moe_policy_rejects_conflicting_explicit_type():
    effective = normalize_builder_config(
        "bf16",
        "cuda",
        target_options={"quant_config": {"moe": {"type": "none"}}},
    )
    effective.extra_options["moe_quant_type"] = "nvfp4"

    with pytest.raises(ValueError, match="conflicts with checkpoint moe_quant_type"):
        validate_model_dependent_config(effective, type("Config", (), {"num_experts": 8})())

    assert effective.extra_options["_quant_config"].moe.type == "none"
    assert effective.target_options["quant_config"]["moe"]["type"] == "none"


def test_speculative_layers_flatten_in_order():
    effective = normalize_builder_config(
        "int4",
        "cuda",
        target_options={"attention": {"implementation": "paged"}},
        speculative_options={"aux_hidden_state_layers": [6, 20, 34]},
    )
    assert effective.extra_options["aux_hidden_state_layers"] == "6,20,34"


@pytest.mark.parametrize("value", [1.5, True, "4"])
def test_structured_state_update_capacity_must_be_an_integer(value):
    with pytest.raises(ValueError, match="state_update_capacity must be an integer"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            speculative_options={"state_update_capacity": value},
        )


@pytest.mark.parametrize("value", [256.0, True, "256"])
def test_structured_paged_block_size_must_be_an_integer(value):
    with pytest.raises(ValueError, match="attention.paged.block_size must be an integer"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged", "paged": {"block_size": value}}},
        )


@pytest.mark.parametrize("value", [3.5, True, "3"])
def test_structured_drafter_integer_fields_reject_coercion(tmp_path, value):
    with pytest.raises(ValueError, match="num_draft_tokens must be an integer"):
        normalize_builder_config(
            "int4",
            "cuda",
            target_options={"attention": {"implementation": "paged"}},
            drafter_options={
                "drafter_type": "dflash2",
                "path": make_drafter_checkpoint(tmp_path),
                "num_draft_tokens": value,
            },
        )


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
        "model": {"decoder": {"session_options": {"provider_options": [{"CUDA": {"gpu_mem_limit": "1024"}}]}}},
        "engine": {"dynamic_batching": {"num_blocks": 128}},
        "search": {"top_k": 1},
    }
    merged = apply_runtime_config(generated, runtime)
    assert merged["model"]["decoder"]["filename"] == "model.onnx"
    assert merged["model"]["decoder"]["session_options"]["provider_options"] == [
        {"CUDA": {"required": "1", "gpu_mem_limit": "1024"}}
    ]
    assert merged["engine"]["dynamic_batching"] == {"block_size": 256, "num_blocks": 128}
    assert merged["search"]["top_k"] == 1
    assert generated == original


def test_runtime_adds_config_only_profile():
    generated = {
        "model": {
            "context_length": 262144,
            "decoder": {"filename": "model.onnx"},
            "dflash2": {"num_draft_tokens": 7},
        },
        "engine": {"dynamic_batching": {"num_blocks": 800, "max_batch_size": 1}},
        "search": {"max_length": 200001, "chunk_size": 512},
        "speculative": {"max_draft_tokens": 7},
    }
    profile = {
        "id": "32gib",
        "eligibility": {"minimum_total_device_memory_bytes": 34359738368},
        "overlay": {
            "engine": {"dynamic_batching": {"num_blocks": 928, "max_batch_size": 8}},
            "search": {"chunk_size": 256},
            "speculative": {"max_draft_tokens": 6},
        },
    }

    updated = apply_runtime_config(generated, {"runtime_profiles": [profile]})

    assert updated["runtime_profiles"] == [profile]
    assert updated["model"]["decoder"]["filename"] == "model.onnx"
    assert updated["engine"]["dynamic_batching"]["num_blocks"] == 800


@pytest.mark.parametrize("in_profile", [False, True])
@pytest.mark.parametrize(
    "engine,field,value,error",
    [
        ({"static_batching": {}}, "num_blocks", 16, "absent dynamic_batching"),
        ({"dynamic_batching": {}}, "num_blocks", 2**31, "at most 2147483647"),
        ({"dynamic_batching": {}}, "max_scheduled_tokens", 2**31, "at most 2147483647"),
        ({"dynamic_batching": {}}, "max_batch_size", 257, "at most 256"),
        ({"dynamic_batching": {}}, "num_blocks", True, "positive integer"),
    ],
)
def test_runtime_batching_validation_is_shared_with_profiles(in_profile, engine, field, value, error):
    generated = {"model": {"decoder": {}}, "engine": engine}
    runtime = {"engine": {"dynamic_batching": {field: value}}}
    if in_profile:
        runtime = {
            "runtime_profiles": [
                {"id": "gpu", "eligibility": {"minimum_total_device_memory_bytes": 1}, "overlay": runtime}
            ]
        }

    with pytest.raises(ValueError, match=error):
        apply_runtime_config(generated, runtime)


@pytest.mark.parametrize("in_profile", [False, True])
@pytest.mark.parametrize(
    "model,max_draft_tokens,error",
    [
        ({"decoder": {}}, 4, "absent speculative configuration"),
        ({"decoder": {}, "dflash2": {"num_draft_tokens": 4}}, 5, "exported drafter/state capacity"),
        (
            {"decoder": {"state_update_capacity": 3}, "dspark": {"num_draft_tokens": 7}},
            4,
            "exported drafter/state capacity",
        ),
        ({"decoder": {}, "dflash2": {"num_draft_tokens": 4}}, 4, None),
        ({"decoder": {"state_update_capacity": 3}, "dspark": {"num_draft_tokens": 7}}, 3, None),
        ({"decoder": {}, "mtp": {}}, 16, None),
    ],
)
def test_runtime_draft_capacity_validation_is_shared_with_profiles(in_profile, model, max_draft_tokens, error):
    generated = {"model": model}
    runtime = {"speculative": {"max_draft_tokens": max_draft_tokens}}
    if in_profile:
        runtime = {
            "runtime_profiles": [
                {"id": "gpu", "eligibility": {"minimum_total_device_memory_bytes": 1}, "overlay": runtime}
            ]
        }

    if error:
        with pytest.raises(ValueError, match=error):
            apply_runtime_config(generated, runtime)
    else:
        updated = apply_runtime_config(generated, runtime)
        for key, value in runtime.items():
            assert updated[key] == value


@pytest.mark.parametrize(
    "overlay",
    [{}, {"model": {}}, {"model": {"decoder": {}}}, {"engine": {"dynamic_batching": {}}}, {"search": {}}],
)
def test_runtime_rejects_profile_without_overlay_fields(overlay):
    generated = {"model": {"decoder": {}}, "engine": {"dynamic_batching": {}}, "search": {}}
    profile = {"id": "empty", "eligibility": {"minimum_total_device_memory_bytes": 1}, "overlay": overlay}
    with pytest.raises(ValueError, match="must contain at least one overlay field"):
        apply_runtime_config(generated, {"runtime_profiles": [profile]})


def test_runtime_rejects_profile_max_length():
    generated = {"model": {"context_length": 4096}, "search": {"max_length": 4096}}
    profile = {
        "id": "request-limit",
        "eligibility": {"minimum_total_device_memory_bytes": 1},
        "overlay": {"search": {"max_length": 2048}},
    }
    with pytest.raises(ValueError, match="unknown runtime_config.runtime_profiles\\[0\\].overlay.search"):
        apply_runtime_config(generated, {"runtime_profiles": [profile]})


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


def test_runtime_adds_speculative_config_for_exported_drafter():
    generated = {
        "model": {
            "decoder": {},
            "dflash2": {"filename": "dflash2.onnx", "num_draft_tokens": 7},
        },
        "search": {},
    }

    updated = apply_runtime_config(generated, {"speculative": {"max_draft_tokens": 7}})

    assert updated["speculative"] == {"max_draft_tokens": 7}


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


@pytest.mark.parametrize(
    "generated_name,runtime_name,generated_options,runtime_options",
    [
        ("cuda", "CUDA", {"enable_cuda_graph": "1", "device_id": "0"}, {"gpu_mem_limit": "1024"}),
        ("WebGPU", "webgpu", {"validationMode": "basic"}, {"validationMode": "disabled"}),
        ("NvTensorRtRtx", "NVTENSORRTRTX", {"enable_cuda_graph": "1"}, {"enable_cuda_graph": "0"}),
    ],
)
def test_runtime_provider_options_merge_case_insensitively(
    generated_name, runtime_name, generated_options, runtime_options
):
    generated = {
        "model": {
            "decoder": {
                "session_options": {"provider_options": [{generated_name: generated_options}]},
            }
        }
    }

    updated = apply_runtime_config(
        generated,
        {"model": {"decoder": {"session_options": {"provider_options": [{runtime_name: runtime_options}]}}}},
    )

    assert updated["model"]["decoder"]["session_options"]["provider_options"] == [
        {generated_name: {**generated_options, **runtime_options}}
    ]


@pytest.mark.parametrize(
    "provider,option_name",
    [
        ("WebGPU", "multiRotaryCacheConcatOffset"),
        ("NvTensorRtRtx", "multi_rotary_cache_concat_offset"),
    ],
)
def test_runtime_rejects_graph_derived_provider_option_changes(provider, option_name):
    generated = {"model": {"decoder": {"session_options": {"provider_options": [{provider: {option_name: "4096"}}]}}}}
    runtime = {"model": {"decoder": {"session_options": {"provider_options": [{provider: {option_name: "1"}}]}}}}

    with pytest.raises(ValueError, match="cannot overwrite graph-derived provider option"):
        apply_runtime_config(generated, runtime)


@pytest.mark.parametrize("runtime_options", [{"unknown": "1"}, {"gpu_mem_limit": 1024}])
def test_runtime_rejects_unsupported_or_non_string_provider_options(runtime_options):
    generated = {
        "model": {"decoder": {"session_options": {"provider_options": [{"CUDA": {"enable_cuda_graph": "0"}}]}}}
    }

    with pytest.raises(ValueError, match="unsupported runtime provider option|must be strings"):
        apply_runtime_config(
            generated,
            {"model": {"decoder": {"session_options": {"provider_options": [{"CUDA": runtime_options}]}}}},
        )


@pytest.mark.parametrize("run_options", ["invalid", [], {"tag": 1}])
def test_runtime_run_options_must_be_an_object_of_strings(run_options):
    generated = {"model": {"decoder": {"session_options": {}}}}

    with pytest.raises(ValueError, match="run_options must be an object of strings"):
        apply_runtime_config(generated, {"model": {"decoder": {"run_options": run_options}}})


@pytest.mark.parametrize("component_name", ["dflash2", "dspark"])
def test_runtime_block_drafter_cannot_disable_execution_provider_synchronization(component_name):
    generated = {"model": {component_name: {"session_options": {}}}}
    runtime = {
        "model": {
            component_name: {
                "run_options": {"disable_synchronize_execution_providers": "1"},
            }
        }
    }

    with pytest.raises(ValueError, match="cannot disable execution-provider synchronization"):
        apply_runtime_config(generated, runtime)


@pytest.mark.parametrize("max_length", ["8192", 0, True])
def test_runtime_search_max_length_must_be_a_positive_integer(max_length):
    generated = {"model": {"context_length": 4096}}

    with pytest.raises(ValueError, match="max_length must be an integer between 1 and 2147483647"):
        apply_runtime_config(generated, {"search": {"max_length": max_length}})


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("do_sample", "false", "must be a boolean"),
        ("top_k", "7", "must be an integer"),
        ("top_k", -1, "must be an integer between 0 and 2147483647"),
        ("top_k", 2**31, "must be an integer between 0 and 2147483647"),
        ("top_p", "0.9", "must be a finite number"),
        ("top_p", 1.1, "must be at most 1"),
        ("temperature", -0.1, "must be at least 0"),
        ("repetition_penalty", 0, "must be greater than 0"),
        ("batch_size", 33, "must be an integer between 1 and 32"),
    ],
)
def test_runtime_search_validates_parser_types_and_ranges(field, value, error):
    generated = {"model": {"context_length": 4096}, "search": {"max_length": 4096}}

    with pytest.raises(ValueError, match=error):
        apply_runtime_config(generated, {"search": {field: value}})


def test_runtime_search_max_length_cannot_exceed_context_length():
    generated = {"model": {"context_length": 4096}, "search": {"max_length": 4096}}

    with pytest.raises(ValueError, match="exceeds the exported model context_length"):
        apply_runtime_config(generated, {"search": {"max_length": 8192}})


def test_runtime_search_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown runtime_config.search field"):
        apply_runtime_config({"model": {"context_length": 4096}}, {"search": {"unsupported": 1}})


def test_runtime_allows_block_drafter_to_repeat_decoder_provider_options():
    generated = {
        "model": {
            "decoder": {"session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "0"}}]}},
            "dflash2": {"session_options": {"ep.cuda.fpa_intb_gemm": "0"}},
        }
    }
    runtime = {"model": {"dflash2": {"session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]}}}}

    updated = apply_runtime_config(generated, runtime)

    assert updated["model"]["dflash2"]["session_options"]["provider_options"] == [{"cuda": {"enable_cuda_graph": "1"}}]


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


def test_runtime_accepts_mtp_without_static_draft_capacity():
    generated = {
        "model": {
            "decoder": {"session_options": {}},
            "mtp": {"filename": "mtp.onnx"},
        }
    }

    updated = apply_runtime_config(generated, {"speculative": {"max_draft_tokens": 8}})

    assert updated["speculative"] == {"max_draft_tokens": 8}


@pytest.mark.parametrize(
    "mtp_options",
    [
        {"session_options": {"intra_op_num_threads": 4}},
        {"run_options": {"tag": "mtp"}},
        {"session_options": {"log_id": "mtp"}, "run_options": {"tag": "mtp"}},
    ],
)
def test_runtime_accepts_mtp_options_without_generated_session_options(mtp_options):
    generated = {"model": {"decoder": {"session_options": {}}, "mtp": {"filename": "mtp.onnx"}}}
    updated = apply_runtime_config(generated, {"model": {"mtp": mtp_options}})

    assert updated["model"]["mtp"] == {"filename": "mtp.onnx", **mtp_options}
    assert generated["model"]["mtp"] == {"filename": "mtp.onnx"}


@pytest.mark.parametrize("mtp_session", [None, {}, {"provider_options": [{"CUDA": {"gpu_mem_limit": "2048"}}]}])
def test_runtime_mtp_provider_options_use_component_or_inherited_providers(mtp_session):
    generated = {
        "model": {
            "decoder": {"session_options": {"provider_options": [{"CUDA": {"gpu_mem_limit": "1024"}}]}},
            "mtp": {"filename": "mtp.onnx"},
        }
    }
    if mtp_session is not None:
        generated["model"]["mtp"]["session_options"] = mtp_session
    original = copy.deepcopy(generated)
    runtime = {"model": {"mtp": {"session_options": {"provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]}}}}
    original_runtime = copy.deepcopy(runtime)

    updated = apply_runtime_config(generated, runtime)

    assert updated["model"]["mtp"]["session_options"]["provider_options"] == [
        {"CUDA": {"gpu_mem_limit": "2048" if mtp_session else "1024", "enable_cuda_graph": "1"}}
    ]
    assert generated == original
    assert runtime == original_runtime


@pytest.mark.parametrize(
    "runtime_providers,error",
    [
        ([{"CPU": {}}], "cannot change execution providers"),
        ([], "cannot change execution providers"),
        ([{"webgpu": {"multiRotaryCacheConcatOffset": "0"}}], "graph-derived provider option"),
    ],
)
def test_runtime_mtp_rejects_incompatible_provider_overrides(runtime_providers, error):
    generated = {
        "model": {
            "decoder": {
                "session_options": {"provider_options": [{"WebGPU": {"multiRotaryCacheConcatOffset": "4096"}}]}
            },
            "mtp": {"filename": "mtp.onnx"},
        }
    }
    with pytest.raises(ValueError, match=error):
        apply_runtime_config(
            generated, {"model": {"mtp": {"session_options": {"provider_options": runtime_providers}}}}
        )


@pytest.mark.parametrize(
    "session_options",
    [
        {"intra_op_num_threads": "4"},
        {"inter_op_num_threads": 1.5},
        {"log_severity_level": True},
        {"log_verbosity_level": 2**31},
        {"enable_cpu_mem_arena": "true"},
        {"enable_mem_pattern": 1},
        {"graph_optimization_level": "ORT_ENABLE_EVERYTHING"},
        {"enable_profiling": 1},
        {"ep.cuda.fpa_intb_gemm": 1},
    ],
)
def test_runtime_rejects_invalid_session_option_values(session_options):
    generated = {"model": {"decoder": {"session_options": {}}}}

    with pytest.raises(ValueError, match="session_options"):
        apply_runtime_config(generated, {"model": {"decoder": {"session_options": session_options}}})


def test_runtime_accepts_parser_typed_session_options():
    generated = {"model": {"decoder": {"session_options": {}}}}
    session_options = {
        "intra_op_num_threads": 4,
        "enable_cpu_mem_arena": False,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "log_id": "decoder",
        "ep.cuda.fpa_intb_gemm": "1",
    }

    updated = apply_runtime_config(generated, {"model": {"decoder": {"session_options": session_options}}})

    assert updated["model"]["decoder"]["session_options"] == session_options
