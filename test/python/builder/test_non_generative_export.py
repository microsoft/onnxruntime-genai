# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import math
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from _builder_test_utils import load_builder_module
from onnx.reference import ReferenceEvaluator

clm = load_builder_module("clm")
kev = load_builder_module("kev")
dispatch = load_builder_module("non_generative")


def _tiny_clm_head(seed):
    generator = torch.Generator().manual_seed(seed)
    state = {
        "inp.weight": torch.randn(4, 3, generator=generator),
        "inp.bias": torch.randn(4, generator=generator),
        "out.weight": torch.randn(2, 4, generator=generator),
        "out.bias": torch.randn(2, generator=generator),
    }
    for index in range(1):
        state[f"hidden.{index}.weight"] = torch.randn(4, 4, generator=generator)
        state[f"hidden.{index}.bias"] = torch.randn(4, generator=generator)
        state[f"norms.{index}.weight"] = torch.randn(4, generator=generator)
        state[f"norms.{index}.bias"] = torch.randn(4, generator=generator)
    return state


def _tiny_clm_checkpoint():
    return {
        "state_head": _tiny_clm_head(1),
        "action_head": _tiny_clm_head(2),
        "logit_scale": torch.tensor(math.log(150.0)),
        "cfg": {
            "width": 4,
            "depth": 3,
            "projection_dim": 2,
            "activation": "gelu",
            "layernorm": True,
            "residual": False,
            "model": "Qwen/Qwen3-8B",
            "hidden_size": 3,
        },
        "hidden_size": 3,
        "projection_dim": 2,
    }


def _torch_clm_head(value, state):
    value = F.normalize(value, dim=-1)
    value = F.gelu(F.linear(value, state["inp.weight"], state["inp.bias"]))
    for index in range(1):
        value = F.linear(value, state[f"hidden.{index}.weight"], state[f"hidden.{index}.bias"])
        value = F.layer_norm(
            value,
            (4,),
            state[f"norms.{index}.weight"],
            state[f"norms.{index}.bias"],
            eps=1e-5,
        )
        value = F.gelu(value)
    return F.normalize(F.linear(value, state["out.weight"], state["out.bias"]), dim=-1)


def test_clm_tiny_graph_structure_weights_and_math():
    checkpoint = _tiny_clm_checkpoint()
    model = clm.build_clm_model(checkpoint)
    op_types = [node.op_type for node in model.graph.node]

    assert op_types.count("Gemm") == 6
    assert op_types.count("Gelu") == 4
    assert op_types.count("LayerNormalization") == 2
    assert op_types.count("ReduceL2") == 4
    assert op_types.count("Max") == 4
    assert {item.name for item in model.graph.initializer} >= {
        "state.inp.weight",
        "state.hidden.0.weight",
        "state.norms.0.weight",
        "state.out.weight",
    }

    state_input = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float32)
    action_input = torch.tensor([[-0.1, 0.5, 0.4]], dtype=torch.float32)
    actual = ReferenceEvaluator(model).run(
        None,
        {
            "state_hidden_states": state_input.numpy(),
            "action_hidden_states": action_input.numpy(),
        },
    )
    np.testing.assert_allclose(actual[0], _torch_clm_head(state_input, checkpoint["state_head"]), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        actual[1], _torch_clm_head(action_input, checkpoint["action_head"]), rtol=2e-5, atol=2e-5
    )
    assert actual[2] == np.float32(100.0)
    scaled = ReferenceEvaluator(model).run(
        None,
        {
            "state_hidden_states": (state_input * 7).numpy(),
            "action_hidden_states": (action_input * 3).numpy(),
        },
    )
    np.testing.assert_allclose(scaled[0], actual[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(scaled[1], actual[1], rtol=2e-5, atol=2e-5)


def test_clm_l2_normalization_matches_f_normalize_epsilon_for_zero_output():
    checkpoint = _tiny_clm_checkpoint()
    for state in (checkpoint["state_head"], checkpoint["action_head"]):
        state["out.weight"].zero_()
        state["out.bias"].zero_()
    outputs = ReferenceEvaluator(clm.build_clm_model(checkpoint)).run(
        None,
        {
            "state_hidden_states": np.zeros((1, 3), dtype=np.float32),
            "action_hidden_states": np.zeros((1, 3), dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(outputs[0], np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(outputs[1], np.zeros((1, 2), dtype=np.float32))


def test_clm_logit_scale_is_finite_and_clamped_before_exp():
    checkpoint = _tiny_clm_checkpoint()
    checkpoint["logit_scale"] = torch.tensor(1000.0)
    outputs = ReferenceEvaluator(clm.build_clm_model(checkpoint)).run(
        None,
        {
            "state_hidden_states": np.ones((1, 3), dtype=np.float32),
            "action_hidden_states": np.ones((1, 3), dtype=np.float32),
        },
    )
    assert outputs[2] == np.float32(100.0)

    checkpoint["logit_scale"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="logit_scale must be finite"):
        clm.build_clm_model(checkpoint)


def _tiny_kev_checkpoint():
    generator = torch.Generator().manual_seed(3)
    return {
        "state": {
            "q.weight": torch.randn(2, 3, generator=generator),
            "q.bias": torch.randn(2, generator=generator),
            "k.weight": torch.randn(2, 3, generator=generator),
            "k.bias": torch.randn(2, generator=generator),
        },
        "head_dim": 2,
        "temperature": 2.5,
    }


def _expanded(shape):
    return torch.zeros(1).expand(shape)


def _real_clm_checkpoint():
    head = {
        "inp.weight": _expanded((1536, 4096)),
        "inp.bias": _expanded((1536,)),
        "hidden.0.weight": _expanded((1536, 1536)),
        "hidden.0.bias": _expanded((1536,)),
        "norms.0.weight": _expanded((1536,)),
        "norms.0.bias": _expanded((1536,)),
        "out.weight": _expanded((512, 1536)),
        "out.bias": _expanded((512,)),
    }
    return {
        "state_head": head,
        "action_head": dict(head),
        "logit_scale": torch.tensor(1.0),
        "cfg": {
            "model": "Qwen/Qwen3-8B",
            "hidden_size": 4096,
            "width": 1536,
            "depth": 3,
            "projection_dim": 512,
            "activation": "gelu",
            "layernorm": True,
            "residual": False,
        },
        "hidden_size": 4096,
        "projection_dim": 512,
    }


def _real_kev_checkpoint(state=None):
    policy = kev.kev_policy()
    return {
        "args": {"seed": 17},
        "suite_sha256": "a" * 64,
        "init_source": "trained",
        "temperature_fit": {"method": "holdout"},
        "base": policy["base_model"],
        "head": state
        or {
            "q.weight": _expanded((256, 2560)),
            "q.bias": _expanded((256,)),
            "k.weight": _expanded((256, 2560)),
            "k.bias": _expanded((256,)),
        },
        "base_revision": policy["base_revision"],
        "lora": {"r": 16},
        "head_dim": 256,
        "option_isolation": False,
        "special_embeddings": {},
        "weights_dtype": "float32",
        "temperature": policy["temperature"],
        "holdout": {"questions": 100},
        "lora_placement": ["q_proj", "v_proj"],
    }


def _write_kev_artifact(path, checkpoint=None):
    policy = kev.kev_policy()
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": policy["base_model"],
            }
        )
    )
    (path / "adapter_model.safetensors").write_bytes(b"synthetic")
    torch.save(checkpoint or _real_kev_checkpoint(), path / "head.pt")


def test_kev_tiny_graph_weights_masked_grouped_softmax_math():
    checkpoint = _tiny_kev_checkpoint()
    model = kev.build_kev_model(checkpoint)
    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.5, 1.0]],
            [[0.2, 0.8, -0.1], [0.4, -0.3, 0.9], [0.7, 0.1, 0.6]],
        ]
    )
    decide = np.asarray([0, 1], dtype=np.int64)
    options = np.asarray([[1, 2], [0, 2]], dtype=np.int64)
    mask = np.asarray([[True, False], [True, True]])

    probabilities, scores = ReferenceEvaluator(model).run(
        None,
        {
            "hidden_states": hidden.numpy(),
            "decide_indices": decide,
            "option_indices": options,
            "option_mask": mask,
        },
    )
    state = checkpoint["state"]
    rows = torch.arange(hidden.shape[0])
    q = F.linear(
        hidden[rows, torch.from_numpy(decide)],
        state["q.weight"],
        state["q.bias"],
    )
    k = F.linear(
        hidden[rows[:, None], torch.from_numpy(options)],
        state["k.weight"],
        state["k.bias"],
    )
    expected_scores = (q[:, None, :] * k).sum(-1) / (math.sqrt(2) * 2.5)
    expected_probabilities = torch.softmax(expected_scores.masked_fill(~torch.from_numpy(mask), -torch.inf), dim=-1)

    np.testing.assert_allclose(scores, expected_scores.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(probabilities, expected_probabilities.numpy(), rtol=1e-6, atol=1e-6)
    assert [node.op_type for node in model.graph.node][-3:] == [
        "Where",
        "Softmax",
        "Where",
    ]
    all_masked, _ = ReferenceEvaluator(model).run(
        None,
        {
            "hidden_states": hidden.numpy(),
            "decide_indices": decide,
            "option_indices": options,
            "option_mask": np.zeros_like(mask),
        },
    )
    np.testing.assert_array_equal(all_masked, np.zeros_like(all_masked))


def test_artifact_layout_dispatch_and_ambiguity(tmp_path):
    clm_dir = tmp_path / "clm"
    clm_dir.mkdir()
    torch.save(_tiny_clm_checkpoint(), clm_dir / "checkpoint.pt")
    assert dispatch.detect_model_specific_artifact(clm_dir) == "clm"

    kev_dir = tmp_path / "kev"
    _write_kev_artifact(kev_dir)
    assert dispatch.detect_model_specific_artifact(kev_dir) == "kev"

    torch.save(_tiny_clm_checkpoint(), kev_dir / "checkpoint.pt")
    with pytest.raises(ValueError, match="ambiguous"):
        dispatch.detect_model_specific_artifact(kev_dir)


def test_kev_artifact_requires_adapter_weights_and_lora_type(tmp_path):
    source = tmp_path / "kev"
    _write_kev_artifact(source)
    (source / "adapter_model.safetensors").unlink()
    assert not kev.is_kev_artifact(source)

    (source / "adapter_model.safetensors").write_bytes(b"synthetic")
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "IA3",
                "base_model_name_or_path": kev.kev_policy()["base_model"],
            }
        )
    )
    with pytest.raises(ValueError, match="peft_type must be LORA"):
        kev.load_kev_checkpoint(source)


def test_dispatch_validates_provenance_and_backbone(tmp_path):
    policy = kev.kev_policy()
    source = tmp_path / "kev"
    _write_kev_artifact(source)
    config = types.SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        hidden_size=2560,
    )
    options = types.SimpleNamespace(
        model_source=str(source),
        artifact_revision=policy["adapter_revision"],
        base_revision=policy["base_revision"],
    )
    extra_options = {}

    assert dispatch.configure_model_specific_export(options, config, extra_options) == "kev"
    assert extra_options["adapter_path"] == str(source)
    assert extra_options["exclude_lm_head"] is True

    options.base_revision = "unpinned"
    with pytest.raises(ValueError, match="base_revision"):
        dispatch.validate_model_specific_backbone(options, config)


def test_model_specific_hf_pipeline_keeps_base_tokenizer_and_component_separate(tmp_path):
    policy = kev.kev_policy()
    kev_source = tmp_path / "kev"
    _write_kev_artifact(kev_source)
    kev_options = types.SimpleNamespace(
        model_source=str(kev_source),
        artifact_revision=policy["adapter_revision"],
        base_revision=policy["base_revision"],
    )
    extra_options = {}

    dispatch.prepare_model_specific_hf(kev_options, extra_options, policy["base_model"])

    assert extra_options["base_revision"] == policy["base_revision"]
    assert extra_options["adapter_path"] == str(kev_source)
    assert extra_options["_model_specific_hf"] == {
        "base_model": policy["base_model"],
        "base_revision": policy["base_revision"],
        "tokenizer": policy["base_model"],
        "adapter": str(kev_source),
        "component": str(kev_source),
    }

    clm_source = tmp_path / "clm"
    clm_source.mkdir()
    torch.save(_real_clm_checkpoint(), clm_source / "checkpoint.pt")
    clm_options = types.SimpleNamespace(
        model_source=str(clm_source),
        artifact_revision=clm.clm_artifact_revision(),
        base_revision="caller-selected-qwen3-revision",
    )
    clm_extra = {}
    dispatch.prepare_model_specific_hf(clm_options, clm_extra, "Qwen/Qwen3-8B")
    assert clm_extra["base_revision"] == "caller-selected-qwen3-revision"
    assert clm_extra["_model_specific_hf"]["adapter"] is None
    assert clm_extra["_model_specific_hf"]["tokenizer"] == "Qwen/Qwen3-8B"

    malformed_source = tmp_path / "malformed-clm"
    malformed_source.mkdir()
    torch.save(_tiny_clm_checkpoint(), malformed_source / "checkpoint.pt")
    malformed_options = types.SimpleNamespace(
        model_source=str(malformed_source),
        artifact_revision=clm.clm_artifact_revision(),
        base_revision="caller-selected-qwen3-revision",
    )
    with pytest.raises(ValueError, match="hidden_size"):
        dispatch.prepare_model_specific_hf(malformed_options, {}, "Qwen/Qwen3-8B")


def test_real_released_clm_layout_is_accepted(tmp_path):
    source = tmp_path / "clm"
    source.mkdir()
    torch.save(_real_clm_checkpoint(), source / "checkpoint.pt")

    loaded = clm.load_clm_checkpoint(source)
    clm.validate_head("state", loaded["state_head"], 4096, 1536, 512, 3)
    assert set(loaded["state_head"]) == {
        "inp.weight",
        "inp.bias",
        "hidden.0.weight",
        "hidden.0.bias",
        "norms.0.weight",
        "norms.0.bias",
        "out.weight",
        "out.bias",
    }


def test_real_released_kev_top_level_layout_is_accepted(tmp_path):
    policy = kev.kev_policy()
    source = tmp_path / "kev"
    _write_kev_artifact(source)
    loaded = kev.load_kev_checkpoint(source)
    assert loaded["head_dim"] == 256
    assert loaded["temperature"] == policy["temperature"]
    assert set(loaded["state"]) == set(policy["head_keys"])


def test_controlled_checkpoint_schema_and_shapes_fail_closed(tmp_path):
    clm_dir = tmp_path / "clm"
    clm_dir.mkdir()
    bad_clm = _tiny_clm_checkpoint()
    bad_clm["unexpected"] = torch.tensor(1)
    torch.save(bad_clm, clm_dir / "checkpoint.pt")
    with pytest.raises(ValueError, match="keys must be exactly"):
        clm.load_clm_checkpoint(clm_dir)

    kev_dir = tmp_path / "kev"
    _write_kev_artifact(
        kev_dir,
        _real_kev_checkpoint(
            {
                "q.weight": torch.zeros(256, 10),
                "q.bias": torch.zeros(256),
                "k.weight": _expanded((256, 2560)),
                "k.bias": torch.zeros(256),
            }
        ),
    )
    with pytest.raises(ValueError, match=r"q.weight.*\(256, 2560\)"):
        kev.load_kev_checkpoint(kev_dir)


def test_model_specific_export_rejects_backbone_collision_and_manifest_symlink(
    tmp_path,
):
    source = tmp_path / "clm"
    source.mkdir()
    torch.save(_real_clm_checkpoint(), source / "checkpoint.pt")
    output = tmp_path / "package"
    output.mkdir()
    options = types.SimpleNamespace(
        model_source=str(source),
        artifact_revision=clm.clm_artifact_revision(),
        base_revision="caller-pin",
        backbone_filename="clm_heads.onnx",
    )
    with pytest.raises(ValueError, match="conflicts with model-specific head"):
        dispatch.export_model_specific_components(options, output)

    options.backbone_filename = "backbone.onnx"
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged")
    (output / "component_manifest.json").symlink_to(outside)
    with pytest.raises(ValueError, match="destination contains symlink"):
        dispatch.export_model_specific_components(options, output)
    assert outside.read_text() == "unchanged"
