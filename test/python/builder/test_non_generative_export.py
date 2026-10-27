# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import math
import sys
import types

import numpy as np
import onnx_ir as ir
import pytest
import torch
import torch.nn.functional as F
from _builder_test_utils import load_builder_module
from onnx import checker
from onnx.reference import ReferenceEvaluator

clm = load_builder_module("clm")
kev = load_builder_module("kev")
dispatch = load_builder_module("non_generative")
qwen = load_builder_module("qwen")
base = load_builder_module("base")


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


@pytest.mark.parametrize(
    "builder_class,build_model,checkpoint_factory,graph_name,opset",
    [
        (clm.CLMModel, clm.build_clm_model, _tiny_clm_checkpoint, "clm_heads", 20),
        (kev.KEVModel, kev.build_kev_model, lambda: _tiny_kev_checkpoint(), "kev_head", 18),
    ],
)
def test_component_builders_use_inherited_model_emitters(
    monkeypatch, builder_class, build_model, checkpoint_factory, graph_name, opset
):
    assert issubclass(builder_class, base.Model)
    calls = {}
    for method_name in ("make_graph", "make_node", "make_value", "make_initializer"):
        original = getattr(base.Model, method_name)
        assert getattr(builder_class, method_name) is original
        calls[method_name] = []

        def track(self, *args, method_name=method_name, original=original, **kwargs):
            calls[method_name].append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(base.Model, method_name, track)

    model = build_model(checkpoint_factory())
    builder = calls["make_graph"][0]
    assert isinstance(builder, builder_class)
    assert builder.model is model
    assert model.graph.name == graph_name
    assert model.graph.opset_imports == {"": opset}
    assert model.ir_version == 10
    assert model.producer_name == "onnxruntime-genai"
    assert len(calls["make_graph"]) == 1
    assert len(calls["make_node"]) == len(model.graph)
    assert len(calls["make_initializer"]) == len(model.graph.initializers)
    for method_calls in calls.values():
        assert method_calls
        assert all(instance is builder for instance in method_calls)
    for value in (*model.graph.inputs, *model.graph.outputs, *model.graph.initializers.values()):
        assert builder.values[value.name] is value
    for node in model.graph:
        assert node.name in builder.node_names
        for value in (*node.inputs, *node.outputs):
            assert builder.values[value.name] is value
    checker.check_model(ir.serde.serialize_model(model))


def test_clm_head_compatibility_wrappers_use_shared_values_and_match_math():
    builder = clm.CLMModel()
    value = builder.make_value("hidden", ir.DataType.FLOAT, [None, 3])
    state = _tiny_clm_head(1)
    nodes, initializers = clm.build_head_nodes(value, "embedding", state, "state", 3, 4, 2, 3)
    assert nodes[0].inputs[0] is value
    builder.model.graph.inputs.append(value)
    builder.model.graph.extend(nodes)
    for initializer in initializers:
        builder.model.graph.register_initializer(initializer)
    output = nodes[-1].outputs[0]
    output.dtype = ir.DataType.FLOAT
    output.shape = ir.Shape([None, 2])
    builder.model.graph.outputs.append(output)
    hidden = torch.tensor([[0.2, -0.3, 0.7], [0.0, 0.0, 0.0]], dtype=torch.float32)
    actual = ReferenceEvaluator(ir.serde.serialize_model(builder.model)).run(None, {"hidden": hidden.numpy()})[0]
    np.testing.assert_allclose(actual, _torch_clm_head(hidden, state), rtol=2e-5, atol=2e-5)
    linear_nodes, linear_initializers = [], []
    linear = clm.append_linear(linear_nodes, linear_initializers, value, state, "inp", "state")
    assert linear_nodes[0].inputs[0] is value
    assert linear_nodes[0].outputs[0] is linear
    assert {initializer.name for initializer in linear_initializers} == {"state.inp.weight", "state.inp.bias"}


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
    assert isinstance(model, ir.Model)
    op_types = [node.op_type for node in model.graph]

    assert op_types.count("Gemm") == 6
    assert op_types.count("Gelu") == 4
    assert op_types.count("LayerNormalization") == 2
    assert op_types.count("ReduceL2") == 4
    assert op_types.count("Max") == 4
    assert set(model.graph.initializers) >= {
        "state.inp.weight",
        "state.hidden.0.weight",
        "state.norms.0.weight",
        "state.out.weight",
    }

    state_input = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float32)
    action_input = torch.tensor([[-0.1, 0.5, 0.4]], dtype=torch.float32)
    actual = ReferenceEvaluator(ir.serde.serialize_model(model)).run(
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
    scaled = ReferenceEvaluator(ir.serde.serialize_model(model)).run(
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
    outputs = ReferenceEvaluator(ir.serde.serialize_model(clm.build_clm_model(checkpoint))).run(
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
    outputs = ReferenceEvaluator(ir.serde.serialize_model(clm.build_clm_model(checkpoint))).run(
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


def test_clm_checkpoint_preflight_validates_heads_and_scale(tmp_path):
    checkpoint = _real_clm_checkpoint()
    checkpoint["state_head"]["inp.weight"] = torch.zeros(5, 4096)
    path = tmp_path / "invalid-head.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="state_head.inp.weight"):
        clm.load_clm_checkpoint(path)

    checkpoint = _real_clm_checkpoint()
    checkpoint["logit_scale"] = torch.tensor(float("inf"))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="logit_scale must be finite"):
        clm.load_clm_checkpoint(path)


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
    assert isinstance(model, ir.Model)
    hidden = torch.tensor(
        [
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.5, 1.0]],
            [[0.2, 0.8, -0.1], [0.4, -0.3, 0.9], [0.7, 0.1, 0.6]],
        ]
    )
    decide = np.asarray([0, 1], dtype=np.int64)
    options = np.asarray([[1, 2], [0, 2]], dtype=np.int64)
    mask = np.asarray([[True, False], [True, True]])

    probabilities, scores = ReferenceEvaluator(ir.serde.serialize_model(model)).run(
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
    assert [node.op_type for node in model.graph][-3:] == [
        "Where",
        "Softmax",
        "Where",
    ]
    all_masked, _ = ReferenceEvaluator(ir.serde.serialize_model(model)).run(
        None,
        {
            "hidden_states": hidden.numpy(),
            "decide_indices": decide,
            "option_indices": options,
            "option_mask": np.zeros_like(mask),
        },
    )
    np.testing.assert_array_equal(all_masked, np.zeros_like(all_masked))


@pytest.mark.parametrize("component", ["clm", "kev"])
def test_component_ir_export_roundtrip_preserves_contract(component, monkeypatch, tmp_path):
    if component == "clm":
        checkpoint = _tiny_clm_checkpoint()
        module, loader, builder, exporter = (clm, "load_clm_checkpoint", clm.build_clm_model, clm.export_clm_components)
        revision, base_revision = clm.clm_artifact_revision(), "caller-pin"
        filename, component_name, opset = "clm_heads.onnx", "clm_heads", 20
        input_contract = [
            ("state_hidden_states", ir.DataType.FLOAT, [None, 3]),
            ("action_hidden_states", ir.DataType.FLOAT, [None, 3]),
        ]
        output_contract = [
            ("state_embedding", ir.DataType.FLOAT, [None, 2]),
            ("action_embedding", ir.DataType.FLOAT, [None, 2]),
            ("effective_logit_scale", ir.DataType.FLOAT, []),
        ]
        weight_name, weight = "state.inp.weight", checkpoint["state_head"]["inp.weight"]
        feeds = {
            "state_hidden_states": np.ones((2, 3), dtype=np.float32),
            "action_hidden_states": np.zeros((2, 3), dtype=np.float32),
        }
    else:
        checkpoint = _tiny_kev_checkpoint()
        module, loader, builder, exporter = (kev, "load_kev_checkpoint", kev.build_kev_model, kev.export_kev_components)
        policy = kev.kev_policy()
        revision, base_revision = policy["adapter_revision"], policy["base_revision"]
        filename, component_name, opset = "kev_head.onnx", "kev_head", 18
        input_contract = [
            ("hidden_states", ir.DataType.FLOAT, [None, None, 3]),
            ("decide_indices", ir.DataType.INT64, [None]),
            ("option_indices", ir.DataType.INT64, [None, None]),
            ("option_mask", ir.DataType.BOOL, [None, None]),
        ]
        output_contract = [
            ("probabilities", ir.DataType.FLOAT, [None, None]),
            ("scores", ir.DataType.FLOAT, [None, None]),
        ]
        weight_name, weight = "q.weight", checkpoint["state"]["q.weight"]
        feeds = {
            "hidden_states": np.ones((1, 2, 3), dtype=np.float32),
            "decide_indices": np.asarray([0], dtype=np.int64),
            "option_indices": np.asarray([[0, 1]], dtype=np.int64),
            "option_mask": np.asarray([[True, False]]),
        }

    monkeypatch.setattr(module, loader, lambda _: checkpoint)
    options = types.SimpleNamespace(
        model_source="synthetic",
        artifact_revision=revision,
        base_revision=base_revision,
        backbone_filename="backbone.onnx",
    )
    expected = builder(checkpoint)
    manifest = exporter(options, tmp_path)
    loaded = ir.load(tmp_path / filename)
    assert isinstance(loaded, ir.Model)
    assert loaded.graph.opset_imports == {"": opset}
    assert loaded.producer_name == "onnxruntime-genai"
    assert loaded.metadata_props == expected.metadata_props
    assert manifest["components"][component_name]["filename"] == filename
    for values, contract in ((loaded.graph.inputs, input_contract), (loaded.graph.outputs, output_contract)):
        assert [(value.name, value.dtype, list(value.shape)) for value in values] == contract
    assert list(loaded.graph.initializers) == list(expected.graph.initializers)
    for name, value in expected.graph.initializers.items():
        np.testing.assert_array_equal(loaded.graph.initializers[name].const_value.numpy(), value.const_value.numpy())
    np.testing.assert_array_equal(loaded.graph.initializers[weight_name].const_value.numpy(), weight.numpy().T)
    # ONNX's checker independently validates the serialized IR at the file boundary.
    checker.check_model(str(tmp_path / filename))
    expected_outputs = ReferenceEvaluator(ir.serde.serialize_model(expected)).run(None, feeds)
    actual_outputs = ReferenceEvaluator(str(tmp_path / filename)).run(None, feeds)
    for actual, expected_output in zip(actual_outputs, expected_outputs, strict=True):
        np.testing.assert_array_equal(actual, expected_output)


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
        text_config=types.SimpleNamespace(hidden_size=2560),
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
    assert extra_options["exclude_mtp"] is True

    options.base_revision = "unpinned"
    with pytest.raises(ValueError, match="base_revision"):
        dispatch.validate_model_specific_backbone(options, config)


def test_kev_adapter_loads_on_text_backbone_and_rejects_missing_weights(monkeypatch):
    base_text = object()
    container = types.SimpleNamespace(model=base_text)
    merged = object()
    load_result = types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class FakeAdapted:
        def merge_and_unload(self):
            return merged

    calls = {}

    class FakeConfig:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["config"] = (source, kwargs)
            return object()

    def fake_get_peft_model(model, config):
        calls["model"] = model
        return FakeAdapted()

    def fake_load_weights(source, device):
        calls["weights"] = (source, device)
        return {"adapter": torch.tensor(1.0)}

    def fake_set_state(model, weights, adapter_name):
        calls["state"] = (model, weights, adapter_name)
        return load_result

    peft = types.ModuleType("peft")
    peft.PeftConfig = FakeConfig
    peft.get_peft_model = fake_get_peft_model
    save_and_load = types.ModuleType("peft.utils.save_and_load")
    save_and_load.load_peft_weights = fake_load_weights
    save_and_load.set_peft_model_state_dict = fake_set_state
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", save_and_load)

    assert base.load_text_backbone_adapter(container, "adapter") is container
    assert calls["model"] is base_text
    assert container.model is merged

    load_result.missing_keys = ["base_model.model.layer.lora_A.default.weight"]
    with pytest.raises(ValueError, match="did not load exactly"):
        base.load_text_backbone_adapter(types.SimpleNamespace(model=base_text), "adapter")

    nested_text = object()
    outer = types.SimpleNamespace(language_model=nested_text)
    nested_container = types.SimpleNamespace(model=outer)
    load_result.missing_keys = []
    assert base.load_text_backbone_adapter(nested_container, "adapter") is nested_container
    assert calls["model"] is nested_text
    assert nested_container.model is outer
    assert outer.language_model is merged


def test_qwen35_linear_attention_uses_adapter_aware_matmul_outputs():
    builder = object.__new__(qwen.Qwen35TextModel)
    builder.use_paged_attention = False
    builder.io_dtype = "float32"
    builder.linear_conv_dim = 6
    calls = {"transpose": [], "gated_norm": []}

    def make_matmul(_projection, name, _input):
        return f"{name}/with_lora"

    builder.make_matmul = make_matmul
    builder.require_dense_linear_attention_gate = lambda *_: None
    builder.exclude_node_from_quantization = lambda *_: None
    builder.make_initializer = lambda *_args, **_kwargs: None
    builder.make_transpose = lambda name, value, *_args: calls["transpose"].append((name, value))
    attention = types.SimpleNamespace(
        in_proj_qkv=object(),
        in_proj_z=object(),
        in_proj_b=types.SimpleNamespace(quant_type="none"),
        in_proj_a=types.SimpleNamespace(quant_type="none"),
        conv1d=types.SimpleNamespace(weight=torch.ones(1)),
    )

    z, b, a, conv, _ = builder.make_linear_attention_input_proj(0, attention, "root")
    assert z.endswith("z_proj/MatMul/with_lora/output_0")
    assert b.endswith("b_proj/MatMul/with_lora/output_0")
    assert a.endswith("a_proj/MatMul/with_lora/output_0")
    assert calls["transpose"][0][1].endswith("qkv_proj/MatMul/with_lora/output_0")
    assert conv.endswith("qkv_proj/Transpose/output_0")

    builder.linear_value_dim = 4
    builder.linear_num_value_heads = 1
    builder.use_paged_attention = False
    builder.layernorm_attrs = {"epsilon": 1e-5}
    builder.make_gated_rms_norm = lambda name, **kwargs: calls["gated_norm"].append((name, kwargs))
    attention.norm = types.SimpleNamespace(weight=torch.ones(4))
    attention.out_proj = object()
    builder.make_linear_attention_output_proj(0, attention, "attention", z)
    assert calls["gated_norm"][0][1]["gate"] == z
    assert builder.layernorm_attrs["skip_input"].endswith("out_proj/MatMul/with_lora/output_0")


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

    conflicting = {"adapter_path": str(tmp_path / "other-adapter")}
    with pytest.raises(ValueError, match="must match adapter_path"):
        dispatch.prepare_model_specific_hf(kev_options, conflicting, policy["base_model"])

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
