# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""CPU embedding execution and token/embedding binding coverage for the Qwen3.8 KV calibrator."""

import json
import sys
import types

import numpy as np
import pytest
from _qwen38_tools_test_utils import load_tools, make_kv_calibration_stubs

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def calibration():
    stubs = make_kv_calibration_stubs()
    stubs["torch"] = torch
    with load_tools("calibrate_qwen38_kv", stubs=stubs) as (module,):
        yield module


def embedding_package(path, rank=1, bfloat16=True):
    table = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 10
    dims = ["tokens"] if rank == 1 else ["batch", "tokens"]
    nodes = [onnx.helper.make_node("Gather", ["table", "ids"], ["gathered"])]
    output_type = onnx.TensorProto.BFLOAT16 if bfloat16 else onnx.TensorProto.FLOAT
    nodes.append(onnx.helper.make_node("Cast", ["gathered"], ["embeds"], to=output_type))
    graph = onnx.helper.make_graph(
        nodes,
        "embedding",
        [onnx.helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, dims)],
        [onnx.helper.make_tensor_value_info("embeds", output_type, [*dims, 8])],
        [onnx.numpy_helper.from_array(table, "table")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path / "embedding.onnx")
    config = {
        "model": {
            "embedding": {
                "filename": "embedding.onnx",
                "inputs": {"input_ids": "ids"},
                "outputs": {"embeddings": "embeds"},
            }
        }
    }
    (path / "genai_config.json").write_text(json.dumps(config))
    return table


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("bfloat16", [False, True])
def test_package_cpu_embedding_preserves_names_shapes_and_dtype(calibration, tmp_path, rank, bfloat16):
    table = embedding_package(tmp_path, rank, bfloat16)
    embedding = calibration.create_embedding_session(ort, tmp_path)
    tokens = np.asarray([[1, 4, 7, 2]], dtype=np.int64)
    result = calibration.run_embeddings(ort, *embedding, tokens, torch.device("cpu"))
    expected = torch.from_numpy(table[tokens]).to(torch.bfloat16 if bfloat16 else torch.float32)
    assert result.shape == (1, 4, 8)
    assert result.dtype == expected.dtype
    assert torch.equal(result, expected)
    later = calibration.run_embeddings(ort, *embedding, tokens[:, ::-1].copy(), torch.device("cpu"))
    assert torch.equal(later, expected.flip(1))
    assert torch.equal(result, expected)


def test_embedding_graph_is_required_for_embedding_only_calibration(calibration, tmp_path):
    (tmp_path / "genai_config.json").write_text('{"model": {}}')
    with pytest.raises(ValueError, match="requires model.embedding.filename"):
        calibration.create_embedding_session(ort, tmp_path)


@pytest.mark.parametrize("dense_input", ["input_ids", "inputs_embeds"])
@pytest.mark.parametrize("head_input", ["input_ids", "inputs_embeds"])
def test_calibrator_binds_prompt_and_shifted_head_embeddings(
    calibration, tmp_path, monkeypatch, dense_input, head_input
):
    table = embedding_package(tmp_path)
    tokens = np.asarray([[1, 2, 4]], dtype=np.int64)

    class Binding:
        def __init__(self, session):
            self.session = session
            self.inputs = {}
            self.outputs = []

        def bind_ortvalue_input(self, name, value):
            assert name in {meta.name for meta in self.session.get_inputs()}, f"unexpected input {name}"
            self.inputs[name] = value

        def bind_output(self, name, *args):
            self.outputs.append(name)

        def get_outputs(self):
            return [self.session.results[name] for name in self.outputs]

    class Session:
        def __init__(self, token_input, head=False):
            self.token_input = token_input
            self.head = head
            self.seen = {}
            self.results = {}

        def get_inputs(self):
            names = [self.token_input, "attention_mask", "position_ids"]
            names += (
                ["past_position_ids", "past_key_values.0.index_key", "hidden_states"]
                if self.head
                else ["logits_indices", "ple_embeddings.0"]
            )
            return [types.SimpleNamespace(name=name, shape=[1, "tokens", 8], type="tensor(bfloat16)") for name in names]

        def get_outputs(self):
            return [types.SimpleNamespace(name=name) for name in ("present.3.key", "present.3.value", "hidden_states")]

        def io_binding(self):
            return Binding(self)

        def run_with_iobinding(self, binding):
            self.seen = {name: torch.from_dlpack(value).clone() for name, value in binding.inputs.items()}
            embedded = self.seen.get("inputs_embeds")
            if embedded is None:
                embedded = torch.from_numpy(table[self.seen["input_ids"].numpy()]).to(torch.bfloat16)
            rows = embedded.float().mean(dim=-1).reshape(-1)
            if self.head:
                rows += self.seen["hidden_states"].float().mean(dim=-1).reshape(-1)
            key = rows.reshape(1, 1, -1, 1).expand(1, 2, len(rows), 256).to(torch.bfloat16).contiguous()
            if self.head:
                self.results = {
                    "key": key,
                    "value": rows[:, None].expand(len(rows), 512).to(torch.bfloat16).contiguous(),
                    "index_key": torch.zeros((1, len(rows), 128), dtype=torch.bfloat16),
                }
            else:
                self.results = {"present.3.key": key, "present.3.value": key * 2, "hidden_states": embedded}

    dense = Session(dense_input)
    head = Session(head_input, head=True)
    monkeypatch.setattr(calibration, "create_calibration_sessions", lambda *args: (dense, head))
    loaded_embeddings = []
    create_embedding_session = calibration.create_embedding_session

    def load_embedding(*args):
        loaded_embeddings.append(True)
        return create_embedding_session(*args)

    monkeypatch.setattr(calibration, "create_embedding_session", load_embedding)
    monkeypatch.setattr(calibration, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda path: object()))
    monkeypatch.setattr(calibration, "_tokenize_corpus", lambda *args: [tokens])
    cpu_torch = types.SimpleNamespace(
        **{
            name: getattr(torch, name)
            for name in (
                "from_numpy",
                "from_dlpack",
                "zeros",
                "ones",
                "arange",
                "tensor",
                "utils",
                "bfloat16",
                "int64",
                "float32",
            )
        }
    )
    cpu_torch.device = lambda name: torch.device("cpu")
    monkeypatch.setattr(calibration, "torch", cpu_torch)

    class Lookup:
        def initial_context(self, batch):
            return None

        def __call__(self, prompts, context):
            return torch.zeros((1, len(prompts[0]), 4), dtype=torch.bfloat16), None

    modules = {
        "mobius._configs": {
            "Qwen4ExpConfig": types.SimpleNamespace(
                from_transformers=lambda config: types.SimpleNamespace(ple_layer_ids=[1])
            )
        },
        "mobius.generation._qwen4_exp_ple": {"Qwen4ExpCPUPLELookup": lambda *args: Lookup()},
        "mobius.integrations.transformers._builder": {
            "_load_transformers_config": lambda *args, **kwargs: (None, None)
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    result = calibration.calibrate_package(
        tmp_path, tmp_path, tmp_path / "scales.json", method="minmax", rotary_dim=0, sample_budget=3
    )
    expected = torch.from_numpy(table[tokens]).to(torch.bfloat16)
    if dense_input == "inputs_embeds":
        assert torch.equal(dense.seen["inputs_embeds"], expected)
        assert "input_ids" not in dense.seen
    else:
        assert torch.equal(dense.seen["input_ids"], torch.from_numpy(tokens))
    if head_input == "inputs_embeds":
        assert torch.equal(head.seen["inputs_embeds"], expected[:, 1:, :])
        assert "input_ids" not in head.seen
    else:
        assert torch.equal(head.seen["input_ids"], torch.from_numpy(tokens[:, 1:]))
    assert torch.equal(head.seen["hidden_states"], expected[:, :-1, :])
    assert len(loaded_embeddings) == int("inputs_embeds" in (dense_input, head_input))
    for payload in result.values():
        target_max = dense.results["present.3.key"].float().abs().amax(dim=(0, 2)).reshape(-1).numpy()
        head_max = head.results["key"].float().abs().amax(dim=(0, 2)).reshape(-1).numpy()
        np.testing.assert_allclose(payload["scales"]["k_scales"][0], target_max / payload["qmax"])
        np.testing.assert_allclose(payload["scales"]["head_k_scales"][0], head_max / payload["qmax"])
    assert (tmp_path / "scales.json").is_file()
    assert (tmp_path / "scales.fp8_per_channel.json").is_file()
