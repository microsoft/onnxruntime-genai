# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import external_data_helper, helper

from models.builders.qwen import Qwen4ExpModel, Qwen35MoEModel
from models.builders.qwen3_8 import Qwen4ExpMTPTextModel


def _make_external_model(path, data_name, tensors):
    initializers = []
    for name, offset, length in tensors:
        tensor = onnx.TensorProto()
        tensor.name = name
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.extend([1])
        tensor.raw_data = b"\0" * length
        external_data_helper.set_external_data(
            tensor,
            location=data_name,
            offset=offset,
            length=length,
        )
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        initializers.append(tensor)

    graph = helper.make_graph([], "test", [], [], initializer=initializers)
    onnx.save(helper.make_model(graph), path)


def _external_info(tensor):
    values = {entry.key: entry.value for entry in tensor.external_data}
    return values["location"], int(values["offset"]), int(values["length"])


def _make_qwen_mtp_model():
    model = object.__new__(Qwen35MoEModel)
    model.mtp_attrs = {
        "shared_initializers": [],
        "shared_initializer_names": {"model.embed_tokens.weight"},
        "shared_initializer_prefixes": ("lm_head.MatMul.",),
    }
    return model


def test_qwen4_exp_mtp_offset_rmsnorm_uses_onnx_domain():
    model = object.__new__(Qwen4ExpMTPTextModel)
    model.io_dtype = onnx.TensorProto.FLOAT16
    model.layernorm_attrs = {"add_offset": 1.0, "epsilon": 1e-6}
    initializers = {}
    nodes = []
    model.make_initializer = lambda value, name, **kwargs: initializers.update({name: value})
    model.make_node = lambda op_type, **kwargs: nodes.append(helper.make_node(op_type, **kwargs))
    model.make_value = lambda *args, **kwargs: None
    model.make_hidden_state_shape = lambda: [1, 1, 3]

    output = model.make_offset_rmsnorm("/mtp/enorm", "input", np.asarray([-1.0, 0.0, 1.0]))

    assert output == "/mtp/enorm/output_0"
    assert len(nodes) == 1
    assert nodes[0].op_type == "SimplifiedLayerNormalization"
    assert nodes[0].domain == ""
    assert initializers["mtp.enorm.weight"].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "model_type,head_type",
    [
        ("decoder", "decoder"),
        ("qwen3_5", "qwen3_5_text"),
        ("qwen3_5_moe", "qwen3_5_moe_text"),
        ("qwen3_5_text", "qwen3_5_text"),
        ("qwen4_exp", "qwen4_exp_text"),
    ],
)
def test_mtp_decoder_overlay_excludes_multimodal_companions(tmp_path, model_type, head_type):
    config = {
        "model": {
            "type": model_type,
            "decoder": {"shared_initializers": []},
            "mtp": {
                "filename": "mtp.onnx",
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "head_size": 256,
                "shared_initializers": [],
                "inputs": {"input_ids": "input_ids", "hidden_states": "hidden_states"},
                "outputs": {"logits": "logits", "hidden_states": "hidden_states_out"},
            },
        }
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(config))
    example_path = Path(__file__).resolve().parents[3] / "examples/python/qwen-3.6-mtp.py"
    spec = importlib.util.spec_from_file_location("qwen_mtp_example", example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    overlay = example.mtp_decoder_overlay(tmp_path)["model"]

    assert overlay["type"] == head_type
    assert overlay["decoder"]["filename"] == "mtp.onnx"
    assert overlay["decoder"]["layer_types"] == []
    assert overlay["decoder"]["conv_cache_size"] == 0
    assert overlay["decoder"]["inputs"] == config["model"]["mtp"]["inputs"]
    for companion in ("embedding", "vision", "engram"):
        assert overlay[companion]["filename"] == ""


@pytest.mark.parametrize("prefix_caching", [None, False, True])
def test_add_mtp_to_genai_config(tmp_path, prefix_caching):
    config_path = tmp_path / "genai_config.json"
    dynamic_batching = {} if prefix_caching is None else {"prefix_caching": prefix_caching}
    config_path.write_text(
        json.dumps(
            {
                "model": {"decoder": {}},
                "engine": {"dynamic_batching": dynamic_batching},
            }
        )
    )
    model = object.__new__(Qwen35MoEModel)
    model.decoder = type("Decoder", (), {"num_kv_heads": 2, "head_size": 128})()
    model.mtp_attrs = {"shared_initializers": []}

    model.add_mtp_to_genai_config(tmp_path)

    config = json.loads(config_path.read_text())
    assert config["model"]["decoder"]["outputs"]["hidden_states"] == "hidden_states"
    assert config["model"]["mtp"]["enabled"] is True
    assert config["model"]["mtp"]["filename"] == "mtp.onnx"
    expected_prefix_caching = False if prefix_caching is None else prefix_caching
    assert config["engine"]["dynamic_batching"]["prefix_caching"] is expected_prefix_caching


def test_add_mtp_to_static_genai_config(tmp_path):
    config_path = tmp_path / "genai_config.json"
    config_path.write_text(json.dumps({"model": {"decoder": {}}}))
    model = object.__new__(Qwen35MoEModel)
    model.decoder = type("Decoder", (), {"num_kv_heads": 2, "head_size": 128})()
    model.mtp_attrs = {"shared_initializers": []}

    model.add_mtp_to_genai_config(tmp_path)

    config = json.loads(config_path.read_text())
    assert "engine" not in config
    assert config["model"]["mtp"]["filename"] == "mtp.onnx"


@pytest.mark.parametrize("fpa_intb_gemm", [None, "0", "1"])
def test_add_qwen4_exp_mtp_to_genai_config(tmp_path, fpa_intb_gemm):
    config_path = tmp_path / "genai_config.json"
    expected_session_options = {"session.use_device_allocator_for_initializers": "1"}
    if fpa_intb_gemm is not None:
        expected_session_options["ep.cuda.fpa_intb_gemm"] = fpa_intb_gemm
    decoder_session_options = {
        **expected_session_options,
        "log_id": "onnxruntime-genai",
        "provider_options": [{"cuda": {"enable_cuda_graph": "0"}}],
        "session.layer_assignment_settings": "cpu(=cpu_embedding)",
        "session.disable_prepacking": "1",
        "ep.cuda.qmoe_row_tile_size": "1",
    }
    config_path.write_text(json.dumps({"model": {"decoder": {"session_options": decoder_session_options}}}))
    model = object.__new__(Qwen4ExpModel)
    model.decoder = type(
        "Decoder",
        (),
        {"num_kv_heads": 2, "head_size": 256, "hc_hidden_size": 10240},
    )()
    model.mtp_attrs = {"shared_initializers": []}

    model.add_mtp_to_genai_config(tmp_path)

    config = json.loads(config_path.read_text())
    assert "hidden_size" not in config["model"]["mtp"]
    assert config["model"]["mtp"]["inputs"]["past_indexer_names"] == "past.%d.indexer_key"
    assert config["model"]["mtp"]["inputs"]["past_sequence_length"] == "past_sequence_length"
    assert config["model"]["mtp"]["outputs"]["present_indexer_names"] == "present.%d.indexer_key"
    assert config["model"]["mtp"]["session_options"] == expected_session_options
    assert config["model"]["decoder"]["session_options"] == decoder_session_options


@pytest.mark.parametrize("qwen38", [False, True])
def test_share_mtp_weights_repacks_data_after_staging_metadata(tmp_path, qwen38):
    main_data = b"samecodescalglob"
    mtp_data = b"samecodescalglobkeep"
    (tmp_path / "model.onnx.data").write_bytes(main_data)
    (tmp_path / "mtp.onnx.data").write_bytes(mtp_data)
    _make_external_model(
        tmp_path / "model.onnx",
        "model.onnx.data",
        [
            ("model.embed_tokens.weight", 0, 4),
            ("lm_head.MatMul.nvfp4_weight", 4, 4),
            ("lm_head.MatMul.nvfp4_weight_scale", 8, 4),
            ("lm_head.MatMul.nvfp4_weight_scale_2", 12, 4),
        ],
    )
    _make_external_model(
        tmp_path / "mtp.onnx",
        "mtp.onnx.data",
        [
            ("model.embed_tokens.weight", 0, 4),
            ("lm_head.MatMul.nvfp4_weight", 4, 4),
            ("lm_head.MatMul.nvfp4_weight_scale", 8, 4),
            ("lm_head.MatMul.nvfp4_weight_scale_2", 12, 4),
            ("mtp.fc.weight", 16, 4),
        ],
    )

    model_builder = _make_qwen_mtp_model()
    if qwen38:
        model_builder = object.__new__(Qwen4ExpModel)
        model_builder.make_mtp_init(SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=1)), {})
    shared_initializers = model_builder.share_initializers(tmp_path, "model.onnx", "mtp.onnx")

    assert (tmp_path / "mtp.onnx.data").read_bytes() == b"keep"
    model = onnx.load(tmp_path / "mtp.onnx", load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert _external_info(initializers["model.embed_tokens.weight"]) == ("model.onnx.data", 0, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight"]) == ("model.onnx.data", 4, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight_scale"]) == ("model.onnx.data", 8, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight_scale_2"]) == ("model.onnx.data", 12, 4)
    assert _external_info(initializers["mtp.fc.weight"]) == ("mtp.onnx.data", 0, 4)
    assert shared_initializers == [
        {
            "name": "model.embed_tokens.weight",
            "data_file": "model.onnx.data",
            "offset": "0",
            "length": "4",
            "data_type": onnx.TensorProto.FLOAT,
            "shape": [1],
        },
        {
            "name": "lm_head.MatMul.nvfp4_weight",
            "data_file": "model.onnx.data",
            "offset": "4",
            "length": "4",
            "data_type": onnx.TensorProto.FLOAT,
            "shape": [1],
        },
        {
            "name": "lm_head.MatMul.nvfp4_weight_scale",
            "data_file": "model.onnx.data",
            "offset": "8",
            "length": "4",
            "data_type": onnx.TensorProto.FLOAT,
            "shape": [1],
        },
        {
            "name": "lm_head.MatMul.nvfp4_weight_scale_2",
            "data_file": "model.onnx.data",
            "offset": "12",
            "length": "4",
            "data_type": onnx.TensorProto.FLOAT,
            "shape": [1],
        },
    ]


def test_share_initializers_can_adopt_source_quantization(tmp_path):
    (tmp_path / "model.onnx.data").write_bytes(b"main")
    (tmp_path / "dflash2.onnx.data").write_bytes(b"diffkeep")
    _make_external_model(
        tmp_path / "model.onnx",
        "model.onnx.data",
        [("lm_head.MatMul.weight_Q4", 0, 4)],
    )
    _make_external_model(
        tmp_path / "dflash2.onnx",
        "dflash2.onnx.data",
        [("lm_head.MatMul.weight_Q4", 0, 4), ("dflash2.fc.weight", 4, 4)],
    )

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(
        tmp_path,
        "model.onnx",
        "dflash2.onnx",
        adopt_source_initializers={"lm_head.MatMul.weight_Q4"},
    )

    assert (tmp_path / "dflash2.onnx.data").read_bytes() == b"keep"
    model = onnx.load(tmp_path / "dflash2.onnx", load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert _external_info(initializers["lm_head.MatMul.weight_Q4"]) == ("model.onnx.data", 0, 4)
    assert shared_initializers[0]["name"] == "lm_head.MatMul.weight_Q4"


def test_excluded_initializer_keeps_its_private_copy(tmp_path):
    (tmp_path / "model.onnx.data").write_bytes(b"same")
    (tmp_path / "dflash2.onnx.data").write_bytes(b"same")
    tensors = [("lm_head.MatMul.weight_Q4", 0, 4)]
    _make_external_model(tmp_path / "model.onnx", "model.onnx.data", tensors)
    _make_external_model(tmp_path / "dflash2.onnx", "dflash2.onnx.data", tensors)
    original_model = (tmp_path / "dflash2.onnx").read_bytes()

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(
        tmp_path,
        "model.onnx",
        "dflash2.onnx",
        excluded_source_initializers={"lm_head.MatMul.weight_Q4"},
    )

    assert shared_initializers == []
    assert (tmp_path / "dflash2.onnx").read_bytes() == original_model
    assert (tmp_path / "dflash2.onnx.data").read_bytes() == b"same"


def test_required_adoption_is_transactional_when_one_initializer_is_incompatible(tmp_path):
    (tmp_path / "model.onnx.data").write_bytes(b"weightscale")
    (tmp_path / "dflash2.onnx.data").write_bytes(b"draft!bad")
    tensors = [("lm_head.MatMul.weight_Q4", 0, 6), ("lm_head.MatMul.weight_scales", 6, 5)]
    _make_external_model(tmp_path / "model.onnx", "model.onnx.data", tensors)
    _make_external_model(
        tmp_path / "dflash2.onnx",
        "dflash2.onnx.data",
        [("lm_head.MatMul.weight_Q4", 0, 6), ("lm_head.MatMul.weight_scales", 6, 3)],
    )
    original_model = (tmp_path / "dflash2.onnx").read_bytes()
    original_data = (tmp_path / "dflash2.onnx.data").read_bytes()
    required = {"lm_head.MatMul.weight_Q4", "lm_head.MatMul.weight_scales"}

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(
        tmp_path,
        "model.onnx",
        "dflash2.onnx",
        adopt_source_initializers=required,
        required_source_initializers=required,
    )

    assert shared_initializers == []
    assert (tmp_path / "dflash2.onnx").read_bytes() == original_model
    assert (tmp_path / "dflash2.onnx.data").read_bytes() == original_data
    assert not (tmp_path / "dflash2.onnx.tmp").exists()
    assert not (tmp_path / "dflash2.onnx.data.tmp").exists()


def test_required_adoption_rejects_a_truncated_source_range(tmp_path):
    (tmp_path / "model.onnx.data").write_bytes(b"abc")
    (tmp_path / "dflash2.onnx.data").write_bytes(b"diff")
    _make_external_model(
        tmp_path / "model.onnx",
        "model.onnx.data",
        [("lm_head.MatMul.weight_Q4", 0, 4)],
    )
    _make_external_model(
        tmp_path / "dflash2.onnx",
        "dflash2.onnx.data",
        [("lm_head.MatMul.weight_Q4", 0, 4)],
    )
    original_model = (tmp_path / "dflash2.onnx").read_bytes()
    required = {"lm_head.MatMul.weight_Q4"}

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(
        tmp_path,
        "model.onnx",
        "dflash2.onnx",
        adopt_source_initializers=required,
        required_source_initializers=required,
    )

    assert shared_initializers == []
    assert (tmp_path / "dflash2.onnx").read_bytes() == original_model
    assert (tmp_path / "dflash2.onnx.data").read_bytes() == b"diff"
    assert not (tmp_path / "dflash2.onnx.tmp").exists()
    assert not (tmp_path / "dflash2.onnx.data.tmp").exists()


def test_share_mtp_weights_leaves_originals_on_truncated_data(tmp_path):
    (tmp_path / "model.onnx.data").write_bytes(b"same")
    (tmp_path / "mtp.onnx.data").write_bytes(b"samexx")
    _make_external_model(
        tmp_path / "model.onnx",
        "model.onnx.data",
        [("model.embed_tokens.weight", 0, 4)],
    )
    _make_external_model(
        tmp_path / "mtp.onnx",
        "mtp.onnx.data",
        [("model.embed_tokens.weight", 0, 4), ("mtp.fc.weight", 4, 4)],
    )
    original_model = (tmp_path / "mtp.onnx").read_bytes()

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(tmp_path, "model.onnx", "mtp.onnx")

    assert (tmp_path / "mtp.onnx.data").read_bytes() == b"samexx"
    assert (tmp_path / "mtp.onnx").read_bytes() == original_model
    assert not (tmp_path / "mtp.onnx.data.tmp").exists()
    assert not (tmp_path / "mtp.onnx.tmp").exists()
    assert shared_initializers == []


def test_share_mtp_weights_restores_originals_when_metadata_replace_fails(tmp_path, monkeypatch):
    (tmp_path / "model.onnx.data").write_bytes(b"same")
    (tmp_path / "mtp.onnx.data").write_bytes(b"samekeep")
    _make_external_model(
        tmp_path / "model.onnx",
        "model.onnx.data",
        [("model.embed_tokens.weight", 0, 4)],
    )
    _make_external_model(
        tmp_path / "mtp.onnx",
        "mtp.onnx.data",
        [("model.embed_tokens.weight", 0, 4), ("mtp.fc.weight", 4, 4)],
    )
    original_data = (tmp_path / "mtp.onnx.data").read_bytes()
    original_model = (tmp_path / "mtp.onnx").read_bytes()
    original_replace = os.replace

    def fail_metadata_replace(source, destination):
        if str(source).endswith("mtp.onnx.tmp") and str(destination).endswith("mtp.onnx"):
            raise OSError("injected metadata replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_metadata_replace)

    model_builder = _make_qwen_mtp_model()
    shared_initializers = model_builder.share_initializers(tmp_path, "model.onnx", "mtp.onnx")

    assert (tmp_path / "mtp.onnx.data").read_bytes() == original_data
    assert (tmp_path / "mtp.onnx").read_bytes() == original_model
    for suffix in ("mtp.onnx.data.tmp", "mtp.onnx.tmp", "mtp.onnx.data.bak", "mtp.onnx.bak"):
        assert not (tmp_path / suffix).exists()
    assert shared_initializers == []
