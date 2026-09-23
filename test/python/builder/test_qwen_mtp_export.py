# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import os

import onnx
import pytest
from onnx import external_data_helper, helper

from models.builders.qwen import Qwen35MoEModel


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


def test_share_mtp_weights_repacks_data_after_staging_metadata(tmp_path):
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
