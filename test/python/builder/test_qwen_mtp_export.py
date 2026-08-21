# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os

import onnx
from onnx import external_data_helper, helper

from models.builders.qwen import Qwen35MoeTextModel


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

    shared_initializers = Qwen35MoeTextModel._share_mtp_embedding_lm_head(tmp_path, "model.onnx")

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

    shared_initializers = Qwen35MoeTextModel._share_mtp_embedding_lm_head(tmp_path, "model.onnx")

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

    shared_initializers = Qwen35MoeTextModel._share_mtp_embedding_lm_head(tmp_path, "model.onnx")

    assert (tmp_path / "mtp.onnx.data").read_bytes() == original_data
    assert (tmp_path / "mtp.onnx").read_bytes() == original_model
    for suffix in ("mtp.onnx.data.tmp", "mtp.onnx.tmp", "mtp.onnx.data.bak", "mtp.onnx.bak"):
        assert not (tmp_path / suffix).exists()
    assert shared_initializers == []
