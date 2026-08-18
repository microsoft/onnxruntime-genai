# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

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

    Qwen35MoeTextModel._share_mtp_embedding_lm_head(tmp_path)

    assert (tmp_path / "mtp.onnx.data").read_bytes() == b"keep"
    model = onnx.load(tmp_path / "mtp.onnx", load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    assert _external_info(initializers["model.embed_tokens.weight"]) == ("model.onnx.data", 0, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight"]) == ("model.onnx.data", 4, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight_scale"]) == ("model.onnx.data", 8, 4)
    assert _external_info(initializers["lm_head.MatMul.nvfp4_weight_scale_2"]) == ("model.onnx.data", 12, 4)
    assert _external_info(initializers["mtp.fc.weight"]) == ("mtp.onnx.data", 0, 4)


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

    Qwen35MoeTextModel._share_mtp_embedding_lm_head(tmp_path)

    assert (tmp_path / "mtp.onnx.data").read_bytes() == b"samexx"
    assert (tmp_path / "mtp.onnx").read_bytes() == original_model
    assert not (tmp_path / "mtp.onnx.data.tmp").exists()
    assert not (tmp_path / "mtp.onnx.tmp").exists()
