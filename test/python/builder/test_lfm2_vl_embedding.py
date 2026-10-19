# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper
from safetensors.torch import save_file

REPO = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location(
    "lfm2_vl_embedding", REPO / "src" / "python" / "py" / "models" / "lfm2_vl_embedding.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

FIXTURE = REPO / "test" / "models" / "lfm2-vl"
HIDDEN = 32  # the fixture's dummy vision model emits 32-wide features
VOCAB = 320
IMAGE_TOKEN = 300


def run(model, ids, features):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"input_ids": ids, "image_features": features})[0]


@pytest.mark.parametrize("table_dtype", [np.float32, np.float16])
@pytest.mark.parametrize("features_dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    "ids",
    [[[1, IMAGE_TOKEN, IMAGE_TOKEN, 5, IMAGE_TOKEN]], [[IMAGE_TOKEN, 2], [3, IMAGE_TOKEN]], [[7, 8, 9]], [[4]]],
)
def test_embedding_matches_numpy(table_dtype, features_dtype, ids):
    rng = np.random.default_rng(0)
    table = rng.standard_normal((VOCAB, HIDDEN)).astype(table_dtype)
    ids = np.array(ids, np.int64)
    features = rng.standard_normal((int((ids == IMAGE_TOKEN).sum()), HIDDEN)).astype(features_dtype)
    model = tool.make_embedding_model(table, IMAGE_TOKEN, helper.np_dtype_to_tensor_dtype(features.dtype))
    onnx.checker.check_model(model)
    expected = table[ids]
    expected[ids == IMAGE_TOKEN] = features.astype(table_dtype)
    actual = run(model, ids, features)
    assert actual.dtype == table_dtype
    np.testing.assert_array_equal(actual, expected)


def make_checkpoint(path, layout="lfm2.5", sharded=False, hidden=HIDDEN):
    """A bf16 checkpoint with only the files the tool reads; returns the embedding table."""
    path.mkdir()
    image_processor = {
        "resample": 2 if layout == "lfm2.5" else 3,
        "size": {"height": 512, "width": 512},
        "rescale_factor": 1 / 255,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "min_image_tokens": 64,
        "max_image_tokens": 256,
        "max_num_patches": 1024,
        "encoder_patch_size": 16,
        "downsample_factor": 2,
    }
    (path / "config.json").write_text(json.dumps({"image_token_id": IMAGE_TOKEN}))
    # LFM2.5-VL nests the image processor in processor_config.json, LFM2-VL keeps it in preprocessor_config.json.
    if layout == "lfm2.5":
        (path / "processor_config.json").write_text(json.dumps({"image_processor": image_processor}))
    else:
        (path / "processor_config.json").write_text(json.dumps({"processor_class": "Lfm2VlProcessor"}))
        (path / "preprocessor_config.json").write_text(json.dumps(image_processor))
    table = torch.randn(VOCAB, hidden, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    if sharded:
        save_file({"other": torch.zeros(1)}, path / "model-00001-of-00002.safetensors")
        save_file({tool.EMBEDDING_WEIGHT: table}, path / "model-00002-of-00002.safetensors")
        weight_map = {
            "other": "model-00001-of-00002.safetensors",
            tool.EMBEDDING_WEIGHT: "model-00002-of-00002.safetensors",
        }
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file({tool.EMBEDDING_WEIGHT: table}, path / "model.safetensors")
    return table


def make_package(path, embeds_type=TensorProto.FLOAT, provider_options=(), model_type="lfm2_vl"):
    """A builder output with an identity decoder and the fixture's vision model."""
    path.mkdir()
    shutil.copy(FIXTURE / "dummy_vision.onnx", path / "vision_encoder.onnx")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["inputs_embeds"], ["logits"])],
        "decoder",
        [helper.make_tensor_value_info("inputs_embeds", embeds_type, ["b", "s", HIDDEN])],
        [helper.make_tensor_value_info("logits", embeds_type, ["b", "s", HIDDEN])],
    )
    onnx.save(helper.make_model(graph), path / "model.onnx")
    decoder = {
        "filename": "model.onnx",
        "hidden_size": HIDDEN,
        "inputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": {"log_id": "onnxruntime-genai", "provider_options": list(provider_options)},
    }
    (path / "genai_config.json").write_text(json.dumps({"model": {"type": model_type, "decoder": decoder}}))


@pytest.mark.parametrize("layout", ["lfm2.5", "lfm2"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("embeds_type", [TensorProto.FLOAT, TensorProto.FLOAT16])
def test_convert_writes_the_package(tmp_path, layout, sharded, embeds_type):
    table = make_checkpoint(tmp_path / "pt", layout, sharded)
    out = tmp_path / "out"
    make_package(out, embeds_type)
    tool.convert(tmp_path / "pt", out)
    tool.convert(tmp_path / "pt", out)  # a rerun replaces the files instead of appending to them

    # The fixture holds the LFM2.5-VL sections, with other file names.
    config = json.loads((out / "genai_config.json").read_text())["model"]
    fixture = json.loads((FIXTURE / "genai_config.json").read_text())["model"]
    assert config["embedding"] == dict(fixture["embedding"], filename="embeddings.onnx")
    assert config["vision"] == dict(fixture["vision"], filename="vision_encoder.onnx")
    expected = json.loads((FIXTURE / "processor_config.json").read_text())
    if layout == "lfm2":
        expected["processor"]["transforms"][1]["operation"]["attrs"]["interpolation"] = "CUBIC"
    assert json.loads((out / "processor_config.json").read_text()) == expected

    dtype = torch.float32 if embeds_type == TensorProto.FLOAT else torch.float16
    assert (out / "embeddings.onnx.data").stat().st_size == VOCAB * HIDDEN * dtype.itemsize
    ids = np.array([[1, IMAGE_TOKEN, 2]], np.int64)
    features = np.full((1, HIDDEN), 0.01, np.float32)  # the vision model's output type
    expected = table.to(dtype).numpy()[ids]
    expected[0, 1] = features[0]
    np.testing.assert_array_equal(run(onnx.load(out / "embeddings.onnx"), ids, features), expected)


@pytest.mark.parametrize(
    "decoder,sub_model",
    [
        ({"cuda": {"enable_cuda_graph": "1"}}, {"cuda": {}}),
        ({"dml": {"enable_graph_capture": "1"}}, {"dml": {}}),
        ({"webgpu": {"enableGraphCapture": "1", "x": "y"}}, {"webgpu": {"x": "y"}}),
    ],
)
def test_sub_models_follow_the_decoder_without_graph_capture(tmp_path, decoder, sub_model):
    make_checkpoint(tmp_path / "pt")
    make_package(tmp_path / "out", TensorProto.FLOAT16, [decoder])
    tool.convert(tmp_path / "pt", tmp_path / "out")

    model = json.loads((tmp_path / "out" / "genai_config.json").read_text())["model"]
    assert model["decoder"]["session_options"]["provider_options"] == [decoder]
    for section in ("embedding", "vision"):
        assert model[section]["session_options"]["provider_options"] == [sub_model]


@pytest.mark.parametrize(
    "case,message",
    [
        ("text", "exclude_embeds=true"),
        ("bf16", "float32 and float16"),
        ("vision", "32-wide image_features"),
        ("table", "table is 16 wide"),
        ("resample", "Unsupported resample 1"),
        ("name", "named embeddings.onnx"),
    ],
)
def test_refuses_before_writing(tmp_path, case, message):
    make_checkpoint(tmp_path / "pt", hidden=16 if case == "table" else HIDDEN)
    out = tmp_path / "out"
    make_package(
        out,
        TensorProto.BFLOAT16 if case == "bf16" else TensorProto.FLOAT,
        (),
        "lfm2_vl_text" if case == "text" else "lfm2_vl",
    )
    if case == "vision":  # a vision model for a 64-wide decoder
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["image_features"])],
            "vision",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 64])],
            [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["n", 64])],
        )
        onnx.save(helper.make_model(graph), out / "vision_encoder.onnx")
    if case == "name":  # a decoder exported with --extra_options filename=embeddings.onnx
        (out / "model.onnx").rename(out / "embeddings.onnx")
        config = json.loads((out / "genai_config.json").read_text())
        config["model"]["decoder"]["filename"] = "embeddings.onnx"
        (out / "genai_config.json").write_text(json.dumps(config))
    if case == "resample":
        path = tmp_path / "pt" / "processor_config.json"
        image_processor = json.loads(path.read_text())["image_processor"]
        path.write_text(json.dumps({"image_processor": dict(image_processor, resample=1)}))
    before = sorted(p.name for p in out.iterdir()), (out / "genai_config.json").read_text()

    with pytest.raises(ValueError, match=message):
        tool.convert(tmp_path / "pt", out)
    assert (sorted(p.name for p in out.iterdir()), (out / "genai_config.json").read_text()) == before
