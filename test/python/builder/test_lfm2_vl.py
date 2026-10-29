# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM2-VL model builder.

LFM2-VL reuses the dense LFM2 builder for its language model. The checkpoint nests that decoder under
`model.language_model`, so the builder has to load it through `Lfm2VlForConditionalGeneration` and find
its `embedding_norm` there. Without `exclude_embeds` the export is a text-only decoder (`lfm2_vl_text`).
With it, the builder writes the whole `lfm2_vl` pipeline: the decoder, the embedding model, the vision
model and their configs. The vision and embedding tests run tiny random transformers models in ONNX
Runtime on CPU and compare them with transformers.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")
# _builder_test_utils, used by other tests in the session, stubs onnxruntime when it is not installed
if not hasattr(ort, "InferenceSession"):
    pytest.skip("onnxruntime is not installed", allow_module_level=True)

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"
BUILDERS_DIR = MODELS_DIR / "builders"
sys.path.insert(0, str(MODELS_DIR))


def _load_builder_module(module_name):
    if f"models.builders.{module_name}" in sys.modules:
        return sys.modules[f"models.builders.{module_name}"]
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def _load_builder_entrypoint_module():
    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", MODELS_DIR / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
builder_module = _load_builder_entrypoint_module()
Model = base_module.Model
lfm2_vl_module = importlib.import_module("builders.lfm2_vl")
ir = lfm2_vl_module.ir


@pytest.mark.parametrize(
    "exclude_embeds,builder_name,expected_type",
    [(True, "LFM2VLModel", "lfm2_vl"), (False, "LFM2Model", "lfm2_vl_text")],
)
def test_lfm2_vl_architecture_picks_the_pipeline_or_the_text_model(
    monkeypatch, tmp_path, exclude_embeds, builder_name, expected_type
):
    captured = {}

    class FakeModel:
        model_type = "lfm2_vl"  # LFM2VLModel labels its decoder itself

        def __init__(self, *args):
            captured["builder"], captured["args"] = type(self).__name__, args

        def make_genai_config(self, *args):
            captured["model_type"] = self.model_type

        def save_processing(self, *args):
            pass

    config = types.SimpleNamespace(architectures=["Lfm2VlForConditionalGeneration"])
    monkeypatch.setattr(builder_module, "LFM2Model", type("LFM2Model", (FakeModel,), {}))
    monkeypatch.setattr(builder_module, "LFM2VLModel", type("LFM2VLModel", (FakeModel,), {}))

    builder_module.create_model(
        "LiquidAI/LFM2.5-VL-1.6B",
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "int4",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        exclude_embeds=exclude_embeds,
        hf_details={"extra_kwargs": {}, "hf_name": "LiquidAI/LFM2.5-VL-1.6B", "hf_config": config},
    )

    assert captured["builder"] == builder_name
    assert captured["args"][0] is config
    assert captured["model_type"] == expected_type


def test_d1_config_with_auto_map_loads_without_remote_code(tmp_path):
    # LiquidAI/d1-3B is a stock Lfm2VlForConditionalGeneration whose config.json also names its own
    # AutoModel class (auto_map). The builder must load it without trust_remote_code and modeling_d1.py.
    auto_map = {"AutoModel": "modeling_d1.D1Model"}
    config = {"architectures": ["Lfm2VlForConditionalGeneration"], "auto_map": auto_map, "model_type": "lfm2_vl"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(Path(__file__).parents[2] / "models" / "lfm2-vl" / name, tmp_path / name)

    hf_config = builder_module.get_hf_details("LiquidAI/d1-3B", str(tmp_path), str(tmp_path / "cache"), {})["hf_config"]

    assert type(hf_config).__name__ == "Lfm2VlConfig"
    assert hf_config.auto_map == auto_map


@pytest.mark.parametrize("model_type", ["lfm2_vl", "lfm2_vl_text"])
def test_lfm2_vl_loads_the_vlm_transformers_model(monkeypatch, model_type):
    # The auto-class lookup is a substring match, so the single "lfm2_vl" entry must also cover
    # "lfm2_vl_text"; falling through to AutoModelForCausalLM cannot load a VLM checkpoint.
    calls = []

    class FakeLfm2VlModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(cls)
            return cls()

    class FakeCausalLM(FakeLfm2VlModel):
        pass

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.Lfm2VlForConditionalGeneration = FakeLfm2VlModel
    transformers_stub.AutoModelForCausalLM = FakeCausalLM
    monkeypatch.setattr(base_module, "transformers", transformers_stub)

    model = Model.__new__(Model)
    model.model_type = model_type
    model.model_name_or_path = "LiquidAI/LFM2.5-VL-1.6B"
    model.cache_dir = "/cache"
    model.hf_token = True
    model.hf_remote = False
    model.quant_type = None
    model.num_layers = 16
    model.extra_options = {}

    loaded_model = model.load_weights("")

    assert calls == [FakeLfm2VlModel]
    assert not isinstance(loaded_model, FakeCausalLM)


def test_has_final_norm_finds_the_nested_lfm2_vl_embedding_norm():
    # LFM2 applies `embedding_norm` after the last layer; in LFM2-VL it lives under `language_model`.
    embedding_norm = object()
    other_module = object()
    language_model = types.SimpleNamespace(embedding_norm=embedding_norm)
    orig_model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=language_model))

    model = Model.__new__(Model)

    assert model.has_final_norm(embedding_norm, orig_model)
    assert not model.has_final_norm(other_module, orig_model)


FIXTURE = Path(__file__).parents[2] / "models" / "lfm2-vl"
# Every width of the tiny model differs from its vocabulary size: the builder takes a Linear with vocab_size
# outputs as the LM head.
VOCAB_SIZE = 320
IMAGE_TOKEN = 300
TEXT_HIDDEN_SIZE = 64
# Patch grids with a side below, at and above the 4x4 position table. A side-2 grid matches transformers only
# with its antialiased resize, and a mixed batch only with a position grid and tokens per image.
GRIDS = [(2, 6), (4, 4), (6, 8)]


def tiny_lfm2_vl(projector_use_layernorm=True):
    text_config = transformers.Lfm2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    vision_config = transformers.Siglip2VisionConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_patches=16,
        patch_size=4,
        vision_use_head=False,
        layer_norm_eps=0.1,
    )
    config = transformers.Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=IMAGE_TOKEN,
        projector_hidden_size=40,
        projector_use_layernorm=projector_use_layernorm,
    )
    # Larger weights than the default 0.02, so a wrong position grid or token order moves the features far past
    # the tolerance.
    config.initializer_range = 0.2
    # Seed the weights without changing the global RNG for later tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = transformers.Lfm2VlForConditionalGeneration(config).eval()
    # A large epsilon, so a LayerNorm built with the wrong one moves the features past the tolerance
    if hf_model.model.multi_modal_projector.layer_norm is not None:
        hf_model.model.multi_modal_projector.layer_norm.eps = 0.1
    return config, hf_model


def write_image_processor(path, layout="lfm2.5", resample=None):
    """The image processor settings of a checkpoint, in the LFM2.5-VL or the LFM2-VL file layout."""
    path.mkdir(exist_ok=True)
    image_processor = {
        "image_processor_type": "Lfm2VlImageProcessorFast",
        "resample": resample or (2 if layout == "lfm2.5" else 3),
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
    # LFM2.5-VL nests the image processor in processor_config.json, LFM2-VL keeps it in preprocessor_config.json.
    if layout == "lfm2.5":
        (path / "processor_config.json").write_text(json.dumps({"image_processor": image_processor}))
    else:
        (path / "processor_config.json").write_text(json.dumps({"processor_class": "Lfm2VlProcessor"}))
        (path / "preprocessor_config.json").write_text(json.dumps(image_processor))


def build(model, hf_model, out_dir):
    model.load_weights = lambda input_path: hf_model
    model.make_model("")
    model.save_model(str(out_dir))


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("projector_use_layernorm", [True, False])
def test_vision_model_matches_transformers(tmp_path, projector_use_layernorm):
    config, hf_model = tiny_lfm2_vl(projector_use_layernorm)
    vision = lfm2_vl_module.LFM2VLVisionEncoder(
        copy.deepcopy(config), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path / "cache"), {}
    )
    build(vision, hf_model, tmp_path)

    # Each image is padded to the longest one, as genai pads every image to max_num_patches.
    rng = np.random.default_rng(0)
    num_patches, patch_dim = max(rows * cols for rows, cols in GRIDS), 3 * 4 * 4
    pixel_values = np.zeros((len(GRIDS), num_patches, patch_dim), np.float32)
    pixel_attention_mask = np.zeros((len(GRIDS), num_patches), np.int64)
    for image, (rows, cols) in enumerate(GRIDS):
        pixel_values[image, : rows * cols] = rng.standard_normal((rows * cols, patch_dim))
        pixel_attention_mask[image, : rows * cols] = 1
    spatial_shapes = np.array(GRIDS, np.int64)
    inputs = {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "spatial_shapes": spatial_shapes,
    }
    (features,) = session(tmp_path / "vision.onnx").run(None, inputs)

    with torch.no_grad():
        expected = hf_model.get_image_features(**{name: torch.from_numpy(value) for name, value in inputs.items()})
    expected = torch.cat(expected.pooler_output).numpy()
    assert features.shape == (sum(rows * cols // 4 for rows, cols in GRIDS), TEXT_HIDDEN_SIZE)
    np.testing.assert_allclose(features, expected, rtol=1e-5, atol=1e-5)
    # WebGPU has no antialiased Resize, and these ops leave it for the CPU with copies of the activations.
    op_types = {node.op_type for node in onnx.load(tmp_path / "vision.onnx", load_external_data=False).graph.node}
    assert not op_types & {"Resize", "NonZero", "Compress", "Loop", "If"}


@pytest.mark.parametrize("table_dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("features_dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize(
    "ids",
    [[[1, IMAGE_TOKEN, IMAGE_TOKEN, 5, IMAGE_TOKEN]], [[IMAGE_TOKEN, 2], [3, IMAGE_TOKEN]], [[7, 8, 9]], [[4]]],
)
def test_embedding_model_matches_transformers(tmp_path, table_dtype, features_dtype, ids):
    config, hf_model = tiny_lfm2_vl()
    embedding = lfm2_vl_module.LFM2VLEmbedding(
        copy.deepcopy(config), table_dtype, table_dtype, "cpu", str(tmp_path / "cache"), {}
    )
    embedding.embed_attrs["image_features_dtype"] = features_dtype
    build(embedding, hf_model, tmp_path)

    ids = np.array(ids, np.int64)
    rng = np.random.default_rng(0)
    features = rng.standard_normal(((ids == IMAGE_TOKEN).sum(), TEXT_HIDDEN_SIZE))
    features = features.astype(np.float32 if features_dtype == ir.DataType.FLOAT else np.float16)
    (embeds,) = session(tmp_path / "embeddings.onnx").run(None, {"input_ids": ids, "image_features": features})

    torch_dtype = torch.float32 if table_dtype == ir.DataType.FLOAT else torch.float16
    with torch.no_grad():
        expected = hf_model.get_input_embeddings()(torch.from_numpy(ids)).to(torch_dtype)
    is_image = torch.from_numpy(ids == IMAGE_TOKEN).unsqueeze(-1)
    expected = expected.masked_scatter(is_image, torch.from_numpy(features).to(torch_dtype)).numpy()
    np.testing.assert_array_equal(embeds, expected)


@pytest.mark.parametrize(
    "ep, io_dtype, onnx_dtype, vision_dtype",
    [
        # int4 quantizes the decoder only
        ("cpu", ir.DataType.FLOAT, ir.DataType.INT4, ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16, ir.DataType.FLOAT16, ir.DataType.FLOAT16),
        # A bf16 vision model drifts much further from fp32 than an fp16 one
        ("cuda", ir.DataType.BFLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT16),
    ],
)
def test_precision_and_quantization_reach_only_the_decoder(tmp_path, ep, io_dtype, onnx_dtype, vision_dtype):
    config, hf_model = tiny_lfm2_vl()
    write_image_processor(tmp_path / "checkpoint")
    config.name_or_path = str(tmp_path / "checkpoint")
    options = {"exclude_embeds": True}
    model = builder_module.LFM2VLModel(
        copy.deepcopy(config), io_dtype, onnx_dtype, ep, str(tmp_path / "cache"), options
    )
    for part in (model.vision, model.embedding, model.decoder):
        part.load_weights = lambda input_path: hf_model
    model.make_model("")

    # The decoder is the one the builder made before the pipeline existed
    decoder = builder_module.LFM2Model(
        copy.deepcopy(config), io_dtype, onnx_dtype, ep, str(tmp_path / "cache"), options
    )
    decoder.load_weights = lambda input_path: hf_model
    decoder.make_model("")
    assert [(node.name, node.op_type) for node in model.decoder.model.graph] == [
        (node.name, node.op_type) for node in decoder.model.graph
    ]

    (tmp_path / "out").mkdir()
    model.save_model(str(tmp_path / "out"))
    graphs = {
        name: onnx.load(tmp_path / "out" / name, load_external_data=False).graph
        for name in ("vision.onnx", "embeddings.onnx", "model.onnx")
    }
    vision, embedding = graphs["vision.onnx"], graphs["embeddings.onnx"]
    assert vision.input[0].type.tensor_type.elem_type == vision_dtype
    assert vision.output[0].type.tensor_type.elem_type == vision_dtype
    # The position table stays fp32 in an fp16 vision model, as its resize runs in fp32
    table = "model.embeddings.position_embedding.weight"
    assert {tensor.data_type for tensor in vision.initializer if tensor.name == table} == {ir.DataType.FLOAT}
    assert embedding.input[1].type.tensor_type.elem_type == vision_dtype
    assert embedding.output[0].type.tensor_type.elem_type == io_dtype
    assert {tensor.data_type for tensor in embedding.initializer if tensor.name == "model.embed_tokens.weight"} == {
        io_dtype
    }
    assert any(node.op_type == "Cast" for node in embedding.node) == (vision_dtype != io_dtype)
    is_quantized = onnx_dtype == ir.DataType.INT4
    assert any(node.op_type == "MatMulNBits" for node in graphs["model.onnx"].node) == is_quantized
    assert not any(node.op_type == "MatMulNBits" for node in [*vision.node, *embedding.node])


@pytest.mark.parametrize("layout, interpolation", [("lfm2.5", "LINEAR"), ("lfm2", "CUBIC")])
def test_genai_config_and_processor_config(tmp_path, layout, interpolation):
    config, _ = tiny_lfm2_vl()
    write_image_processor(tmp_path / "checkpoint", layout)
    config.name_or_path = str(tmp_path / "checkpoint")
    model = builder_module.LFM2VLModel(
        config, ir.DataType.FLOAT, ir.DataType.INT4, "cpu", str(tmp_path / "cache"), {"exclude_embeds": True}
    )
    (tmp_path / "out").mkdir()
    model.make_genai_config(config, {}, str(tmp_path / "out"))

    # The fixture holds the LFM2.5-VL sections, with other file names
    genai_config = json.loads((tmp_path / "out" / "genai_config.json").read_text())["model"]
    fixture = json.loads((FIXTURE / "genai_config.json").read_text())["model"]
    assert genai_config["type"] == "lfm2_vl"
    assert genai_config["decoder"]["filename"] == "model.onnx"
    assert genai_config["embedding"] == dict(fixture["embedding"], filename="embeddings.onnx")
    assert genai_config["vision"] == dict(fixture["vision"], filename="vision.onnx")
    expected = json.loads((FIXTURE / "processor_config.json").read_text())
    expected["processor"]["transforms"][1]["operation"]["attrs"]["interpolation"] = interpolation
    assert json.loads((tmp_path / "out" / "processor_config.json").read_text()) == expected


@pytest.mark.parametrize(
    "ep, extra_options, decoder_options, sub_model_options",
    [
        ("cuda", {"enable_cuda_graph": True}, {"cuda": {"enable_cuda_graph": "1"}}, {"cuda": {}}),
        ("dml", {}, {"dml": {"enable_graph_capture": "1"}}, {"dml": {}}),
        (
            "webgpu",
            {"enable_webgpu_graph": True},
            {"webgpu": {"enableGraphCapture": "1", "validationMode": "disabled"}},
            {"webgpu": {"validationMode": "disabled"}},
        ),
        ("trt-rtx", {}, {"NvTensorRtRtx": {"enable_cuda_graph": "1"}}, {"NvTensorRtRtx": {}}),
    ],
)
def test_sub_models_follow_the_decoder_without_graph_capture(
    tmp_path, ep, extra_options, decoder_options, sub_model_options
):
    # A captured graph replays the shapes of its first run, and the vision and embedding inputs change with
    # every prompt.
    config, _ = tiny_lfm2_vl()
    write_image_processor(tmp_path / "checkpoint")
    config.name_or_path = str(tmp_path / "checkpoint")
    model = builder_module.LFM2VLModel(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        ep,
        str(tmp_path / "cache"),
        {"exclude_embeds": True, **extra_options},
    )
    (tmp_path / "out").mkdir()
    model.make_genai_config(config, {}, str(tmp_path / "out"))

    genai_config = json.loads((tmp_path / "out" / "genai_config.json").read_text())["model"]
    assert genai_config["decoder"]["session_options"]["provider_options"] == [decoder_options]
    for section in ("embedding", "vision"):
        assert genai_config[section]["session_options"]["provider_options"] == [sub_model_options]


@pytest.mark.parametrize(
    "extra_options, resample, message",
    [
        ({"filename": "embeddings.onnx"}, None, "cannot be named embeddings.onnx"),
        ({"filename": "vision.onnx"}, None, "cannot be named vision.onnx"),
        ({}, 1, "Unsupported resample 1"),
    ],
)
def test_refuses_before_building(tmp_path, extra_options, resample, message):
    config, _ = tiny_lfm2_vl()
    write_image_processor(tmp_path / "checkpoint", resample=resample)
    config.name_or_path = str(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match=message):
        builder_module.LFM2VLModel(
            config,
            ir.DataType.FLOAT,
            ir.DataType.FLOAT,
            "cpu",
            str(tmp_path / "cache"),
            {"exclude_embeds": True, **extra_options},
        )


def test_refuses_a_projector_activation_other_than_gelu(tmp_path):
    config, _ = tiny_lfm2_vl()
    config.projector_hidden_act = "gelu_pytorch_tanh"
    with pytest.raises(ValueError, match="Unsupported projector_hidden_act gelu_pytorch_tanh"):
        lfm2_vl_module.LFM2VLVisionEncoder(
            config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path / "cache"), {}
        )
