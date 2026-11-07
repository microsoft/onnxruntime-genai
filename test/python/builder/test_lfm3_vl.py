# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM3-VL model builder.

With `exclude_embeds` the builder writes the whole `lfm3_vl` pipeline: the LFM3 decoder, the embedding model, the
vision model and their configs. No transformers release has the LFM3-VL classes, so the vision graph is compared with
a stand-in that has the module names of the checkpoints' remote code and the vision math written out here. Real
checkpoints are compared outside this suite, since their remote code cannot run in CI.
"""

from __future__ import annotations

import copy
import importlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
from _builder_test_utils import check_node_shapes
from test_lfm2_vl import builder_module, ir
from test_lfm3 import _stand_in_lfm3

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

LFM3VLModel = builder_module.LFM3VLModel
LFM3VLVisionEncoder = importlib.import_module("builders.lfm3_vl").LFM3VLVisionEncoder

FIXTURE = Path(__file__).parents[2] / "models" / "lfm3-vl"
# Every width differs from the vocabulary size: the builder takes a Linear with vocab_size outputs as the LM head.
VOCAB_SIZE = 320
IMAGE_TOKEN = 300
TEXT_HIDDEN_SIZE = 72
VISION_HIDDEN_SIZE = 64
HEAD_DIM = 16
PATCH_SIZE = 4
PATCH_DIM = 3 * 2 * PATCH_SIZE**2
MERGE_SIZE = 2
PROJECTOR_EPS = 0.05
NORM_EPS = 0.1
ROPE_THETA = 1e6
LAYER_TYPES = ["conv", "full_attention", "conv"]
GRIDS = [(2, 2), (4, 6), (8, 4), (6, 10)]


def stand_in_config(tmp_path):
    """The attributes of an LFM3-VL config.json that the builder reads, at a tiny size."""
    text_config = transformers.PretrainedConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        intermediate_size=96,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "sliding_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=TEXT_HIDDEN_SIZE // 4,
        max_position_embeddings=128,
        sliding_window=4,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        },
        norm_eps=1e-3,
        conv_L_cache=3,
        tie_word_embeddings=False,
    )
    vision_config = transformers.PretrainedConfig(
        hidden_size=VISION_HIDDEN_SIZE,
        intermediate_size=96,
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        num_attention_heads=4,
        num_key_value_heads=2,
        in_channels=3,
        patch_size=PATCH_SIZE,
        temporal_patch_size=2,
        spatial_merge_size=MERGE_SIZE,
        conv_kernel_size=3,
        norm_eps=NORM_EPS,
        rope_parameters={"rope_type": "axial", "rope_theta": ROPE_THETA},
    )
    config = transformers.PretrainedConfig(
        architectures=["Lfm3VlForConditionalGeneration"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=IMAGE_TOKEN,
        downsample_factor=MERGE_SIZE,
        projector_hidden_act="gelu",
        projector_hidden_size=40,
        projector_bias=True,
        projector_layer_norm_eps=PROJECTOR_EPS,
    )
    config.name_or_path = str(tmp_path / "checkpoint")
    return config


def write_image_processor(tmp_path, **overrides):
    """The image processor settings of a checkpoint, in the checkpoint's nested file layout."""
    path = tmp_path / "checkpoint"
    path.mkdir(exist_ok=True)
    image_processor = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "Lfm3VlImageProcessor",
        "patch_size": PATCH_SIZE,
        "merge_size": MERGE_SIZE,
        "temporal_patch_size": 2,
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 4096, "longest_edge": 262144},
        **overrides,
    }
    (path / "processor_config.json").write_text(json.dumps({"image_processor": image_processor}), encoding="utf-8")


class Lfm3VlVisionLayer(torch.nn.Module):
    """A layer with the module names of the checkpoints' remote code. The builder finds layers by this class name's
    suffix and reads their weights by attribute name."""

    def __init__(self, config, layer_type):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        if layer_type == "conv":
            self.conv = torch.nn.Module()
            self.conv.in_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
            kernel = config.conv_kernel_size
            self.conv.spatial_conv = torch.nn.Conv2d(
                hidden, hidden, kernel, padding=kernel // 2, groups=hidden, bias=False
            )
            self.conv.out_proj = torch.nn.Linear(hidden, hidden, bias=False)
        else:
            heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(hidden, heads * HEAD_DIM, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(hidden, kv_heads * HEAD_DIM, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(hidden, kv_heads * HEAD_DIM, bias=False)
            self.self_attn.out_proj = torch.nn.Linear(heads * HEAD_DIM, hidden, bias=False)
            self.self_attn.q_layernorm = torch.nn.RMSNorm(HEAD_DIM, eps=config.norm_eps)
            self.self_attn.k_layernorm = torch.nn.RMSNorm(HEAD_DIM, eps=config.norm_eps)
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w1 = torch.nn.Linear(hidden, intermediate, bias=False)
        self.feed_forward.w3 = torch.nn.Linear(hidden, intermediate, bias=False)
        self.feed_forward.w2 = torch.nn.Linear(intermediate, hidden, bias=False)
        self.operator_norm = torch.nn.RMSNorm(hidden, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(hidden, eps=config.norm_eps)


def stand_in_model(config):
    """A module tree in the checkpoint's order, with larger weights than the default 0.02 so that an error in any
    part moves the features far past the tolerance, and norm weights and biases away from their defaults."""
    vision, text = config.vision_config, config.text_config
    hf_model = torch.nn.Module()
    hf_model.model = torch.nn.Module()
    tower = hf_model.model.vision_tower = torch.nn.Module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        tower.patchifier = torch.nn.Linear(PATCH_DIM, vision.hidden_size, bias=False)
        tower.embedding_norm = torch.nn.RMSNorm(vision.hidden_size, eps=NORM_EPS)
        tower.layers = torch.nn.ModuleList(Lfm3VlVisionLayer(vision, layer_type) for layer_type in vision.layer_types)
        projector = hf_model.model.multi_modal_projector = torch.nn.Module()
        projector.layer_norm = torch.nn.LayerNorm(vision.hidden_size, eps=PROJECTOR_EPS)
        token_dim = MERGE_SIZE**2 * vision.hidden_size
        projector.linear_1 = torch.nn.Linear(token_dim, config.projector_hidden_size)
        projector.linear_2 = torch.nn.Linear(config.projector_hidden_size, text.hidden_size)
        for module in (tower, projector):
            for name, parameter in module.named_parameters():
                is_norm = "norm" in name and name.endswith("weight")
                torch.nn.init.normal_(parameter, mean=1.0 if is_norm else 0.0, std=0.2)
        text_model = _stand_in_lfm3(
            types.SimpleNamespace(
                vocab_size=VOCAB_SIZE,
                hidden_size=text.hidden_size,
                intermediate_size=text.intermediate_size,
                num_attention_heads=text.num_attention_heads,
                num_key_value_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
                conv_L_cache=text.conv_L_cache,
                layer_types=text.layer_types,
            ),
            use_sandwich_norm=True,
        )
    hf_model.model.language_model = text_model.model
    hf_model.lm_head = text_model.lm_head
    hf_model.get_input_embeddings = lambda: text_model.model.embed_tokens
    return hf_model


def block_order(rows, cols):
    # The raster index of each row of pixel_values and of each merged token's patches
    return [
        (block_row * MERGE_SIZE + patch_row) * cols + block_col * MERGE_SIZE + patch_col
        for block_row in range(rows // MERGE_SIZE)
        for block_col in range(cols // MERGE_SIZE)
        for patch_row in range(MERGE_SIZE)
        for patch_col in range(MERGE_SIZE)
    ]


def rotate(states, cos, sin):
    half = states.shape[-1] // 2
    first, second = states[..., :half], states[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def run_attention(attention, states, cos, sin):
    heads, kv_heads = attention.q_proj.out_features // HEAD_DIM, attention.k_proj.out_features // HEAD_DIM
    query = rotate(attention.q_layernorm(attention.q_proj(states).reshape(-1, heads, HEAD_DIM)), cos, sin)
    key = rotate(attention.k_layernorm(attention.k_proj(states).reshape(-1, kv_heads, HEAD_DIM)), cos, sin)
    value = attention.v_proj(states).reshape(-1, kv_heads, HEAD_DIM)
    key = key.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    value = value.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
    scores = query.transpose(0, 1) @ key.transpose(1, 2) * HEAD_DIM**-0.5
    return attention.out_proj((scores.softmax(-1) @ value).transpose(0, 1).reshape(states.shape[0], -1))


def run_conv(conv, states, rows, cols):
    gate, scale, values = conv.in_proj(states).chunk(3, dim=-1)
    grid = (gate * values).T.reshape(1, -1, rows, cols)
    return conv.out_proj(scale * conv.spatial_conv(grid).reshape(states.shape[1], -1).T)


def expected_features(hf_model, pixel_values, rows, cols):
    """The vision math on the rows of one image. Every layer runs in raster order, as in the graph."""
    tower, projector = hf_model.model.vision_tower, hf_model.model.multi_modal_projector
    order = torch.tensor(block_order(rows, cols))
    raster = torch.empty(pixel_values.shape)
    raster[order] = pixel_values
    states = tower.embedding_norm(tower.patchifier(raster))

    inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, HEAD_DIM // 2, 2).float() / (HEAD_DIM // 2))
    positions = torch.arange(rows * cols)
    angles = torch.cat(
        (torch.outer((positions // cols).float(), inv_freq), torch.outer((positions % cols).float(), inv_freq)), -1
    )
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    for layer in tower.layers:
        normed = layer.operator_norm(states)
        if hasattr(layer, "conv"):
            states = states + run_conv(layer.conv, normed, rows, cols)
        else:
            states = states + run_attention(layer.self_attn, normed, cos, sin)
        normed = layer.ffn_norm(states)
        mlp = layer.feed_forward
        states = states + mlp.w2(torch.nn.functional.silu(mlp.w1(normed)) * mlp.w3(normed))

    tokens = projector.layer_norm(states)[order].reshape(-1, MERGE_SIZE**2 * states.shape[-1])
    return projector.linear_2(torch.nn.functional.gelu(projector.linear_1(tokens)))


def build_vision(tmp_path, hf_model, config, cache_length=64):
    dtype = ir.DataType.FLOAT
    vision = LFM3VLVisionEncoder(copy.deepcopy(config), dtype, dtype, "cpu", str(tmp_path / "cache"), {})
    vision.rope_attrs["cache_length"] = cache_length
    vision.load_weights = lambda input_path: hf_model
    vision.make_model("")
    vision.save_model(str(tmp_path))
    return ort.InferenceSession(str(tmp_path / "vision.onnx"), providers=["CPUExecutionProvider"]), vision


def run_vision(session, pixel_values, rows, cols):
    grid = np.array([[1, rows, cols]], np.int64)
    return session.run(None, {"pixel_values": pixel_values.astype(np.float32), "image_grid_thw": grid})[0]


def random_pixels(rows, cols):
    return np.random.default_rng(0).standard_normal((rows * cols, PATCH_DIM), dtype=np.float32)


@pytest.mark.parametrize("rows, cols", GRIDS)
def test_vision_model_matches_the_vision_math(tmp_path, rows, cols):
    config = stand_in_config(tmp_path)
    hf_model = stand_in_model(config)
    session, _ = build_vision(tmp_path, hf_model, config)

    pixels = random_pixels(rows, cols)
    features = run_vision(session, pixels, rows, cols)
    with torch.no_grad():
        expected = expected_features(hf_model, torch.from_numpy(pixels), rows, cols).numpy()
    assert features.shape == (rows * cols // MERGE_SIZE**2, TEXT_HIDDEN_SIZE)
    np.testing.assert_allclose(features, expected, rtol=1e-4, atol=1e-5)


def test_vision_graph_declares_every_shape_and_avoids_cpu_only_ops(tmp_path):
    config = stand_in_config(tmp_path)
    _, vision = build_vision(tmp_path, stand_in_model(config), config)

    model_proto = ir.serde.serialize_model(vision.model)
    check_node_shapes(model_proto)
    ops = {node.op_type for node in model_proto.graph.node}
    assert {"Conv", "MultiHeadAttention", "RotaryEmbedding"} <= ops
    # WebGPU has no antialiased Resize, and these ops leave it for the CPU with copies of the activations.
    assert not ops & {"Resize", "NonZero", "Compress", "Loop", "If", "Identity"}


@pytest.mark.parametrize("layer_id", range(len(LAYER_TYPES)))
def test_each_vision_layer_reaches_the_features(tmp_path, layer_id):
    config = stand_in_config(tmp_path)
    hf_model = stand_in_model(config)
    rows, cols = 4, 6
    pixels = random_pixels(rows, cols)
    baseline, _ = build_vision(tmp_path, hf_model, config)
    features = run_vision(baseline, pixels, rows, cols)

    layer = hf_model.model.vision_tower.layers[layer_id]
    with torch.no_grad():
        operator = layer.conv if hasattr(layer, "conv") else layer.self_attn
        operator.out_proj.weight.zero_()
        layer.feed_forward.w2.weight.zero_()
    (tmp_path / "zeroed").mkdir()
    session, _ = build_vision(tmp_path / "zeroed", hf_model, config)
    assert np.abs(features - run_vision(session, pixels, rows, cols)).max() > 1e-3


def test_rope_table_covers_the_longest_side(tmp_path):
    # The checkpoint's smart_resize is Qwen2-VL's: it rounds each side to a multiple of 8 pixels (half to even) and
    # scales down only above max_pixels. 4000x20 becomes 4000x16, 64,000 pixels, a long side past
    # sqrt(200 * max_pixels) = 3620. Aspect ratios above 200 are refused, so a short side of 20 or less allows at most
    # 4000. One of 21 or more rounds to 24 or more, so a long side past 2730 goes over max_pixels and is scaled down to
    # at most 3620.
    min_pixels, max_pixels = 4096, 65536
    longest_side = 4000

    config = stand_in_config(tmp_path)
    write_image_processor(tmp_path, size={"shortest_edge": min_pixels, "longest_edge": max_pixels})
    model = LFM3VLModel(
        config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path / "cache"), {"exclude_embeds": True}
    )
    cache_length = model.vision.rope_attrs["cache_length"]
    session, _ = build_vision(tmp_path, stand_in_model(config), config, cache_length)
    rows, cols = longest_side // PATCH_SIZE, 16 // PATCH_SIZE
    features = run_vision(session, random_pixels(rows, cols), rows, cols)
    assert features.shape == (rows * cols // MERGE_SIZE**2, TEXT_HIDDEN_SIZE)
    assert cache_length == (3620 // 8 + 101) * MERGE_SIZE


def build_pipeline(tmp_path, config, io_dtype, onnx_dtype, ep, extra_options):
    write_image_processor(tmp_path)
    hf_model = stand_in_model(config)
    model = LFM3VLModel(copy.deepcopy(config), io_dtype, onnx_dtype, ep, str(tmp_path / "cache"), extra_options)
    for part in (model.vision, model.embedding, model.decoder):
        part.load_weights = lambda input_path: hf_model
    model.make_model("")
    return model, hf_model


@pytest.mark.parametrize(
    "ep, io_dtype, onnx_dtype, vision_dtype",
    [
        ("cpu", ir.DataType.FLOAT, ir.DataType.FLOAT, ir.DataType.FLOAT),
        # int4 quantizes the decoder only
        ("cpu", ir.DataType.FLOAT, ir.DataType.INT4, ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16, ir.DataType.FLOAT16, ir.DataType.FLOAT16),
        # A bf16 vision model drifts much further from fp32 than an fp16 one
        ("cuda", ir.DataType.BFLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT16),
        # WebGPU's MatMul and Gemm sum fp16 in fp16
        ("webgpu", ir.DataType.FLOAT16, ir.DataType.FLOAT16, ir.DataType.FLOAT),
        ("webgpu", ir.DataType.FLOAT, ir.DataType.FLOAT, ir.DataType.FLOAT),
    ],
)
def test_precision_and_quantization_reach_only_the_decoder(tmp_path, ep, io_dtype, onnx_dtype, vision_dtype):
    config = stand_in_config(tmp_path)
    options = {"exclude_embeds": True}
    model, hf_model = build_pipeline(tmp_path, config, io_dtype, onnx_dtype, ep, options)

    # The decoder is the one LFM3Model makes on its own
    decoder = builder_module.LFM3Model(
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
    assert embedding.input[1].type.tensor_type.elem_type == vision_dtype
    assert embedding.output[0].type.tensor_type.elem_type == io_dtype
    assert any(node.op_type == "Cast" for node in embedding.node) == (vision_dtype != io_dtype)
    is_quantized = onnx_dtype == ir.DataType.INT4
    assert any(node.op_type == "MatMulNBits" for node in graphs["model.onnx"].node) == is_quantized
    assert not any(node.op_type == "MatMulNBits" for node in [*vision.node, *embedding.node])


def test_embedding_model_writes_the_features_over_the_image_tokens(tmp_path):
    config = stand_in_config(tmp_path)
    model, hf_model = build_pipeline(
        tmp_path, config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", {"exclude_embeds": True}
    )
    (tmp_path / "out").mkdir()
    model.save_model(str(tmp_path / "out"))
    session = ort.InferenceSession(str(tmp_path / "out" / "embeddings.onnx"), providers=["CPUExecutionProvider"])

    ids = np.array([[1, IMAGE_TOKEN, IMAGE_TOKEN, 5, IMAGE_TOKEN]], np.int64)
    features = np.random.default_rng(0).standard_normal((3, TEXT_HIDDEN_SIZE), dtype=np.float32)
    (embeds,) = session.run(None, {"input_ids": ids, "image_features": features})

    with torch.no_grad():
        expected = hf_model.get_input_embeddings()(torch.from_numpy(ids))
    mask = torch.from_numpy(ids == IMAGE_TOKEN).unsqueeze(-1)
    np.testing.assert_array_equal(embeds, expected.masked_scatter(mask, torch.from_numpy(features)).numpy())


@pytest.mark.parametrize(
    "exclude_embeds, builder_name, model_type",
    [(True, "LFM3VLModel", "lfm3_vl"), (False, "LFM3Model", "lfm3")],
)
def test_architecture_picks_the_pipeline_or_the_text_model(
    monkeypatch, tmp_path, exclude_embeds, builder_name, model_type
):
    config = stand_in_config(tmp_path)
    write_image_processor(tmp_path)
    captured = {}

    def recording(parent):
        class Recording(parent):
            def make_genai_config(self, *args):
                parts = (self.vision, self.decoder) if isinstance(self, LFM3VLModel) else (self,)
                for part in parts:
                    part.resolve_transformers_class = lambda class_name: class_name
                captured["builder"] = parent.__name__
                captured["loaders"] = {part.resolve_model_loader() for part in parts}
                captured["model_type"] = (self.decoder if isinstance(self, LFM3VLModel) else self).model_type

            def save_processing(self, *args):
                pass

        return Recording

    monkeypatch.setattr(builder_module, "LFM3VLModel", recording(builder_module.LFM3VLModel))
    monkeypatch.setattr(builder_module, "LFM3Model", recording(builder_module.LFM3Model))
    builder_module.create_model(
        config.name_or_path,
        str(tmp_path / "input"),
        str(tmp_path / "output"),
        "fp32",
        "cpu",
        str(tmp_path / "cache"),
        config_only=True,
        exclude_embeds=exclude_embeds,
        hf_details={"extra_kwargs": {}, "hf_name": config.name_or_path, "hf_config": config},
    )

    assert captured == {
        "builder": builder_name,
        "loaders": {"AutoModelForImageTextToText"},
        "model_type": model_type,
    }


def test_genai_config_and_processor_config(tmp_path):
    # The fixture holds the sections of the released LFM3-VL-2B, with other file names
    config = stand_in_config(tmp_path)
    config.vision_config.patch_size = 16
    write_image_processor(tmp_path, patch_size=16, size={"shortest_edge": 65536, "longest_edge": 16777216})
    model = LFM3VLModel(
        config, ir.DataType.FLOAT, ir.DataType.INT4, "cpu", str(tmp_path / "cache"), {"exclude_embeds": True}
    )
    (tmp_path / "out").mkdir()
    model.make_genai_config(config, {}, str(tmp_path / "out"))

    genai_config = json.loads((tmp_path / "out" / "genai_config.json").read_text(encoding="utf-8"))["model"]
    fixture = json.loads((FIXTURE / "genai_config.json").read_text(encoding="utf-8"))["model"]
    assert genai_config["type"] == "lfm3_vl"
    assert genai_config["decoder"]["filename"] == "model.onnx"
    assert genai_config["decoder"]["layer_types"] == config.text_config.layer_types
    assert genai_config["embedding"] == dict(fixture["embedding"], filename="embeddings.onnx")
    assert genai_config["vision"] == dict(fixture["vision"], filename="vision.onnx")
    assert json.loads((tmp_path / "out" / "processor_config.json").read_text(encoding="utf-8")) == json.loads(
        (FIXTURE / "processor_config.json").read_text(encoding="utf-8")
    )


FP16, FP32 = ir.DataType.FLOAT16, ir.DataType.FLOAT


@pytest.mark.parametrize(
    "case, error, message",
    [
        ({"ep": "dml", "dtype": FP16}, NotImplementedError, "not dml"),
        ({"ep": "trt-rtx", "dtype": FP16}, NotImplementedError, "not trt-rtx"),
        ({"ep": "cuda", "dtype": FP32}, NotImplementedError, "selects Attention"),
        ({"dtype": FP16}, NotImplementedError, "selects MultiHeadAttention"),
        ({"ep": "cuda", "dtype": FP16, "options": {"use_paged_attention": True}}, NotImplementedError, "use_paged"),
        ({"options": {"filename": "vision.onnx"}}, ValueError, "cannot be named vision.onnx"),
        ({"image_processor": {"resample": 2}}, ValueError, "resample"),
        ({"image_processor": {"do_resize": False}}, ValueError, "do_resize"),
        ({"image_processor": {"do_rescale": False}}, ValueError, "do_rescale"),
        ({"image_processor": {"do_normalize": False}}, ValueError, "do_normalize"),
        ({"image_processor": {"do_convert_rgb": False}}, ValueError, "do_convert_rgb"),
        ({"image_processor": {"patch_size": 8}}, ValueError, "patch_size"),
        ({"image_processor": {"merge_size": 1}}, ValueError, "merge_size"),
        ({"image_processor": {"temporal_patch_size": 1}}, ValueError, "temporal_patch_size"),
        ({"image_processor": {"max_pixels": 65536}}, ValueError, "max_pixels"),
        (
            {"image_processor": {"temporal_patch_size": 1}, "vision_config": {"temporal_patch_size": 1}},
            ValueError,
            "temporal_patch_size 1: expected 2",
        ),
        ({"config": {"downsample_factor": 1}}, ValueError, "downsample_factor"),
        ({"config": {"projector_hidden_act": "relu"}}, ValueError, "projector_hidden_act relu"),
        ({"vision_config": {"rope_parameters": {"rope_type": "default"}}}, ValueError, "vision RoPE default"),
        ({"vision_config": {"in_channels": 1}}, ValueError, "in_channels 1"),
        ({"vision_config": {"layer_types": ["conv", "sliding_attention"]}}, ValueError, "vision layer types"),
    ],
)
def test_refuses_before_building(tmp_path, case, error, message):
    config = stand_in_config(tmp_path)
    for target, section in ((config, "config"), (config.vision_config, "vision_config")):
        for name, value in case.get(section, {}).items():
            setattr(target, name, value)
    write_image_processor(tmp_path, **case.get("image_processor", {}))
    dtype = case.get("dtype", FP32)
    options = {"exclude_embeds": True, **case.get("options", {})}
    with pytest.raises(error, match=message):
        LFM3VLModel(config, dtype, dtype, case.get("ep", "cpu"), str(tmp_path / "cache"), options)
