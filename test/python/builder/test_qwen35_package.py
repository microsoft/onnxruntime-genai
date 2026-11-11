# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import importlib
import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _builder_test_utils import load_builder_module
from huggingface_hub import split_torch_state_dict_into_shards


@pytest.mark.parametrize(
    ("component", "class_name"),
    [
        ("text", "Qwen35TextModel"),
        ("text", "Qwen35MoETextModel"),
        ("model", "Qwen35Model"),
        ("model", "Qwen35MoEModel"),
        ("mtp", "Qwen35MTPModel"),
        ("mtp", "Qwen35DenseMTPModel"),
    ],
)
def test_qwen35_package_preserves_public_class_exports(component, class_name):
    package = load_builder_module("qwen3_5")
    owner = load_builder_module(f"qwen3_5.{component}")
    legacy = load_builder_module("qwen")
    builder_class = getattr(owner, class_name)

    assert getattr(package, class_name) is builder_class
    assert getattr(legacy, class_name) is builder_class
    assert class_name in package.__all__
    assert builder_class.__module__ == f"models.builders.qwen3_5.{component}"


@pytest.mark.parametrize(
    ("class_name", "decoder_name", "mtp_name"),
    [
        ("Qwen35Model", "Qwen35TextModel", "Qwen35DenseMTPModel"),
        ("Qwen35MoEModel", "Qwen35MoETextModel", "Qwen35MTPModel"),
    ],
)
def test_qwen35_composite_dispatch_retains_constructor_globals(monkeypatch, class_name, decoder_name, mtp_name):
    package = load_builder_module("qwen3_5")
    builder_class = getattr(package, class_name)
    decoder, mtp = object(), object()
    monkeypatch.setitem(builder_class.__init__.__globals__, decoder_name, decoder)
    monkeypatch.setitem(builder_class.__init__.__globals__, mtp_name, mtp)
    model = builder_class.__new__(builder_class)

    assert model.get_decoder_model_class() is decoder
    assert model.get_mtp_model_class() is mtp


@pytest.mark.parametrize("class_name", ["Qwen35MTPModel", "Qwen35DenseMTPModel"])
def test_qwen35_mtp_hidden_states_output_uses_normalization_producer(monkeypatch, class_name):
    package = load_builder_module("qwen3_5")
    builder_class = getattr(package, class_name)
    model = builder_class.__new__(builder_class)
    model.model = SimpleNamespace(graph=ir.Graph(inputs=[], outputs=[], nodes=[]))
    model.values = {}
    model.node_names = set()
    model.layernorm_attrs = {}
    model.io_dtype = ir.DataType.FLOAT
    model.mtp_weights = SimpleNamespace(layers=[object()], norm=object(), lm_head=object())
    hidden_shape = [1, 2, 4]
    placeholder = model.make_value("hidden_states_out", model.io_dtype, hidden_shape)

    monkeypatch.setattr(model, "make_inputs_and_outputs", lambda: None)
    monkeypatch.setattr(model, "load_mtp_weights", lambda path: None)
    monkeypatch.setattr(model, "make_preprocessing_nodes", lambda: None)
    monkeypatch.setattr(model, "make_mtp_input_projection", lambda: "projected")
    monkeypatch.setattr(model, "make_layer", lambda *args: None)
    monkeypatch.setattr(model, "make_postprocessing_nodes", lambda: None)
    monkeypatch.setattr(model, "make_hidden_state_shape", lambda: hidden_shape)

    def make_layernorm(*args, **kwargs):
        model.make_node("Sigmoid", ["projected"], ["normalized"], name="normalization")
        model.layernorm_attrs["output_0"] = "normalized"

    def make_lm_head(weight):
        model.make_node("Relu", ["normalized"], ["logits"], name="lm_head")

    monkeypatch.setattr(model, "make_layernorm", make_layernorm)
    monkeypatch.setattr(model, "make_lm_head", make_lm_head)
    model.make_model("unused-checkpoint")

    graph = model.model.graph
    hidden = graph.outputs[0]
    assert hidden.name == "hidden_states_out"
    assert hidden is model.values["hidden_states_out"]
    assert hidden is not placeholder
    assert hidden.producer() is graph[0]
    assert graph[1].inputs[0] is hidden
    assert "normalized" not in model.values
    assert hidden.dtype == model.io_dtype
    assert list(hidden.shape) == hidden_shape
    assert [node.op_type for node in graph] == ["Sigmoid", "Relu"]


@pytest.mark.parametrize("grid", [(1, 4, 6), (1, 6, 4), (1, 8, 8)])
@pytest.mark.parametrize(
    ("source_module", "config_name", "model_name"),
    [
        ("qwen3_5", "Qwen3_5VisionConfig", "Qwen3_5VisionModel"),
        ("qwen4_exp", "Qwen4ExpVisionConfig", "Qwen4ExpVisionModel"),
    ],
)
def test_qwen35_vision_export_matches_source_features(tmp_path, grid, source_module, config_name, model_name):
    source = pytest.importorskip(f"transformers.models.{source_module}.modeling_{source_module}")
    config = getattr(source, config_name)(
        depth=2,
        hidden_size=32,
        intermediate_size=48,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    visual = getattr(source, model_name)(config).eval()
    builder = load_builder_module("qwen3_8.vision").Qwen4ExpVisionModel(config, visual, ir.DataType.FLOAT)
    builder.save_model(str(tmp_path))

    patches = torch.randn(np.prod(grid), 3 * 2 * 2 * 2)
    image_grid = torch.tensor([grid])
    with torch.inference_mode():
        expected = visual(patches, grid_thw=image_grid).pooler_output.numpy()
    session = ort.InferenceSession(str(tmp_path / "vision.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pixel_values": patches.numpy(), "image_grid_thw": image_grid.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    assert (tmp_path / "vision.onnx.data").exists()


def test_vision_rotary_preserves_float32_intermediates_with_float16_io():
    vision = load_builder_module("qwen3_8.vision")
    builder = vision.Qwen4ExpVisionModel.__new__(vision.Qwen4ExpVisionModel)
    vision.Qwen4ExpGraphModel.__init__(builder, ir.DataType.FLOAT16, "vision.onnx", "vision_rotary")
    builder.num_heads, builder.head_size = 2, 8
    builder.visual = SimpleNamespace(rotary_pos_emb=SimpleNamespace(inv_freq=torch.tensor([1.0, 0.01])))
    for name in ("rows", "cols"):
        builder.graph.inputs.append(builder.make_value(name, ir.DataType.INT64, ["num_patches"]))
    builder.graph.inputs.append(builder.make_value("query", ir.DataType.FLOAT16, ["num_patches", 2, 8]))
    cos, sin = builder.make_rotary_embeddings("rows", "cols")
    output = builder.apply_rotary(0, "q", "query", cos, sin)
    builder.graph.outputs.append(builder.values[output])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        ir.to_proto(builder.model).SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rows = np.array([1, 2, 7, 9, 24, 43], dtype=np.int64)
    cols = np.array([0, 1, 7, 9, 21, 31], dtype=np.int64)
    query = np.random.default_rng(7).normal(size=(6, 2, 8)).astype(np.float16) * np.float16(100)
    frequencies = np.stack([rows, cols], axis=-1).astype(np.float32)[..., None] * np.array([1, 0.01], np.float32)
    angles = np.tile(frequencies.reshape(6, 4), (1, 2))[:, None, :]
    rotated = np.concatenate([-query[..., 4:], query[..., :4]], axis=-1).astype(np.float32)
    expected = (query.astype(np.float32) * np.cos(angles) + rotated * np.sin(angles)).astype(np.float16)
    actual = session.run(None, {"rows": rows, "cols": cols, "query": query})[0]
    np.testing.assert_array_equal(actual, expected)


def tiny_qwen35_config(hybrid=False):
    source = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    return source.Qwen3_5Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2 if hybrid else 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 101,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "layer_types": ["linear_attention", "full_attention"] if hybrid else ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 128,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
        image_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_qwen35_visual_pytorch_loading_maps_only_compatible_checkpoints(monkeypatch, tmp_path, legacy):
    source = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    load_builder_module("base")
    vision = importlib.import_module("models.loaders.vision")

    config = tiny_qwen35_config().vision_config
    expected = source.Qwen3_5VisionModel(config).eval()
    state = {f"model.visual.{name}": tensor for name, tensor in expected.state_dict().items()}
    state["model.language_model.embed_tokens.weight"] = torch.ones(8, 32)
    torch.save(state, tmp_path / "pytorch_model.bin", _use_new_zipfile_serialization=not legacy)
    load = torch.load
    calls = []

    def load_checkpoint(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load_checkpoint)
    actual = vision.load_qwen35_visual(source.Qwen3_5VisionModel, config, str(tmp_path))

    assert len(calls) == 1
    assert calls[0]["mmap"] == (not legacy)
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["safetensors", "pytorch", "pytorch-sharded"])
def test_qwen35_packed_vision_emits_pure_lookup_without_drafter(monkeypatch, tmp_path, hybrid, checkpoint_format):
    source = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    config = tiny_qwen35_config(hybrid)
    weights = source.Qwen3_5ForConditionalGeneration(config).eval()
    source_path = tmp_path / "source"
    if checkpoint_format == "safetensors":
        weights.save_pretrained(str(source_path))
    else:
        config.save_pretrained(str(source_path))
        state = weights.state_dict()
        shards = split_torch_state_dict_into_shards(
            state,
            filename_pattern="pytorch_model{suffix}.bin",
            max_shard_size="10KB" if checkpoint_format == "pytorch-sharded" else "5GB",
        )
        for filename, names in shards.filename_to_tensors.items():
            torch.save({name: state[name] for name in names}, source_path / filename)
        if shards.is_sharded:
            (source_path / "pytorch_model.bin.index.json").write_text(
                json.dumps(
                    {
                        "metadata": shards.metadata,
                        "weight_map": shards.tensor_to_filename,
                    }
                )
            )
    builder = load_builder_module("qwen3_5").Qwen35Model(
        config,
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        "cuda",
        "",
        {"use_paged_attention": True, "exclude_embeds": True, "state_update_capacity": 7},
    )
    monkeypatch.setattr(builder.decoder, "load_weights", lambda path: weights)
    builder.make_model(str(source_path))
    patches = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        expected_features = weights.model.visual(patches, grid_thw=grid).pooler_output
        actual_features = builder.vision.visual(patches, grid_thw=grid).pooler_output
    torch.testing.assert_close(actual_features, expected_features)
    builder.save_model(str(tmp_path))
    builder.make_genai_config(config, {}, str(tmp_path))

    decoder = ir.load(str(tmp_path / "model.onnx"))
    embeds = next(value for value in decoder.graph.inputs if value.name == "inputs_embeds")
    assert list(embeds.shape) == ["num_tokens", 32]
    session = ort.InferenceSession(str(tmp_path / "embedding.onnx"), providers=["CPUExecutionProvider"])
    ids = np.array([3, 7, 9], dtype=np.int64)
    actual = session.run(None, {"input_ids": ids})[0]
    expected = weights.model.language_model.embed_tokens(torch.from_numpy(ids)).detach().numpy()
    np.testing.assert_array_equal(actual, expected.astype(np.float16))
    metadata = json.loads((tmp_path / "genai_config.json").read_text())["model"]
    assert metadata["type"] == "qwen3_5"
    assert metadata["embedding"]["inputs"] == {"input_ids": "input_ids"}
    assert metadata["vision"]["patch_size"] == 2
    assert metadata["decoder"]["inputs"]["inputs_embeds"] == "inputs_embeds"
    assert (tmp_path / "vision.onnx").exists()
    if hybrid:
        assert metadata["decoder"]["state_update_capacity"] == 7
        assert metadata["decoder"]["inputs"]["state_update_active"] == "state_update_active"
        conv = next(node for node in decoder.graph if node.op_type == "VarlenCausalConvWithState")
        recurrent = next(node for node in decoder.graph if node.op_type == "GatedDeltaNet")
        assert len(conv.inputs) == 6
        assert len(recurrent.inputs) == 11


@pytest.mark.parametrize("exclude_embeds", [False, True])
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen35_export_type_distinguishes_text_from_complete_vision(monkeypatch, tmp_path, exclude_embeds, model_type):
    from models import builder as builder_module  # noqa: PLC0415

    config = tiny_qwen35_config()
    builder_class = builder_module.Qwen35Model
    if model_type == "qwen3_5_moe":
        transformers = pytest.importorskip("transformers")
        options = config.to_dict()
        options.pop("model_type")
        options["text_config"].pop("model_type")
        options["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        config = transformers.Qwen3_5MoeConfig(**options)
        builder_class = builder_module.Qwen35MoEModel
    source_path = tmp_path / "source"
    config.save_pretrained(str(source_path))
    output_path = tmp_path / "output"
    monkeypatch.setattr(builder_class, "save_processing", lambda *args: None)
    builder_module.create_model(
        str(source_path),
        str(source_path),
        str(output_path),
        "fp16",
        "cuda",
        str(tmp_path / "cache"),
        config_only=True,
        use_paged_attention=True,
        exclude_embeds=exclude_embeds,
        hf_details={"extra_kwargs": {}, "hf_name": str(source_path), "hf_config": config},
    )
    metadata = json.loads((output_path / "genai_config.json").read_text())["model"]
    assert metadata["type"] == (model_type if exclude_embeds else model_type + "_text")
    assert ("vision" in metadata) == exclude_embeds
    assert ("embedding" in metadata) == exclude_embeds


def test_qwen35_text_only_cannot_export_packed_vision(tmp_path):
    owner = load_builder_module("qwen3_5")
    with pytest.raises(ValueError, match="cannot use text_only"):
        owner.Qwen35Model(
            tiny_qwen35_config(),
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT16,
            "cuda",
            str(tmp_path),
            {"use_paged_attention": True, "exclude_embeds": True, "text_only": True},
        )


@pytest.mark.parametrize("exclude_embeds", [False, True])
def test_qwen35_mtp_remains_text_only(tmp_path, exclude_embeds):
    config = tiny_qwen35_config()
    config.text_config.mtp_num_hidden_layers = 1
    builder_class = load_builder_module("qwen3_5").Qwen35Model
    options = {"use_paged_attention": True, "exclude_embeds": exclude_embeds}
    if exclude_embeds:
        with pytest.raises(ValueError, match="vision does not support MTP"):
            builder_class(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", str(tmp_path), options)
    else:
        builder = builder_class(config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", str(tmp_path), options)
        assert builder.mtp is not None


def test_qwen35_processor_download_honors_source_revision(monkeypatch, tmp_path):
    owner = load_builder_module("qwen3_8.vision")
    metadata = tmp_path / "preprocessor.json"
    metadata.write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 16, "longest_edge": 256},
                "patch_size": 2,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    downloads = []

    def download(repository, filename, **kwargs):
        downloads.append((repository, filename, kwargs))
        return str(metadata)

    monkeypatch.setattr(owner, "hf_hub_download", download)
    owner.save_image_processor(
        "remote/qwen", {"revision": "pinned", "cache_dir": str(tmp_path)}, str(tmp_path), "token"
    )
    assert downloads == [
        (
            "remote/qwen",
            "preprocessor_config.json",
            {"revision": "pinned", "cache_dir": str(tmp_path), "token": "token"},
        )
    ]


@pytest.mark.parametrize("component", ["visual", "language_model"])
def test_qwen35_adapter_does_not_silently_drop_visual_weights(monkeypatch, tmp_path, component):
    pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    owner = load_builder_module("qwen3_5.model")
    target = "visual.blocks.0.attn.qkv" if component == "visual" else "language_model.layers.0.self_attn.q_proj"
    safetensors.save_file(
        {
            f"base_model.model.model.{target}.lora_A.weight": torch.ones(2, 32),
            f"base_model.model.model.{target}.lora_B.weight": torch.ones(96 if component == "visual" else 32, 2),
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    calls = []
    model = owner.Qwen35Model.__new__(owner.Qwen35Model)
    model.decoder = SimpleNamespace(
        use_paged_attention=True,
        exclude_embeds=True,
        adapter_path=str(tmp_path),
        cache_dir=str(tmp_path),
        model_name_or_path=str(tmp_path),
        io_dtype=ir.DataType.FLOAT16,
        make_model=lambda path: calls.append(path),
        resolve_transformers_class=lambda name: object,
    )
    model.hf_token = None
    model.vision_config = object()
    model.mtp = None
    monkeypatch.setattr(owner, "load_qwen35_visual", lambda *args, **kwargs: object())
    monkeypatch.setattr(owner, "Qwen4ExpVisionModel", lambda *args: object())
    monkeypatch.setattr(model, "make_dflash2_model", lambda path: None)
    monkeypatch.setattr(model, "make_dspark_model", lambda path: None)
    if component == "visual":
        with pytest.raises(ValueError, match="vision-targeted adapters"):
            model.make_model(str(tmp_path))
        assert not calls
    else:
        model.make_model(str(tmp_path))
        assert calls == [str(tmp_path)]
        assert model.vision is not None
