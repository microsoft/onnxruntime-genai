from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase, main, mock, skipIf

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from transformers import GenerationConfig

try:
    import onnxruntime as ort
except ImportError:
    ort = None


BUILDERS_DIR = (
    Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
)
REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))
sys.path.insert(0, str(BUILDERS_DIR.parent))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(
        f"models.builders.{module_name}",
        BUILDERS_DIR / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault(
    "models.builders", types.ModuleType("models.builders")
)
builders_package.__path__ = [str(BUILDERS_DIR)]

nemotron_parse = _load_builder_module("nemotron_parse")
NemotronParseModel = nemotron_parse.NemotronParseModel


def _make_builder_config(*, image_size=(32, 32), max_sequence_length=16):
    return types.SimpleNamespace(
        _name_or_path="tiny-nemotron-parse",
        architectures=["NemotronParseForConditionalGeneration"],
        decoder_start_token_id=2,
        max_sequence_length=max_sequence_length,
        image_size=list(image_size),
        encoder=types.SimpleNamespace(patch_size=4),
        decoder=types.SimpleNamespace(
            activation_function="gelu",
            d_model=8,
            decoder_attention_heads=2,
            decoder_ffn_dim=16,
            decoder_layers=1,
            eos_token_id=2,
            pad_token_id=1,
            scale_embedding=True,
            tie_word_embeddings=False,
            vocab_size=32,
        ),
    )


def _make_builder(
    *,
    config=None,
    io_dtype=nemotron_parse.ir.DataType.FLOAT16,
    onnx_dtype=None,
    ep="cuda",
    **extra_options,
):
    return NemotronParseModel(
        config if config is not None else _make_builder_config(),
        io_dtype=io_dtype,
        onnx_dtype=io_dtype if onnx_dtype is None else onnx_dtype,
        ep=ep,
        cache_dir=None,
        extra_options=extra_options,
    )


class _TinyAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)


class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.self_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.self_attn = _TinyAttention(hidden_size)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.encoder_attn = _TinyAttention(hidden_size)
        self.final_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)


class _TinyDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.decoder.vocab_size, config.decoder.d_model
        )
        self.embed_tokens.embed_scale = config.decoder.d_model**0.5
        self.layernorm_embedding = torch.nn.LayerNorm(config.decoder.d_model)
        self.layers = torch.nn.ModuleList(
            [
                _TinyDecoderLayer(
                    config.decoder.d_model,
                    config.decoder.decoder_ffn_dim,
                )
                for _ in range(config.decoder.decoder_layers)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(config.decoder.d_model)


class _TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = _TinyDecoder(config)
        self.lm_head = torch.nn.Linear(
            config.decoder.d_model,
            config.decoder.vocab_size,
            bias=False,
        )


class _TinyPatchGenerator(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.patch_size = 4
        self.num_rows = 8
        self.num_cols = 8
        self.num_skip = 4
        self.num_cls_tokens = 3
        self.num_registers = 1
        self.cpe_mode = True
        self.embedder = torch.nn.Linear(3 * 4 * 4, hidden_size, bias=False)
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, self.num_rows * self.num_cols, hidden_size)
        )
        self.cls_token = types.SimpleNamespace(
            token=torch.nn.Parameter(torch.randn(self.num_skip, hidden_size))
        )
        self.patch_normalizer = torch.nn.Identity()

    def forward(self, pixel_values):
        patches = torch.nn.functional.unfold(
            pixel_values, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        patch_rows = pixel_values.shape[-2] // self.patch_size
        patch_cols = pixel_values.shape[-1] // self.patch_size
        position = self.pos_embed.reshape(
            1, self.num_rows, self.num_cols, -1
        ).permute(0, 3, 1, 2)
        max_dim = max(patch_rows, patch_cols)
        position = torch.nn.functional.interpolate(
            position,
            size=(max_dim, max_dim),
            mode="bilinear",
            align_corners=True,
        )
        position = position[..., :patch_rows, :patch_cols]
        position = position.permute(0, 2, 3, 1).reshape(
            1, patch_rows * patch_cols, -1
        )
        patches = self.embedder(patches) + position
        prefix = self.cls_token.token.unsqueeze(0).expand(
            pixel_values.shape[0], -1, -1
        )
        return torch.cat((prefix, patches), dim=1)


class _TinyRadioAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_size // num_heads) ** -0.5
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch, sequence_length, hidden_size = hidden_states.shape
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(
                batch,
                sequence_length,
                3,
                self.num_heads,
                hidden_size // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=self.scale
        )
        return self.proj(
            context.transpose(1, 2).reshape(
                batch, sequence_length, hidden_size
            )
        )


class _TinyRadioBlock(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = _TinyRadioAttention(hidden_size, num_heads)
        self.ls1 = torch.nn.Identity()
        self.drop_path1 = torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = types.SimpleNamespace(
            fc1=torch.nn.Linear(hidden_size, intermediate_size),
            fc2=torch.nn.Linear(intermediate_size, hidden_size),
        )
        self.ls2 = torch.nn.Identity()
        self.drop_path2 = torch.nn.Identity()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        feed_forward = self.mlp.fc2(
            torch.nn.functional.gelu(
                self.mlp.fc1(self.norm2(hidden_states)), approximate="none"
            )
        )
        return hidden_states + feed_forward


class _TinyRadio(torch.nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=16, num_heads=2):
        super().__init__()
        self.model = types.SimpleNamespace(
            embed_dim=hidden_size,
            patch_generator=_TinyPatchGenerator(hidden_size),
            blocks=torch.nn.ModuleList(
                [
                    _TinyRadioBlock(
                        hidden_size, intermediate_size, num_heads
                    )
                ]
            ),
            norm=torch.nn.Identity(),
        )
        self.feature_normalizer = torch.nn.Identity()
        self.adaptors = torch.nn.ModuleDict()
        self.summary_idxs = torch.tensor([0, 1, 2])

    def forward(self, pixel_values):
        hidden_states = self.model.patch_generator(pixel_values)
        for block in self.model.blocks:
            hidden_states = block(hidden_states)
        summary = hidden_states[:, self.summary_idxs].flatten(1)
        features = hidden_states[:, self.model.patch_generator.num_skip :]
        return summary, features


class _TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.model_encoder = types.SimpleNamespace(
            radio_model=_TinyRadio(hidden_size)
        )
        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, 1)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.conv2 = torch.nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(1, 4),
            stride=(1, 4),
            bias=False,
        )
        self.layer_norm2 = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.sum_proj = torch.nn.Linear(hidden_size * 3, hidden_size)
        self.layer_norm3 = torch.nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, pixel_values):
        summary, features = self.model_encoder.radio_model(pixel_values)
        output = self.conv1(features.transpose(1, 2)).transpose(1, 2)
        output = self.layer_norm1(output)
        patch_rows = pixel_values.shape[-2] // 4
        patch_cols = pixel_values.shape[-1] // 4
        output = output.reshape(
            1, patch_rows, patch_cols, output.shape[-1]
        ).permute(0, 3, 1, 2)
        output = self.conv2(output).permute(0, 2, 3, 1).flatten(1, 2)
        output = self.layer_norm2(output)
        summary = self.layer_norm3(self.sum_proj(summary)).unsqueeze(1)
        return torch.cat((output, summary), dim=1)


class _TinyFullModel(_TinyModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = _TinyEncoder(config.decoder.d_model)


def _build_component():
    builder = _make_builder()
    component = builder.make_decoder_component()
    component.build(_TinyModel(builder.config))
    model = nemotron_parse.ir.to_proto(component.model)
    return builder, component, model


def _shape_dims(value):
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def split_heads(value, num_heads):
    batch_size, sequence_length, hidden_size = value.shape
    return value.reshape(
        batch_size,
        sequence_length,
        num_heads,
        hidden_size // num_heads,
    ).transpose(1, 2)


def _reference_attention(
    attention,
    hidden_states,
    key_value_states,
    attention_mask=None,
):
    num_heads = 2
    query = split_heads(attention.q_proj(hidden_states), num_heads)
    key = split_heads(attention.k_proj(key_value_states), num_heads)
    value = split_heads(attention.v_proj(key_value_states), num_heads)
    scores = torch.matmul(query, key.transpose(-1, -2))
    scores *= 1.0 / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        scores += attention_mask
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.matmul(probabilities, value)
    context = context.transpose(1, 2).reshape(hidden_states.shape)
    return attention.out_proj(context), key, value


def _reference_decoder(model, input_ids, attention_mask, encoder_states):
    decoder = model.decoder
    hidden_states = decoder.embed_tokens(input_ids)
    hidden_states *= decoder.embed_tokens.embed_scale
    hidden_states = decoder.layernorm_embedding(hidden_states)

    sequence_length = input_ids.shape[1]
    mask_value = torch.finfo(hidden_states.dtype).min
    padding_mask = torch.where(
        attention_mask[:, None, None, :] == 0,
        mask_value,
        0.0,
    )
    causal_mask = torch.zeros(
        (1, 1, sequence_length, sequence_length),
        dtype=hidden_states.dtype,
    )
    causal_mask.masked_fill_(
        torch.triu(
            torch.ones_like(causal_mask, dtype=torch.bool),
            diagonal=1,
        ),
        mask_value,
    )
    self_attention_mask = padding_mask + causal_mask

    caches = []
    for layer in decoder.layers:
        self_norm = layer.self_attn_layer_norm(hidden_states)
        self_attention, self_key, self_value = _reference_attention(
            layer.self_attn,
            self_norm,
            self_norm,
            self_attention_mask,
        )
        hidden_states = hidden_states + self_attention

        cross_norm = layer.encoder_attn_layer_norm(hidden_states)
        cross_attention, cross_key, cross_value = _reference_attention(
            layer.encoder_attn,
            cross_norm,
            encoder_states,
        )
        hidden_states = hidden_states + cross_attention

        final_norm = layer.final_layer_norm(hidden_states)
        feed_forward = layer.fc2(
            torch.nn.functional.gelu(
                layer.fc1(final_norm),
                approximate="none",
            )
        )
        hidden_states = hidden_states + feed_forward
        caches.append((self_key, self_value, cross_key, cross_value))

    hidden_states = decoder.layer_norm(hidden_states)
    return model.lm_head(hidden_states), caches


def _make_image_processor(**overrides):
    padding = type("PadIfNeeded", (), {"fill": 255, "border_mode": 0, "position": "center"})()
    attributes = {
        "do_normalize": True,
        "transform": types.SimpleNamespace(transforms=[padding]),
        "torch_transform": types.SimpleNamespace(transforms=[type("ToTensor", (), {})()]),
        **overrides,
    }
    return type("NemotronParseImageProcessor", (), attributes)()


class NemotronParseBuilderTests(TestCase):
    def setUp(self):
        generation = mock.patch.object(
            GenerationConfig, "from_pretrained",
            return_value=GenerationConfig(repetition_penalty=1.1),
        )
        self.generation_loader = generation.start()
        self.addCleanup(generation.stop)
        self.tokenizer = mock.Mock()
        self.tokenizer.encode.return_value = [0, 2, 0, 4, 5, 6, 2]
        processor = mock.patch.object(
            nemotron_parse.AutoProcessor, "from_pretrained",
            return_value=types.SimpleNamespace(tokenizer=self.tokenizer, image_processor=_make_image_processor()),
        )
        self.processor_loader = processor.start()
        self.addCleanup(processor.stop)

    def test_builder_method_conventions(self):
        for builder_class in (
            NemotronParseModel,
            nemotron_parse.NemotronParseEncoderComponent,
            nemotron_parse.NemotronParseDecoderComponent,
        ):
            for name, member in vars(builder_class).items():
                if name.startswith("__"):
                    continue
                with self.subTest(builder=builder_class.__name__, member=name):
                    self.assertFalse(name.startswith("_"))
                    self.assertNotIsInstance(member, staticmethod)

    def test_defaults_to_block32(self):
        builder = _make_builder()

        self.assertEqual(builder.cache_sequence_length, builder.config.max_sequence_length)
        self.assertEqual([builder.image_height, builder.image_width], builder.config.image_size)
        self.assertEqual(builder.extra_options["block_size"], 32)
        self.assertIs(builder.hf_remote, False)

    def test_inherits_model_without_mutating_checkpoint_or_options(self):
        config = _make_builder_config(max_sequence_length=32)
        original = copy.deepcopy(config)
        options = {"hf_token": "test-token"}
        builder = NemotronParseModel(config, None, None, "cpu", None, options)

        self.assertIsInstance(builder, nemotron_parse.Model)
        self.assertEqual(config, original)
        self.assertNotIn("block_size", options)
        self.assertEqual(builder.context_length, 32)
        self.assertEqual(builder.hidden_size, config.decoder.d_model)
        self.assertEqual(builder.num_layers, config.decoder.decoder_layers)
        self.assertEqual(builder.io_dtype, nemotron_parse.ir.DataType.FLOAT16)
        self.assertEqual(builder.onnx_dtype, builder.io_dtype)
        self.assertEqual(builder.hf_token, "test-token")
        self.assertEqual(builder.model_name_or_path, config._name_or_path)
        self.assertEqual(builder.make_decoder_component().context_length, builder.context_length)
        self.assertEqual(config, original)

    def test_image_dimensions_come_from_checkpoint(self):
        builder = _make_builder(config=_make_builder_config(image_size=(64, 32)))

        self.assertEqual(builder.image_height, 64)
        self.assertEqual(builder.image_width, 32)

    def test_cache_must_leave_room_for_decode(self):
        builder = _make_builder(config=_make_builder_config(max_sequence_length=8))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "leave room for at least one decoded token"
        ):
            builder.make_genai_config(builder.config, {}, tmp)
        self.tokenizer.encode.assert_called_once_with(builder.default_user_prompt, add_special_tokens=True)

    def test_requires_valid_checkpoint_dimensions_and_capacity(self):
        for image_size in (None, 768, [], [32], [32, 32, 3], [0, 32], [32, -1]):
            with self.subTest(image_size=image_size), self.assertRaisesRegex(ValueError, "image_size"):
                config = _make_builder_config()
                config.image_size = image_size
                _make_builder(config=config)
        for capacity in (0, -1):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, "max_sequence_length"):
                _make_builder(config=_make_builder_config(max_sequence_length=capacity))

    def test_decoder_component_preserves_base_output_policy(self):
        builder = _make_builder(io_dtype=nemotron_parse.ir.DataType.BFLOAT16)

        decoder = builder.make_decoder_component()

        self.assertEqual(decoder.filename, "decoder.onnx")
        self.assertEqual(
            decoder.output_types["logits"],
            nemotron_parse.ir.DataType.FLOAT,
        )
        self.assertEqual(
            decoder.output_shapes["logits"],
            [
                1,
                1,
                builder.config.decoder.vocab_size,
            ],
        )

    def test_rejects_removed_model_specific_options(self):
        options = {
            "image_height": 64, "image_width": 64, "cache_sequence_length": 32,
            "prefill_sequence_length": 8, "export_components": "encoder,decoder", "torch_dtype": "auto",
        }
        for name, value in options.items():
            with self.subTest(option=name), self.assertRaisesRegex(ValueError, f"no longer accepts extra options: {name}"):
                _make_builder(**{name: value})

    def test_save_model_always_exports_both_components(self):
        builder = _make_builder()
        builder.weights = mock.Mock()
        weights = builder.weights
        with mock.patch.object(builder, "make_encoder_component") as encoder_factory, \
                mock.patch.object(builder, "make_decoder_component") as decoder_factory:
            with tempfile.TemporaryDirectory() as tmp:
                builder.save_model(tmp)
            encoder_factory.assert_called_once_with(weights)
            encoder_factory.return_value.build.assert_called_once_with()
            encoder_factory.return_value.save_model.assert_called_once_with(tmp)
            decoder_factory.assert_called_once_with()
            decoder_factory.return_value.build.assert_called_once_with(weights)
            decoder_factory.return_value.save_model.assert_called_once_with(tmp)
        self.assertIsNone(weights.encoder)
        self.assertFalse(hasattr(builder, "weights"))

    def test_load_precision_matches_export(self):
        for dtype, expected in (
            (nemotron_parse.ir.DataType.FLOAT, torch.float32),
            (nemotron_parse.ir.DataType.FLOAT16, torch.float16),
            (nemotron_parse.ir.DataType.BFLOAT16, torch.bfloat16),
            (nemotron_parse.ir.DataType.INT4, "auto"),
        ):
            with self.subTest(dtype=dtype):
                builder = _make_builder(onnx_dtype=dtype)
                with mock.patch.object(nemotron_parse.AutoModel, "from_pretrained") as loader:
                    builder.load_model("missing-local-path")
                self.assertEqual(loader.call_args.kwargs["torch_dtype"], expected)

    def test_decoder_emits_dynamic_tensor_scatter_graph(self):
        builder, _, model = _build_component()

        self.assertEqual(
            next(
                opset.version
                for opset in model.opset_import
                if opset.domain == ""
            ),
            24,
        )
        update_nodes = [
            node
            for node in model.graph.node
            if node.op_type == "TensorScatter"
        ]
        self.assertEqual(len(update_nodes), 2)
        self.assertEqual({node.domain for node in update_nodes}, {""})
        self.assertEqual(
            list(update_nodes[0].input),
            [
                "past_key_values.0.key",
                "/model/layers.0/attn/k/Transpose/output_0",
                "cache_write_indices",
            ],
        )
        self.assertEqual(list(update_nodes[0].output), ["present.0.key"])
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in update_nodes[0].attribute
        }
        self.assertEqual(attrs, {"axis": -2, "mode": b"linear"})

        inputs = {value.name: value for value in model.graph.input}
        self.assertNotIn("encoder_hidden_states", inputs)
        self.assertEqual(
            inputs["cache_write_indices"].type.tensor_type.elem_type,
            TensorProto.INT64,
        )
        self.assertEqual(
            _shape_dims(inputs["past_key_values.0.key"]),
            [1, 2, builder.cache_sequence_length, 4],
        )
        self.assertEqual(
            _shape_dims(inputs["decoder_input_ids"]),
            [1, "sequence_length"],
        )
        self.assertEqual(
            _shape_dims(inputs["decoder_attention_mask"]),
            [1, builder.cache_sequence_length],
        )
        outputs = {value.name: value for value in model.graph.output}
        self.assertEqual(_shape_dims(inputs["cache_write_indices"]), [1])
        self.assertEqual(_shape_dims(outputs["logits"]), [1, 1, builder.config.decoder.vocab_size])
        for kind in ("key", "value"):
            self.assertEqual(
                _shape_dims(inputs[f"cross_past_key_values.0.{kind}"]),
                [1, 2, builder.encoder_sequence_length, 4],
            )
            self.assertEqual(
                _shape_dims(outputs[f"present.0.{kind}"]),
                [1, 2, builder.cache_sequence_length, 4],
            )
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
            self.assertNotIn("batch_size", _shape_dims(value), value.name)
            self.assertNotIn("encoder_sequence_length", _shape_dims(value), value.name)
        output_names = {value.name for value in model.graph.output}
        self.assertIn("present.0.key", output_names)
        self.assertNotIn("cross_present.0.key", output_names)
        self.assertTrue(
            any(node.op_type == "Range" for node in model.graph.node)
        )
        self.assertTrue(
            any(node.op_type == "Greater" for node in model.graph.node)
        )
        onnx.checker.check_model(model)

    def test_decoder_fixed_cross_cache_shapes_match_encoder(self):
        for image_size in ((32, 16), (32, 32), (64, 32)):
            with self.subTest(image_size=image_size):
                config = _make_builder_config(image_size=image_size)
                config.decoder.decoder_layers = 2
                builder = _make_builder(config=config)
                weights = _TinyFullModel(config)
                encoder = builder.make_encoder_component(weights)
                encoder.build()
                decoder = builder.make_decoder_component()
                decoder.build(weights)
                encoder_model = nemotron_parse.ir.to_proto(encoder.model)
                decoder_model = nemotron_parse.ir.to_proto(decoder.model)
                encoder_outputs = {value.name: value for value in encoder_model.graph.output}
                decoder_inputs = {value.name: value for value in decoder_model.graph.input}
                for layer in range(config.decoder.decoder_layers):
                    for kind in ("key", "value"):
                        expected = [1, 2, builder.encoder_sequence_length, 4]
                        self.assertEqual(_shape_dims(encoder_outputs[f"cross_present.{layer}.{kind}"]), expected)
                        self.assertEqual(_shape_dims(decoder_inputs[f"cross_past_key_values.{layer}.{kind}"]), expected)
                self.assertEqual(_shape_dims(decoder_inputs["decoder_input_ids"]), [1, "sequence_length"])
                onnx.checker.check_model(decoder_model)

    def test_decoder_follows_builder_node_and_output_namespaces(self):
        config = _make_builder_config()
        config.decoder.decoder_layers = 2
        builder = NemotronParseModel(
            config, nemotron_parse.ir.DataType.FLOAT, nemotron_parse.ir.DataType.FLOAT,
            "cpu", None, {},
        )
        component = builder.make_decoder_component()
        component.build(_TinyModel(config))
        model = nemotron_parse.ir.to_proto(component.model)
        public_outputs = {value.name for value in model.graph.output}
        self.assertEqual(public_outputs, {
            "logits", "present.0.key", "present.0.value", "present.1.key", "present.1.value",
        })
        self.assertEqual({value.name for value in model.graph.input}, {
            "decoder_input_ids", "decoder_attention_mask", "cache_write_indices",
            *(f"{prefix}.{layer}.{kind}" for prefix in ("past_key_values", "cross_past_key_values")
              for layer in range(2) for kind in ("key", "value")),
        })
        names = [node.name for node in model.graph.node]
        self.assertEqual(len(names), len(set(names)))
        for layer in range(2):
            self.assertIn(f"/model/layers.{layer}/attn/q_proj/MatMul", names)
            self.assertIn(f"/model/layers.{layer}/cross_attn/q_proj/MatMul", names)
            self.assertIn(f"/model/layers.{layer}/mlp/fc1/MatMul", names)
        self.check_graph_namespaces(model)

    def test_encoder_follows_builder_node_and_output_namespaces(self):
        builder = _make_builder(io_dtype=nemotron_parse.ir.DataType.FLOAT, config=_make_builder_config(image_size=(32, 16)))
        component = builder.make_encoder_component(_TinyFullModel(builder.config))
        component.build()
        model = nemotron_parse.ir.to_proto(component.model)
        self.assertEqual({value.name for value in model.graph.input}, {"pixel_values"})
        self.assertEqual({value.name for value in model.graph.output}, {
            "encoder_hidden_states",
            *(f"cross_present.{layer}.{kind}" for layer in range(builder.config.decoder.decoder_layers)
              for kind in ("key", "value")),
        })
        self.assertTrue(any(node.op_type == "Split" for node in model.graph.node))
        self.check_graph_namespaces(model)

    def check_graph_namespaces(self, model):
        public_outputs = {value.name for value in model.graph.output}
        names = [node.name for node in model.graph.node]
        self.assertEqual(len(names), len(set(names)))
        for node in model.graph.node:
            with self.subTest(node=node.name):
                # Shared constants have their own documented namespace; the base LM head uses /lm_head.
                if node.op_type == "Constant":
                    self.assertTrue(node.output[0].startswith("/model/constants/"))
                    self.assertEqual(node.name, node.output[0].replace("/constants/", "/constant_nodes/"))
                    continue
                self.assertTrue(node.name.startswith(("/model/", "/lm_head/")))
                self.assertEqual(node.name.rsplit("/", 1)[-1], node.op_type)
                for index, output in enumerate(node.output):
                    if output not in public_outputs:
                        self.assertEqual(output, f"{node.name}/output_{index}")
        onnx.checker.check_model(model)

    @skipIf(ort is None, "onnxruntime is required for numerical graph validation")
    def test_unified_decoder_matches_reference_for_prefill_and_decode(self):
        torch.manual_seed(0)
        builder = _make_builder(
            io_dtype=nemotron_parse.ir.DataType.FLOAT,
            config=_make_builder_config(max_sequence_length=8),
        )
        model = _TinyModel(builder.config).eval()
        input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        encoder_states = torch.randn(
            1,
            builder.encoder_sequence_length,
            builder.config.decoder.d_model,
        )

        with torch.no_grad():
            reference_prefill_logits, reference_prefill_caches = (
                _reference_decoder(
                    model,
                    input_ids,
                    attention_mask,
                    encoder_states,
                )
            )

        decoder = builder.make_decoder_component()
        decoder.build(model)
        session = ort.InferenceSession(
            nemotron_parse.ir.to_proto(decoder.model).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "decoder_input_ids": input_ids.numpy(),
            "decoder_attention_mask": np.array(
                [[1] * input_ids.shape[1] + [0] * 4], dtype=np.int64
            ),
            "cache_write_indices": np.array([0], dtype=np.int64),
        }
        for layer_id, (_, _, cross_key, cross_value) in enumerate(
            reference_prefill_caches
        ):
            cache = np.zeros((1, 2, 8, 4), dtype=np.float32)
            feeds[f"past_key_values.{layer_id}.key"] = cache.copy()
            feeds[f"past_key_values.{layer_id}.value"] = cache.copy()
            feeds[f"cross_past_key_values.{layer_id}.key"] = (
                cross_key.detach().numpy()
            )
            feeds[f"cross_past_key_values.{layer_id}.value"] = (
                cross_value.detach().numpy()
            )

        prefill_outputs = dict(
            zip(
                [output.name for output in session.get_outputs()],
                session.run(None, feeds),
            )
        )
        np.testing.assert_allclose(
            prefill_outputs["logits"],
            reference_prefill_logits[:, -1:, :].detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        for layer_id, (key, value, _, _) in enumerate(
            reference_prefill_caches
        ):
            for slot, expected in (("key", key), ("value", value)):
                np.testing.assert_allclose(
                    prefill_outputs[f"present.{layer_id}.{slot}"][
                        :, :, : input_ids.shape[1], :
                    ],
                    expected.detach().numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )

        next_token = torch.tensor([[6]], dtype=torch.int64)
        full_input_ids = torch.cat((input_ids, next_token), dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        with torch.no_grad():
            reference_decode_logits, reference_decode_caches = (
                _reference_decoder(
                    model,
                    full_input_ids,
                    full_attention_mask,
                    encoder_states,
                )
            )

        feeds["decoder_input_ids"] = next_token.numpy()
        feeds["decoder_attention_mask"] = np.array(
            [[1] * full_input_ids.shape[1] + [0] * 3],
            dtype=np.int64,
        )
        feeds["cache_write_indices"] = np.array(
            [input_ids.shape[1]], dtype=np.int64
        )
        for layer_id in range(builder.config.decoder.decoder_layers):
            feeds[f"past_key_values.{layer_id}.key"] = prefill_outputs[
                f"present.{layer_id}.key"
            ]
            feeds[f"past_key_values.{layer_id}.value"] = prefill_outputs[
                f"present.{layer_id}.value"
            ]

        decode_outputs = dict(
            zip(
                [output.name for output in session.get_outputs()],
                session.run(None, feeds),
            )
        )
        np.testing.assert_allclose(
            decode_outputs["logits"],
            reference_decode_logits[:, -1:, :].detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        active_length = full_input_ids.shape[1]
        for layer_id, (key, value, _, _) in enumerate(reference_decode_caches):
            np.testing.assert_allclose(
                decode_outputs[f"present.{layer_id}.key"][
                    :, :, :active_length, :
                ],
                key.detach().numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                decode_outputs[f"present.{layer_id}.value"][
                    :, :, :active_length, :
                ],
                value.detach().numpy(),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_int4_component_uses_common_qdq_block32_config(self):
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            builder = NemotronParseModel(
                _make_builder_config(),
                io_dtype=nemotron_parse.ir.DataType.FLOAT16,
                onnx_dtype=nemotron_parse.ir.DataType.INT4,
                ep="trt-rtx",
                cache_dir=str(cache_dir),
                extra_options={"use_qdq": True},
            )
            component = builder.make_decoder_component()

            self.assertIs(component.quant_attrs["use_qdq"], True)
            self.assertEqual(component.quant_attrs["matmul_block_size"], 32)

            component.build(_TinyModel(builder.config))
            component.save_model(tmp)
            serialized_model = onnx.load(
                Path(tmp) / "decoder.onnx",
                load_external_data=False,
            )
            reshape_heads = next(
                node
                for node in serialized_model.graph.node
                if node.op_type == "Constant" and node.output[0] == component.constant_names["reshape_heads"]
            )
            reshape_tensor = helper.get_attribute_value(reshape_heads.attribute[0])
            self.assertNotEqual(
                reshape_tensor.data_location,
                onnx.TensorProto.EXTERNAL,
            )
            np.testing.assert_array_equal(onnx.numpy_helper.to_array(reshape_tensor), [0, 0, 2, 4])
            self.assertTrue(all(
                initializer.data_location == TensorProto.EXTERNAL
                for initializer in serialized_model.graph.initializer
            ))
            if ort is not None:
                ort.InferenceSession(
                    str(Path(tmp) / "decoder.onnx"),
                    providers=["CPUExecutionProvider"],
                )
            model = onnx.load(
                Path(tmp) / "decoder.onnx",
                load_external_data=True,
            )

        dequantize_nodes = [
            node
            for node in model.graph.node
            if node.op_type == "DequantizeLinear"
        ]
        self.assertGreater(len(dequantize_nodes), 0)
        for node in dequantize_nodes:
            attrs = {
                attr.name: helper.get_attribute_value(attr)
                for attr in node.attribute
            }
            self.assertEqual(attrs["block_size"], 32)
            self.assertEqual(attrs["axis"], 0)
        consumers = {
            input_name: node
            for node in model.graph.node
            for input_name in node.input
        }
        for dequantize in dequantize_nodes:
            matmul = consumers[dequantize.output[0]]
            self.assertEqual(matmul.op_type, "MatMul")
            self.assertEqual(matmul.input[1], dequantize.output[0])
        onnx.checker.check_model(model)

    def test_decoder_constants_use_shared_namespace_and_precision(self):
        for dtype, torch_dtype in (
            (nemotron_parse.ir.DataType.FLOAT, torch.float32),
            (nemotron_parse.ir.DataType.FLOAT16, torch.float16),
            (nemotron_parse.ir.DataType.BFLOAT16, torch.bfloat16),
        ):
            with self.subTest(dtype=dtype):
                builder = _make_builder(io_dtype=dtype)
                component = builder.make_decoder_component()
                component.build(_TinyModel(builder.config))
                model = nemotron_parse.ir.to_proto(component.model)
                constants = {
                    node.output[0]: helper.get_attribute_value(node.attribute[0])
                    for node in model.graph.node if node.op_type == "Constant"
                }
                for key, name in component.constant_names.items():
                    if key != "key_positions":
                        self.assertTrue(name.startswith("/model/constants/"))
                        self.assertIn(name, constants)
                for key in ("mask_value", "float_zero", "attention_scale"):
                    tensor = constants[component.constant_names[key]]
                    self.assertEqual(tensor.data_type, int(dtype))
                scale = builder.config.decoder.d_model**0.5
                scale_name = f"/model/constants/{component.to_str_dtype(dtype)}/{scale}"
                self.assertIn(scale_name, constants)
                self.assertEqual(constants[scale_name].data_type, int(dtype))
                self.assertEqual(
                    float(onnx.numpy_helper.to_array(constants[scale_name])),
                    float(torch.tensor(scale, dtype=torch_dtype)),
                )
                embedding_mul = next(node for node in model.graph.node if node.name == "/model/embed_tokens/Mul")
                self.assertEqual(embedding_mul.input[1], scale_name)
                mask_value = onnx.numpy_helper.to_array(constants[component.constant_names["mask_value"]])
                self.assertEqual(float(mask_value), float(torch.finfo(torch_dtype).min))
                self.assertEqual(
                    sum(node.op_type == "Constant" for node in model.graph.node), len(constants),
                )
                onnx.checker.check_model(model)

    def test_encoder_constants_use_shared_namespace_and_precision(self):
        for dtype, torch_dtype in (
            (nemotron_parse.ir.DataType.FLOAT, torch.float32),
            (nemotron_parse.ir.DataType.FLOAT16, torch.float16),
            (nemotron_parse.ir.DataType.BFLOAT16, torch.bfloat16),
        ):
            with self.subTest(dtype=dtype):
                builder = _make_builder(io_dtype=dtype, config=_make_builder_config(image_size=(32, 16)))
                weights = _TinyFullModel(builder.config)
                component = builder.make_encoder_component(weights)
                component.build()
                model = nemotron_parse.ir.to_proto(component.model)
                constants = {
                    node.output[0]: helper.get_attribute_value(node.attribute[0])
                    for node in model.graph.node if node.op_type == "Constant"
                }
                self.assertTrue(constants)
                self.assertEqual(
                    sum(node.op_type == "Constant" for node in model.graph.node), len(constants),
                )
                for name, tensor in constants.items():
                    self.assertTrue(name.startswith("/model/constants/"))
                    self.assertNotEqual(tensor.data_location, TensorProto.EXTERNAL)
                for node in model.graph.node:
                    if node.op_type in {"Reshape", "Squeeze", "Split", "Slice", "Gather"}:
                        for name in node.input[1:]:
                            self.assertIn(name, constants)
                            self.assertEqual(constants[name].data_type, TensorProto.INT64)
                    elif node.op_type == "Mul":
                        scale = math.sqrt(float(component.vit.blocks[0].attn.scale))
                        tensor = constants[node.input[1]]
                        self.assertEqual(tensor.data_type, int(dtype))
                        self.assertEqual(
                            float(onnx.numpy_helper.to_array(tensor)),
                            float(torch.tensor(scale, dtype=torch_dtype)),
                        )
                onnx.checker.check_model(model)

    def test_trt_rtx_int4_matches_fp_reference_without_cpu_fallback(self):
        self.check_trt_rtx_int4_matches_fp_reference(dynamic_prefill=False)

    def test_trt_rtx_dynamic_int4_matches_fp_reference_without_cpu_fallback(self):
        self.check_trt_rtx_int4_matches_fp_reference(dynamic_prefill=True)

    def check_trt_rtx_int4_matches_fp_reference(self, dynamic_prefill):
        if os.environ.get("NEMOTRON_PARSE_REQUIRE_TRT_RTX") != "1":
            self.skipTest("Set NEMOTRON_PARSE_REQUIRE_TRT_RTX=1 on a TRT-RTX test host")
        self.assertIsNotNone(ort, "TRT-RTX validation requires onnxruntime")
        provider = "NvTensorRTRTXExecutionProvider"
        provider_names = {provider, "NvTensorRtRtx"}
        library = os.environ.get("ORT_TRT_RTX_EP_LIBRARY")
        devices = None
        if library:
            if not any(device.ep_name in provider_names for device in ort.get_ep_devices()):
                ort.register_execution_provider_library(provider, library)
            devices = [device for device in ort.get_ep_devices()
                       if device.ep_name in provider_names]
            self.assertTrue(devices, "The TRT-RTX plugin did not expose a supported device")
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            builder = NemotronParseModel(
                _make_builder_config(max_sequence_length=8), nemotron_parse.ir.DataType.FLOAT16,
                nemotron_parse.ir.DataType.INT4, "trt-rtx", str(cache),
                {"use_qdq": True, "block_size": 32},
            )
            weights = _TinyModel(builder.config).eval()
            component = builder.make_decoder_component()
            component.build(weights)
            component.save_model(tmp)
            exported = onnx.load(str(Path(tmp) / "decoder.onnx"))
            self.assertTrue(any(weight.data_type == TensorProto.INT4 for weight in exported.graph.initializer))
            self.assertTrue(any(node.op_type == "DequantizeLinear" for node in exported.graph.node))
            output_dtypes = {
                output.name: helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type)
                for output in exported.graph.output
            }
            reference_builder = _make_builder(
                io_dtype=nemotron_parse.ir.DataType.FLOAT, config=_make_builder_config(max_sequence_length=8),
            )
            reference = reference_builder.make_decoder_component()
            reference.build(weights)
            reference_session = ort.InferenceSession(
                nemotron_parse.ir.to_proto(reference.model).SerializeToString(), providers=["CPUExecutionProvider"],
            )
            rng = np.random.default_rng(0)
            prefill_length = 6 if dynamic_prefill else 4
            attention_mask = np.zeros((1, 8), dtype=np.int64)
            attention_mask[:, :prefill_length] = 1
            feeds = {
                "decoder_input_ids": np.arange(2, 2 + prefill_length, dtype=np.int64)[None, :],
                "decoder_attention_mask": attention_mask,
                "cache_write_indices": np.array([0], dtype=np.int64),
            }
            for kind in ("key", "value"):
                feeds[f"past_key_values.0.{kind}"] = np.zeros((1, 2, 8, 4), dtype=np.float32)
                feeds[f"cross_past_key_values.0.{kind}"] = rng.standard_normal(
                    (1, 2, builder.encoder_sequence_length, 4)
                ).astype(np.float32)

            # Keep the target's own prefill caches resident for decode, independently of the FP32 reference.
            device_feeds = {
                name: ort.OrtValue.ortvalue_from_numpy(value.astype(np.float16), "cuda", 0)
                for name, value in feeds.items() if value.dtype == np.float32
            }
            for sequence_length in (prefill_length, 1):
                options = ort.SessionOptions()
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(tmp) / f"trt_{sequence_length}")
                provider_options = {}
                if dynamic_prefill and sequence_length != 1:
                    for kind, length in (("min", 1), ("opt", 4), ("max", 7)):
                        shapes = {name: list(value.shape) for name, value in feeds.items()}
                        shapes["decoder_input_ids"][1] = length
                        provider_options[f"nv_profile_{kind}_shapes"] = ",".join(
                            name + ":" + "x".join(map(str, shape)) for name, shape in shapes.items()
                        )
                else:
                    options.add_free_dimension_override_by_name("sequence_length", sequence_length)
                if library:
                    options.add_provider_for_devices([devices[0]], provider_options)
                    session = ort.InferenceSession(str(Path(tmp) / "decoder.onnx"), sess_options=options)
                else:
                    session = ort.InferenceSession(
                        str(Path(tmp) / "decoder.onnx"), sess_options=options,
                        providers=[(provider, provider_options)],
                    )
                session.disable_fallback()
                self.assertTrue(provider_names.intersection(session.get_providers()))
                expected = dict(zip(
                    [output.name for output in reference_session.get_outputs()], reference_session.run(None, feeds)
                ))
                for name, value in feeds.items():
                    if value.dtype != np.float32:
                        device_feeds[name] = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
                binding = session.io_binding()
                for name, value in device_feeds.items():
                    binding.bind_ortvalue_input(name, value)
                output_names = [output.name for output in session.get_outputs()]
                device_outputs = {}
                for name in output_names:
                    if name.startswith("present."):
                        past_name = name.replace("present.", "past_key_values.", 1)
                        device_outputs[name] = device_feeds[past_name]
                    else:
                        device_outputs[name] = ort.OrtValue.ortvalue_from_numpy(
                            np.zeros(expected[name].shape, dtype=output_dtypes[name]), "cuda", 0
                        )
                    binding.bind_ortvalue_output(name, device_outputs[name])
                binding.synchronize_inputs()
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                actual = {name: value.numpy() for name, value in device_outputs.items()}
                # Bound absolute error relative to the FP logits scale, including values near zero.
                np.testing.assert_allclose(actual["logits"], expected["logits"], rtol=0.1, atol=0.1)
                self.assertTrue(np.isfinite(actual["logits"]).all())
                for kind in ("key", "value"):
                    present_name = f"present.0.{kind}"
                    self.assertEqual(
                        device_outputs[present_name].data_ptr(), device_feeds[f"past_key_values.0.{kind}"].data_ptr()
                    )
                    np.testing.assert_allclose(actual[present_name], expected[present_name], rtol=0.1, atol=0.1)
                    consumed = int(feeds["cache_write_indices"][0]) + sequence_length
                    np.testing.assert_array_equal(actual[present_name][:, :, consumed:, :], 0)
                events = json.loads(Path(session.end_profiling()).read_text())
                assignments = [event.get("args", {}).get("provider") for event in events
                               if event.get("cat") == "Node" and event.get("args", {}).get("provider")]
                self.assertTrue(provider_names.intersection(assignments))
                self.assertNotIn("CPUExecutionProvider", assignments)
                feeds["decoder_input_ids"] = np.array([[int(expected["logits"].argmax())]], dtype=np.int64)
                feeds["decoder_attention_mask"][0, prefill_length] = 1
                feeds["cache_write_indices"][0] = prefill_length
                for kind in ("key", "value"):
                    feeds[f"past_key_values.0.{kind}"] = expected[f"present.0.{kind}"]

    @skipIf(ort is None, "onnxruntime is required for numerical graph validation")
    def test_external_weight_exports_preserve_encoder_and_multistep_decoder(self):
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            builder = NemotronParseModel(
                _make_builder_config(image_size=(32, 16)), nemotron_parse.ir.DataType.FLOAT,
                nemotron_parse.ir.DataType.FLOAT, "cpu", str(cache),
                {},
            )
            weights = _TinyFullModel(builder.config).eval()
            encoder = builder.make_encoder_component(weights)
            encoder.build()
            decoder = builder.make_decoder_component()
            decoder.build(weights)
            saved_sessions = []
            memory_sessions = []
            for component in (encoder, decoder):
                memory_sessions.append(ort.InferenceSession(
                    nemotron_parse.ir.to_proto(component.model).SerializeToString(),
                    providers=["CPUExecutionProvider"],
                ))
                component.save_model(tmp)
                path = Path(tmp) / component.filename
                serialized = onnx.load(path, load_external_data=False)
                self.assertTrue(serialized.graph.initializer)
                self.assertTrue(all(
                    tensor.data_location == TensorProto.EXTERNAL for tensor in serialized.graph.initializer
                ))
                constants = {
                    node.output[0]: helper.get_attribute_value(node.attribute[0])
                    for node in serialized.graph.node if node.op_type == "Constant"
                }
                for node in serialized.graph.node:
                    if node.op_type in {"Reshape", "Unsqueeze", "Squeeze", "Split"}:
                        with self.subTest(component=component.filename, node=node.name):
                            self.assertIn(node.input[1], constants)
                            self.assertNotEqual(constants[node.input[1]].data_location, TensorProto.EXTERNAL)
                onnx.checker.check_model(str(path))
                saved_sessions.append(ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]))

            pixels = torch.randn(1, 3, builder.image_height, builder.image_width).numpy()
            expected_encoder = memory_sessions[0].run(None, {"pixel_values": pixels})
            actual_encoder = saved_sessions[0].run(None, {"pixel_values": pixels})
            for actual, expected in zip(actual_encoder, expected_encoder, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            encoder_outputs = dict(zip(
                [output.name for output in saved_sessions[0].get_outputs()], actual_encoder, strict=True,
            ))
            feeds = {
                "decoder_input_ids": np.array([[2, 3, 4, 5]], dtype=np.int64),
                "decoder_attention_mask": np.zeros((1, builder.cache_sequence_length), dtype=np.int64),
                "cache_write_indices": np.array([0], dtype=np.int64),
            }
            feeds["decoder_attention_mask"][:, :4] = 1
            for kind in ("key", "value"):
                feeds[f"past_key_values.0.{kind}"] = np.zeros((1, 2, builder.cache_sequence_length, 4), dtype=np.float32)
                feeds[f"cross_past_key_values.0.{kind}"] = encoder_outputs[f"cross_present.0.{kind}"]
            output_names = [output.name for output in saved_sessions[1].get_outputs()]
            for step in range(3):
                expected = memory_sessions[1].run(None, feeds)
                actual = saved_sessions[1].run(None, feeds)
                for name, value, reference in zip(output_names, actual, expected, strict=True):
                    with self.subTest(step=step, output=name):
                        np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-6)
                outputs = dict(zip(output_names, actual, strict=True))
                length = int(feeds["cache_write_indices"][0]) + feeds["decoder_input_ids"].shape[1]
                feeds["cache_write_indices"][0] = length
                feeds["decoder_attention_mask"][:, :length + 1] = 1
                feeds["decoder_input_ids"] = np.array([[int(outputs["logits"].argmax())]], dtype=np.int64)
                for kind in ("key", "value"):
                    feeds[f"past_key_values.0.{kind}"] = outputs[f"present.0.{kind}"]

    @skipIf(ort is None, "onnxruntime is required for numerical graph validation")
    def test_manual_encoder_matches_radio_neck_and_cross_cache(self):
        torch.manual_seed(1)
        builder = _make_builder(
            io_dtype=nemotron_parse.ir.DataType.FLOAT,
            config=_make_builder_config(image_size=(32, 16)),
        )
        model = _TinyFullModel(builder.config).eval()
        pixel_values = torch.randn(
            1, 3, builder.image_height, builder.image_width
        )
        with torch.no_grad():
            encoder_hidden_states = model.encoder(pixel_values)

        component = builder.make_encoder_component(model)
        component.build()
        exported = nemotron_parse.ir.to_proto(component.model)
        onnx.checker.check_model(exported)
        session = ort.InferenceSession(
            exported.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = dict(
            zip(
                [output.name for output in session.get_outputs()],
                session.run(None, {"pixel_values": pixel_values.numpy()}),
            )
        )

        self.assertEqual(
            _shape_dims(exported.graph.input[0]),
            [1, 3, builder.image_height, builder.image_width],
        )
        np.testing.assert_allclose(
            outputs["encoder_hidden_states"],
            encoder_hidden_states.numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        for layer_id, layer in enumerate(model.decoder.layers):
            for kind, projection in (
                ("key", layer.encoder_attn.k_proj),
                ("value", layer.encoder_attn.v_proj),
            ):
                expected = split_heads(
                    projection(encoder_hidden_states), 2
                ).detach().numpy()
                np.testing.assert_allclose(
                    outputs[f"cross_present.{layer_id}.{kind}"],
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                )

        self.assertFalse(
            any(
                node.op_type.startswith("ATen")
                or node.domain.startswith("pkg.torch")
                for node in exported.graph.node
            )
        )

    def test_genai_config_describes_native_cached_pipeline(self):
        builder = _make_builder()
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            tmp_path = Path(tmp)
            builder.make_genai_config("local", {}, tmp_path)
            config = json.loads(
                (tmp_path / "genai_config.json").read_text()
            )

        model_config = config["model"]
        vision = model_config["vision"]
        decoder = model_config["decoder"]
        self.assertEqual(model_config["type"], "nemotron_parse")
        self.assertEqual(
            model_config["context_length"],
            builder.cache_sequence_length,
        )
        self.assertEqual(
            vision["num_visual_tokens"],
            builder.encoder_sequence_length,
        )
        self.assertEqual(
            vision["config_filename"], "vision_processing.json"
        )
        self.assertEqual(
            model_config["encoder"]["outputs"],
            {
                "cross_present_key_names": "cross_present.%d.key",
                "cross_present_value_names": "cross_present.%d.value",
            },
        )
        self.assertNotIn("prefill_filename", decoder)
        self.assertNotIn("encoder_hidden_states", decoder["inputs"])
        self.assertEqual(
            decoder["prefill_sequence_length"],
            8,
        )
        self.assertEqual(
            decoder["inputs"]["cache_write_indices"],
            "cache_write_indices",
        )
        self.assertIs(
            config["search"]["past_present_share_buffer"], True
        )
        self.assertEqual(config["search"]["repetition_penalty"], 1.1)
        self.assertEqual(config["model"]["default_user_prompt"], NemotronParseModel.default_user_prompt)

    def test_config_uses_loaded_generation_defaults(self):
        builder = _make_builder()
        builder.generation_config = GenerationConfig(
            repetition_penalty=1.2, do_sample=True, temperature=0.8, top_k=7, top_p=0.9,
        )
        with tempfile.TemporaryDirectory() as tmp:
            builder.make_genai_config(builder.config, {}, tmp)
            search = json.loads((Path(tmp) / "genai_config.json").read_text())["search"]
        for key in ("repetition_penalty", "do_sample", "temperature", "top_k", "top_p"):
            self.assertEqual(search[key], getattr(builder.generation_config, key))
        self.generation_loader.assert_not_called()

    def test_config_reuses_shared_writer_and_provider_options(self):
        for ep, provider in (("cpu", []), ("cuda", [{"cuda": {"enable_cuda_graph": "0"}}]),
                             ("trt-rtx", [{"NvTensorRtRtx": {"enable_cuda_graph": "1"}}])):
            with self.subTest(ep=ep), tempfile.TemporaryDirectory() as tmp:
                builder = _make_builder(ep=ep, config=_make_builder_config(max_sequence_length=32))
                self.tokenizer.encode.return_value = list(range(15))
                original = copy.deepcopy(builder.config)
                with mock.patch.object(
                    nemotron_parse.Model, "make_genai_config", autospec=True,
                    side_effect=nemotron_parse.Model.make_genai_config,
                ) as shared_writer:
                    builder.make_genai_config(builder.config, {}, tmp)
                shared_writer.assert_called_once()
                config = json.loads((Path(tmp) / "genai_config.json").read_text())

                self.assertEqual(builder.config, original)
                model = config["model"]
                self.assertEqual((model["bos_token_id"], model["eos_token_id"], model["pad_token_id"]), (2, 2, 1))
                self.assertEqual(model["context_length"], 32)
                self.assertEqual(model["decoder"]["prefill_sequence_length"], 16)
                self.assertEqual(model["decoder"]["session_options"]["provider_options"], provider)
                self.assertEqual(model["decoder"]["filename"], "decoder.onnx")
                self.assertEqual(model["decoder"]["inputs"], {
                    "input_ids": "decoder_input_ids",
                    "attention_mask": "decoder_attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                    "cross_past_key_names": "cross_past_key_values.%d.key",
                    "cross_past_value_names": "cross_past_key_values.%d.value",
                    "cache_write_indices": "cache_write_indices",
                })
                self.assertEqual(model["decoder"]["outputs"], {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                })
                self.assertEqual(config["search"]["max_length"], 32)
                self.assertTrue(config["search"]["past_present_share_buffer"])

    def test_config_only_preserves_single_sequence_and_shared_cache(self):
        builder = _make_builder(config_only=True)
        builder.config.decoder.num_beams = 4
        builder.config.decoder.num_return_sequences = 2
        builder.generation_config = GenerationConfig(num_beams=4, num_return_sequences=2)
        with tempfile.TemporaryDirectory() as tmp:
            builder.make_genai_config(builder.config, {}, tmp)
            search = json.loads((Path(tmp) / "genai_config.json").read_text())["search"]
        self.assertEqual(search["num_beams"], 1)
        self.assertEqual(search["num_return_sequences"], 1)
        self.assertTrue(search["past_present_share_buffer"])

    def test_config_and_processing_reuse_checkpoint_processor(self):
        builder = _make_builder(hf_token="test-token", hf_remote=True, config_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            builder.make_genai_config(builder.config, {"local_files_only": True}, tmp)
            builder.save_processing(builder.model_name_or_path, {"local_files_only": True}, tmp)
        self.processor_loader.assert_called_once_with(
            builder.model_name_or_path, token="test-token", trust_remote_code=True, local_files_only=True,
        )
        self.tokenizer.encode.assert_called_once_with(builder.default_user_prompt, add_special_tokens=True)
        self.assertEqual(builder.prefill_sequence_length, 8)

    def test_config_requires_processor_tokenizer(self):
        self.processor_loader.return_value = types.SimpleNamespace()
        builder = _make_builder()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "does not expose a tokenizer"):
            builder.make_genai_config(builder.config, {}, tmp)

    def test_generation_config_load_uses_shared_auth_and_source(self):
        builder = _make_builder(hf_token="test-token", hf_remote=True)
        builder.model_name_or_path = "local-checkpoint"
        with tempfile.TemporaryDirectory() as tmp:
            builder.make_genai_config(builder.config, {"local_files_only": True}, tmp)
        self.generation_loader.assert_called_once_with(
            "local-checkpoint", token="test-token", trust_remote_code=True, local_files_only=True,
        )

    def test_missing_generation_config_falls_back_with_warning(self):
        builder = _make_builder()
        builder.config.decoder.repetition_penalty = 1.15
        self.generation_loader.side_effect = OSError("missing generation config")
        with tempfile.TemporaryDirectory() as tmp, self.assertWarnsRegex(UserWarning, "decoder configuration defaults"):
            builder.make_genai_config(builder.config, {}, tmp)
            search = json.loads((Path(tmp) / "genai_config.json").read_text())["search"]
        self.assertEqual(search["repetition_penalty"], 1.15)

    def test_cuda_int4_config_keeps_matmul_bias_separate(self):
        builder = _make_builder(onnx_dtype=nemotron_parse.ir.DataType.INT4)
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            builder.make_genai_config("local", {}, tmp)
            config = json.loads(
                (Path(tmp) / "genai_config.json").read_text()
            )

        session_options = config["model"]["decoder"]["session_options"]
        self.assertEqual(
            session_options["optimization.disable_specified_optimizers"],
            "MatMulAddFusion",
        )

    def test_trt_rtx_int4_config_allows_matmul_add_fusion(self):
        builder = _make_builder(
            onnx_dtype=nemotron_parse.ir.DataType.INT4,
            ep="trt-rtx",
        )
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            builder.make_genai_config("local", {}, tmp)
            config = json.loads(
                (Path(tmp) / "genai_config.json").read_text()
            )

        session_options = config["model"]["decoder"]["session_options"]
        self.assertNotIn(
            "optimization.disable_specified_optimizers", session_options
        )

    def test_trt_rtx_export_supplies_graph_specific_profiles(self):
        for image_size in ((32, 16), (32, 32), (64, 32)):
            with self.subTest(image_size=image_size), tempfile.TemporaryDirectory() as tmp:
                builder = _make_builder(ep="trt-rtx", config=_make_builder_config(image_size=image_size))
                builder.make_genai_config(builder.config, {}, tmp)
                model = json.loads((Path(tmp) / "genai_config.json").read_text())["model"]
                decoder_options = model["decoder"]["session_options"]
                vision_options = model["vision"]["session_options"]
                self.assertEqual(vision_options["provider_options"], decoder_options["provider_options"])
                for kind in ("min", "opt", "max"):
                    key = f"ep.nvtensorrtrtxexecutionprovider.nv_profile_{kind}_shapes"
                    self.assertEqual(decoder_options[key], "decoder_input_ids:1x1")
                    self.assertEqual(vision_options[key], f"pixel_values:1x3x{image_size[0]}x{image_size[1]}")

    def test_non_trt_export_does_not_add_profiles(self):
        for ep in ("cpu", "cuda"):
            with self.subTest(ep=ep), tempfile.TemporaryDirectory() as tmp:
                builder = _make_builder(ep=ep)
                builder.make_genai_config(builder.config, {}, tmp)
                model = json.loads((Path(tmp) / "genai_config.json").read_text())["model"]
                self.assertNotIn("session_options", model["vision"])
                self.assertFalse(any("nv_profile" in key for key in model["decoder"]["session_options"]))

    def test_save_processing_writes_native_config_and_tokenizer(self):
        builder = _make_builder()
        tokenizer = mock.Mock()
        image_processor = _make_image_processor()
        processor = types.SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            with mock.patch.object(
                nemotron_parse.AutoProcessor,
                "from_pretrained",
                return_value=processor,
            ):
                builder.save_processing("local", {}, tmp)

            tokenizer.save_pretrained.assert_called_once_with(tmp)
            self.assertEqual((Path(tmp) / "chat_template.jinja").read_text(), NemotronParseModel.chat_template)
            config = json.loads(
                (Path(tmp) / "vision_processing.json").read_text()
            )

        operations = [
            transform["operation"]["type"]
            for transform in config["processor"]["transforms"]
        ]
        self.assertEqual(operations, ["DecodeImage"])
        self.assertEqual(config["image_mean"], [0.48145466, 0.4578275, 0.40821073])
        self.assertEqual(config["image_std"], [0.26862954, 0.26130258, 0.27577711])
        self.assertEqual((config["image_height"], config["image_width"]), (32, 32))
        self.assertEqual(
            config["processor"]["transforms"][0]["operation"]["attrs"],
            {"color_space": "RGB"},
        )

    def test_save_processing_preserves_resolved_normalization(self):
        mean = [0.48145465, 0.4578275, 0.40821073]
        std = [0.26862955, 0.26130258, 0.27577711]
        processor = types.SimpleNamespace(
            tokenizer=mock.Mock(),
            image_processor=_make_image_processor(image_mean=mean, image_std=std),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(nemotron_parse.AutoProcessor, "from_pretrained", return_value=processor):
                _make_builder().save_processing("local", {}, tmp)
            config = json.loads((Path(tmp) / "vision_processing.json").read_text())
        self.assertEqual(config["image_mean"], mean)
        self.assertEqual(config["image_std"], std)

    def test_rejects_incompatible_preprocessing_before_saving(self):
        cases = {
            "do_normalize": False, "do_rescale": False, "do_resize": False, "do_pad": False,
            "rescale_factor": 1.0, "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
            "resample": 0, "interpolation": 0, "padding_value": 0, "padding_position": "top_left",
        }
        for name, value in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                image_processor = _make_image_processor(**{name: value})
                tokenizer = mock.Mock()
                processor = types.SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)
                with mock.patch.object(nemotron_parse.AutoProcessor, "from_pretrained", return_value=processor):
                    with self.assertRaisesRegex(ValueError, name):
                        _make_builder().save_processing("local", {}, tmp)
                tokenizer.save_pretrained.assert_not_called()
                self.assertFalse((Path(tmp) / "vision_processing.json").exists())

    def test_rejects_padding_library_default_changes(self):
        processor = _make_image_processor()
        processor.transform.transforms[0].fill = 0
        with self.assertRaisesRegex(ValueError, "white padding"):
            _make_builder().validate_image_processor(processor)


if __name__ == "__main__":
    main()
