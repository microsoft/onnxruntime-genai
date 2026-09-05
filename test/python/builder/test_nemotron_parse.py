from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import types
from pathlib import Path
from unittest import TestCase, main, mock, skipIf

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

try:
    import onnxruntime as ort
except ImportError:
    ort = None


BUILDERS_DIR = (
    Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
)
REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


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


def _make_builder_config():
    return types.SimpleNamespace(
        _name_or_path="tiny-nemotron-parse",
        architectures=["NemotronParseForConditionalGeneration"],
        decoder_start_token_id=2,
        max_sequence_length=16,
        image_size=[32, 32],
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
    io_dtype=nemotron_parse.ir.DataType.FLOAT16,
    onnx_dtype=None,
    ep="cuda",
    **extra_options,
):
    return NemotronParseModel(
        _make_builder_config(),
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
    component = builder._make_decoder_component()
    component.build(_TinyModel(builder.config))
    model = nemotron_parse.ir.to_proto(component.model)
    return builder, component, model


def _shape_dims(value):
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def _split_heads(value, num_heads):
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
    query = _split_heads(attention.q_proj(hidden_states), num_heads)
    key = _split_heads(attention.k_proj(key_value_states), num_heads)
    value = _split_heads(attention.v_proj(key_value_states), num_heads)
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


class NemotronParseBuilderTests(TestCase):
    def test_defaults_to_block32(self):
        builder = _make_builder()

        self.assertEqual(builder.prefill_sequence_length, 8)
        self.assertEqual(builder.extra_options["block_size"], 32)
        self.assertIs(builder.hf_remote, False)

    def test_single_image_dimension_override_preserves_checkpoint_default(self):
        builder = _make_builder(image_height=64)

        self.assertEqual(builder.image_height, 64)
        self.assertEqual(builder.image_width, 32)

    def test_cache_must_leave_room_for_decode(self):
        with self.assertRaisesRegex(
            ValueError, "leave room for at least one decoded token"
        ):
            _make_builder(
                cache_sequence_length=8,
                prefill_sequence_length=8,
            )

    def test_decoder_component_preserves_base_output_policy(self):
        builder = _make_builder(io_dtype=nemotron_parse.ir.DataType.BFLOAT16)

        decoder = builder._make_decoder_component()

        self.assertEqual(decoder.filename, "decoder.onnx")
        self.assertEqual(
            decoder.output_types["logits"],
            nemotron_parse.ir.DataType.FLOAT,
        )
        self.assertEqual(
            decoder.output_shapes["logits"],
            [
                "batch_size",
                1,
                builder.config.decoder.vocab_size,
            ],
        )

    def test_rejects_unknown_export_component(self):
        with self.assertRaisesRegex(
            ValueError, "only encoder and/or decoder"
        ):
            _make_builder(export_components="encoder,tokenizer")

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
                "/decoder/layers.0/self_attn/k/Transpose/output_0",
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
            ["batch_size", 2, builder.cache_sequence_length, 4],
        )
        self.assertEqual(
            _shape_dims(inputs["decoder_input_ids"]),
            ["batch_size", "sequence_length"],
        )
        self.assertEqual(
            _shape_dims(inputs["decoder_attention_mask"]),
            ["batch_size", builder.cache_sequence_length],
        )
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

    @skipIf(ort is None, "onnxruntime is required for numerical graph validation")
    def test_unified_decoder_matches_reference_for_prefill_and_decode(self):
        torch.manual_seed(0)
        builder = _make_builder(
            io_dtype=nemotron_parse.ir.DataType.FLOAT,
            prefill_sequence_length=4,
            cache_sequence_length=8,
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

        decoder = builder._make_decoder_component()
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
            component = builder._make_decoder_component()

            self.assertIs(component.quant_attrs["use_qdq"], True)
            self.assertEqual(component.quant_attrs["matmul_block_size"], 32)

            component.build(_TinyModel(builder.config))
            component.save_model(tmp)
            serialized_model = onnx.load(
                Path(tmp) / "decoder.onnx",
                load_external_data=False,
            )
            reshape_heads = next(
                initializer
                for initializer in serialized_model.graph.initializer
                if initializer.name.endswith("/reshape_heads")
            )
            self.assertNotEqual(
                reshape_heads.data_location,
                onnx.TensorProto.EXTERNAL,
            )
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

    @skipIf(ort is None, "onnxruntime is required for numerical graph validation")
    def test_manual_encoder_matches_radio_neck_and_cross_cache(self):
        torch.manual_seed(1)
        builder = _make_builder(
            io_dtype=nemotron_parse.ir.DataType.FLOAT,
            image_width=16,
        )
        model = _TinyFullModel(builder.config).eval()
        pixel_values = torch.randn(
            1, 3, builder.image_height, builder.image_width
        )
        with torch.no_grad():
            encoder_hidden_states = model.encoder(pixel_values)

        component = builder._make_encoder_component(model)
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
                expected = _split_heads(
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
            vision["config_filename"], "processor_config.json"
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
            builder.prefill_sequence_length,
        )
        self.assertEqual(
            decoder["inputs"]["cache_write_indices"],
            "cache_write_indices",
        )
        self.assertIs(
            config["search"]["past_present_share_buffer"], True
        )

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

    def test_save_processing_writes_native_config_and_tokenizer(self):
        builder = _make_builder()
        tokenizer = mock.Mock()
        processor = types.SimpleNamespace(tokenizer=tokenizer)
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
            config = json.loads(
                (Path(tmp) / "processor_config.json").read_text()
            )

        operations = [
            transform["operation"]["type"]
            for transform in config["processor"]["transforms"]
        ]
        self.assertEqual(operations, ["DecodeImage"])
        self.assertEqual(
            config["processor"]["transforms"][0]["operation"]["attrs"],
            {"color_space": "RGB"},
        )


if __name__ == "__main__":
    main()
