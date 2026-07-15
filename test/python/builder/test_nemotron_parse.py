from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import onnx
from onnx import TensorProto, helper


BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(
        f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

nemotron_parse = _load_builder_module("nemotron_parse")
NemotronParseModel = nemotron_parse.NemotronParseModel


def _make_cache_value(name: str, sequence_dim):
    return helper.make_tensor_value_info(
        name,
        TensorProto.FLOAT16,
        ["batch_size", 2, sequence_dim, 4],
    )


def _make_decoder_model():
    graph = helper.make_graph(
        [
            helper.make_node(
                "Concat",
                ["past_key_values.0.key", "new_key"],
                ["present.0.key"],
                axis=2,
            ),
            helper.make_node(
                "Concat",
                ["past_key_values.0.value", "new_value"],
                ["present.0.value"],
                axis=2,
            ),
            helper.make_node(
                "Identity",
                ["cross_past_key_values.0.key"],
                ["cross_present.0.key"],
            ),
            helper.make_node(
                "Identity",
                ["cross_past_key_values.0.value"],
                ["cross_present.0.value"],
            ),
            helper.make_node("Identity", ["logits_in"], ["logits"]),
        ],
        "decoder",
        [
            helper.make_tensor_value_info("decoder_attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]),
            _make_cache_value("past_key_values.0.key", "past_sequence_length"),
            _make_cache_value("past_key_values.0.value", "past_sequence_length"),
            _make_cache_value("cross_past_key_values.0.key", "encoder_sequence_length"),
            _make_cache_value("cross_past_key_values.0.value", "encoder_sequence_length"),
            _make_cache_value("new_key", 1),
            _make_cache_value("new_value", 1),
            helper.make_tensor_value_info("logits_in", TensorProto.FLOAT16, ["batch_size", 1, 8]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch_size", 1, 8]),
            _make_cache_value("present.0.key", "total_sequence_length"),
            _make_cache_value("present.0.value", "total_sequence_length"),
            _make_cache_value("cross_present.0.key", "encoder_sequence_length"),
            _make_cache_value("cross_present.0.value", "encoder_sequence_length"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def _shape_dims(value):
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return dims


def _make_builder_config():
    return types.SimpleNamespace(
        decoder_start_token_id=2,
        max_sequence_length=1032,
        image_size=[2048, 1648],
        encoder=types.SimpleNamespace(patch_size=16),
        decoder=types.SimpleNamespace(
            d_model=768,
            decoder_attention_heads=8,
            decoder_layers=1,
            eos_token_id=0,
            pad_token_id=1,
            vocab_size=50016,
        ),
    )


def _make_tensor_scatter_builder():
    return NemotronParseModel(
        _make_builder_config(),
        io_dtype=None,
        onnx_dtype=None,
        ep="cuda",
        cache_dir=None,
        extra_options={
            "decoder_cache_mode": "tensor_scatter",
            "cache_sequence_length": 16,
        },
    )


class NemotronParseTensorScatterTests(unittest.TestCase):
    def test_prefill_returns_only_last_token_logits(self):
        class Decoder(nemotron_parse.torch.nn.Module):
            def forward(
                self,
                input_ids,
                attention_mask,
                encoder_hidden_states,
                use_cache,
                return_dict,
            ):
                del attention_mask, encoder_hidden_states, use_cache, return_dict
                batch_size, sequence_length = input_ids.shape
                hidden_states = nemotron_parse.torch.arange(
                    batch_size * sequence_length * 4,
                    dtype=nemotron_parse.torch.float32,
                ).reshape(batch_size, sequence_length, 4)
                cache = nemotron_parse.torch.zeros((batch_size, 1, sequence_length, 4))
                return types.SimpleNamespace(
                    last_hidden_state=hidden_states,
                    past_key_values=[(cache, cache, cache, cache)],
                )

        model = nemotron_parse.torch.nn.Module()
        model.decoder = Decoder()
        model.lm_head = nemotron_parse.torch.nn.Identity()
        wrapper = nemotron_parse._NemotronParseDecoderPrefill(model)
        decoder_input_ids = nemotron_parse.torch.ones((1, 3), dtype=nemotron_parse.torch.long)
        outputs = wrapper(
            decoder_input_ids,
            nemotron_parse.torch.ones_like(decoder_input_ids),
            nemotron_parse.torch.zeros((1, 2, 4)),
        )

        self.assertEqual(tuple(outputs[0].shape), (1, 1, 4))
        nemotron_parse.torch.testing.assert_close(
            outputs[0], nemotron_parse.torch.tensor([[[8.0, 9.0, 10.0, 11.0]]])
        )

    def test_direct_tensor_scatter_export_requires_torch_2_12_or_newer(self):
        with mock.patch.object(nemotron_parse.torch, "__version__", "2.11.9+cu130"):
            self.assertIs(nemotron_parse._supports_direct_tensor_scatter_export(), False)

        for version in ("2.12.0", "2.12.1+cu130", "2.13.0", "3.0.0"):
            with self.subTest(version=version), mock.patch.object(nemotron_parse.torch, "__version__", version):
                self.assertIs(nemotron_parse._supports_direct_tensor_scatter_export(), True)

    def test_tensor_scatter_uses_compatibility_rewrite_before_torch_2_12(self):
        with mock.patch.object(nemotron_parse.torch, "__version__", "2.11.9+cu130"):
            model = _make_tensor_scatter_builder()

        self.assertEqual(model.decoder_cache_mode, "tensor_scatter")
        self.assertIs(model.use_direct_tensor_scatter_export, False)

    def test_tensor_scatter_cache_must_leave_room_for_decode(self):
        with self.assertRaisesRegex(
            ValueError, "leave room for at least one decoded token"
        ):
            NemotronParseModel(
                _make_builder_config(),
                io_dtype=None,
                onnx_dtype=None,
                ep="cuda",
                cache_dir=None,
                extra_options={
                    "decoder_cache_mode": "tensor_scatter",
                    "cache_sequence_length": 8,
                    "prefill_sequence_length": 8,
                },
            )

    def test_nemotron_parse_defaults_to_tensor_scatter_cache(self):
        with mock.patch.object(nemotron_parse.torch, "__version__", "2.12.0"):
            model = NemotronParseModel(
                _make_builder_config(),
                io_dtype=None,
                onnx_dtype=None,
                ep="cuda",
                cache_dir=None,
                extra_options={},
            )

        self.assertEqual(model.decoder_cache_mode, "tensor_scatter")
        self.assertEqual(model.prefill_sequence_length, 8)

    def test_nemotron_parse_rejects_unsupported_cache_mode(self):
        with self.assertRaisesRegex(ValueError, "supported modes: tensor_scatter"):
            NemotronParseModel(
                _make_builder_config(),
                io_dtype=None,
                onnx_dtype=None,
                ep="cuda",
                cache_dir=None,
                extra_options={"decoder_cache_mode": "concat"},
            )

    def test_export_components_rejects_unknown_component(self):
        with self.assertRaisesRegex(ValueError, "only encoder and/or decoder"):
            NemotronParseModel(
                _make_builder_config(),
                io_dtype=None,
                onnx_dtype=None,
                ep="cpu",
                cache_dir=None,
                extra_options={"export_components": "encoder,tokenizer"},
            )

    def test_external_data_save_removes_previous_data_file(self):
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            model_path = Path(tmp) / "decoder.onnx"
            external_data_path = Path(f"{model_path}.data")
            external_data_path.write_bytes(b"stale")

            with mock.patch.object(nemotron_parse.onnx, "save_model") as save_model:
                nemotron_parse._save_model_with_external_data(
                    object(), str(model_path)
                )

            self.assertFalse(external_data_path.exists())
            self.assertEqual(
                save_model.call_args.kwargs["location"], "decoder.onnx.data"
            )

    def test_encoder_export_has_fully_static_input_shape(self):
        class Encoder(nemotron_parse.torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nemotron_parse.torch.nn.Parameter(
                    nemotron_parse.torch.ones(1)
                )

            def forward(self, pixel_values, return_dict=True):
                del return_dict
                return types.SimpleNamespace(last_hidden_state=pixel_values)

        builder = _make_tensor_scatter_builder()
        model = types.SimpleNamespace(encoder=Encoder())
        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            with mock.patch.object(nemotron_parse.torch.onnx, "export") as export:
                builder._export_encoder(model, tmp)

        exported_input = export.call_args.args[1][0]
        self.assertEqual(
            tuple(exported_input.shape),
            (1, 3, builder.image_height, builder.image_width),
        )
        self.assertNotIn("dynamic_axes", export.call_args.kwargs)

    def test_tensor_scatter_rewrite_replaces_self_cache_concat_and_static_shapes(self):
        model = _make_decoder_model()

        summary = nemotron_parse._apply_tensor_scatter_to_decoder_model(
            model,
            cache_sequence_length=16,
        )

        self.assertEqual(summary["patched_self_updates"], 2)
        self.assertEqual(summary["removed_cross_outputs"], 2)
        self.assertIs(summary["added_cache_write_indices"], True)
        cache_write_input = next(value for value in model.graph.input if value.name == "cache_write_indices")
        self.assertEqual(cache_write_input.type.tensor_type.elem_type, TensorProto.INT64)

        update_nodes = [node for node in model.graph.node if node.op_type == "TensorScatter"]
        self.assertEqual(len(update_nodes), 2)
        self.assertEqual({node.domain for node in update_nodes}, {""})
        self.assertEqual(
            list(update_nodes[0].input),
            ["past_key_values.0.key", "new_key", "cache_write_indices"],
        )
        self.assertEqual(
            list(update_nodes[1].input),
            ["past_key_values.0.value", "new_value", "cache_write_indices"],
        )

        output_names = {value.name for value in model.graph.output}
        self.assertIn("present.0.key", output_names)
        self.assertIn("present.0.value", output_names)
        self.assertNotIn("cross_present.0.key", output_names)
        self.assertNotIn("cross_present.0.value", output_names)

        self.assertFalse(any(function.name == "TensorScatter" for function in model.functions))
        self.assertEqual(
            next(opset.version for opset in model.opset_import if opset.domain == ""),
            24,
        )
        self.assertFalse(any(opset.domain == "trt" for opset in model.opset_import))
        onnx.checker.check_model(model)

        shapes = {
            value.name: _shape_dims(value)
            for value in list(model.graph.input) + list(model.graph.output)
            if value.name in {"decoder_attention_mask", "past_key_values.0.key", "present.0.key"}
        }
        self.assertEqual(shapes["decoder_attention_mask"], ["batch_size", 16])
        self.assertEqual(shapes["past_key_values.0.key"], ["batch_size", 2, 16, 4])
        self.assertEqual(shapes["present.0.key"], ["batch_size", 2, 16, 4])

    def test_genai_config_marks_tensor_scatter_decoder(self):
        with mock.patch.object(nemotron_parse.torch, "__version__", "2.12.0"):
            model = _make_tensor_scatter_builder()

        self.assertIs(model.use_direct_tensor_scatter_export, True)

        tmp_root = REPO_ROOT / "build" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            tmp_path = Path(tmp)
            model.make_genai_config("local", {}, tmp_path)
            config = json.loads((tmp_path / "genai_config.json").read_text())
            model_config = config["model"]
            vision = model_config["vision"]
            decoder = model_config["decoder"]

        self.assertEqual(model_config["type"], "nemotron_parse")
        self.assertEqual(model_config["context_length"], 16)
        self.assertEqual(vision["num_visual_tokens"], model.encoder_sequence_length)
        self.assertEqual(
            vision["outputs"]["image_features"], "encoder_hidden_states"
        )
        self.assertNotIn("image_height", vision)
        self.assertNotIn("image_width", vision)
        self.assertNotIn("encoder_sequence_length", vision)
        self.assertEqual(config["search"]["max_length"], 16)
        self.assertIs(config["search"]["past_present_share_buffer"], True)
        self.assertNotIn("session_options", model_config["vision"])
        self.assertEqual(decoder["prefill_filename"], "decoder_prefill.onnx")
        self.assertEqual(decoder["prefill_sequence_length"], 8)
        self.assertEqual(decoder["cache_update_mode"], "tensor_scatter")
        self.assertEqual(
            decoder["inputs"]["cache_write_indices"], "cache_write_indices"
        )
        self.assertNotIn("cache_sequence_length", decoder)
        for stale_field in (
            "use_cache",
            "inplace_kv_cache_update",
            "self_cache_outputs_only",
            "kv_cache_update_op_type",
            "kv_cache_update_op_domain",
            "removed_cross_present_outputs",
        ):
            self.assertNotIn(stale_field, decoder)


if __name__ == "__main__":
    unittest.main()
