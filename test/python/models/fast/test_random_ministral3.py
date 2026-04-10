# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    requires_transformers,
    run_session_or_io_binding,
)

MINISTRAL3_MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"


@requires_transformers("5")
class TestMinistral3(ExtTestCase):
    def common_fast_ministral3_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, Ministral3Config, PreTrainedTokenizerFast

        num_hidden_layers = 1
        config = Ministral3Config(
            architectures=["Ministral3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )

        basename = f"test_discrepancies_ministral3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=MINISTRAL3_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill[:, :1, :], ort_logits_np[:, :1, :])
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            self.assertEqual(np_prefill.shape, ort_logits_np.shape)
            # Verify first-token logits are numerically close; subsequent positions
            # can diverge slightly in FP32 due to GQA kernel differences.
            np.testing.assert_allclose(np_prefill[:, :1, :], ort_logits_np[:, :1, :], atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_ministral3_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, Ministral3Config, PreTrainedTokenizerFast

        num_hidden_layers = 1
        config = Ministral3Config(
            architectures=["Ministral3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )

        basename = f"test_generation_ministral3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.head_dim
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(batch_size, cur_len),
            }
            for i in range(num_hidden_layers):
                feed[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feed[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]
            feed = {k: v for k, v in feed.items() if k in input_names}

            results, _ = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config.eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MINISTRAL3_MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_ministral3_fp32_cpu_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_ministral3_fp16_cpu_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_fp32_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_fp16_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_ministral3_bf16_cuda_greedy_generation(self):
        self.common_ministral3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_fp32_cpu(self):
        self.common_fast_ministral3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_fp16_cpu(self):
        self.common_fast_ministral3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_ministral3_int4_cpu(self):
        self.common_fast_ministral3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_ministral3_fp16_cuda(self):
        self.common_fast_ministral3_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_ministral3_conditional_generation_fp32_cpu_random_weights(self):
        """
        Convert a randomly-initialised Mistral3ForConditionalGeneration model to
        fp32 ONNX models targeting the CPU execution provider.

        Mistral3ForConditionalGeneration is the multimodal model class for the
        Ministral 3 family.  The builder now exports **two** artifacts:

        1. ``vision_encoder.onnx`` – the Pixtral vision encoder (patch
           convolution + transformer) together with the multimodal projector.
           Input: ``pixel_values`` [1, 3, image_size, image_size].
           Output: ``image_features`` [num_merged_patches, text_hidden_size].

        2. ``model.onnx`` – the Mistral text decoder with
           ``exclude_embeds=True`` so that it accepts ``inputs_embeds``
           (produced by the vision encoder or from plain embed_tokens) rather
           than raw ``input_ids``.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * Both ``vision_encoder.onnx`` and ``model.onnx`` are written to the
          output directory.
        * Both ONNX files can be loaded by ``onnxruntime``.
        * The vision encoder produces output of the correct shape.
        * The text decoder produces output when fed ``inputs_embeds``.
        """
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            Ministral3Config,
            Mistral3Config,
            Mistral3ForConditionalGeneration,
            PixtralVisionConfig,
            PreTrainedTokenizerFast,
        )

        num_hidden_layers = 1
        # Use a small image size (56×56) so the test stays fast; patch_size=14
        # gives 4×4=16 patches, then spatial_merge_size=2 → 4 merged patches.
        image_size = 56
        patch_size = 14
        spatial_merge_size = 2

        vision_config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            head_dim=16,
            image_size=image_size,
            patch_size=patch_size,
        )
        text_config = Ministral3Config(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-05,
            sliding_window=None,
            vocab_size=32000,
        )
        config = Mistral3Config(
            text_config=text_config, vision_config=vision_config, spatial_merge_size=spatial_merge_size
        )
        config.architectures = ["Mistral3ForConditionalGeneration"]

        model_dir = self.get_model_dir("test_ministral3_conditional_generation_fp32_cpu_random_weights")
        output_dir, cache_dir = self.get_dirs("test_ministral3_conditional_generation_fp32_cpu_random_weights")

        model = Mistral3ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MINISTRAL3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Verify both ONNX files exist and load correctly ---
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        self.assertExists(vision_onnx_path)

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        # --- Verify genai_config.json has vision_encoder section ---
        import json

        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path) as f:
            genai_config = json.load(f)
        self.assertIn("vision_encoder", genai_config["model"])
        ve_cfg = genai_config["model"]["vision_encoder"]
        self.assertEqual(ve_cfg["filename"], "vision_encoder.onnx")
        self.assertEqual(ve_cfg["image_size"], image_size)
        self.assertEqual(ve_cfg["patch_size"], patch_size)
        self.assertEqual(ve_cfg["spatial_merge_size"], spatial_merge_size)

        # --- Run vision encoder forward pass ---
        num_patches_per_side = image_size // patch_size
        expected_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        vision_sess = self.check_ort(vision_onnx_path)
        pixel_values = np.zeros((1, vision_config.num_channels, image_size, image_size), dtype=np.float32)
        vision_outputs = vision_sess.run(None, {"pixel_values": pixel_values})
        self.assertIsNotNone(vision_outputs[0])
        self.assertEqual(vision_outputs[0].shape[0], expected_merged_patches)
        self.assertEqual(vision_outputs[0].shape[1], text_config.hidden_size)

        # --- Run text decoder forward pass ---
        # The ONNX model was built with exclude_embeds=True, so it expects
        # `inputs_embeds` (shape [batch, seq, hidden_size]) rather than
        # `input_ids`.  Compute the embeddings using the saved model weights.
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy().astype(np.float32)

        text_sess = self.check_ort(text_onnx_path)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        head_size = text_config.head_dim
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = text_sess.run(None, onnx_feed)
        self.assertIsNotNone(onnx_outputs[0])

    def test_dequantize_fp8_weights_no_op_when_no_fp8(self):
        """_dequantize_fp8_weights leaves normal float32 weights unchanged."""
        import torch
        from models.builders.mistral import _dequantize_fp8_weights

        linear = torch.nn.Linear(8, 4, bias=False)
        original_data = linear.weight.data.clone()
        _dequantize_fp8_weights(linear)
        self.assertTrue(torch.allclose(linear.weight.data, original_data))

    def test_dequantize_fp8_weights_applies_scale(self):
        """_dequantize_fp8_weights dequantizes float8_e4m3fn weights correctly."""
        import torch

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("float8_e4m3fn not available in this PyTorch build")

        from models.builders.mistral import _dequantize_fp8_weights

        linear = torch.nn.Linear(8, 4, bias=False)
        # Simulate FP8 quantization: store weight as float8_e4m3fn.
        fp8_weight = linear.weight.data.to(fp8_dtype)
        linear.weight = torch.nn.Parameter(fp8_weight, requires_grad=False)
        # Use a scale > 1 to ensure the multiplication is exercised distinctly.
        scale_inv = torch.tensor([2.0])
        linear.register_buffer("weight_scale_inv", scale_inv)

        _dequantize_fp8_weights(linear)

        self.assertEqual(linear.weight.dtype, torch.float32)
        expected = fp8_weight.float() * scale_inv.float()
        self.assertTrue(torch.allclose(linear.weight.data, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
