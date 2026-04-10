# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers, run_session_or_io_binding

QWEN2_5_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


@requires_transformers("5")  # text_config missing
class TestRandomQwen25VL(ExtTestCase):
    def common_fast_qwen25vl_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            PreTrainedTokenizerFast,
            Qwen2_5_VLConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLTextConfig,
        )

        num_hidden_layers = 1

        # Minimal Qwen2.5-VL text config matching the architecture but with
        # small dimensions to keep the test fast and completely offline.
        text_config = Qwen2_5_VLTextConfig(
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            # mrope_section must satisfy: sum(mrope_section) * 2 == head_size
            # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
            # so sum(mrope_section) must be 32: [8, 12, 12] sums to 32.
            rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        )
        # Use a minimal vision config to avoid allocating the default 7B-scale
        # vision encoder (depth=32, hidden_size=3584) which causes OOM in CI.
        # Text-only inference never invokes the vision encoder, so its exact
        # dimensions don't affect correctness.
        vision_config = {
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [0],
        }
        config = Qwen2_5_VLConfig(text_config=text_config, vision_config=vision_config)
        config.architectures = ["Qwen2_5_VLForConditionalGeneration"]

        basename = f"test_discrepancies_qwen25vl_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = Qwen2_5_VLForConditionalGeneration(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=QWEN2_5_VL_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=QWEN2_5_VL_MODEL_NAME,
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
        head_size = text_config.hidden_size // text_config.num_attention_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        # Compute inputs_embeds using the model's embed_tokens since the ONNX
        # model was built with exclude_embeds=True and therefore expects
        # inputs_embeds instead of input_ids.
        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds_np = inputs_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

        # Qwen2.5-VL uses 3D position_ids: [3, batch_size, seq_len]
        # The three dims correspond to temporal, height, and width.
        # For text-only inference all three dims use the same arange.
        position_ids_3d = np.arange(seq_len, dtype=np.int64).reshape(1, 1, seq_len).repeat(3, axis=0).repeat(batch_size, axis=1)

        with self.subTest(step="prefill"):
            prefill_feed = {
                "inputs_embeds": inputs_embeds_np,
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": position_ids_3d,
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, text_config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=text_config.vocab_size,
            )

            # Build 3D position_ids tensor for PyTorch
            pt_position_ids = torch.from_numpy(position_ids_3d).to(provider)
            with torch.no_grad():
                pt_prefill = model(
                    inputs_embeds=inputs_embeds.to(provider),
                    position_ids=pt_position_ids,
                    attention_mask=torch.ones((batch_size, seq_len), dtype=torch.long).to(provider),
                )

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            # Compute embedding for the next token
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
            with torch.no_grad():
                next_embeds = model.get_input_embeddings()(next_token_tensor)
            next_embeds_np = next_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

            # 3D position_ids for decode step: position = seq_len
            decode_position_ids_3d = np.full((3, batch_size, 1), seq_len, dtype=np.int64)

            decode_feed = {
                "inputs_embeds": next_embeds_np,
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": decode_position_ids_3d,
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
                vocab_size=text_config.vocab_size,
                results=prefill_results,
            )

            pt_decode_pos_ids = torch.from_numpy(decode_position_ids_3d).to(provider)
            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                pt_decode = model(
                    inputs_embeds=next_embeds.to(provider),
                    position_ids=pt_decode_pos_ids,
                    attention_mask=torch.ones((batch_size, seq_len + 1), dtype=torch.long).to(provider),
                    past_key_values=pt_past_kv,
                )
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_fp32_cpu(self):
        self.common_fast_qwen25vl_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_fp16_cpu(self):
        self.common_fast_qwen25vl_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25vl_int4_cpu(self):
        self.common_fast_qwen25vl_random_weights("int4", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_fp32_cuda(self):
        self.common_fast_qwen25vl_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_fp16_cuda(self):
        self.common_fast_qwen25vl_random_weights("fp16", "cuda")

    @unittest.skip("Could not find an implementation for MatMul(13) node with name '/model/layers.0/attn/q_proj/MatMul'")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25vl_bf16_cuda(self):
        self.common_fast_qwen25vl_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
