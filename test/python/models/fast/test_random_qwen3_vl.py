# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers, run_session_or_io_binding

QWEN3_VL_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


@requires_transformers("5")
class TestRandomQwen3VL(ExtTestCase):
    def common_fast_qwen3_vl_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

        num_hidden_layers = 1

        # Minimal Qwen3-VL text config with small dimensions for fast, offline testing.
        # mrope_section=[12, 10, 10]: sum=32, and sum*2=64 == head_dim.
        text_config = Qwen3VLTextConfig(
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=2048,
            vocab_size=32000,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [12, 10, 10]},
            pad_token_id=0,
        )
        config = Qwen3VLConfig(text_config=text_config)
        config.architectures = ["Qwen3VLForConditionalGeneration"]
        config.bos_token_id = 1
        config.eos_token_id = 2

        basename = f"test_discrepancies_qwen3_vl_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Qwen3VLForConditionalGeneration(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=QWEN3_VL_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=QWEN3_VL_MODEL_NAME,
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
        head_size = text_config.head_dim
        num_kv_heads = text_config.num_key_value_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        np_dtype = self.get_input_np_dtype(precision)

        # Qwen3-VL text model is built with exclude_embeds=True, so the ONNX
        # model takes inputs_embeds (pre-computed token embeddings) rather than
        # raw input_ids.  Compute the embeddings from the HF model.
        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids)  # [B, S, H]

        with self.subTest(step="prefill"):
            # Sequential 3D position_ids for text-only MRoPE: [3, B, S]
            position_ids = np.tile(np.arange(seq_len, dtype=np.int64), (3, batch_size, 1))

            prefill_feed = {
                "inputs_embeds": embeds.cpu().numpy().astype(np_dtype),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": position_ids,
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_kv_heads, 0, head_size), dtype=np_dtype)
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_kv_heads, 0, head_size), dtype=np_dtype)
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=text_config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids=input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            # Embed the single decode token
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
            with torch.no_grad():
                decode_embeds = model.get_input_embeddings()(next_token_tensor)  # [B, 1, H]

            # position_ids for decode step: [3, B, 1] with value = seq_len
            decode_position_ids = np.full((3, batch_size, 1), seq_len, dtype=np.int64)

            decode_feed = {
                "inputs_embeds": decode_embeds.cpu().numpy().astype(np_dtype),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": decode_position_ids,
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

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                pt_decode = model(input_ids=next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_fp32_cpu(self):
        self.common_fast_qwen3_vl_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_fp16_cpu(self):
        self.common_fast_qwen3_vl_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_int4_cpu(self):
        self.common_fast_qwen3_vl_random_weights("int4", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_fp32_cuda(self):
        self.common_fast_qwen3_vl_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_fp16_cuda(self):
        self.common_fast_qwen3_vl_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_bf16_cuda(self):
        self.common_fast_qwen3_vl_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
