# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, requires_transformers, run_session_or_io_binding

MODEL_NAME = "google/gemma-3-4b-it"


@requires_transformers("5")
class TestRandomGemma3Conditional(ModelBuilderTestCase):
    """Fast offline tests for the Gemma3ForConditionalGeneration text decoder.

    The builder exports only the text component of the VLM (with
    ``exclude_embeds=True``), so the resulting ONNX model takes
    ``inputs_embeds`` rather than ``input_ids``.  The tests verify that
    the exported model produces logits numerically close to those from the
    original ``Gemma3ForConditionalGeneration`` HF model, and that greedy
    token generation is identical.
    """

    @staticmethod
    def _make_config():
        """Return a minimal ``Gemma3Config`` for offline testing."""
        from transformers import Gemma3Config, Gemma3TextConfig

        text_config = Gemma3TextConfig(
            head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=1,
            num_key_value_heads=4,
            query_pre_attn_scalar=64,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
        )
        return Gemma3Config(architectures=["Gemma3ForConditionalGeneration"], text_config=text_config)

    def common_fast_gemma3_conditional_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Gemma3ForConditionalGeneration, PreTrainedTokenizerFast

        config = self._make_config()
        num_hidden_layers = config.text_config.num_hidden_layers

        basename = f"test_discrepancies_gemma3_cond_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Gemma3ForConditionalGeneration(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<bos>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        # create_model detects Gemma3ForConditionalGeneration and automatically
        # sets exclude_embeds=True so the ONNX model takes ``inputs_embeds``.
        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=MODEL_NAME,
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
        head_size = config.text_config.head_dim
        vocab_size = config.text_config.vocab_size
        num_kv_heads = config.text_config.num_key_value_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        # Compute input embeddings using the language model's embed_tokens so
        # that ONNX and PyTorch operate on the same tensor.
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids)

        inputs_embeds_np = inputs_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

        with self.subTest(step="prefill"):
            prefill_feed = {
                "inputs_embeds": inputs_embeds_np,
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(inputs_embeds=inputs_embeds)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))
            next_ids = torch.tensor([[next_token]], dtype=torch.long).to(provider)

            with torch.no_grad():
                next_embeds = model.model.language_model.embed_tokens(next_ids)

            next_embeds_np = next_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

            decode_feed = {
                "inputs_embeds": next_embeds_np,
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            _, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_decode = model(inputs_embeds=next_embeds, past_key_values=pt_prefill.past_key_values)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_gemma3_conditional_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Gemma3ForConditionalGeneration, PreTrainedTokenizerFast

        config = self._make_config()
        num_hidden_layers = config.text_config.num_hidden_layers

        basename = f"test_generation_gemma3_cond_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = Gemma3ForConditionalGeneration(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<bos>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
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
        head_size = config.text_config.head_dim
        vocab_size = config.text_config.vocab_size
        num_kv_heads = config.text_config.num_key_value_heads
        eos_token_id = config.text_config.eos_token_id
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, vocab_size, (batch_size, 5)).to(provider)

        # PyTorch reference: greedy generation using the full model directly
        # (text-only path, no image required).
        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=eos_token_id
            )
        pt_tokens = pt_output[0].tolist()

        # ONNX greedy generation (manual auto-regressive loop).
        # The ONNX model expects ``inputs_embeds`` (exclude_embeds=True).
        embed_tokens = model.model.language_model.embed_tokens
        with torch.no_grad():
            current_embeds = embed_tokens(prompt_ids)
        current_embeds_np = current_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, num_kv_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, num_kv_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = prompt_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_embeds_np.shape[1]

            feed = {
                "inputs_embeds": current_embeds_np,
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
                vocab_size=vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            with torch.no_grad():
                next_ids = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                current_embeds = embed_tokens(next_ids)
            current_embeds_np = current_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

            if next_token == eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            # fp16/bf16 precision can cause small numerical errors that
            # accumulate over long sequences, leading the last few generated
            # tokens to diverge. Trim the trailing 5 tokens to avoid flaky
            # comparisons while still validating the majority of the output.
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_fp32_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_fp16_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_int4_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma3_cond_fp16_cuda(self):
        self.common_fast_gemma3_conditional_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gemma3_cond_fp32_cpu_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma3_cond_fp16_cpu_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_cond_fp16_cuda_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_cond_bf16_cuda_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
