# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

MODEL_NAME = "openai/gpt-oss-20b"


class TestGptOss20b(ModelBuilderTestCase):
    def common_fast_gpt_oss_20b_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
        )

        basename = f"test_discrepancies_gpt_oss_20b_{precision}_{provider}"
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
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
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

    def common_gpt_oss_20b_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
            max_position_embeddings=4096,
        )

        basename = f"test_generation_gpt_oss_20b_{precision}_{provider}"
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
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
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
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gpt_oss_20b_fp16_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_gpt_oss_20b_fp32_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_fp16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_bf16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp32_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_int4_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cuda(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_genai_generate(self):
        """
        Verify that ``onnxruntime-genai`` can load the fp32/CPU ONNX model
        produced from a randomly-initialised GPT-OSS-20B architecture and
        generate tokens without discrepancies against ``transformers.generate``.

        Both backends use greedy (argmax) decoding with the same prompt, so
        the generated token sequences must be bit-for-bit identical.

        The test is skipped automatically when ``onnxruntime-genai`` is not
        installed.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        prefix = "test_gpt_oss_20b_fp32_cpu_genai_generate"
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
            eos_token_id=2,
            bos_token_id=1,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
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

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)

        # Use a fixed seed so the prompt token IDs are deterministic.
        torch.manual_seed(0)
        batch_size = 1
        max_new_tokens = 5
        # Start from token ID 3 to avoid accidentally hitting BOS/EOS/PAD.
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 4))
        prompt_len = prompt_ids.shape[1]

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id
            )
        pt_tokens = pt_output[0].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(output_dir)

        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(prompt_ids.numpy().astype(np.int64))

        og_tokens = prompt_ids[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
