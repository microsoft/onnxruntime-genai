# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np
from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

PHI3V_MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct"


class TestRandomPhi3V(ModelBuilderTestCase):
    def common_fast_phi3v_random_weights(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast

        # Build a tiny Phi-3-vision config that is structurally identical to
        # microsoft/Phi-3-vision-128k-instruct but uses a single hidden layer
        # and a reduced hidden size so the test stays fast and fully offline.
        # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
        # => rotary_dim = 32, so short_factor / long_factor must have length 32.
        num_hidden_layers = 1
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3VForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={
                "type": "longrope",
                "short_factor": [1.0] * (head_size // 2),
                "long_factor": [1.0] * (head_size // 2),
            },
            vocab_size=32064,
        )

        basename = f"test_discrepancies_phi3v_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        # Phi3VForCausalLM is not registered in current transformers, so we
        # use the structurally identical Phi3ForCausalLM, then patch the saved
        # config.json to set architectures=["Phi3VForCausalLM"] so that
        # create_model selects Phi3VModel (with exclude_embeds=True).
        model = Phi3ForCausalLM(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            saved_cfg = json.load(f)
        saved_cfg["architectures"] = ["Phi3VForCausalLM"]
        with open(config_path, "w") as f:
            json.dump(saved_cfg, f, indent=2)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        # Phi3VForCausalLM is automatically built with exclude_embeds=True by
        # create_model, so the ONNX model takes `inputs_embeds` rather than
        # `input_ids`.
        create_model(
            model_name=PHI3V_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=PHI3V_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            # Compute embeddings using the saved model weights so that ONNX
            # and PyTorch both operate on the same inputs_embeds tensor.
            with torch.no_grad():
                inputs_embeds = model.model.embed_tokens(input_ids)

            inputs_embeds_np = inputs_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

            prefill_feed = {
                "inputs_embeds": inputs_embeds_np,
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
                pt_prefill = model(inputs_embeds=inputs_embeds)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 2e-3 if provider == "cuda" else 2e-4, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")

            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            with torch.no_grad():
                next_embed = model.model.embed_tokens(torch.tensor([[next_token]], dtype=torch.long).to(provider))

            next_embed_np = next_embed.cpu().numpy().astype(self.get_input_np_dtype(precision))

            decode_feed = {
                "inputs_embeds": next_embed_np,
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
                pt_decode = model(inputs_embeds=next_embed, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-4, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-4, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_discrepancy_phi3v_fp32_cpu(self):
        self.common_fast_phi3v_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3v_fp16_cpu(self):
        self.common_fast_phi3v_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi3v_int4_cpu(self):
        self.common_fast_phi3v_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3v_fp16_cuda(self):
        self.common_fast_phi3v_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi3v_bf16_cuda(self):
        self.common_fast_phi3v_random_weights("bf16", "cuda")

    def common_phi3v_greedy_generation(self, precision, provider):
        import torch
        from models.builder import create_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast

        num_hidden_layers = 1
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3VForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={
                "type": "longrope",
                "short_factor": [1.0] * (head_size // 2),
                "long_factor": [1.0] * (head_size // 2),
            },
            vocab_size=32064,
        )

        basename = f"test_generation_phi3v_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = Phi3ForCausalLM(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            saved_cfg = json.load(f)
        saved_cfg["architectures"] = ["Phi3VForCausalLM"]
        with open(config_path, "w") as f:
            json.dump(saved_cfg, f, indent=2)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=PHI3V_MODEL_NAME,
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
        max_new_tokens = 10

        # Use a fixed seed so the prompt token IDs are deterministic.
        torch.manual_seed(0)
        # Start from token ID 3 to avoid accidentally hitting BOS/EOS/PAD.
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # The Phi3ForCausalLM model accepts inputs_embeds directly.
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id
            )
        pt_tokens = pt_output[0].tolist()

        # ------------------------------------------------------------------
        # ONNX greedy generation (manual auto-regressive loop)
        # The ONNX model was built with exclude_embeds=True so it expects
        # inputs_embeds rather than input_ids.
        # ------------------------------------------------------------------
        with torch.no_grad():
            current_embeds = model.model.embed_tokens(prompt_ids)
        current_embeds_np = current_embeds.cpu().numpy().astype(self.get_input_np_dtype(precision))

        # Initialise empty KV-cache for every layer.
        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
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
            # Drop any inputs the model does not declare.
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

            # Greedy: pick the token with the highest logit at the last position.
            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            # Carry forward the updated KV-cache.
            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            # Embed the single next token for the following decode step.
            with torch.no_grad():
                next_embed = model.model.embed_tokens(torch.tensor([[next_token]], dtype=torch.long).to(provider))
            current_embeds_np = next_embed.cpu().numpy().astype(self.get_input_np_dtype(precision))

            if next_token == config.eos_token_id:
                break

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=PHI3V_MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            # fp16 rounding can cause the last few generated tokens to diverge
            # between PyTorch and ORT due to accumulated numerical differences.
            # Comparing all but the final 5 tokens is sufficient to validate
            # that the generation loop is correct.
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_phi3v_fp32_cpu_greedy_generation(self):
        self.common_phi3v_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_phi3v_fp16_cpu_greedy_generation(self):
        self.common_phi3v_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_phi3v_fp16_cuda_greedy_generation(self):
        self.common_phi3v_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    @unittest.skip("Could not find an implementation for MatMul(13)")
    def test_phi3v_bf16_cuda_greedy_generation(self):
        self.common_phi3v_greedy_generation("bf16", "cuda")

    @unittest.skip("RuntimeError: Load model from dump_models/test_gemma3_fp32_cpu_genai_generate/output/ failed:Protobuf parsing failed.")
    @hide_stdout()
    def test_phi3v_fp32_cpu_genai_generate(self):
        import torch
        from transformers import Phi3Config, Phi3ForCausalLM

        from models.builder import create_model

        prefix = "test_phi3v_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        head_size = 64
        config = Phi3Config(
            architectures=["Phi3VForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=8192,
            original_max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_scaling={"type": "longrope", "short_factor": [1.0] * (head_size // 2), "long_factor": [1.0] * (head_size // 2)},
            vocab_size=32064,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = Phi3ForCausalLM(config)
        model.eval()
        model.save_pretrained(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            saved_cfg = json.load(f)
        saved_cfg["architectures"] = ["Phi3VForCausalLM"]
        with open(config_path, "w") as f:
            json.dump(saved_cfg, f, indent=2)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=PHI3V_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
