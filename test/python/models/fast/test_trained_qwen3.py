# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
This test file is ignore unless you set ``LONGTEST=1``.
It verifies a pretrained text model, discrepancies at prefill and decoding steps.
It runs the token generation through transformers and ort-genai.
"""
import os
import unittest
from ext_test_case import ExtTestCase, long_test, requires_cuda, hide_stdout

QWEN3_MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
]


class TestTrainedQwen3(ExtTestCase):
    def _common_part(self, model_id, precision, dtype, provider="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from modelbuilder.builder import create_model

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "What is machine learning?"
        inputs = tokenizer(text, return_tensors="pt")

        smodel_id = model_id.lower().replace("/", "_").replace(".", "_").replace("-", "_")
        output_dir, cache_dir = self.get_dirs(f"test_trained_{smodel_id}_{precision}_{provider}", clean=False)
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            create_model(
                model_name=model_id,
                input_path="",
                precision=precision,
                execution_provider=provider,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        model = AutoModelForCausalLM.from_pretrained(model_id, ignore_mismatched_sizes=True, dtype=dtype)
        model.eval().to(provider).to(dtype)

        return (
            onnx_path,
            model,
            dict(input_ids=inputs["input_ids"].to(provider), attention_mask=inputs["attention_mask"].to(provider)),
            dict(input_ids=inputs["input_ids"].detach().cpu().numpy(), attention_mask=inputs["attention_mask"].detach().cpu().numpy()),
            tokenizer,
        )

    def _common_trained_discrepancies(self, model_id, precision, provider):
        import torch

        dtype = self.get_input_torch_dtype(precision)

        onnx_path, model, torch_feed, onnx_feed, tokenizer = self._common_part(model_id, precision, dtype, provider=provider)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")
        self.fill_with_empty_cache(onnx_feed, sess, provider)

        with torch.no_grad():
            pt_logits = model(**torch_feed).logits
        pt_logits = pt_logits.detach().cpu().numpy()

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        smodel_id = model_id.lower().replace("/", "_").replace(".", "_").replace("-", "_")
        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision=precision,
                model_id=model_id,
                experiment="forward",
                provider="cuda",
                test=f"test_trained_{smodel_id}_discrepancies_{precision}_{provider}",
                input_type="text",
                kind="prefill",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 3)

    @long_test()
    @hide_stdout()
    def test_trained_qwen3_discrepancies_fp32_cpu(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_discrepancies(model_id, "fp32", "cpu")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_qwen3_discrepancies_fp16_cuda(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_discrepancies(model_id, "fp16", "cuda")

    def _common_trained_generate(self, model_id, precision, provider):
        import torch
        import onnxruntime_genai as og

        dtype = self.get_input_torch_dtype(precision)

        onnx_path, model, torch_feed, onnx_feed, tokenizer = self._common_part(model_id, precision, dtype, provider=provider)

        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        prompt_len = torch_feed["input_ids"].shape[1]
        with torch.no_grad():
            pt_output = model.generate(**torch_feed, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        # Keep only the newly generated tokens (exclude the prompt).
        pt_tokens = pt_output[0][prompt_len:].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(onnx_feed["input_ids"])

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        min_length = min(len(pt_tokens), len(og_tokens))
        pt_tokens = pt_tokens[:min_length]
        og_tokens = og_tokens[:min_length]
        disc = self.first_token_diff(pt_tokens, og_tokens)
        smodel_id = model_id.lower().replace("/", "_").replace(".", "_").replace("-", "_")
        disc.update(
            dict(
                precision=precision,
                model_id=model_id,
                experiment="generate",
                provider="cuda",
                test=f"test_trained_{smodel_id}_genai_generate_{precision}_{provider}",
                expected_text=tokenizer.decode(pt_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        length = 2 if precision == "int4" else len(pt_tokens)
        self.assertEqual(pt_tokens[:length], og_tokens[:length])

    @long_test()
    @hide_stdout()
    def test_trained_qwen3_generate_fp32_cpu(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_generate(model_id, "fp32", "cpu")

    @long_test()
    @hide_stdout()
    def test_trained_qwen3_generate_int4_cpu(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_generate(model_id, "int4", "cpu")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_qwen3_generate_fp16_cuda(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_generate(model_id, "fp16", "cuda")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_qwen3_generate_bf16_cuda(self):
        for model_id in QWEN3_MODEL_NAMES:
            with self.subTest(model_id=model_id):
                self._common_trained_generate(model_id, "bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
