# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :func:`modelbuilder.helpers.rt_helper.onnx_generate`.
"""

import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

from ext_test_case import ExtTestCase, hide_stdout, onnx_generate

TINT64 = onnx.TensorProto.INT64
TFLOAT = onnx.TensorProto.FLOAT
VOCAB = 8


def _make_tiny_lm_no_cache(fixed_token: int = 3) -> onnx.ModelProto:
    """Returns a minimal ONNX LM with no KV cache.

    The model always predicts ``fixed_token`` at every step (the logit for
    ``fixed_token`` is set to 10.0; all others are 0.0).
    """
    fixed_logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
    fixed_logits[0, 0, fixed_token] = 10.0

    return oh.make_model(
        oh.make_graph(
            [oh.make_node("Constant", [], ["logits"], value=onh.from_array(fixed_logits))],
            "tiny_lm_no_cache",
            [oh.make_tensor_value_info("input_ids", TINT64, [1, None])],
            [oh.make_tensor_value_info("logits", TFLOAT, [1, 1, VOCAB])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )


class TestOnnxGenerate(ExtTestCase):
    # ------------------------------------------------------------------
    # No-KV-cache model tests
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_generate_stops_at_eos(self):
        """Generation stops as soon as EOS is predicted."""
        model = _make_tiny_lm_no_cache(fixed_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)

        tokens = onnx_generate(model, prompt, max_new_tokens=10, eos_token_id=3)

        # The model always predicts token 3 (EOS), so it should stop after
        # the very first generated token.
        np.testing.assert_array_equal(tokens, np.array([[1, 2, 3]], dtype=np.int64))

    @hide_stdout()
    def test_generate_max_new_tokens(self):
        """Generation runs for exactly max_new_tokens when EOS is never produced."""
        # Use a non-EOS fixed token so the loop runs to the limit.
        model = _make_tiny_lm_no_cache(fixed_token=5)
        prompt = np.array([[1, 2]], dtype=np.int64)
        max_new_tokens = 4

        tokens = onnx_generate(model, prompt, max_new_tokens=max_new_tokens, eos_token_id=3)

        self.assertEqual(tokens.shape, (1, 2 + max_new_tokens))
        # Verify all generated tokens are 5.
        np.testing.assert_array_equal(tokens[0, 2:], np.full(max_new_tokens, 5, dtype=np.int64))

    @hide_stdout()
    def test_generate_no_eos_argument(self):
        """Without eos_token_id the loop runs for max_new_tokens."""
        model = _make_tiny_lm_no_cache(fixed_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)
        max_new_tokens = 5

        tokens = onnx_generate(model, prompt, max_new_tokens=max_new_tokens)

        self.assertEqual(tokens.shape, (1, 2 + max_new_tokens))

    @hide_stdout()
    def test_generate_returns_numpy_array(self):
        """Return type is always a NumPy integer array."""
        model = _make_tiny_lm_no_cache(fixed_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)

        tokens = onnx_generate(model, prompt, max_new_tokens=3, eos_token_id=3)

        self.assertIsInstance(tokens, np.ndarray)
        self.assertEqual(tokens.dtype, np.int64)

    @hide_stdout()
    def test_generate_return_session(self):
        """return_session=True returns a 3-tuple (tokens, session, feeds)."""
        from onnxruntime import InferenceSession

        model = _make_tiny_lm_no_cache(fixed_token=3)
        prompt = np.array([[1, 2]], dtype=np.int64)

        result = onnx_generate(model, prompt, max_new_tokens=3, eos_token_id=3, return_session=True)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        tokens, session, feeds = result
        self.assertIsInstance(tokens, np.ndarray)
        self.assertIsInstance(session, InferenceSession)
        self.assertIsInstance(feeds, dict)

    @hide_stdout()
    def test_generate_accepts_inference_session(self):
        """Passing an existing InferenceSession avoids re-creating the session."""
        from onnxruntime import InferenceSession

        model = _make_tiny_lm_no_cache(fixed_token=3)
        session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        prompt = np.array([[1, 2]], dtype=np.int64)

        tokens = onnx_generate(session, prompt, max_new_tokens=3, eos_token_id=3)

        np.testing.assert_array_equal(tokens, np.array([[1, 2, 3]], dtype=np.int64))

    @hide_stdout()
    def test_generate_accepts_path(self):
        """Passing a file path also works."""
        import os
        import tempfile

        model = _make_tiny_lm_no_cache(fixed_token=3)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name
        try:
            prompt = np.array([[1, 2]], dtype=np.int64)
            tokens = onnx_generate(path, prompt, max_new_tokens=3, eos_token_id=3)
            np.testing.assert_array_equal(tokens, np.array([[1, 2, 3]], dtype=np.int64))
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # KV-cache model (full Tiny-LLM round-trip)
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_generate_kv_cache_matches_manual_loop(self):
        """
        onnx_generate with KV cache must produce the same tokens as the
        manual auto-regressive loop already tested in test_random_tiny_llm.py.
        """
        import os

        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

        from models.builder import create_model

        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir("test_onnx_generate_kv_cache")
        output_dir, cache_dir = self.get_dirs("test_onnx_generate_kv_cache")

        torch.manual_seed(42)
        pt_model = AutoModelForCausalLM.from_config(config)
        pt_model.eval()
        pt_model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name="arnir0/Tiny-LLM",
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        batch_size = 1
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5))
        prompt_np = prompt_ids.numpy().astype(np.int64)

        # --- transformers greedy reference ---
        with torch.no_grad():
            pt_output = pt_model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        # --- onnx_generate helper ---
        tokens = onnx_generate(onnx_path, prompt_np, max_new_tokens=max_new_tokens, eos_token_id=config.eos_token_id, do_sample=False)

        self.assertEqual(pt_tokens, tokens[0].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
