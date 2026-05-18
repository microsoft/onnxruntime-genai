# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
import onnx_ir as ir
from model_builder_test_case import ModelBuilderTestCase, hide_stdout, requires_transformers
from models.builders.llama import LlamaModel

# LlamaForCausalLM architecture matching arnir0/Tiny-LLM but with a single
# hidden layer and smaller dimensions to keep tests fast and completely offline.
_LLAMA_CONFIG_KWARGS = dict(
    architectures=["LlamaForCausalLM"],
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=512,
    intermediate_size=1376,
    max_position_embeddings=2048,
    model_type="llama",
    num_attention_heads=8,
    num_hidden_layers=1,
    num_key_value_heads=4,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
    vocab_size=32000,
)

_MODEL_ID = "arnir0/Tiny-LLM"


class _AttentionOnlyLlamaModel(LlamaModel):
    """Build an ONNX model containing only the LlamaAttention subgraph.

    Unlike the full pipeline (embedding → LayerNorm → attention → MLP →
    final norm → LM head), this builder creates a minimal ONNX model whose
    single computation is the attention layer:

        hidden_states  →  Q/K/V projections  →  GQA (with fused RoPE)  →  O proj  →  attn_output

    The model also accepts an ``attention_mask`` used to derive
    ``seqlens_k`` / ``total_seq_len`` for the GroupQueryAttention kernel and
    optional past / present KV-cache tensors.
    """

    def make_inputs_and_outputs(self):
        g_inputs = self.model.graph.inputs
        g_outputs = self.model.graph.outputs

        # Input: hidden_states (output of LayerNorm in the full pipeline)
        g_inputs.append(
            self.make_value(
                "hidden_states", dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
            )
        )
        # Input: attention_mask needed to compute seqlens_k / total_seq_len
        g_inputs.append(
            self.make_value("attention_mask", dtype=ir.DataType.INT64, shape=["batch_size", "total_sequence_length"])
        )
        # Input: past KV cache for layer 0
        kv_shape = ["batch_size", self.num_kv_heads, "past_sequence_length", self.head_size]
        g_inputs.append(self.make_value("past_key_values.0.key", dtype=self.io_dtype, shape=kv_shape))
        g_inputs.append(self.make_value("past_key_values.0.value", dtype=self.io_dtype, shape=kv_shape))

        # Output: attention output (before residual connection)
        g_outputs.append(
            self.make_value(
                "attn_output", dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
            )
        )
        # Output: updated KV cache
        kv_out_shape = ["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size]
        g_outputs.append(self.make_value("present.0.key", dtype=self.io_dtype, shape=kv_out_shape))
        g_outputs.append(self.make_value("present.0.value", dtype=self.io_dtype, shape=kv_out_shape))

    def build_attention_model(self, attn_module, out_dir):
        """Build and save an attention-only ONNX model.

        Args:
            attn_module: The PyTorch ``LlamaAttention`` module whose weights
                are used to initialise the ONNX operators.
            out_dir: Directory where ``model.onnx`` will be written.
        """
        self.make_inputs_and_outputs()

        # Build the mask-reformatting subgraph that computes seqlens_k and
        # total_seq_len from the 2-D attention_mask.  These values are
        # consumed by the GroupQueryAttention operator.
        self.make_preprocessing_nodes()

        # Set the root input for make_attention (normally LayerNorm output).
        self.layernorm_attrs["output_0"] = "hidden_states"

        if attn_module is None:
            attn_module = getattr(self, "attn_module", None)
        if attn_module is None:
            raise ValueError("attn_module must be provided to build attention-only model.")

        # Build the attention subgraph: Q/K/V projections → GQA → O projection.
        self.make_attention(0, attn_module, root_input="hidden_states")

        # Connect the final attention output (stored in layernorm_attrs after
        # make_attention_output_proj) to the named graph output.
        self.make_node(
            "Identity",
            inputs=[self.layernorm_attrs["skip_input"]],
            outputs=["attn_output"],
            name="/model/attn_output_identity",
        )

        self.save_model(out_dir)


class TestLlamaAttentionDiscrepancies(ModelBuilderTestCase):
    """Fast discrepancy tests for LlamaAttention using randomly initialised weights.

    Only the attention layer is tested (Q/K/V projections, fused rotary
    embeddings via GQA, output projection).  No embedding layer, LayerNorm,
    MLP, or LM head is created.

    Random weights are used so every test runs completely offline.
    Discrepancy metrics are recorded via ``log_results`` and hard thresholds
    are enforced with ``assertLess`` to catch regressions in the builder's
    attention conversion.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_attention_onnx(self, test_name):
        """Create a random LlamaAttention and build an attention-only ONNX model.

        Returns ``(config, inner_model, attn_module, ort_session)``.
        """
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(**_LLAMA_CONFIG_KWARGS)
        output_dir, cache_dir = self.get_dirs(test_name)

        torch.manual_seed(42)
        pt_model = AutoModelForCausalLM.from_config(config)
        pt_model.eval()

        inner = pt_model.model
        attn_module = inner.layers[0].self_attn

        builder = _AttentionOnlyLlamaModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", cache_dir, {})
        builder.build_attention_model(attn_module, output_dir)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)
        return config, inner, attn_module, sess

    @staticmethod
    def _run_pt_attn(inner, attn_module, hidden_states, pos_ids, past_kv):
        """Run a single PyTorch attention forward pass.

        Computes position embeddings via the model's ``rotary_emb`` and
        returns the attention output as a NumPy array.
        """
        import torch

        pos_emb = inner.rotary_emb(hidden_states, position_ids=pos_ids)
        with torch.no_grad():
            attn_out, _ = attn_module(hidden_states, position_embeddings=pos_emb, past_key_values=past_kv)
        return attn_out.numpy()

    @staticmethod
    def _empty_past_kv(config, batch_size, head_size):
        """Return zero-length KV-cache arrays for layer 0."""
        shape = (batch_size, config.num_key_value_heads, 0, head_size)
        return {
            "past_key_values.0.key": np.zeros(shape, dtype=np.float32),
            "past_key_values.0.value": np.zeros(shape, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Test: prefill discrepancies
    # ------------------------------------------------------------------

    @requires_transformers("5.0")
    @hide_stdout()
    def test_llama_attention_prefill_discrepancies(self):
        """Check numerical discrepancies for the LlamaAttention prefill pass.

        The test builds an ONNX model containing *only* the attention layer
        (no embedding, LayerNorm, MLP, or LM head) using randomly initialised
        weights.  Both backends receive the same ``hidden_states`` tensor and
        the attention outputs are compared with ``get_numpy_discrepancy``.
        """
        import torch
        from transformers.cache_utils import DynamicCache

        test_name = "test_llama_attention_prefill_discrepancies"
        config, inner, attn_module, sess = self._build_attention_onnx(test_name)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        torch.manual_seed(0)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        pos_ids = torch.arange(seq_len).unsqueeze(0)

        # PyTorch reference (no past KV cache)
        past_kv = DynamicCache(config=config)
        pt_out = self._run_pt_attn(inner, attn_module, hidden_states, pos_ids, past_kv)

        # ONNX inference
        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        feed = {
            "hidden_states": hidden_states.numpy(),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        }
        feed.update(self._empty_past_kv(config, batch_size, head_size))
        feed = {k: v for k, v in feed.items() if k in onnx_input_names}

        onnx_out = sess.run(None, feed)[0]

        disc = self.get_numpy_discrepancy(pt_out, onnx_out)
        disc.update(
            dict(
                precision="fp32",
                model_id=_MODEL_ID,
                experiment="attention_prefill",
                provider="cpu",
                test=test_name,
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 1e-3)

    # ------------------------------------------------------------------
    # Test: decode-step discrepancies
    # ------------------------------------------------------------------

    @requires_transformers("5.0")
    @hide_stdout()
    def test_llama_attention_decode_discrepancies(self):
        """Check numerical discrepancies for the LlamaAttention decode step.

        Two inference passes are run on the attention-only ONNX model:

        1. **Prefill** – ``seq_len`` hidden states are fed with an empty
           KV cache.  The ``present.*`` outputs are kept for the next step.
        2. **Decode** – a single hidden-state vector is fed together with
           the KV cache from step 1.

        The decode-step attention outputs from ONNX and PyTorch are compared
        and a hard threshold on ``max_abs_err`` is enforced.
        """
        import torch
        from transformers.cache_utils import DynamicCache

        test_name = "test_llama_attention_decode_discrepancies"
        config, inner, attn_module, sess = self._build_attention_onnx(test_name)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        onnx_output_names = [out.name for out in sess.get_outputs()]

        torch.manual_seed(0)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        hidden_states_dec = torch.randn(batch_size, 1, config.hidden_size)

        # ------------------------------------------------------------------
        # Step 1: ONNX prefill (populate KV cache)
        # ------------------------------------------------------------------
        prefill_feed = {
            "hidden_states": hidden_states.numpy(),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        }
        prefill_feed.update(self._empty_past_kv(config, batch_size, head_size))
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}
        prefill_results = dict(zip(onnx_output_names, sess.run(None, prefill_feed)))

        # ------------------------------------------------------------------
        # Step 2: ONNX decode (single-token, with KV cache from step 1)
        # ------------------------------------------------------------------
        decode_feed = {
            "hidden_states": hidden_states_dec.numpy(),
            "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            "past_key_values.0.key": prefill_results["present.0.key"],
            "past_key_values.0.value": prefill_results["present.0.value"],
        }
        decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}
        onnx_decode_out = sess.run(None, decode_feed)[0]

        # ------------------------------------------------------------------
        # PyTorch reference: same two steps
        # ------------------------------------------------------------------
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        past_kv_pt = DynamicCache(config=config)
        # Prefill: populate the cache
        self._run_pt_attn(inner, attn_module, hidden_states, pos_ids, past_kv_pt)

        pos_ids_dec = torch.arange(seq_len, seq_len + 1).unsqueeze(0)
        pt_decode_out = self._run_pt_attn(inner, attn_module, hidden_states_dec, pos_ids_dec, past_kv_pt)

        disc = self.get_numpy_discrepancy(pt_decode_out, onnx_decode_out)
        disc.update(
            dict(
                precision="fp32",
                model_id=_MODEL_ID,
                experiment="attention_decode",
                provider="cpu",
                test=test_name,
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
