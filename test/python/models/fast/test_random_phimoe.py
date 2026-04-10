# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from ext_test_case import ExtTestCase, hide_stdout, requires_cuda, run_session_or_io_binding

_MODEL_NAME = "microsoft/Phi-3.5-MoE-instruct"

# Dimensions chosen to keep the test fast while exercising the full pipeline.
# head_size = hidden_size // num_attention_heads = 64 // 4 = 16
# rotary_dim_half = head_size // 2 = 8  (size of short_factor / long_factor arrays)
_HIDDEN_SIZE = 64
_NUM_ATTN_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_SIZE = _HIDDEN_SIZE // _NUM_ATTN_HEADS
_ROTARY_DIM_HALF = _HEAD_SIZE // 2
_INTERMEDIATE_SIZE = 32  # per-expert hidden dimension
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOK = 2
_MAX_POS = 128
_ORIG_MAX_POS = 64
_VOCAB_SIZE = 1000


def _make_phimoe_config():
    """Create a tiny Phi3Config that satisfies the PhiMoE builder requirements.

    ``Phi3MoELongRoPEModel`` inherits from ``MistralModel`` and requires:
    * ``max_position_embeddings != original_max_position_embeddings`` so the
      builder dispatches to ``Phi3MoELongRoPEModel``.
    * ``rope_scaling`` containing ``short_factor`` and ``long_factor`` so
      ``make_rotary_embedding_multi_cache()`` can build the dual cos/sin caches.
    * ``num_local_experts`` and ``num_experts_per_tok`` for the QMoE/MoE op.

    ``rope_scaling`` is set via direct attribute assignment to avoid
    version-specific validation in transformers 4.x / 5.x.  The ``"type"``
    key is used so that the builder's ``make_rope_init`` recognises it.
    """
    from transformers import Phi3Config

    # Construct with "Phi3ForCausalLM" first so that transformers 5.x
    # rope_parameters validation accepts the config; the architectures value is
    # overridden below *after* rope_scaling is patched in.  Both steps follow
    # the same two-step pattern used in test_random_phi4mm.py.
    config = Phi3Config(
        architectures=["Phi3ForCausalLM"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=_NUM_ATTN_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        max_position_embeddings=_MAX_POS,
        original_max_position_embeddings=_ORIG_MAX_POS,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB_SIZE,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
    )
    # Set rope_scaling directly to bypass version-specific constructor
    # validation.  Both transformers 4.x and 5.x accept attribute assignment.
    config.rope_scaling = {"type": "longrope", "short_factor": [1.0] * _ROTARY_DIM_HALF, "long_factor": [1.0] * _ROTARY_DIM_HALF}
    # Override architectures so the builder dispatches to Phi3MoELongRoPEModel.
    # Done after rope_scaling is set to avoid version-specific validation.
    config.architectures = ["PhiMoEForCausalLM"]
    # MoE-specific attributes read by the base Model.__init__.
    config.num_local_experts = _NUM_EXPERTS
    config.num_experts_per_tok = _NUM_EXPERTS_PER_TOK
    return config


def _make_phimoe_torch_model(config):
    """Build a minimal PhiMoEForCausalLM-like PyTorch model.

    The class names and attribute names match what the builder accesses:
    * Layer class name must end in ``DecoderLayer`` (``is_layer`` check).
    * ``layer.input_layernorm`` / ``layer.post_attention_layernorm`` – each is
      a ``nn.LayerNorm`` because ``Phi3MoELongRoPEModel`` sets
      ``layernorm_attrs["simple"] = False`` (expects both weight and bias).
    * ``layer.self_attn.{q,k,v,o}_proj`` – ``nn.Linear`` without bias (matches
      the real model's attention projection layers).
    * ``layer.block_sparse_moe.gate`` – ``nn.Linear`` (routing).
    * ``layer.block_sparse_moe.experts[i].{w1,w2,w3}`` – ``nn.Linear`` without
      bias (SwiGLU-style expert: w1 gate, w3 up-project, w2 down-project).
    * ``model.embed_tokens`` – ``nn.Embedding`` with ``vocab_size`` rows so the
      builder's embedding detection finds it.
    * ``model.norm`` – ``nn.LayerNorm`` (has_final_norm check).
    * ``lm_head`` – ``nn.Linear`` with ``out_features == vocab_size``
      (has_lm_head check).
    """
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attn_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = hidden_size // num_attn_heads
    num_experts = config.num_local_experts

    class _PhiMoEExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

        def forward(self, x):
            return self.w2(torch.silu(self.w1(x)) * self.w3(x))

    class _PhiMoEBlockSparseMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)
            self.experts = nn.ModuleList([_PhiMoEExpert() for _ in range(num_experts)])

        def forward(self, x):
            router_logits = self.gate(x)
            routing_weights = torch.softmax(router_logits, dim=-1)
            top_k = config.num_experts_per_tok
            routing_weights, selected = torch.topk(routing_weights, top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            out = torch.zeros_like(x)
            for i in range(num_experts):
                mask = (selected == i).any(dim=-1)
                if mask.any():
                    expert_out = self.experts[i](x[mask])
                    w = routing_weights[mask][selected[mask] == i].unsqueeze(-1)
                    out[mask] += w * expert_out
            return out

    class _PhiMoEAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, num_attn_heads * head_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_size, bias=False)
            self.o_proj = nn.Linear(num_attn_heads * head_size, hidden_size, bias=False)

        def forward(self, x, past_key_values=None):
            b, s, _ = x.shape
            q = self.q_proj(x).view(b, s, num_attn_heads, head_size).transpose(1, 2)
            k = self.k_proj(x).view(b, s, num_kv_heads, head_size).transpose(1, 2)
            v = self.v_proj(x).view(b, s, num_kv_heads, head_size).transpose(1, 2)
            # Expand k/v for GQA
            repeat = num_attn_heads // num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            attn = torch.matmul(q, k.transpose(-2, -1)) / (head_size**0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
            return self.o_proj(out), (k[:, ::repeat], v[:, ::repeat])

    class _PhiMoEDecoderLayer(nn.Module):
        """Class name MUST end with 'DecoderLayer' for the builder's is_layer() check."""

        def __init__(self):
            super().__init__()
            # nn.LayerNorm (with both weight and bias) is used intentionally:
            # Phi3MoELongRoPEModel sets layernorm_attrs["simple"] = False,
            # which maps to SkipLayerNorm / LayerNorm ops that require a bias.
            # The epsilon is taken from config.rms_norm_eps, the only eps field
            # provided by Phi3Config.
            self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)
            self.self_attn = _PhiMoEAttention()
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)
            self.block_sparse_moe = _PhiMoEBlockSparseMoE()

        def forward(self, x, past_key_values=None):
            residual = x
            x, pkv = self.self_attn(self.input_layernorm(x), past_key_values)
            x = x + residual
            residual = x
            x = self.block_sparse_moe(self.post_attention_layernorm(x))
            x = x + residual
            return x, pkv

    class _PhiMoEInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
            self.layers = nn.ModuleList([_PhiMoEDecoderLayer() for _ in range(config.num_hidden_layers)])
            # nn.LayerNorm is used here for the same reason as in the decoder
            # layers: simple=False means the builder reads both .weight and
            # .bias from the normalisation module.
            self.norm = nn.LayerNorm(hidden_size, eps=config.rms_norm_eps)

    class PhiMoEForCausalLM(PreTrainedModel):
        config_class = config.__class__

        def __init__(self, cfg):
            super().__init__(cfg)
            self.model = _PhiMoEInnerModel()
            self.lm_head = nn.Linear(hidden_size, cfg.vocab_size, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def forward(self, input_ids=None, past_key_values=None, **kwargs):
            x = self.model.embed_tokens(input_ids).float()
            new_pkv = []
            for i, layer in enumerate(self.model.layers):
                past = past_key_values[i] if past_key_values is not None else None
                x, pkv = layer(x, past)
                new_pkv.append(pkv)
            x = self.model.norm(x)
            logits = self.lm_head(x)
            return CausalLMOutputWithPast(logits=logits, past_key_values=tuple(new_pkv))

    return PhiMoEForCausalLM(config)


def _make_phimoe_builder(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
    """Return a ``Phi3MoELongRoPEModel`` subclass whose ``load_weights`` returns
    a freshly constructed synthetic model instead of loading from disk.

    This keeps the test completely offline and avoids the ``trust_remote_code``
    complexity of the real ``microsoft/Phi-3.5-MoE-instruct`` model.
    """
    import torch
    from models.builders.phi import Phi3MoELongRoPEModel

    class _PhiMoEBuilderWithSyntheticWeights(Phi3MoELongRoPEModel):
        """Subclass that replaces disk-based weight loading with an in-memory
        synthetic PyTorch model created at test time.  Everything else (ONNX
        graph construction, rotary caches, QMoE quantisation) runs as in
        production."""

        def load_weights(self, input_path):
            torch.manual_seed(0)
            model = _make_phimoe_torch_model(config)
            model.eval()
            return model

    return _PhiMoEBuilderWithSyntheticWeights(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class TestPhiMoE(ExtTestCase):
    def _build_phimoe_onnx(self, output_dir, cache_dir):
        """Build the PhiMoE ONNX model with synthetic random weights.

        ``Phi3MoELongRoPEModel`` asserts ``io_dtype == FLOAT16`` and forces
        ``onnx_dtype = INT4`` (QMoE), which is the only supported combination
        for this builder.
        """
        from models.builder import set_io_dtype, set_onnx_dtype

        config = _make_phimoe_config()
        extra_options = {}
        io_dtype = set_io_dtype("fp16", "cuda", extra_options)
        onnx_dtype = set_onnx_dtype("int4", extra_options)

        builder = _make_phimoe_builder(config, io_dtype, onnx_dtype, "cuda", cache_dir, extra_options)
        builder.make_model(cache_dir)
        builder.save_model(output_dir)
        return config

    @hide_stdout()
    @requires_cuda()  # for tensorrt-llm
    def test_phimoe_onnx_build_cpu(self):
        """Build the PhiMoE ONNX model with synthetic random weights (CPU safe).

        The ONNX graph construction (weight extraction and ONNX IR building)
        runs entirely on CPU.  Only ORT inference of the QMoE operator requires
        CUDA, so this test can run in any environment.
        """
        basename = "test_phimoe_build_cpu"
        output_dir, cache_dir = self.get_dirs(basename)

        config = self._build_phimoe_onnx(output_dir, cache_dir)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        # Verify that the ONNX model has the expected inputs and outputs by
        # loading it with the onnx library (no ORT / CUDA required).
        import onnx

        model_proto = onnx.load(onnx_path)
        input_names = {inp.name for inp in model_proto.graph.input}
        output_names = {out.name for out in model_proto.graph.output}

        # Standard inputs
        self.assertIn("input_ids", input_names)
        self.assertIn("attention_mask", input_names)
        self.assertIn("position_ids", input_names)

        # KV-cache inputs and outputs
        num_hidden_layers = config.num_hidden_layers
        for i in range(num_hidden_layers):
            self.assertIn(f"past_key_values.{i}.key", input_names)
            self.assertIn(f"past_key_values.{i}.value", input_names)
            self.assertIn(f"present.{i}.key", output_names)
            self.assertIn(f"present.{i}.value", output_names)

        # Logits output
        self.assertIn("logits", output_names)

    @hide_stdout()
    @requires_cuda()
    def test_phimoe_int4_cuda_build_and_run(self):
        """Build the PhiMoE ONNX model and run prefill + decode on CUDA.

        The QMoE operator is only supported on CUDA, so this test is gated
        by ``@requires_cuda()``.  We check that:
        * The ONNX file is created without errors.
        * An ORT inference session can be created on CUDA.
        * Prefill produces logits of the correct shape.
        * Decode (using KV-cache from prefill) also succeeds.
        """
        basename = "test_phimoe_int4_cuda"
        output_dir, cache_dir = self.get_dirs(basename)

        config = self._build_phimoe_onnx(output_dir, cache_dir)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        sess = self._check_with_ort(onnx_path, cpu=False)
        onnx_input_names = {i.name for i in sess.get_inputs()}

        batch_size = 1
        seq_len = 5
        head_size = _HEAD_SIZE
        num_hidden_layers = config.num_hidden_layers
        precision = "fp16"

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, _NUM_KV_HEADS, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, _NUM_KV_HEADS, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, prefill_logits = run_session_or_io_binding(
                use_iobinding=False, precision=precision, provider="cuda", feed=prefill_feed, sess=sess, vocab_size=config.vocab_size
            )
            self.assertEqual(prefill_logits.shape, (batch_size, seq_len, config.vocab_size))

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
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

            _decode_results, decode_logits = run_session_or_io_binding(
                use_iobinding=False,
                precision=precision,
                provider="cuda",
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )
            self.assertEqual(decode_logits.shape, (batch_size, 1, config.vocab_size))


if __name__ == "__main__":
    unittest.main(verbosity=2)
