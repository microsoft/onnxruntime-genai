# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from models.builders.phi import Phi4MMModel
from ext_test_case import ExtTestCase, hide_stdout, requires_cuda

# Use a small Phi-3 configuration as the base for the synthetic Phi4MM model.
# Phi4MM extends Phi3MiniLongRoPE (which requires short/long RoPE factors) and
# adds packed-QKV LoRA adapters for "default" and "vision" modalities.
_MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"

# head_size = hidden_size // num_attention_heads = 64 // 4 = 16
# rotary_dim_half = head_size // 2 = 8  (number of short/long factors required)
_HEAD_SIZE = 16
_ROTARY_DIM_HALF = _HEAD_SIZE // 2


def _make_phi4mm_config():
    """Create a tiny Phi3Config that satisfies the Phi4MM builder requirements.

    Phi4MMModel inherits from Phi3MiniLongRoPEModel, which calls
    make_rotary_embedding_multi_cache() unconditionally.  That function needs
    ``rope_attrs["multi_cache"]``, which is populated only when
    ``config.rope_scaling`` contains ``short_factor`` / ``long_factor``.

    The ``rope_scaling`` attribute is set directly after construction to work
    with both transformers 4.x (plain attribute) and 5.x (property backed by
    ``rope_parameters``).  The dict uses the ``"type"`` key so that the
    builder's ``make_rope_init`` finds it without relying on the newer
    ``"rope_type"`` alias.
    """
    from transformers import Phi3Config

    config = Phi3Config(
        architectures=["Phi3ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        original_max_position_embeddings=64,
        rms_norm_eps=1e-5,
        vocab_size=1000,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )
    # Set rope_scaling directly to bypass version-specific constructor
    # validation.  Both transformers 4.x and 5.x accept attribute assignment.
    config.rope_scaling = {
        "type": "longrope",
        "short_factor": [1.0] * _ROTARY_DIM_HALF,
        "long_factor": [1.0] * _ROTARY_DIM_HALF,
    }
    # Override architectures so the builder dispatches to Phi4MMModel.
    config.architectures = ["Phi4MMForCausalLM"]
    return config


def _make_peft_model(config):
    """Build a Phi3ForCausalLM base and wrap it in a PEFT LoRA model.

    Two adapters are created: "default" and "vision".  Because lora_B is
    zero-initialised by PEFT, the combined model produces identical logits to
    the plain base model.  Phi4MMModel.make_layer reassigns the vision adapter
    to default before building the ONNX graph, so both outputs agree at init.

    The base config intentionally omits ``rope_scaling`` (uses default RoPE)
    to avoid version-specific rope-validation across transformers 4.x / 5.x.
    The weight shapes are identical regardless of the rotary-embedding variant.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Phi3Config

    # Build the base model using a plain Phi3 config without LongRoPE so that
    # AutoModelForCausalLM.from_config succeeds on both transformers 4.x and 5.x.
    base_cfg = Phi3Config(
        architectures=["Phi3ForCausalLM"],
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        original_max_position_embeddings=config.original_max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
    base_model = AutoModelForCausalLM.from_config(base_cfg)

    lora_cfg = LoraConfig(r=4, target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    peft_model = get_peft_model(base_model, lora_cfg, adapter_name="default")
    peft_model.add_adapter("vision", lora_cfg)
    return peft_model


class _Phi4MMModelWithSyntheticWeights(Phi4MMModel):
    """Thin subclass that replaces disk-based weight loading with an in-memory
    PEFT model created at test time.  Everything else (ONNX graph construction,
    rotary embedding caches, etc.) is exercised exactly as in production."""

    def __init__(self, *args, synthetic_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._synthetic_weights = synthetic_weights

    def load_weights(self, input_path):
        return self._synthetic_weights


class TestPhi4MM(ExtTestCase):
    def common_fast_phi4mm_random_weights(self, precision, provider):
        """Build the ONNX model and run prefill + decode, comparing against PyTorch.

        The Phi4MM builder sets ``exclude_embeds=True``, so the ONNX model
        takes ``inputs_embeds`` rather than ``input_ids``.  We feed the same
        ``inputs_embeds`` tensor to both the PyTorch model and the ONNX
        session to enable numerical comparison.

        Because PEFT initialises ``lora_B`` to zero, the PEFT model output is
        identical to the base model at initialisation, and Phi4MMModel.make_layer
        copies the vision adapter to the default slot before constructing the
        ONNX graph.  We capture PyTorch reference outputs before any builder
        mutation and use a fresh PEFT model for the ONNX build so that the two
        forward passes share the same weights.
        """
        import torch

        config = _make_phi4mm_config()
        num_hidden_layers = config.num_hidden_layers
        basename = f"phi4mm_discrepancies_{precision}_{provider}"
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        peft_model = _make_peft_model(config)
        peft_model.eval()

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        # Compute inputs_embeds via the base model's embedding layer so that
        # both PyTorch and ONNX receive exactly the same tensor.
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        base_model = peft_model.base_model.model
        with torch.no_grad():
            inputs_embeds = base_model.model.embed_tokens(input_ids)
            pt_prefill = peft_model(inputs_embeds=inputs_embeds)

        np_embeds = inputs_embeds.detach().cpu().numpy().astype(self.get_input_np_dtype(precision))
        np_pt_logits = pt_prefill.logits.detach().cpu().numpy()

        # --- Build ONNX --------------------------------------------------------
        from models.builder import set_io_dtype, set_onnx_dtype

        extra_options = {"exclude_embeds": True}
        io_dtype = set_io_dtype(precision, provider, extra_options)
        onnx_dtype = set_onnx_dtype(precision, extra_options)

        # Re-create PEFT model for the builder (it will be mutated internally).
        torch.manual_seed(0)
        peft_model_for_onnx = _make_peft_model(config)
        peft_model_for_onnx.eval()

        onnx_builder = _Phi4MMModelWithSyntheticWeights(
            config, io_dtype, onnx_dtype, provider, cache_dir, extra_options, synthetic_weights=peft_model_for_onnx
        )
        onnx_builder.make_model(cache_dir)
        onnx_builder.save_model(output_dir)

        log_data = dict(
            precision=precision,
            model_id=_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_output_names = [i.name for i in sess.get_outputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "inputs_embeds": np_embeds,
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
            prefill_outputs = sess.run(None, prefill_feed)
            prefill_results = dict(zip(onnx_output_names, prefill_outputs))

            disc = self.get_numpy_discrepancy(np_pt_logits, prefill_outputs[0])
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_pt_logits, prefill_outputs[0], atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            # Single-token decode using the KV-cache from prefill.
            next_token_ids = torch.tensor([[next_token]], dtype=torch.long)
            with torch.no_grad():
                next_embeds = base_model.model.embed_tokens(next_token_ids)
            np_next_embeds = next_embeds.detach().cpu().numpy().astype(self.get_input_np_dtype(precision))

            decode_feed = {
                "inputs_embeds": np_next_embeds,
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            decode_outputs = sess.run(None, decode_feed)
            onnx_decode_logits = decode_outputs[0]

            pt_past_kv = pt_prefill.past_key_values
            with torch.no_grad():
                pt_decode = peft_model(inputs_embeds=next_embeds, past_key_values=pt_past_kv)
            pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_phi4mm_fp32_cpu(self):
        self.common_fast_phi4mm_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_phi4mm_fp16_cpu(self):
        self.common_fast_phi4mm_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_phi4mm_int4_cpu(self):
        self.common_fast_phi4mm_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_phi4mm_fp16_cuda(self):
        self.common_fast_phi4mm_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
