# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
#
# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# --------------------------------------------------------------------------
"""OGA model builder for Meta **Muse Glimmer** (public; `meta-models/Muse-Glimmer-30B`).

Text-only LLM path for the Muse Glimmer text decoder (a Llama-4-family arch, ~27.85B text
params). This is the PUBLIC, Apache-2.0 release of the model previously enabled here as the
private beta "onyx_early" (see builders/onyx.py). The architecture is IDENTICAL; only the
config schema and (critically) the RoPE weight layout differ. This builder targets the public
model; prefer it over the onyx builder.

Non-standard features (vs. a vanilla Llama), all verified against the official transformers
`modeling_muse_glimmer.py`:
  1. Sandwich / double norm  - 4 norms/layer: input_layernorm, post_attention_layernorm (on the
     attn output, eps post_norm_eps), pre_feedforward_layernorm (pre-FFN, eps rms_norm_eps),
     post_feedforward_layernorm (on the MLP output, eps post_norm_eps). `(1+weight)` RMSNorm
     (MuseGlimmerTextCenteredRMSNorm) -> add_offset=1.
  2. iRoPE / NoPE layers      - full_attention layers (every 4th) get NO positional encoding
     (layer_rope_theta[i]==0); sliding_attention layers use RoPE.
  3. QK-norm (scaleless)      - one shared scaleless RMSNorm on Q and K per head; query is scaled
     by qk_scale_factor (already = raw/sqrt(head_dim) = 3.87 in the public config) BEFORE the
     standard 1/sqrt(head_dim) softmax scale -> combined attention scale = qk_scale_factor/sqrt(hd).
  4. Attention output gate    - attn_out *= sigmoid(gate_proj(x)); `gate_proj` is a separate
     hidden->q_size projection in the attention module.
  5. Normalized token embeds  - scaleless RMSNorm on the embedding output (MuseGlimmerTextNormedEmbedding).
  6. Final-logit soft-cap     - final_logit_softcapping * tanh(logits * output_multiplier / cap).
  7. Final norm weight-AS-IS  - the output norm (self.norm) is a plain MuseGlimmerRMSNorm (no +1),
     unlike the per-layer centered norms; add_offset must be 0 for it.

*** RoPE convention (THE key difference vs the onyx/beta builder) ***
The public HF weights are PERMUTED for `rotate_half` by the official convert script
(`_permute_for_rope` on q_proj/k_proj). With permuted weights, the correct rotary is the
HALF-SPLIT convention == the OGA base default `rope_attrs["interleaved"] = 0`. Do NOT set
interleaved=1 here (that was correct only for the beta's UN-permuted, view_as_complex weights;
applying it to the public permuted weights produces garbage).
"""
import math

import torch

from .base import Model


class MuseGlimmerModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # (1 + weight) RMSNorm for the decoder (centered) norms -> pre-bake the +1.
        self.layernorm_attrs["add_offset"] = 1

        # --- iRoPE / NoPE + sliding/full schedule ---
        # Public config: full_attention layers are the NoPE layers. layer_rope_theta[i] == 0 marks
        # a NoPE layer; a nonzero theta marks a RoPE (local/sliding) layer.
        self.layer_types = list(
            getattr(config, "layer_types", ["full_attention"] * self.num_layers)
        )
        layer_rope_theta = getattr(config, "layer_rope_theta", None)
        if layer_rope_theta is not None:
            self.rope_layer_mask = [1 if t else 0 for t in layer_rope_theta]
        else:
            # Fallback: sliding_attention -> RoPE, full_attention -> NoPE.
            self.rope_layer_mask = [
                0 if lt == "full_attention" else 1 for lt in self.layer_types
            ]

        # --- dual eps for the sandwich norms ---
        self.rms_norm_eps = config.rms_norm_eps
        self.post_norm_eps = getattr(config, "post_norm_eps", config.rms_norm_eps)

        # --- QK norm (scaleless) + query scale ---
        # Public: query_states = qk_norm(q) * qk_scale_factor (=3.87, already raw/sqrt(hd));
        # attention then applies scaling = head_dim**-0.5. Combined = qk_scale_factor / sqrt(hd).
        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            self.attention_attrs["q_norm"] = True
            self.attention_attrs["k_norm"] = True
            qk_scale_factor = getattr(config, "qk_scale_factor", math.sqrt(self.head_size))
            # NOTE: public qk_scale_factor is ALREADY divided by sqrt(head_dim) (=3.87), so only
            # one more 1/sqrt(head_dim) is applied here (vs the beta which stored the raw 43.78).
            self.attention_attrs["scale"] = qk_scale_factor / math.sqrt(self.head_size)

        # --- RoPE convention: PUBLIC weights are permuted -> HALF-SPLIT == base default (0). ---
        # Do NOT override rope_attrs["interleaved"] (leave the base default 0). See module docstring.
        # rope theta lives under rope_parameters in the public config; make sure the base sees it
        # (dispatch hoists text_config, but rope_theta is nested one level deeper).
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params and "rope_theta" in rope_params:
            self.rope_attrs["theta"] = float(rope_params["rope_theta"])

        # --- Attention output gate ---
        self.use_attn_output_gate = getattr(config, "use_attn_output_gate", True)

        # --- Normalized token embeddings (scaleless RMSNorm on embed output) ---
        self.normalize_tok_embeddings = getattr(config, "normalize_tok_embeddings", True)

        # --- Final-logit soft cap: cap * tanh(logits * output_multiplier / cap) ---
        self.output_multiplier = getattr(config, "output_multiplier", 1.0)
        # Public field name is final_logit_softcapping (beta used output_soft_cap_temp).
        self.output_soft_cap_temp = (
            getattr(config, "final_logit_softcapping", None)
            or getattr(config, "output_soft_cap_temp", 0.0)
            or 0.0
        )
        self.lm_head_attrs["scale"] = self.output_multiplier
        self.lm_head_attrs["softcap"] = self.output_soft_cap_temp

    # ------------------------------------------------------------------ #
    # Weight loading: Muse Glimmer is a multimodal `MuseGlimmerForConditionalGeneration`, which is
    # registered under AutoModelForImageTextToText (NOT AutoModelForCausalLM, which the base uses).
    # Load the full model; the base module-walk finds the text decoder layers, embed, the
    # `model.model.language_model.norm` final norm (the same path the base already handles for Gemma-3
    # multimodal), and the vocab-sized lm_head, while skipping the vision tower.
    # ------------------------------------------------------------------ #
    def load_weights(self, input_path):
        from transformers import AutoModelForImageTextToText

        extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            dtype="auto",
            low_cpu_mem_usage=True,
            **extra_kwargs,
        )
        return model

    # ------------------------------------------------------------------ #
    # Layer schedule helpers
    # ------------------------------------------------------------------ #
    def is_local(self, layer_id):
        return self.layer_types[layer_id] == "sliding_attention"

    def uses_rope(self, layer_id):
        return bool(self.rope_layer_mask[layer_id])

    # ------------------------------------------------------------------ #
    # MIGraphX GQA-arity fix: drop trailing-empty GQA optional inputs (sinks etc.) that the
    # MIGraphX ONNX parser rejects. Muse Glimmer has no sinks.
    # ------------------------------------------------------------------ #
    def make_group_query_attention(self, name, **kwargs):
        super().make_group_query_attention(name, **kwargs)
        for node in self.model.graph:
            if node.name == name and node.op_type == "GroupQueryAttention":
                inputs = list(node.inputs)
                while inputs and (inputs[-1] is None or inputs[-1].name == ""):
                    inputs.pop()
                node.resize_inputs(len(inputs))
                break

    # ------------------------------------------------------------------ #
    # NoPE: toggle rope per layer; toggle window for sliding/full schedule.
    # ------------------------------------------------------------------ #
    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_rope = self.attention_attrs["rope"]
        original_use_rope_in_attn = self.attention_attrs["use_rope_in_attn"]
        original_window = self.window_size

        self.attention_attrs["rope"] = original_rope and self.uses_rope(layer_id)
        if not self.uses_rope(layer_id):
            self.attention_attrs["use_rope_in_attn"] = False
        self.window_size = original_window if self.is_local(layer_id) else -1

        super().make_attention(layer_id, attention, root_input, **kwargs)

        self.attention_attrs["rope"] = original_rope
        self.attention_attrs["use_rope_in_attn"] = original_use_rope_in_attn
        self.window_size = original_window

    # ------------------------------------------------------------------ #
    # Sandwich norm layer with PUBLIC Muse Glimmer module names + dual eps.
    #   input_layernorm -> attn -> post_attention_layernorm(on attn out)
    #   -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm(on mlp out)
    # ------------------------------------------------------------------ #
    def make_layer(self, layer_id, layer):
        # 1) input_layernorm (eps = rms_norm_eps)
        self.layernorm_attrs["epsilon"] = self.rms_norm_eps
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )

        # 2) attention
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # 3) post_attention_layernorm (eps = post_norm_eps) on the attn output before the residual add.
        self.layernorm_attrs["epsilon"] = self.post_norm_eps
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_attn",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # 4) pre_feedforward_layernorm == pre-FFN norm (eps = rms_norm_eps)
        self.layernorm_attrs["epsilon"] = self.rms_norm_eps
        self.make_layernorm(
            layer_id,
            layer.pre_feedforward_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )

        # 5) MLP
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        # 6) post_feedforward_layernorm (eps = post_norm_eps) on MLP output before the residual add.
        self.layernorm_attrs["epsilon"] = self.post_norm_eps
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_feedforward_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_ffn",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # restore main eps + advance layer flags
        self.layernorm_attrs["epsilon"] = self.rms_norm_eps
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ------------------------------------------------------------------ #
    # Final norm exception: the output norm (self.norm) is a plain MuseGlimmerRMSNorm
    # (weight AS-IS, NO +1) unlike the per-layer centered norms. Zero add_offset for it.
    # ------------------------------------------------------------------ #
    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if location == "final_norm":
            saved_offset = self.layernorm_attrs["add_offset"]
            self.layernorm_attrs["add_offset"] = 0
            try:
                super().make_layernorm(layer_id, layernorm, skip, simple, location)
            finally:
                self.layernorm_attrs["add_offset"] = saved_offset
        else:
            super().make_layernorm(layer_id, layernorm, skip, simple, location)

    # ------------------------------------------------------------------ #
    # QK norm: SCALELESS RMSNorm (no learnable weight) on Q and K per head. Emit ones-weight
    # SimplifiedLayerNormalization (== scaleless RMSNorm) directly (module has a single shared
    # qk_norm with no weight; we don't read it).
    # ------------------------------------------------------------------ #
    def make_qk_norm(self, layer_id, attention):
        layernorm_kwargs = {"epsilon": self.rms_norm_eps, "axis": -1, "stash_type": 1}
        ones = torch.ones(self.head_size, dtype=torch.float32)

        for proj, path_key, heads, dim_name in (
            ("q", "q_path", self.num_attn_heads, self.q_size),
            ("k", "k_path", self.num_kv_heads, self.kv_size),
        ):
            reshape_1_name = f"/model/layers.{layer_id}/attn/{proj}_norm/Reshape_1"
            reshape_1_output = f"{reshape_1_name}/output_0"
            self.make_reshape(
                reshape_1_name,
                [self.attention_attrs[path_key], f"/model/constants/INT64/[0, -1, {self.head_size}]"],
                dtype=self.io_dtype,
                shape=["batch_size", f"sequence_length * {heads}", self.head_size],
            )

            ln_name = f"/model/layers.{layer_id}/attn/{proj}_norm/SimplifiedLayerNormalization"
            ln_output = f"{ln_name}/output_0"
            weight_name = f"model.layers.{layer_id}.attn.{proj}_norm.weight"
            self.make_initializer(ones, weight_name, to=self.io_dtype)
            self.make_node(
                "SimplifiedLayerNormalization",
                inputs=[reshape_1_output, weight_name],
                outputs=[ln_output],
                name=ln_name,
                **layernorm_kwargs,
            )
            self.make_value(
                ln_output, self.io_dtype,
                shape=["batch_size", f"sequence_length * {heads}", self.head_size],
            )

            reshape_2_name = f"/model/layers.{layer_id}/attn/{proj}_norm/Reshape_2"
            self.make_reshape(
                reshape_2_name,
                [ln_output, f"/model/constants/INT64/[0, -1, {dim_name}]"],
                dtype=self.io_dtype,
                shape=["batch_size", "sequence_length", dim_name],
            )
            self.attention_attrs[path_key] = f"{reshape_2_name}/output_0"

    # ------------------------------------------------------------------ #
    # Attention output gate: attn_out *= sigmoid(gate_proj(root_input)), between GQA and o_proj.
    # PUBLIC attribute name is `gate_proj` (beta used `output_gate_proj`).
    # ------------------------------------------------------------------ #
    def make_attention_output_proj(self, layer_id, attention, root_input, **kwargs):
        gate_attr = "gate_proj" if hasattr(attention, "gate_proj") else "output_gate_proj"
        if not (self.use_attn_output_gate and hasattr(attention, gate_attr)):
            return super().make_attention_output_proj(layer_id, attention, root_input, **kwargs)

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        attn_output = f"{attn_name}/output_0"
        q_size = self.q_size

        gate_matmul = f"/model/layers.{layer_id}/attn/output_gate/MatMul"
        gate_matmul_name = self.make_matmul(getattr(attention, gate_attr), gate_matmul, root_input)
        gate_path = f"{gate_matmul_name}/output_0"

        sig_name = f"/model/layers.{layer_id}/attn/output_gate/Sigmoid"
        self.make_sigmoid(sig_name, gate_path, self.io_dtype,
                          ["batch_size", "sequence_length", q_size])

        gated_name = f"/model/layers.{layer_id}/attn/output_gate/Mul"
        self.make_mul(gated_name, [attn_output, f"{sig_name}/output_0"], self.io_dtype,
                      ["batch_size", "sequence_length", q_size])
        gated_output = f"{gated_name}/output_0"

        o_proj = "o_proj" if hasattr(attention, "o_proj") else "out_proj" if hasattr(attention, "out_proj") else "dense"
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_matmul_name = self.make_matmul(getattr(attention, o_proj), o_matmul_basename, gated_output)

        o_bias_exists = getattr(attention, o_proj).bias is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            self.make_add_bias(getattr(attention, o_proj).bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_matmul_name if not o_bias_exists else o_add_name}/output_0"

    # ------------------------------------------------------------------ #
    # Normalized token embeddings: append a scaleless RMSNorm after the embedding Gather.
    # (The public MuseGlimmerTextNormedEmbedding applies this internally; the base builder
    # only reads the raw embedding weight, so we re-emit the norm here.)
    # ------------------------------------------------------------------ #
    def make_embedding(self, embedding):
        super().make_embedding(embedding)
        if not self.normalize_tok_embeddings:
            return

        root = self.layernorm_attrs["root_input"]
        ones = torch.ones(self.hidden_size, dtype=torch.float32)
        weight_name = "model.embed_norm.weight"
        self.make_initializer(ones, weight_name, to=self.io_dtype)

        ln_name = "/model/embed_norm/SimplifiedLayerNormalization"
        ln_output = f"{ln_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root, weight_name],
            outputs=[ln_output],
            name=ln_name,
            epsilon=self.rms_norm_eps, axis=-1, stash_type=1,
        )
        self.make_value(ln_output, self.io_dtype,
                        shape=["batch_size", "sequence_length", self.hidden_size])
        self.layernorm_attrs["root_input"] = ln_output
        self.layernorm_attrs["skip_input"] = ln_output
