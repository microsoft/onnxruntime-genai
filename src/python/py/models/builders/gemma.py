# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os

import numpy as np
import torch
from onnx_ir.tensor_adapters import to_torch_dtype

from .mistral import MistralModel


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = np.round(np.sqrt(self.hidden_size), decimals=2)
        self.layernorm_attrs["add_offset"] = 1


# TODO: integrate extra LayerNorms into make_layer in base class
class Gemma2Model(GemmaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = False
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = False
        self.attention_attrs["scale"] = config.query_pre_attn_scalar**-0.5

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_layer(self, layer_id, layer):
        # Gemma-2 decoder layer is typically defined as:
        # input_layernorm --> attention --> post_attention_layernorm --> pre_ffn_layernorm --> MLP --> post_ffn_layernorm

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_attention_layernorm
        # 2. Set skip_input to output of post_attention_layernorm
        # 3. Do not cast outputs from post_attention_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.pre_feedforward_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="pre_feedforward",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_feedforward_layernorm
        # 2. Set skip_input to output of post_feedforward_layernorm
        # 3. Do not cast outputs from post_feedforward_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_feedforward_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_feedforward",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

class Gemma3Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.rope_local_theta = config.rope_local_base_freq
        self.make_rotary_embedding_multi_cache()

    def make_attention_init(self, config):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init(config)

    def make_rotary_embedding_multi_cache(self):
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name
        )

        # Create the new cos/sin caches for local attention layers with its own theta value
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name
        )

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get(
            "cos_cache_name", self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name
        )
        sin_cache_name = kwargs.get(
            "sin_cache_name", self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name
        )
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)


class Gemma4Model(Gemma3Model):
    """Builder for the text decoder of Gemma4Unified (gemma4-12b-it).

    Differs from Gemma3 in several structural ways (see below). Only the text
    component is built; vision/audio configs are ignored.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Per-layer-type geometry. Sliding layers are the base profile; full
        # ("global") layers swap in their own head_dim / kv_heads per layer.
        self.layer_types = list(config.layer_types)
        self.sliding_head_dim = config.head_dim
        self.global_head_dim = config.global_head_dim
        self.sliding_num_kv_heads = config.num_key_value_heads
        self.global_num_kv_heads = config.num_global_key_value_heads
        self.attention_k_eq_v = getattr(config, "attention_k_eq_v", False)

        # RoPE parameters differ per layer type (nested dict in the HF config).
        rope_params = config.rope_parameters
        self.global_rope_theta = rope_params["full_attention"]["rope_theta"]
        self.global_partial_rotary_factor = rope_params["full_attention"]["partial_rotary_factor"]
        self.local_rope_theta = rope_params["sliding_attention"]["rope_theta"]

        # Base __init__ reads config.head_dim (= sliding) for self.head_size and
        # config.num_key_value_heads (= sliding) for self.num_kv_heads, which is
        # the default (sliding) profile. It also builds the RoPE caches via the
        # Gemma3 multi-cache path, so set rope_local_base_freq for that.
        config.rope_local_base_freq = self.local_rope_theta
        # Gemma2Model.__init__ derives its attention scale from
        # query_pre_attn_scalar, which Gemma4 does not have (its Q/K are
        # RMS-normed, so the scale is 1.0). Provide a placeholder so the parent
        # chain runs; the scale is overridden to 1.0 below.
        if not hasattr(config, "query_pre_attn_scalar"):
            config.query_pre_attn_scalar = config.head_dim
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Gemma4 RMSNorm uses the weight directly (no `1 + weight` offset that
        # Gemma1/2/3 apply). This also governs q_norm/k_norm weights.
        self.layernorm_attrs["add_offset"] = 0

        # Q/K are RMS-normed, so HF sets attention scaling to 1.0 (not
        # 1/sqrt(head_dim)).
        self.attention_attrs["scale"] = 1.0

        # Partial rotary on the global layers requires the standalone
        # RotaryEmbedding op (GQA's fused rope has no rotary_embedding_dim
        # attribute). Force external RoPE and keep position_ids as an input.
        self.attention_attrs["use_rope_in_attn"] = False
        if "position_ids" not in self.input_names:
            self.input_names["position_ids"] = "position_ids"
            self.input_types["position_ids"] = self.input_types.get("attention_mask")
            self.input_shapes["position_ids"] = ["batch_size", "sequence_length"]

        # Per-layer residual output multipliers (`layer_scalar`), read from the
        # weights in make_layer.
        self.layer_scalars = {}

    def make_rope_init(self, config):
        # Gemma4 stores rope_parameters as a per-layer-type nested dict
        # ({"full_attention": {...}, "sliding_attention": {...}}) rather than the
        # flat {"rope_type": ...} form the base reader expects. The per-type theta
        # / partial_rotary_factor are applied directly in __init__ and via the
        # Gemma3 multi-cache path, so skip the base initializer.
        return

    def load_weights(self, input_path):
        # The checkpoint is a full multimodal Gemma4Unified model whose text
        # weights live under the `model.language_model.` prefix. Loading the full
        # ConditionalGeneration model (as the base does) also pulls in the vision
        # and audio towers, which are out of scope and roughly double the memory.
        # Instead, build the text-only CausalLM and load just the remapped text
        # weights.
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        import glob

        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.gemma4_unified import Gemma4UnifiedForCausalLM

        config = AutoConfig.from_pretrained(
            self.model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote
        )
        text_config = config.text_config
        text_config.num_hidden_layers = self.num_layers
        text_config.layer_types = text_config.layer_types[: self.num_layers]

        with torch.device("meta"):
            model = Gemma4UnifiedForCausalLM(text_config)

        # Remap `model.language_model.*` -> `model.*` and drop layers beyond the
        # (possibly truncated) layer count.
        prefix = "model.language_model."
        state_dict = {}
        for shard in sorted(glob.glob(os.path.join(self.model_name_or_path, "*.safetensors"))):
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if not key.startswith(prefix):
                        continue
                    new_key = key[len(prefix) :]
                    if new_key.startswith("layers."):
                        if int(new_key.split(".")[1]) >= self.num_layers:
                            continue
                    state_dict["model." + new_key] = f.get_tensor(key)

        if getattr(text_config, "tie_word_embeddings", False):
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise ValueError(f"Missing weights while loading Gemma4 text model: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected weights while loading Gemma4 text model: {unexpected}")

        return model

    def is_local(self, layer_id):
        return self.layer_types[layer_id] == "sliding_attention"

    def layer_head_dim(self, layer_id):
        return self.sliding_head_dim if self.is_local(layer_id) else self.global_head_dim

    def layer_num_kv_heads(self, layer_id):
        return self.sliding_num_kv_heads if self.is_local(layer_id) else self.global_num_kv_heads

    def make_key_value_cache_shape(self, layer_id, shape):
        # Emit concrete kv_heads (dim 1) and head_dim (dim 3) per layer so the
        # runtime's DefaultKeyValueCache detects the per-layer variation.
        shape = super().make_key_value_cache_shape(layer_id, shape)
        return [shape[0], self.layer_num_kv_heads(layer_id), shape[2], self.layer_head_dim(layer_id)]

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Swap in this layer's geometry (head_dim, kv_heads, q/kv sizes) around
        # the base implementation, restoring afterward. Window handling is done
        # by Gemma2Model.make_attention (super) via is_local.
        original = (self.head_size, self.num_kv_heads, self.q_size, self.kv_size)
        self.head_size = self.layer_head_dim(layer_id)
        self.num_kv_heads = self.layer_num_kv_heads(layer_id)
        self.q_size = self.num_attn_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        # RoPE uses a full split-half rotation (rotary_embedding_dim=0) for both
        # layer types. The global layers' partial rotary is baked into the global
        # cache as a zero-padded NoPE tail (see make_proportional_rope_caches), so
        # the op rotates the full head_dim with that pre-zeroed cache — matching
        # HF's proportional RoPE. The external RotaryEmbedding op needs position_ids.
        super().make_attention(
            layer_id, attention, root_input, position_ids=self.input_names["position_ids"], **kwargs
        )
        self.head_size, self.num_kv_heads, self.q_size, self.kv_size = original

    def make_attention_input_proj(self, layer_id, attention, root_input, **kwargs):
        if self.attention_k_eq_v and not self.is_local(layer_id):
            # Full-attention layers share the K projection as V (no v_proj).
            attention.v_proj = attention.k_proj
        super().make_attention_input_proj(layer_id, attention, root_input, **kwargs)
        # Insert the scaleless value RMSNorm (v_norm) on the V path.
        self.make_v_norm(layer_id)

    def make_v_norm(self, layer_id):
        # Scaleless SimplifiedLayerNorm (weight = ones) applied per-head on V,
        # matching HF's Gemma4UnifiedRMSNorm(with_scale=False) on value_states.
        head_size = self.head_size
        kv_size = self.kv_size

        reshape_1_name = f"/model/layers.{layer_id}/attn/v_norm/Reshape_1"
        reshape_1_inputs = [self.attention_attrs["v_path"], f"/model/constants/INT64/[0, -1, {head_size}]"]
        self.make_reshape(
            reshape_1_name,
            reshape_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length * num_key_value_heads", head_size],
        )

        weight_name = f"model.layers.{layer_id}.attn.v_norm.layernorm.weight"
        self.make_initializer(torch.ones(head_size), weight_name, to=self.io_dtype)

        layernorm_name = f"/model/layers.{layer_id}/attn/v_norm/SimplifiedLayerNormalization"
        layernorm_output = f"{layernorm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[f"{reshape_1_name}/output_0", weight_name],
            outputs=[layernorm_output],
            name=layernorm_name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(
            layernorm_output,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length * num_key_value_heads", head_size],
        )

        reshape_2_name = f"/model/layers.{layer_id}/attn/v_norm/Reshape_2"
        reshape_2_inputs = [layernorm_output, f"/model/constants/INT64/[0, -1, {kv_size}]"]
        self.make_reshape(
            reshape_2_name,
            reshape_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", kv_size],
        )
        self.attention_attrs["v_path"] = f"{reshape_2_name}/output_0"

    def make_rotary_embedding_multi_cache(self):
        # Build the global cache with the proportional RoPE variant and the
        # local cache with default RoPE (theta = local_rope_theta on the sliding
        # head_dim). Overrides Gemma3's default-both-caches behavior.
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        cos_global, sin_global = self.make_proportional_rope_caches()
        self.make_initializer(cos_global, self.cos_cache_global_name, to=self.io_dtype)
        self.make_initializer(sin_global, self.sin_cache_global_name, to=self.io_dtype)

        # Local (sliding) cache: default RoPE on the sliding head_dim.
        original = (self.head_size, self.rope_attrs["theta"], self.rope_attrs["partial_rotary_factor"])
        self.head_size = self.sliding_head_dim
        self.rope_attrs["theta"] = self.local_rope_theta
        self.rope_attrs["partial_rotary_factor"] = 1.0
        self.rope_attrs["create_caches"] = True
        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super(Gemma3Model, self).make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name
        )
        self.head_size, self.rope_attrs["theta"], self.rope_attrs["partial_rotary_factor"] = original

    def make_proportional_rope_caches(self):
        # Replicates transformers' _compute_proportional_rope_parameters:
        # partial rotary applied on the *global* head_dim, with a zero-padded
        # NoPE tail so the emitted rotary_embedding_dim spans the full head_dim.
        head_dim = self.global_head_dim
        base = self.global_rope_theta
        rope_angles = int(self.global_partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)), dim=0)
        else:
            inv_freq = inv_freq_rotated

        t = torch.arange(self.context_length, dtype=torch.int64).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cache, sin_cache = emb.cos(), emb.sin()
        cos_cache = cos_cache.squeeze().to(to_torch_dtype(self.io_dtype))
        sin_cache = sin_cache.squeeze().to(to_torch_dtype(self.io_dtype))
        # Halve to (M, head_dim/2) as the RotaryEmbedding kernel expects. The
        # NoPE tail contributes zero-frequency (cos=1, sin=0) entries.
        cos_cache = cos_cache[:, : (head_dim // 2)]
        sin_cache = sin_cache[:, : (head_dim // 2)]
        return cos_cache, sin_cache

    def make_layer(self, layer_id, layer):
        super().make_layer(layer_id, layer)

        # Apply the per-layer residual multiplier (`layer_scalar`) to the whole
        # layer output, matching HF's `hidden_states *= self.layer_scalar`.
        #
        # In the SkipLayerNorm design the layer output is carried as
        # (root_input + skip_input), summed inside the next SkipLayerNorm (or the
        # final norm). To scale the sum, scale both operands. `last_layernorm` is
        # set by the super() call for the final layer; the final norm consumes the
        # same (root_input, skip_input) pair, so scaling both is correct there too.
        #
        # Gemma2/3 keep the residual (root_input, via SkipLayerNorm output_3) in
        # fp32 while skip_input is io_dtype, so each Mul must use its operand's own
        # recorded dtype (and a matching scalar constant).
        scalar = float(layer.layer_scalar.item())

        for suffix, attr in (("skip", "skip_input"), ("root", "root_input")):
            operand = self.layernorm_attrs[attr]
            operand_dtype = self.values[operand].dtype
            mul_name = f"/model/layers.{layer_id}/layer_scalar/Mul_{suffix}"
            self.make_mul(
                mul_name,
                [operand, f"/model/constants/{self.to_str_dtype(operand_dtype)}/{scalar}"],
                dtype=operand_dtype,
                shape=["batch_size", "sequence_length", self.hidden_size],
            )
            self.layernorm_attrs[attr] = f"{mul_name}/output_0"
