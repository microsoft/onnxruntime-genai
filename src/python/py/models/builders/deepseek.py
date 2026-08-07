# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
"""ONNX model builder for DeepSeek V4 (DeepseekV4ForCausalLM architecture).

DeepSeek V4 introduces several novel components compared to prior DeepSeek models:

  * Manifold-Constrained Hyper-Connections (mHC): 4-D multi-stream hidden state
    [B, S, hc_mult, D] propagated through every decoder layer.  A pair of
    ``DeepseekV4HyperConnection`` modules handle stream collapse/expand per layer;
    a ``DeepseekV4HyperHead`` collapses the streams before the final norm.
  * Shared-KV Multi-Query Attention (MQA) with ``num_key_value_heads=1``;  K and V
    are the same tensor.  RoPE uses the interleaved convention and is applied to
    the *trailing* ``qk_rope_head_dim`` channels of each head.
  * Per-head learnable attention sinks.
  * Grouped low-rank output projection (``o_a_proj`` / ``o_b_proj``).
  * MoE blocks with two router types: ``DeepseekV4HashRouter`` (frozen lookup by
    token-id) and ``DeepseekV4TopKRouter`` (learned), plus one always-active shared
    expert.
    * Stateful Heavily Compressed Attention (HCA) and Compressed Sparse Attention
        (CSA), including the Lightning Indexer used by CSA.

Reference implementation
------------------------
The numerics implemented here follow the HuggingFace ``DeepseekV4`` modeling code
(``modeling_deepseek_v4.py``): ``DeepseekV4HyperConnection``, ``DeepseekV4Attention``,
``DeepseekV4RotaryEmbedding`` / ``apply_rotary_pos_emb``, ``DeepseekV4SparseMoeBlock``
and its two routers.  When updating this builder, diff against that file rather than
against earlier DeepSeek generations, whose MLA/RoPE conventions differ.

The compression, attention, hash-routing, and hyper-connection kernels are provided
by ONNX Runtime contrib ops. Remaining projections are emitted as standard ONNX
subgraphs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import to_torch_dtype
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

from .base import Model


class DeepSeekV4Model(Model):
    """ONNX builder for ``DeepseekV4ForCausalLM`` (DeepSeek V4 Flash and Pro)."""

    # Name of the ONNX value that carries the 4-D HC-stream tensor throughout
    # model construction; updated by ``make_layer`` and consumed by ``make_hc_head``.
    hc_streams: str

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # ------------------------------------------------------------------ #
        # Patch the config so the base-class __init__ picks up the right values
        # ------------------------------------------------------------------ #

        # partial_rotary_factor is None by default in DeepseekV4Config; compute it
        # from qk_rope_head_dim / head_dim.
        if getattr(config, "partial_rotary_factor", None) is None:
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            config.partial_rotary_factor = qk_rope_head_dim / head_dim

        # The base class uses config.intermediate_size.  DeepSeek V4 stores the MoE
        # intermediate size in config.moe_intermediate_size and maps it via
        # attribute_map; ensure the canonical name is always available.
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = config.moe_intermediate_size

        # Some config fields the base class expects
        if not hasattr(config, "hidden_act"):
            config.hidden_act = "silu"

        # Base class reads config.activation; if absent, fall back to hidden_act.
        # (DeepseekV4Config uses hidden_act; no hidden_activation attribute.)

        self.dequantize_fp8 = (
            hasattr(config, "quantization_config")
            and config.quantization_config.get("quant_method") == "fp8"
        )
        if self.dequantize_fp8:
            delattr(config, "quantization_config")

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.local_cache_capacity = self.window_size if self.window_size > 0 else self.context_length
        fixed_kv_shape = ["batch_size", 1, self.local_cache_capacity, self.head_size]
        self.input_shapes["past_key_values.key"] = fixed_kv_shape
        self.input_shapes["past_key_values.value"] = fixed_kv_shape
        self.output_shapes["present.key"] = fixed_kv_shape
        self.output_shapes["present.value"] = fixed_kv_shape

        # ------------------------------------------------------------------ #
        # DeepSeek V4 – specific hyper-parameters
        # ------------------------------------------------------------------ #
        self.hc_mult: int = getattr(config, "hc_mult", 4)
        self.hc_sinkhorn_iters: int = getattr(config, "hc_sinkhorn_iters", 20)
        self.hc_eps: float = getattr(config, "hc_eps", 1e-6)

        # Attention
        self.q_lora_rank: int = getattr(config, "q_lora_rank", 1024)
        self.qk_rope_head_dim: int = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        self.qk_nope_head_dim: int = self.head_size - self.qk_rope_head_dim
        self.o_groups: int = getattr(config, "o_groups", 8)
        self.o_lora_rank: int = getattr(config, "o_lora_rank", 1024)
        self.compress_rates: dict[str, int] = getattr(
            config,
            "compress_rates",
            {
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
        )
        self.compress_rope_theta: float = getattr(config, "compress_rope_theta", 160000.0)
        self.index_n_heads: int = getattr(config, "index_n_heads", 64)
        self.index_head_dim: int = getattr(config, "index_head_dim", 128)
        self.index_topk: int = getattr(config, "index_topk", 512)
        self.rms_norm_epsilon: float = getattr(config, "rms_norm_eps", 1e-6)

        # MoE
        self.moe_intermediate_size: int = getattr(config, "moe_intermediate_size", config.intermediate_size)
        self.n_shared_experts: int = getattr(config, "n_shared_experts", 1)
        self.swiglu_limit: float = getattr(config, "swiglu_limit", 10.0)
        self.routed_scaling_factor: float = getattr(config, "routed_scaling_factor", 1.5)
        # Router activation applied to the raw gate logits before top-k selection and
        # weighting (`DeepseekV4TopKRouter`/`DeepseekV4HashRouter.score_fn` in the reference).
        # `norm_topk_prob` is deliberately not read here: the reference router always
        # renormalizes the selected weights to sum to 1, regardless of that config value.
        self.scoring_func: str = getattr(config, "scoring_func", "sqrtsoftplus")

        # Per-layer type info
        raw_layer_types = getattr(config, "layer_types", None)
        raw_mlp_layer_types = getattr(config, "mlp_layer_types", None)
        default_layer_type = "sliding_attention"
        default_mlp_layer_type = "moe"
        self.layer_types = raw_layer_types or [default_layer_type] * self.num_layers
        self.mlp_layer_types = raw_mlp_layer_types or [default_mlp_layer_type] * self.num_layers

        self.compression_state_names: list[tuple[str, str]] = []
        self.initialize_compression_states()

        # Track the name of the current HC-stream ONNX value (updated in make_layer)
        self.hc_streams = ""

        # When True, RoPE is emitted as the decomposed reference subgraph instead of a
        # com.microsoft::RotaryEmbedding node.  Only the parity test flips this; it is
        # deliberately not exposed as an extra_option.
        self.use_manual_deepseek_rope = False

    def load_weights(self, input_path):
        if not self.dequantize_fp8:
            return super().load_weights(input_path)

        extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            dtype="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True),
            **extra_kwargs,
        )

        if "adapter_path" in self.extra_options:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, self.extra_options["adapter_path"], cache_dir=self.cache_dir, token=self.hf_token
            )

        return model

    def add_compression_state(self, layer_id: int, name: str, shape: list) -> tuple[str, str]:
        past_name = f"past_compression.{layer_id}.{name}"
        present_name = f"present_compression.{layer_id}.{name}"
        self.input_names[past_name] = past_name
        self.input_types[past_name] = self.io_dtype
        self.input_shapes[past_name] = shape
        self.output_names[present_name] = present_name
        self.output_types[present_name] = self.io_dtype
        self.output_shapes[present_name] = shape
        self.compression_state_names.append((past_name, present_name))
        return past_name, present_name

    def initialize_compression_states(self) -> None:
        for layer_id, layer_type in enumerate(self.layer_types):
            if layer_type == "heavily_compressed_attention":
                rate = self.compress_rates[layer_type]
                entry_capacity = (self.context_length + rate - 1) // rate
                self.add_compression_state(layer_id, "pending_kv", ["batch_size", rate - 1, self.head_size])
                self.add_compression_state(layer_id, "pending_gate", ["batch_size", rate - 1, self.head_size])
                self.add_compression_state(layer_id, "entries", ["batch_size", 1, entry_capacity, self.head_size])
            elif layer_type == "compressed_sparse_attention":
                rate = self.compress_rates[layer_type]
                entry_capacity = (self.context_length + rate - 1) // rate
                for prefix, width, rank4 in (
                    ("compressor", self.head_size, True),
                    ("indexer", self.index_head_dim, False),
                ):
                    self.add_compression_state(
                        layer_id, f"{prefix}_pending_kv", ["batch_size", rate - 1, 2 * width]
                    )
                    self.add_compression_state(
                        layer_id, f"{prefix}_pending_gate", ["batch_size", rate - 1, 2 * width]
                    )
                    entries_shape = (
                        ["batch_size", 1, entry_capacity, width]
                        if rank4
                        else ["batch_size", entry_capacity, width]
                    )
                    self.add_compression_state(layer_id, f"{prefix}_entries", entries_shape)
                    self.add_compression_state(layer_id, f"{prefix}_overlap_kv", ["batch_size", rate, width])
                    self.add_compression_state(layer_id, f"{prefix}_overlap_gate", ["batch_size", rate, width])

    def update_genai_config(self, genai_config):
        if self.compression_state_names:
            decoder = genai_config["model"]["decoder"]
            decoder["inputs"]["past_state_names"] = [names[0] for names in self.compression_state_names]
            decoder["outputs"]["present_state_names"] = [names[1] for names in self.compression_state_names]

    def compression_state(self, layer_id: int, name: str) -> tuple[str, str]:
        return f"past_compression.{layer_id}.{name}", f"present_compression.{layer_id}.{name}"

    def make_compression_weight(self, tensor, name: str) -> str:
        self.make_initializer(tensor.data.T.contiguous(), name, to=self.io_dtype)
        return name

    def make_compression_rope_caches(self) -> None:
        if hasattr(self, "compression_rope_inited"):
            return
        rope_dim = self.qk_rope_head_dim
        inv_freq = 1.0 / (self.compress_rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        positions = torch.arange(self.rope_attrs["cache_length"], dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        dtype = to_torch_dtype(self.io_dtype)
        self.make_initializer(frequencies.cos().to(dtype), "deepseek_compress_cos_cache")
        self.make_initializer(frequencies.sin().to(dtype), "deepseek_compress_sin_cache")
        self.compression_rope_inited = True

    def make_compressor(self, layer_id: int, attn, collapsed: str, q_residual: str) -> tuple[str, str, str]:
        layer_type = self.layer_types[layer_id]
        if layer_type == "sliding_attention":
            return "", "", ""

        self.make_compression_rope_caches()
        base = f"/model/layers.{layer_id}/attn/compressor"
        compressor = attn.compressor
        common_inputs = [
            collapsed,
            self.input_names["position_ids"],
            "deepseek_compress_cos_cache",
            "deepseek_compress_sin_cache",
        ]
        common_attrs = {
            "compress_rate": self.compress_rates[layer_type],
            "entry_capacity": (
                self.context_length + self.compress_rates[layer_type] - 1
            )
            // self.compress_rates[layer_type],
            "rotary_dim": self.qk_rope_head_dim,
            "rms_norm_epsilon": self.rms_norm_epsilon,
        }

        if layer_type == "heavily_compressed_attention":
            weight_names = [
                self.make_compression_weight(
                    compressor.kv_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.kv_proj.weight"
                ),
                self.make_compression_weight(
                    compressor.gate_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.gate_proj.weight"
                ),
            ]
            position_bias = f"model.layers.{layer_id}.self_attn.compressor.position_bias"
            norm_weight = f"model.layers.{layer_id}.self_attn.compressor.kv_norm.weight"
            self.make_initializer(compressor.position_bias.data, position_bias, to=self.io_dtype)
            self.make_initializer(compressor.kv_norm.weight.data, norm_weight, to=self.io_dtype)
            states = [self.compression_state(layer_id, name) for name in ("pending_kv", "pending_gate", "entries")]
            outputs = [f"{base}/output_0", f"{base}/output_1"] + [state[1] for state in states]
            self.make_node(
                "HeavilyCompressedAttention",
                common_inputs + weight_names + [position_bias, norm_weight] + [state[0] for state in states],
                outputs,
                name=base,
                domain="com.microsoft",
                **common_attrs,
            )
            entry_capacity = common_attrs["entry_capacity"]
            self.make_value(outputs[0], self.io_dtype, ["batch_size", 1, entry_capacity, self.head_size])
            self.make_value(outputs[1], self.io_dtype, ["batch_size", 1, "sequence_length", entry_capacity])
            return outputs[0], outputs[1], ""

        states = [
            self.compression_state(layer_id, f"compressor_{name}")
            for name in ("pending_kv", "pending_gate", "entries", "overlap_kv", "overlap_gate")
        ]
        weight_names = [
            self.make_compression_weight(
                compressor.kv_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.kv_proj.weight"
            ),
            self.make_compression_weight(
                compressor.gate_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.gate_proj.weight"
            ),
        ]
        position_bias = f"model.layers.{layer_id}.self_attn.compressor.position_bias"
        norm_weight = f"model.layers.{layer_id}.self_attn.compressor.kv_norm.weight"
        self.make_initializer(compressor.position_bias.data, position_bias, to=self.io_dtype)
        self.make_initializer(compressor.kv_norm.weight.data, norm_weight, to=self.io_dtype)
        outputs = [f"{base}/output_0"] + [state[1] for state in states]
        self.make_node(
            "CompressedSparseAttention",
            common_inputs + weight_names + [position_bias, norm_weight] + [state[0] for state in states],
            outputs,
            name=base,
            domain="com.microsoft",
            **common_attrs,
        )
        self.make_value(
            outputs[0], self.io_dtype, ["batch_size", 1, common_attrs["entry_capacity"], self.head_size]
        )

        indexer = compressor.indexer
        index_base = f"{base}/indexer"
        index_states = [
            self.compression_state(layer_id, f"indexer_{name}")
            for name in ("pending_kv", "pending_gate", "entries", "overlap_kv", "overlap_gate")
        ]
        index_weights = [
            self.make_compression_weight(
                indexer.kv_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.indexer.kv_proj.weight"
            ),
            self.make_compression_weight(
                indexer.gate_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.indexer.gate_proj.weight"
            ),
        ]
        index_position_bias = f"model.layers.{layer_id}.self_attn.compressor.indexer.position_bias"
        index_norm_weight = f"model.layers.{layer_id}.self_attn.compressor.indexer.kv_norm.weight"
        q_weight = self.make_compression_weight(
            indexer.q_b_proj.weight, f"model.layers.{layer_id}.self_attn.compressor.indexer.q_b_proj.weight"
        )
        score_weight = self.make_compression_weight(
            indexer.scorer.weights_proj.weight,
            f"model.layers.{layer_id}.self_attn.compressor.indexer.scorer.weights_proj.weight",
        )
        self.make_initializer(indexer.position_bias.data, index_position_bias, to=self.io_dtype)
        self.make_initializer(indexer.kv_norm.weight.data, index_norm_weight, to=self.io_dtype)
        index_outputs = [f"{index_base}/output_0"] + [state[1] for state in index_states]
        self.make_node(
            "LightningIndexer",
            [
                collapsed,
                q_residual,
                self.input_names["position_ids"],
                "deepseek_compress_cos_cache",
                "deepseek_compress_sin_cache",
            ]
            + index_weights
            + [index_position_bias, index_norm_weight, q_weight, score_weight]
            + [state[0] for state in index_states],
            index_outputs,
            name=index_base,
            domain="com.microsoft",
            num_heads=self.index_n_heads,
            head_size=self.index_head_dim,
            index_topk=self.index_topk,
            **common_attrs,
        )
        self.make_value(index_outputs[0], ir.DataType.INT64, ["batch_size", "sequence_length", self.index_topk])
        return outputs[0], "", index_outputs[0]

    def make_compressed_attention_bias(self, layer_id: int, compressed_kv: str, block_bias: str) -> str:
        if not compressed_kv:
            return ""
        if not block_bias:
            base = f"/model/layers.{layer_id}/attn/compressed_bias"
            compressed_shape = f"{base}/compressed/Shape"
            query_shape = f"{base}/query/Shape"
            self.make_shape(compressed_shape, compressed_kv, shape=[4])
            self.make_shape(query_shape, self.input_names["position_ids"], shape=[2])
            batch_name = f"{base}/batch/Gather"
            sequence_name = f"{base}/sequence/Gather"
            entries_name = f"{base}/entries/Gather"
            self.make_gather(
                batch_name,
                [f"{compressed_shape}/output_0", "/model/constants/INT64/[0]"],
                ir.DataType.INT64,
                [1],
                axis=0,
            )
            self.make_gather(
                sequence_name, [f"{query_shape}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, [1], axis=0
            )
            self.make_gather(
                entries_name,
                [f"{compressed_shape}/output_0", "/model/constants/INT64/[2]"],
                ir.DataType.INT64,
                [1],
                axis=0,
            )
            bias_shape = f"{base}/shape/Concat"
            self.make_concat(
                bias_shape,
                [
                    f"{batch_name}/output_0",
                    "/model/constants/INT64/[1]",
                    f"{sequence_name}/output_0",
                    f"{entries_name}/output_0",
                ],
                ir.DataType.INT64,
                [4],
                axis=0,
            )
            zeros_name = f"{base}/ConstantOfShape"
            self.make_constant_of_shape(
                zeros_name,
                f"{bias_shape}/output_0",
                ir.tensor([0.0], dtype=self.io_dtype),
                self.io_dtype,
                ["batch_size", 1, "sequence_length", "compressed_length"],
            )
            block_bias = f"{zeros_name}/output_0"
        return block_bias

    def make_compressed_attention(
        self,
        layer_id: int,
        query: str,
        local_kv: str,
        compressed_kv: str,
        attention_bias: str,
        selected_indices: str,
        head_sink: str,
        past_local_kv: str = "",
        present_local_kv: str = "",
    ) -> str:
        name = f"/model/layers.{layer_id}/attn/CompressedAttention"
        output = f"{name}/output_0"
        inputs = [query, local_kv, compressed_kv, attention_bias, selected_indices, head_sink]
        outputs = [output]
        if past_local_kv:
            inputs.extend([past_local_kv, self.input_names["position_ids"]])
            outputs.append(present_local_kv)
        self.make_node(
            "CompressedAttention",
            inputs,
            outputs,
            name=name,
            domain="com.microsoft",
            scale=self.head_size**-0.5,
        )
        self.make_value(output, self.io_dtype, ["batch_size", self.num_attn_heads, "sequence_length", self.head_size])
        if present_local_kv:
            self.make_value(
                present_local_kv,
                self.io_dtype,
                ["batch_size", 1, self.local_cache_capacity, self.head_size],
            )
        return output

    def make_layernorm_no_skip(self, layer_id: int, layernorm, root_input: str, location: str) -> str:
        """Emit a no-skip SimplifiedLayerNorm via the base make_layernorm path."""
        saved_root = self.layernorm_attrs["root_input"]
        saved_skip = self.layernorm_attrs["skip_input"]
        saved_output_0 = self.layernorm_attrs.get("output_0", "")
        saved_output_3 = self.layernorm_attrs.get("output_3", "")

        self.layernorm_attrs["root_input"] = root_input
        self.layernorm_attrs["skip_input"] = root_input
        self.make_layernorm(layer_id, layernorm, skip=False, simple=True, location=location)
        output = self.layernorm_attrs["output_0"]

        self.layernorm_attrs["root_input"] = saved_root
        self.layernorm_attrs["skip_input"] = saved_skip
        self.layernorm_attrs["output_0"] = saved_output_0
        self.layernorm_attrs["output_3"] = saved_output_3
        return output

    # ------------------------------------------------------------------ #
    # Override: force MultiHeadAttention so the correct 4D causal mask is
    # built by make_preprocessing_nodes.  DeepSeek V4 uses its own manual
    # attention subgraph; the fused GQA op is incompatible with sinks.
    # ------------------------------------------------------------------ #

    def make_attention_init(self, config):
        # Sizes
        self.q_size = self.num_attn_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size

        # Always use MultiHeadAttention to get the right mask reformatting subgraph.
        # The actual attention is implemented manually in make_deepseek_attention.
        self.attention_attrs["op_type"] = "MultiHeadAttention"
        self.past_present_share_buffer = True

    # ------------------------------------------------------------------ #
    # Helpers: RMS norm without learnable weight
    # ------------------------------------------------------------------ #

    def make_unweighted_rms_norm(self, name: str, root_input: str, eps: float, shape: list) -> str:
        """Emit an UnweightedRMSNorm subgraph (no scale parameter).

        Returns the output value name.
        """
        float_dtype = ir.DataType.FLOAT

        # Cast input to fp32 for stability (matching the PyTorch reference which
        # calls the norm in float32 regardless of the model IO dtype)
        cast_in_name = f"{name}/to_fp32/Cast"
        self.make_cast(cast_in_name, root_input, float_dtype, shape)

        sq_name = f"{name}/sq/Mul"
        self.make_mul(sq_name, [f"{cast_in_name}/output_0", f"{cast_in_name}/output_0"],
                      float_dtype, shape)

        mean_name = f"{name}/ReduceMean"
        self.make_reduce_mean(mean_name,
                              [f"{sq_name}/output_0",
                               f"/model/constants/INT64/[-1]"],
                              float_dtype, shape[:-1] + [1], keepdims=True)

        eps_const = f"/model/constants/FLOAT/{eps}"
        add_eps_name = f"{name}/eps/Add"
        self.make_add(add_eps_name, [f"{mean_name}/output_0", eps_const],
                      float_dtype, shape[:-1] + [1])

        rsqrt_name = f"{name}/RSqrt"
        self.make_rsqrt(rsqrt_name,
                        [f"{add_eps_name}/output_0"],
                        float_dtype, shape[:-1] + [1])

        normed_name = f"{name}/normed/Mul"
        self.make_mul(normed_name,
                      [f"{cast_in_name}/output_0", f"{rsqrt_name}/output_0"],
                      float_dtype, shape)

        # Cast back to model IO dtype
        cast_out_name = f"{name}/to_io/Cast"
        self.make_cast(cast_out_name, f"{normed_name}/output_0", self.io_dtype, shape)
        return f"{cast_out_name}/output_0"

    # ------------------------------------------------------------------ #
    # Helpers: interleaved RoPE applied to the *trailing* rope_dim channels
    # ------------------------------------------------------------------ #

    def build_deepseek_rope_caches(self, expanded: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin caches for DeepSeek V4 interleaved RoPE.

        V4 rotates the trailing ``qk_rope_head_dim`` channels of each head using the
        interleaved convention, so consecutive channel pairs ``(2i, 2i+1)`` share one
        angle.  The caches are returned with one entry per rotated *pair*, i.e. shape
        ``[cache_length, qk_rope_head_dim // 2]``.  That is the layout the
        ``com.microsoft::RotaryEmbedding`` kernel expects (it does the pairing itself),
        and it matches ``Model.make_rotary_embedding_caches`` in the base class, which
        also halves the cache width.

        When ``expanded`` is True the caches are ``repeat_interleave``-expanded to
        ``[cache_length, qk_rope_head_dim]``, one entry per channel.  That is the layout
        consumed by the decomposed reference subgraph in ``make_deepseek_rope_manual``.
        """
        rope_dim = self.qk_rope_head_dim
        theta = self.rope_attrs["theta"]
        # Matches DeepseekV4RotaryEmbedding.compute_default_rope_parameters, which uses
        # dim = head_dim * partial_rotary_factor = qk_rope_head_dim.
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        ctx = self.rope_attrs["cache_length"]
        t = torch.arange(ctx, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [ctx, rope_dim // 2]
        cos_cache, sin_cache = freqs.cos(), freqs.sin()
        if expanded:
            cos_cache = cos_cache.repeat_interleave(2, dim=-1)
            sin_cache = sin_cache.repeat_interleave(2, dim=-1)
        torch_dtype = to_torch_dtype(self.io_dtype)
        return cos_cache.to(torch_dtype), sin_cache.to(torch_dtype)

    def make_deepseek_rope_init(self):
        """Create the cos/sin cache initializers (called once).

        A negated sine cache is created alongside the regular one so that the conjugate
        de-rotation applied to the attention output costs no extra graph nodes.
        """
        if hasattr(self, "deepseek_rope_inited"):
            return
        cos_cache, sin_cache = self.build_deepseek_rope_caches(expanded=self.use_manual_deepseek_rope)
        self.make_initializer(cos_cache, "deepseek_cos_cache")
        self.make_initializer(sin_cache, "deepseek_sin_cache")
        self.make_initializer(-sin_cache, "deepseek_sin_cache_neg")
        self.deepseek_rope_inited = True

    def make_deepseek_rotary_embedding(
        self, name: str, root_input: str, sin_cache: str, num_heads: int, rope_dim: int
    ) -> str:
        """Emit a ``com.microsoft::RotaryEmbedding`` node over a [B, H, S, rope_dim] tensor.

        The kernel infers ``num_heads`` and ``head_size`` from the rank-4 input, and with
        the default ``rotary_embedding_dim=0`` it rotates the full width of that input.
        DeepSeek V4's partial rotation is therefore expressed by slicing the trailing
        channels before the op rather than by the ``rotary_embedding_dim`` attribute,
        which selects the *leading* channels.
        """
        output = f"{name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            inputs=[root_input, self.input_names["position_ids"], "deepseek_cos_cache", sin_cache],
            outputs=[output],
            name=name,
            domain="com.microsoft",
            interleaved=1,
        )
        self.make_value(output, self.io_dtype, shape=["batch_size", num_heads, "sequence_length", rope_dim])
        return output

    def make_deepseek_rope(
        self,
        name: str,
        root_input: str,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        neg_sin: bool = False,
    ) -> str:
        """Apply interleaved RoPE to the last ``rope_dim`` channels of heads.

        ``root_input`` has shape [B, num_heads, S, head_dim].
        Returns an ONNX value name of the same shape.

        When ``neg_sin=True``, the conjugate rotation (-sin) is applied.  This
        is used for the output projection's de-rotation step.
        """
        if self.use_manual_deepseek_rope:
            return self.make_deepseek_rope_manual(name, root_input, num_heads, head_dim, rope_dim, neg_sin)

        self.make_deepseek_rope_init()

        nope_dim = head_dim - rope_dim
        sin_cache = "deepseek_sin_cache_neg" if neg_sin else "deepseek_sin_cache"

        if nope_dim == 0:
            # Whole head is rotated; no slice/concat plumbing needed.
            return self.make_deepseek_rotary_embedding(
                f"{name}/RotaryEmbedding", root_input, sin_cache, num_heads, head_dim
            )

        rope_in_name = f"{name}/rope_in/Slice"
        self.make_slice(
            rope_in_name,
            [root_input,
             f"/model/constants/INT64/[{nope_dim}]",
             f"/model/constants/INT64/[{head_dim}]",
             "/model/constants/INT64/[3]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim],
        )

        rope_out = self.make_deepseek_rotary_embedding(
            f"{name}/RotaryEmbedding", f"{rope_in_name}/output_0", sin_cache, num_heads, rope_dim
        )

        nope_name = f"{name}/nope/Slice"
        self.make_slice(
            nope_name,
            [root_input,
             "/model/constants/INT64/[0]",
             f"/model/constants/INT64/[{nope_dim}]",
             "/model/constants/INT64/[3]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", nope_dim],
        )

        final_concat_name = f"{name}/final/Concat"
        self.make_concat(
            final_concat_name,
            [f"{nope_name}/output_0", rope_out],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", head_dim],
            axis=3,
        )
        return f"{final_concat_name}/output_0"

    def make_deepseek_rope_manual(
        self,
        name: str,
        root_input: str,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        neg_sin: bool = False,
    ) -> str:
        """Decomposed reference implementation of ``make_deepseek_rope``.

        Kept as the numerical reference that the ``com.microsoft::RotaryEmbedding``
        emission is validated against.  It is not reachable from a normal build; only
        the parity test sets ``use_manual_deepseek_rope``.
        """
        self.make_deepseek_rope_init()

        nope_dim = head_dim - rope_dim

        # ---- Extract nope / rope slices along the last (head_dim) axis ----
        # Slice(input, starts=[0], ends=[nope_dim], axes=[3])
        nope_name = f"{name}/nope/Slice"
        self.make_slice(
            nope_name,
            [root_input,
             f"/model/constants/INT64/[0]",
             f"/model/constants/INT64/[{nope_dim}]",
             f"/model/constants/INT64/[3]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", nope_dim],
        )

        rope_in_name = f"{name}/rope_in/Slice"
        self.make_slice(
            rope_in_name,
            [root_input,
             f"/model/constants/INT64/[{nope_dim}]",
             f"/model/constants/INT64/[{head_dim}]",
             f"/model/constants/INT64/[3]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim],
        )

        # ---- Gather cos/sin for the current positions ----
        # deepseek_cos_cache: [ctx, rope_dim].  position_ids: [B, S].
        # Gather(cos_cache, position_ids, axis=0) → [B, S, rope_dim]
        pos_ids = self.input_names["position_ids"]  # [B, S]

        cos_gathered_name = f"{name}/cos/Gather"
        self.make_gather(
            cos_gathered_name,
            ["deepseek_cos_cache", pos_ids],
            self.io_dtype,
            ["batch_size", "sequence_length", rope_dim],
            axis=0,
        )
        sin_gathered_name = f"{name}/sin/Gather"
        self.make_gather(
            sin_gathered_name,
            ["deepseek_sin_cache", pos_ids],
            self.io_dtype,
            ["batch_size", "sequence_length", rope_dim],
            axis=0,
        )

        # Unsqueeze to [B, 1, S, rope_dim] for head broadcasting
        cos_u_name = f"{name}/cos/Unsqueeze"
        self.make_unsqueeze(
            cos_u_name,
            [f"{cos_gathered_name}/output_0", "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["batch_size", 1, "sequence_length", rope_dim],
        )
        sin_u_name = f"{name}/sin/Unsqueeze"
        self.make_unsqueeze(
            sin_u_name,
            [f"{sin_gathered_name}/output_0", "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["batch_size", 1, "sequence_length", rope_dim],
        )

        # ---- Interleaved rotate_half ----
        # Even indices: rope_in[..., 0::2]
        even_name = f"{name}/even/Slice"
        self.make_slice(
            even_name,
            [f"{rope_in_name}/output_0",
             "/model/constants/INT64/[0]",
             f"/model/constants/INT64/[{rope_dim}]",
             "/model/constants/INT64/[3]",
             "/model/constants/INT64/[2]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2],
        )
        # Odd indices: rope_in[..., 1::2]
        odd_name = f"{name}/odd/Slice"
        self.make_slice(
            odd_name,
            [f"{rope_in_name}/output_0",
             "/model/constants/INT64/[1]",
             f"/model/constants/INT64/[{rope_dim}]",
             "/model/constants/INT64/[3]",
             "/model/constants/INT64/[2]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2],
        )

        # Negate odd slice → -x2
        neg_odd_name = f"{name}/odd/Neg"
        self.make_neg(
            neg_odd_name,
            f"{odd_name}/output_0",
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2],
        )

        # Unsqueeze even and neg_odd along last axis for concat-interleave
        neg_odd_u_name = f"{name}/odd/Unsqueeze"
        self.make_unsqueeze(
            neg_odd_u_name,
            [f"{neg_odd_name}/output_0", "/model/constants/INT64/[4]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2, 1],
        )
        even_u_name = f"{name}/even/Unsqueeze"
        self.make_unsqueeze(
            even_u_name,
            [f"{even_name}/output_0", "/model/constants/INT64/[4]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2, 1],
        )

        # Concat → [B, H, S, rope_dim//2, 2]
        rotated_stacked_name = f"{name}/rotated/Concat"
        self.make_concat(
            rotated_stacked_name,
            [f"{neg_odd_u_name}/output_0", f"{even_u_name}/output_0"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim // 2, 2],
            axis=4,
        )

        # Reshape → [B, H, S, rope_dim]
        rotated_name = f"{name}/rotated/Reshape"
        self.make_reshape(
            rotated_name,
            [f"{rotated_stacked_name}/output_0",
             f"/model/constants/INT64/[0, {num_heads}, 0, {rope_dim}]"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", rope_dim],
        )

        # ---- rope_out = rope_in * cos + rotated * sin ----
        cos_mul_name = f"{name}/cos/Mul"
        self.make_mul(cos_mul_name,
                      [f"{rope_in_name}/output_0", f"{cos_u_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", num_heads, "sequence_length", rope_dim])

        sin_mul_name = f"{name}/sin/Mul"
        if neg_sin:
            # conjugate rotation: rope * cos - rotated * sin
            neg_sin_u_name = f"{name}/sin/Neg"
            self.make_neg(
                neg_sin_u_name,
                f"{sin_u_name}/output_0",
                self.io_dtype,
                ["batch_size", 1, "sequence_length", rope_dim],
            )
            self.make_mul(sin_mul_name,
                          [f"{rotated_name}/output_0", f"{neg_sin_u_name}/output_0"],
                          self.io_dtype,
                          ["batch_size", num_heads, "sequence_length", rope_dim])
        else:
            self.make_mul(sin_mul_name,
                          [f"{rotated_name}/output_0", f"{sin_u_name}/output_0"],
                          self.io_dtype,
                          ["batch_size", num_heads, "sequence_length", rope_dim])

        rope_out_name = f"{name}/rope_out/Add"
        self.make_add(rope_out_name,
                      [f"{cos_mul_name}/output_0", f"{sin_mul_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", num_heads, "sequence_length", rope_dim])

        # ---- Concat [nope, rope_out] → full head ----
        final_concat_name = f"{name}/final/Concat"
        self.make_concat(
            final_concat_name,
            [f"{nope_name}/output_0", f"{rope_out_name}/output_0"],
            self.io_dtype,
            ["batch_size", num_heads, "sequence_length", head_dim],
            axis=3,
        )
        return f"{final_concat_name}/output_0"

    # ------------------------------------------------------------------ #
    # HyperConnection
    # ------------------------------------------------------------------ #

    def make_hyper_connection_decomposed(
        self,
        layer_id: int,
        which: str,        # "attn" or "ffn"
        hc_module,         # DeepseekV4HyperConnection instance
        hc_streams: str,   # [B, S, hc_mult, D]
    ) -> tuple[str, str, str]:
        """Emit the mHC mapping subgraph.

        Returns ``(post_name, comb_name, collapsed_name)`` where:
          * ``post_name``      → [B, S, hc_mult]   (scale for sublayer output)
          * ``comb_name``      → [B, S, hc_mult, hc_mult]  (stream mixer)
          * ``collapsed_name`` → [B, S, D]           (input to sublayer)
        """
        base = f"/model/layers.{layer_id}/{which}_hc"
        hc = self.hc_mult
        d = self.hidden_size
        mix_out_dim = (2 + hc) * hc   # pre+post+comb outputs

        # 1. Flatten streams: [B, S, hc_mult, D] → [B, S, hc_mult * D]
        flat_name = f"{base}/flatten/Reshape"
        self.make_reshape(
            flat_name,
            [hc_streams, f"/model/constants/INT64/[0, 0, {hc * d}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", hc * d],
        )

        # 2. UnweightedRMSNorm (in fp32) on the flattened streams
        normed_name = self.make_unweighted_rms_norm(
            f"{base}/InputNorm",
            f"{flat_name}/output_0",
            self.hc_eps,
            ["batch_size", "sequence_length", hc * d],
        )

        # 3. Linear: [B, S, hc*D] @ fn.T → [B, S, (2+hc)*hc]
        fn_weight = f"model.layers.{layer_id}.{which}_hc.fn"
        # fn has shape [mix_out_dim, hc * D]; stored transposed for MatMul
        self.make_initializer(hc_module.fn.data.T.float(), fn_weight, to=ir.DataType.FLOAT)

        fn_matmul_name = f"{base}/fn/MatMul"
        fn_matmul_output = f"{fn_matmul_name}/output_0"
        self.make_node("MatMul", inputs=[normed_name, fn_weight], outputs=[fn_matmul_output], name=fn_matmul_name)
        self.make_value(fn_matmul_output, ir.DataType.FLOAT, shape=["batch_size", "sequence_length", mix_out_dim])

        # 4. Split into (pre_w, post_w, comb_w)
        pre_w_name = f"{base}/PreW"
        post_w_name = f"{base}/PostW"
        comb_w_name = f"{base}/CombW"
        self.make_split(
            f"{base}/Split",
            inputs=[f"{fn_matmul_name}/output_0",
                    f"/model/constants/INT64/[{hc}, {hc}, {hc * hc}]"],
            outputs=[f"{pre_w_name}/output_0", f"{post_w_name}/output_0", f"{comb_w_name}/output_0"],
            dtypes=[ir.DataType.FLOAT] * 3,
            shapes=[
                ["batch_size", "sequence_length", hc],
                ["batch_size", "sequence_length", hc],
                ["batch_size", "sequence_length", hc * hc],
            ],
            axis=-1,
        )

        # 5. Apply bias + scale
        base_init = hc_module.base.data.float()         # [mix_out_dim]
        scale_init = hc_module.scale.data.float()       # [3]

        pre_b_name = f"model.layers.{layer_id}.{which}_hc.pre_base"
        post_b_name = f"model.layers.{layer_id}.{which}_hc.post_base"
        comb_b_name = f"model.layers.{layer_id}.{which}_hc.comb_base"
        pre_scale_name = f"model.layers.{layer_id}.{which}_hc.pre_scale"
        post_scale_name = f"model.layers.{layer_id}.{which}_hc.post_scale"
        comb_scale_name = f"model.layers.{layer_id}.{which}_hc.comb_scale"

        self.make_initializer(base_init[:hc], pre_b_name, to=ir.DataType.FLOAT)
        self.make_initializer(base_init[hc:2 * hc], post_b_name, to=ir.DataType.FLOAT)
        self.make_initializer(base_init[2 * hc:].reshape(hc, hc), comb_b_name, to=ir.DataType.FLOAT)
        self.make_initializer(scale_init[0:1], pre_scale_name, to=ir.DataType.FLOAT)
        self.make_initializer(scale_init[1:2], post_scale_name, to=ir.DataType.FLOAT)
        self.make_initializer(scale_init[2:3], comb_scale_name, to=ir.DataType.FLOAT)

        # pre_logits = pre_w * pre_scale + pre_b
        pre_scaled_name = f"{base}/pre_logits/Mul"
        self.make_mul(pre_scaled_name,
                      [f"{pre_w_name}/output_0", pre_scale_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])
        pre_logits_name = f"{base}/pre_logits/Add"
        self.make_add(pre_logits_name,
                      [f"{pre_scaled_name}/output_0", pre_b_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # post_logits = post_w * post_scale + post_b
        post_scaled_name = f"{base}/post_logits/Mul"
        self.make_mul(post_scaled_name,
                      [f"{post_w_name}/output_0", post_scale_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])
        post_logits_name = f"{base}/post_logits/Add"
        self.make_add(post_logits_name,
                      [f"{post_scaled_name}/output_0", post_b_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # comb_logits = reshape(comb_w) * comb_scale + comb_b
        comb_w_4d_name = f"{base}/comb_logits/Reshape"
        self.make_reshape(
            comb_w_4d_name,
            [f"{comb_w_name}/output_0", f"/model/constants/INT64/[0, 0, {hc}, {hc}]"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", hc, hc],
        )
        comb_scaled_name = f"{base}/comb_logits/Mul"
        self.make_mul(comb_scaled_name,
                      [f"{comb_w_4d_name}/output_0", comb_scale_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, hc])
        comb_logits_name = f"{base}/comb_logits/Add"
        self.make_add(comb_logits_name,
                      [f"{comb_scaled_name}/output_0", comb_b_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, hc])

        # 6. Compute pre: sigmoid(pre_logits) + eps
        pre_sig_name = f"{base}/pre/Sigmoid"
        self.make_sigmoid(pre_sig_name, f"{pre_logits_name}/output_0",
                          ir.DataType.FLOAT,
                          ["batch_size", "sequence_length", hc])
        pre_eps_name = f"model.layers.{layer_id}.{which}_hc.pre_eps"
        self.make_initializer(torch.tensor(self.hc_eps, dtype=torch.float32), pre_eps_name, to=ir.DataType.FLOAT)
        pre_name = f"{base}/pre/Add"
        self.make_add(pre_name,
                      [f"{pre_sig_name}/output_0", pre_eps_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # 7. Compute post: 2 * sigmoid(post_logits)
        post_sig_name = f"{base}/post/Sigmoid"
        self.make_sigmoid(post_sig_name, f"{post_logits_name}/output_0",
                          ir.DataType.FLOAT,
                          ["batch_size", "sequence_length", hc])
        two_const_name = f"model.layers.{layer_id}.{which}_hc.post_two"
        self.make_initializer(torch.tensor(2.0, dtype=torch.float32), two_const_name, to=ir.DataType.FLOAT)
        post_name = f"{base}/post/Mul"
        self.make_mul(post_name,
                      [f"{post_sig_name}/output_0", two_const_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # 8. Compute comb via Sinkhorn-Knopp (unrolled hc_sinkhorn_iters iterations)
        # Start: softmax(comb_logits, dim=-1) + eps
        comb_sfmx_name = f"{base}/comb/Softmax"
        self.make_softmax(comb_sfmx_name, f"{comb_logits_name}/output_0",
                          ir.DataType.FLOAT,
                          ["batch_size", "sequence_length", hc, hc],
                          axis=-1)
        comb_eps_name = f"model.layers.{layer_id}.{which}_hc.comb_eps"
        self.make_initializer(torch.tensor(self.hc_eps, dtype=torch.float32), comb_eps_name, to=ir.DataType.FLOAT)
        comb_cur_name = f"{base}/comb_init/Add"
        self.make_add(comb_cur_name,
                      [f"{comb_sfmx_name}/output_0", comb_eps_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, hc])
        comb_cur = f"{comb_cur_name}/output_0"

        # First col-normalize (before the loop alternation)
        comb_cur = self.sinkhorn_col_normalize(base, "init", comb_cur, hc)

        for it in range(self.hc_sinkhorn_iters - 1):
            comb_cur = self.sinkhorn_row_normalize(base, f"it{it}", comb_cur, hc)
            comb_cur = self.sinkhorn_col_normalize(base, f"it{it}", comb_cur, hc)

        comb_name = comb_cur   # [B, S, hc, hc]

        # 9. Cast post and comb back to IO dtype
        post_cast_name = f"{base}/post/Cast"
        self.make_cast(post_cast_name, f"{post_name}/output_0", self.io_dtype,
                       ["batch_size", "sequence_length", hc])

        comb_cast_name = f"{base}/comb/Cast"
        self.make_cast(comb_cast_name, comb_name, self.io_dtype,
                       ["batch_size", "sequence_length", hc, hc])

        # 10. Compute collapsed = (pre * hidden_streams).sum(dim=2)
        # pre: [B, S, hc]; hidden_streams: [B, S, hc, D]
        # pre.unsqueeze(-1): [B, S, hc, 1]
        pre_u_name = f"{base}/pre/Unsqueeze"
        self.make_unsqueeze(
            pre_u_name,
            [f"{pre_name}/output_0", "/model/constants/INT64/[3]"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", hc, 1],
        )

        # Cast hidden_streams to fp32 for the weighted sum
        hc_float_name = f"{base}/hc_streams/Cast"
        self.make_cast(hc_float_name, hc_streams, ir.DataType.FLOAT,
                       ["batch_size", "sequence_length", hc, d])

        weighted_name = f"{base}/weighted/Mul"
        self.make_mul(weighted_name,
                      [f"{pre_u_name}/output_0", f"{hc_float_name}/output_0"],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, d])

        # sum over axis 2 (hc_mult dimension)
        collapsed_fp32_name = f"{base}/collapsed/ReduceSum"
        self.make_reduce_sum(collapsed_fp32_name,
                             [f"{weighted_name}/output_0",
                              "/model/constants/INT64/[2]"],
                             ir.DataType.FLOAT,
                             ["batch_size", "sequence_length", d],
                             keepdims=False)

        collapsed_cast_name = f"{base}/collapsed/Cast"
        self.make_cast(collapsed_cast_name, f"{collapsed_fp32_name}/output_0",
                       self.io_dtype, ["batch_size", "sequence_length", d])

        return (
            f"{post_cast_name}/output_0",   # post [B, S, hc]
            f"{comb_cast_name}/output_0",   # comb [B, S, hc, hc]
            f"{collapsed_cast_name}/output_0",  # collapsed [B, S, D]
        )

    def sinkhorn_row_normalize(self, base: str, tag: str, comb: str, hc: int) -> str:
        """Divide comb by its row sum: comb / comb.sum(dim=-1, keepdim=True)."""
        row_sum_name = f"{base}/row_{tag}/ReduceSum"
        self.make_reduce_sum(row_sum_name, [comb, "/model/constants/INT64/[-1]"],
                             ir.DataType.FLOAT,
                             ["batch_size", "sequence_length", hc, 1], keepdims=True)
        eps_name = f"{base}/row_{tag}/Add"
        self.make_add(eps_name,
                      [f"{row_sum_name}/output_0",
                       f"model.layers._shared.hc_eps"],   # shared eps const (created once)
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, 1])
        out_name = f"{base}/row_{tag}/Div"
        self.make_div(out_name, [comb, f"{eps_name}/output_0"],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, hc])
        return f"{out_name}/output_0"

    def sinkhorn_col_normalize(self, base: str, tag: str, comb: str, hc: int) -> str:
        """Divide comb by its column sum: comb / comb.sum(dim=-2, keepdim=True)."""
        col_sum_name = f"{base}/col_{tag}/ReduceSum"
        self.make_reduce_sum(col_sum_name, [comb, "/model/constants/INT64/[-2]"],
                             ir.DataType.FLOAT,
                             ["batch_size", "sequence_length", 1, hc], keepdims=True)
        eps_name = f"{base}/col_{tag}/Add"
        self.make_add(eps_name,
                      [f"{col_sum_name}/output_0",
                       f"model.layers._shared.hc_eps"],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", 1, hc])
        out_name = f"{base}/col_{tag}/Div"
        self.make_div(out_name, [comb, f"{eps_name}/output_0"],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, hc])
        return f"{out_name}/output_0"

    def ensure_shared_hc_eps(self):
        """Create the shared HC eps initializer (once)."""
        name = "model.layers._shared.hc_eps"
        if name not in self.node_names:
            self.make_initializer(
                torch.tensor(self.hc_eps, dtype=torch.float32),
                name,
                to=ir.DataType.FLOAT,
            )
            self.node_names.add(name)

    def make_hyper_connection(
        self,
        layer_id: int,
        which: str,
        hc_module,
        hc_streams: str,
    ) -> tuple[str, str, str]:
        base = f"/model/layers.{layer_id}/{which}_hc/HyperConnection"
        weight_name = f"model.layers.{layer_id}.{which}_hc.fn"
        bias_name = f"model.layers.{layer_id}.{which}_hc.base"
        scale_name = f"model.layers.{layer_id}.{which}_hc.scale"
        self.make_initializer(hc_module.fn.data.float(), weight_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_module.base.data.float(), bias_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_module.scale.data.float(), scale_name, to=ir.DataType.FLOAT)
        outputs = [f"{base}/output_{index}" for index in range(3)]
        self.make_node(
            "HyperConnection",
            [hc_streams, weight_name, bias_name, scale_name],
            outputs,
            name=base,
            domain="com.microsoft",
            epsilon=self.hc_eps,
            sinkhorn_iterations=self.hc_sinkhorn_iters,
        )
        self.make_value(outputs[0], self.io_dtype, ["batch_size", "sequence_length", self.hc_mult])
        self.make_value(outputs[1], self.io_dtype, ["batch_size", "sequence_length", self.hc_mult, self.hc_mult])
        self.make_value(outputs[2], self.io_dtype, ["batch_size", "sequence_length", self.hidden_size])
        return outputs[0], outputs[1], outputs[2]

    def make_hyper_connection_mix(
        self,
        layer_id: int,
        which: str,
        hc_module,
        layernorm,
        sublayer_out: str,
        hc_streams: str,
        post: str,
        comb: str,
    ) -> tuple[str, str, str, str]:
        base = f"/model/layers.{layer_id}/{which}_hc/HyperConnectionMix"
        weight_name = f"model.layers.{layer_id}.{which}_hc.fn"
        scale_name = f"model.layers.{layer_id}.{which}_hc.scale"
        bias_name = f"model.layers.{layer_id}.{which}_hc.base"
        norm_weight_name = f"model.layers.{layer_id}.{which}_hc.norm_weight"
        self.make_initializer(hc_module.fn.data.T.contiguous().float(), weight_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_module.scale.data.float(), scale_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_module.base.data.float(), bias_name, to=ir.DataType.FLOAT)
        self.make_initializer(layernorm.weight.data.float(), norm_weight_name, to=ir.DataType.FLOAT)
        outputs = [f"{base}/output_{index}" for index in range(4)]
        self.make_node(
            "HyperConnectionMix",
            [sublayer_out, hc_streams, post, comb, weight_name, scale_name, bias_name, norm_weight_name],
            outputs,
            name=base,
            domain="com.microsoft",
            sinkhorn_iterations=self.hc_sinkhorn_iters,
            epsilon=self.rms_norm_epsilon,
            hc_epsilon=self.hc_eps,
            sinkhorn_epsilon=self.hc_eps,
            post_alpha=2.0,
        )
        self.make_value(outputs[0], self.io_dtype, ["batch_size", "sequence_length", self.hc_mult, self.hidden_size])
        self.make_value(outputs[1], ir.DataType.FLOAT, ["batch_size", "sequence_length", self.hc_mult])
        self.make_value(
            outputs[2],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", self.hc_mult, self.hc_mult],
        )
        self.make_value(outputs[3], self.io_dtype, ["batch_size", "sequence_length", self.hidden_size])
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def cast_hyper_mix_state(self, name: str, shape: list, to: ir.DataType) -> str:
        if self.io_dtype == ir.DataType.FLOAT:
            return name
        cast_name = f"{name}/to_{self.to_str_dtype(to).lower()}/Cast"
        self.make_cast(cast_name, name, to, shape)
        return f"{cast_name}/output_0"

    def make_hc_combine(
        self,
        layer_id: int,
        which: str,
        post: str,       # [B, S, hc_mult]
        comb: str,       # [B, S, hc_mult, hc_mult]
        hc_streams: str, # [B, S, hc_mult, D]
        sublayer_out: str,  # [B, S, D]
    ) -> str:
        """Compute the HC residual update.

        new_hidden = post[...,None] * sublayer_out[...,None,:] + comb.T @ hidden_streams

        Returns name of [B, S, hc_mult, D] tensor.
        """
        base = f"/model/layers.{layer_id}/{which}_hc_combine"
        hc = self.hc_mult
        d = self.hidden_size

        # post.unsqueeze(-1): [B, S, hc, 1]
        post_u_name = f"{base}/post/Unsqueeze"
        self.make_unsqueeze(
            post_u_name,
            [post, "/model/constants/INT64/[-1]"],
            self.io_dtype,
            ["batch_size", "sequence_length", hc, 1],
        )

        # sublayer_out.unsqueeze(-2): [B, S, 1, D]
        sub_u_name = f"{base}/sub/Unsqueeze"
        self.make_unsqueeze(
            sub_u_name,
            [sublayer_out, "/model/constants/INT64/[-2]"],
            self.io_dtype,
            ["batch_size", "sequence_length", 1, d],
        )

        # post_u * sub_u → [B, S, hc, D]
        scaled_sub_name = f"{base}/scaled_sub/Mul"
        self.make_mul(scaled_sub_name,
                      [f"{post_u_name}/output_0", f"{sub_u_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", "sequence_length", hc, d])

        # comb.transpose(-1,-2): [B, S, hc, hc] → [B, S, hc, hc] (transpose last 2 dims)
        comb_t_name = f"{base}/comb/Transpose"
        self.make_transpose(
            comb_t_name, comb, self.io_dtype,
            ["batch_size", "sequence_length", hc, hc],
            perm=[0, 1, 3, 2],
        )

        # matmul(comb_t, hidden_streams): [B, S, hc, hc] @ [B, S, hc, D] = [B, S, hc, D]
        mixed_name = f"{base}/mixed/MatMul"
        mixed_output = f"{mixed_name}/output_0"
        self.make_node("MatMul", inputs=[f"{comb_t_name}/output_0", hc_streams], outputs=[mixed_output], name=mixed_name)
        self.make_value(mixed_output, self.io_dtype, shape=["batch_size", "sequence_length", hc, d])

        # new_hidden = scaled_sub + mixed
        out_name = f"{base}/new_streams/Add"
        self.make_add(out_name,
                      [f"{scaled_sub_name}/output_0", f"{mixed_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", "sequence_length", hc, d])
        return f"{out_name}/output_0"

    # ------------------------------------------------------------------ #
    # Attention subgraph
    # ------------------------------------------------------------------ #

    def make_deepseek_attention(
        self,
        layer_id: int,
        attn,
        collapsed: str,  # [B, S, D] input after input_layernorm
    ) -> str:
        """Build the full DeepSeek V4 attention subgraph.

        Returns the attention output [B, S, D].
        """
        base = f"/model/layers.{layer_id}/attn"
        H = self.num_attn_heads  # 64
        D = self.hidden_size     # 4096
        head_dim = self.head_size  # 512
        q_lora = self.q_lora_rank  # 1024
        rope_dim = self.qk_rope_head_dim  # 64
        nope_dim = self.qk_nope_head_dim  # 448

        # ---------------------------------------------------------------- #
        # Q path: q_a_proj → q_a_norm → q_b_proj → q_b_norm → RoPE
        # ---------------------------------------------------------------- #

        # q_a_proj: Linear(D, q_lora)
        q_a_name = self.make_matmul(attn.q_a_proj, f"{base}/q_a_proj/MatMul", collapsed)

        # q_a_norm: RMSNorm with weight (standard SimplifiedLayerNorm subgraph)
        q_a_norm_output = self.make_layernorm_no_skip(layer_id, attn.q_a_norm, f"{q_a_name}/output_0", "q_a")

        # q_b_proj: Linear(q_lora, H * head_dim)
        q_b_name = self.make_matmul(attn.q_b_proj, f"{base}/q_b_proj/MatMul",
                                    q_a_norm_output)

        # Reshape for per-head q_b_norm: [B, S, H*head_dim] → [B, S, H, head_dim]
        q_b_4d_name = f"{base}/q_b/Reshape"
        self.make_reshape(
            q_b_4d_name,
            [f"{q_b_name}/output_0", f"/model/constants/INT64/[0, 0, {H}, {head_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", H, head_dim],
        )

        # q_b_norm: UnweightedRMSNorm applied per head
        q_b_normed_name = self.make_unweighted_rms_norm(
            f"{base}/q_b_norm",
            f"{q_b_4d_name}/output_0",
            self.layernorm_attrs["epsilon"],
            ["batch_size", "sequence_length", H, head_dim],
        )

        # Transpose to [B, H, S, head_dim] for RoPE
        q_t_name = f"{base}/q/Transpose"
        self.make_transpose(
            q_t_name, q_b_normed_name, self.io_dtype,
            ["batch_size", H, "sequence_length", head_dim],
            perm=[0, 2, 1, 3],
        )

        # Apply RoPE (interleaved, trailing rope_dim channels)
        q_rope_name = self.make_deepseek_rope(
            f"{base}/q_rope", f"{q_t_name}/output_0",
            H, head_dim, rope_dim,
        )
        # q_rope_name: [B, H, S, head_dim]

        # ---------------------------------------------------------------- #
        # KV path: kv_proj → kv_norm → RoPE
        # ---------------------------------------------------------------- #

        kv_proj_name = self.make_matmul(attn.kv_proj, f"{base}/kv_proj/MatMul", collapsed)

        kv_norm_output = self.make_layernorm_no_skip(layer_id, attn.kv_norm, f"{kv_proj_name}/output_0", "kv")

        # Unsqueeze kv from [B, S, head_dim] → [B, 1, S, head_dim] for RoPE + cache
        kv_u_name = f"{base}/kv/Unsqueeze"
        self.make_unsqueeze(
            kv_u_name,
            [kv_norm_output, "/model/constants/INT64/[1]"],
            self.io_dtype,
            ["batch_size", 1, "sequence_length", head_dim],
        )

        # Apply RoPE to kv (1 KV head)
        kv_rope_name = self.make_deepseek_rope(
            f"{base}/kv_rope", f"{kv_u_name}/output_0",
            1, head_dim, rope_dim,
        )
        # kv_rope_name: [B, 1, S, head_dim]

        # ---------------------------------------------------------------- #
        # KV cache: fixed local ring owned by CompressedAttention
        # ---------------------------------------------------------------- #
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)

        compressed_kv, block_bias, selected_indices = self.make_compressor(layer_id, attn, collapsed, q_a_norm_output)
        attention_bias = self.make_compressed_attention_bias(layer_id, compressed_kv, block_bias)

        sinks_init_name = f"model.layers.{layer_id}.self_attn.sinks"
        self.make_initializer(attn.sinks.data, sinks_init_name, to=self.io_dtype)
        attn_out_output = self.make_compressed_attention(
            layer_id,
            q_rope_name,
            kv_rope_name,
            compressed_kv,
            attention_bias,
            selected_indices,
            sinks_init_name,
            past_k,
            present_k,
        )

        self.make_identity(
            f"{base}/present_v/Identity",
            present_k,
            self.io_dtype,
            ["batch_size", 1, self.local_cache_capacity, head_dim],
            output_name=present_v,
        )

        # Transpose to [B, S, H, head_dim]
        attn_t_name = f"{base}/attn/Transpose"
        self.make_transpose(
            attn_t_name,
            attn_out_output,
            self.io_dtype,
            ["batch_size", "sequence_length", H, head_dim],
            perm=[0, 2, 1, 3],
        )

        # ---------------------------------------------------------------- #
        # Conjugate de-rotation on the attention output
        # ---------------------------------------------------------------- #
        # apply_rotary_pos_emb(attn_output.T(1,2), cos, -sin).T(1,2)
        # i.e., transpose to [B, H, S, head_dim], apply neg-sin RoPE, transpose back

        attn_t2_name = f"{base}/attn_t2/Transpose"
        self.make_transpose(
            attn_t2_name, f"{attn_t_name}/output_0", self.io_dtype,
            ["batch_size", H, "sequence_length", head_dim],
            perm=[0, 2, 1, 3],
        )
        attn_derope_name = self.make_deepseek_rope(
            f"{base}/out_rope",
            f"{attn_t2_name}/output_0",
            H, head_dim, rope_dim,
            neg_sin=True,
        )
        # Transpose back to [B, S, H, head_dim]
        attn_final_t_name = f"{base}/attn_final/Transpose"
        self.make_transpose(
            attn_final_t_name, attn_derope_name, self.io_dtype,
            ["batch_size", "sequence_length", H, head_dim],
            perm=[0, 2, 1, 3],
        )

        # ---------------------------------------------------------------- #
        # Grouped output projection
        # ---------------------------------------------------------------- #
        # grouped = attn_output.reshape(B, S, o_groups, H*head_dim//o_groups)
        in_per_group = H * head_dim // self.o_groups  # 64*512//8 = 4096
        grouped_reshape_name = f"{base}/grouped/Reshape"
        self.make_reshape(
            grouped_reshape_name,
            [f"{attn_final_t_name}/output_0",
             f"/model/constants/INT64/[0, 0, {self.o_groups}, {in_per_group}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.o_groups, in_per_group],
        )

        # o_a_proj (GroupedLinear): [B, S, o_groups, in_per_group] → [B, S, o_groups, o_lora_rank]
        o_a_out = self.make_grouped_linear(
            layer_id,
            attn.o_a_proj,
            f"{grouped_reshape_name}/output_0",
            in_per_group,
            self.o_lora_rank,
            self.o_groups,
        )

        # Flatten: [B, S, o_groups, o_lora_rank] → [B, S, o_groups*o_lora_rank]
        flat_name = f"{base}/oa_flat/Reshape"
        self.make_reshape(
            flat_name,
            [o_a_out,
             f"/model/constants/INT64/[0, 0, {self.o_groups * self.o_lora_rank}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.o_groups * self.o_lora_rank],
        )

        # o_b_proj: Linear(o_groups * o_lora_rank, hidden_size)
        o_b_name = self.make_matmul(attn.o_b_proj, f"{base}/o_b_proj/MatMul",
                                    f"{flat_name}/output_0")
        return f"{o_b_name}/output_0"

    def make_grouped_linear(
        self,
        layer_id: int,
        o_a_proj,   # DeepseekV4GroupedLinear instance
        root_input: str,  # [B, S, n_groups, in_per_group]
        in_per_group: int,
        out_per_group: int,
        n_groups: int,
    ) -> str:
        """Batched matmul implementation of DeepseekV4GroupedLinear.

        weight has shape [n_groups * out_per_group, in_per_group] in PyTorch.
        ONNX plan:
          w → [n_groups, in_per_group, out_per_group]      (reshape + transpose)
          x → [B*S, n_groups, in_per_group]                (reshape)
              → [n_groups, B*S, in_per_group]              (transpose)
          y = BatchedMatMul(x, w)  [n_groups, B*S, out_per_group]
              → [B*S, n_groups, out_per_group]             (transpose)
              → [B, S, n_groups, out_per_group]            (reshape)
        """
        base = f"/model/layers.{layer_id}/attn/o_a_proj"

        # Weight initializer: store as [n_groups, in_per_group, out_per_group]
        w_name = f"model.layers.{layer_id}.self_attn.o_a_proj.weight"
        # PyTorch weight: [n_groups * out_per_group, in_per_group]
        # reshape to [n_groups, out_per_group, in_per_group], then transpose 1,2
        w_data = o_a_proj.weight.data.view(n_groups, out_per_group, in_per_group).transpose(1, 2)
        # w_data: [n_groups, in_per_group, out_per_group]
        self.make_initializer(w_data, w_name, to=self.io_dtype)

        # Reshape weight in graph: it's stored with the right shape already
        # No reshape needed - initializer shape = [n_groups, in_per_group, out_per_group]

        # x: [B, S, n_groups, in_per_group] → [B*S, n_groups, in_per_group]
        x_flat_name = f"{base}/x_flat/Reshape"
        self.make_reshape(
            x_flat_name,
            [root_input, f"/model/constants/INT64/[-1, {n_groups}, {in_per_group}]"],
            self.io_dtype,
            [None, n_groups, in_per_group],
        )

        # Transpose: [B*S, n_groups, in_per_group] → [n_groups, B*S, in_per_group]
        x_t_name = f"{base}/x/Transpose"
        self.make_transpose(
            x_t_name, f"{x_flat_name}/output_0", self.io_dtype,
            [n_groups, None, in_per_group],
            perm=[1, 0, 2],
        )

        # BatchMatMul: [n_groups, B*S, in_per_group] @ [n_groups, in_per_group, out_per_group]
        y_t_name = f"{base}/y_t/MatMul"
        y_t_output = f"{y_t_name}/output_0"
        self.make_node("MatMul", inputs=[f"{x_t_name}/output_0", w_name], outputs=[y_t_output], name=y_t_name)
        self.make_value(y_t_output, self.io_dtype, shape=[n_groups, None, out_per_group])

        # Transpose back: [n_groups, B*S, out_per_group] → [B*S, n_groups, out_per_group]
        y_name = f"{base}/y/Transpose"
        self.make_transpose(
            y_name, f"{y_t_name}/output_0", self.io_dtype,
            [None, n_groups, out_per_group],
            perm=[1, 0, 2],
        )

        # Reshape: [B*S, n_groups, out_per_group] → [B, S, n_groups, out_per_group]
        y_out_name = f"{base}/y_out/Reshape"
        self.make_reshape(
            y_out_name,
            [f"{y_name}/output_0",
             f"/model/constants/INT64/[0, 0, {n_groups}, {out_per_group}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", n_groups, out_per_group],
        )
        return f"{y_out_name}/output_0"

    # ------------------------------------------------------------------ #
    # MoE / FFN subgraph
    # ------------------------------------------------------------------ #

    def make_deepseek_moe(
        self,
        layer_id: int,
        mlp,
        collapsed: str,  # [B, S, D]
    ) -> str:
        """Build MoE subgraph (hash-routing or top-k routing + shared expert).

        Returns output [B, S, D].
        """
        base = f"/model/layers.{layer_id}/moe"
        is_hash = getattr(mlp, "is_hash", False) or (
            self.mlp_layer_types[layer_id] == "hash_moe"
        )
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        # ---- Router ----
        if is_hash:
            # Hash router: routing based on input_ids lookup (tid2eid buffer)
            moe_out = self.make_hash_moe(layer_id, mlp, collapsed, op_type, moe_weight_type)
        else:
            # Learned top-k router
            moe_out = self.make_topk_moe(layer_id, mlp, collapsed, op_type, moe_weight_type)

        # ---- Shared expert (always-active SwiGLU-MLP) ----
        shared_out = self.make_shared_expert(layer_id, mlp.shared_experts, collapsed)

        # ---- Combine routed + shared ----
        combine_name = f"{base}/combine/Add"
        self.make_add(combine_name,
                      [moe_out, shared_out],
                      self.io_dtype,
                      ["batch_size", "sequence_length", self.hidden_size])
        return f"{combine_name}/output_0"

    def make_raw_matmul(
        self, weight_tensor: "torch.Tensor", init_name: str, node_name: str, root_input: str
    ) -> str:
        """Emit a single MatMul from a raw weight tensor (not an nn.Linear).

        Stores weight.T as an initializer ``init_name`` (cast to io_dtype) and
        emits a MatMul node ``node_name``.  Returns the node's basename so that
        callers can access ``{node_name}/output_0``.
        """
        raw_linear = SimpleNamespace(weight=weight_tensor)
        self.make_matmul(raw_linear, node_name, root_input)
        return node_name

    def make_moe_router_scores(
        self, layer_id: int, gate_weight: "torch.Tensor", root_input: str, base: str
    ) -> str:
        """Compute the router scores shared by both router types.

        This is ``DeepseekV4{TopK,Hash}Router.score_fn(F.linear(hidden_states, gate.weight))``:
        a raw (bias-free) Linear followed by ``config.scoring_func``.  Returns the flattened
        ``[num_rows, num_experts]`` scores value name.
        """
        num_e = self.moe_attrs["num_experts"]

        # gate.weight is a raw nn.Parameter, not an nn.Linear, so use make_raw_matmul.
        logits_name = self.make_raw_matmul(
            gate_weight,
            f"model.layers.{layer_id}.moe.gate.weight",
            f"{base}/router/MatMul",
            root_input,
        )
        logits_shape = ["batch_size", "sequence_length", num_e]
        if self.scoring_func == "sigmoid":
            scores_name = f"{base}/router/Sigmoid"
            self.make_sigmoid(scores_name, f"{logits_name}/output_0", self.io_dtype, logits_shape)
        elif self.scoring_func == "sqrtsoftplus":
            # Default: sqrt(softplus(x)).  Always strictly positive, which
            # make_moe_masked_router_logits below relies on to take its log.
            softplus_name = f"{base}/router/Softplus"
            self.make_softplus(softplus_name, f"{logits_name}/output_0", self.io_dtype, logits_shape)
            scores_name = f"{base}/router/Sqrt"
            self.make_sqrt(scores_name, [f"{softplus_name}/output_0"], self.io_dtype, logits_shape)
        else:
            # `ACT2FN` in the HuggingFace reference only registers "sigmoid" and
            # "sqrtsoftplus" for this purpose (`transformers/activations.py`); any other
            # `config.scoring_func` value would raise a `KeyError` there too, so fail
            # loudly here rather than silently emitting a graph with different numerics.
            raise NotImplementedError(
                f"Unsupported DeepSeek V4 scoring_func '{self.scoring_func}'; expected "
                "'sigmoid' or 'sqrtsoftplus'."
            )

        # Reshape: [B, S, E] → [B*S, E]
        flat_name = f"{base}/router/Reshape"
        self.make_reshape(
            flat_name,
            [f"{scores_name}/output_0", f"/model/constants/INT64/[-1, {num_e}]"],
            self.io_dtype,
            [None, num_e],
        )
        return f"{flat_name}/output_0"

    def make_moe_masked_router_logits(self, base: str, scores_flat: str, indices: str, top_k: int) -> str:
        """Build the ``[num_rows, num_experts]`` router_probs input fed to the MoE op.

        The reference computes ``weights = scores.gather(indices); weights /= weights.sum()``
        (optionally scaled afterwards) — a plain renormalization of the *selected* scores, with
        no softmax over the full expert set. The ``com.microsoft::MoE`` op instead always
        applies its own softmax over all experts before selecting the top-k and (optionally)
        renormalizing them. To reproduce the reference exactly through that op, this places
        ``log(selected_score)`` at each selected column and a large negative fill everywhere
        else: softmax normalizes any non-selected (~0-weight) columns away, so with
        ``normalize_routing_weights=1`` the op's top-k selection lands on exactly these
        columns and its output weights reduce to ``selected_score / sum(selected_scores)``.
        """
        num_e = self.moe_attrs["num_experts"]

        selected_name = f"{base}/router/GatherElements"
        self.make_gather_elements(
            selected_name, [scores_flat, indices], self.io_dtype, [None, top_k], axis=-1
        )
        log_name = f"{base}/router/Log"
        self.make_log(log_name, f"{selected_name}/output_0", self.io_dtype, [None, top_k])

        # A [num_rows, num_experts] tensor filled with a large negative value (effectively -inf
        # after softmax), the same shape as scores_flat.
        shape_name = f"{base}/router/Shape"
        self.make_shape(shape_name, scores_flat, shape=[2])
        fill_name = f"{base}/router/NegFill"
        self.make_expand(
            fill_name,
            [f"/model/constants/{self.to_str_dtype(self.io_dtype)}/-10000.0", f"{shape_name}/output_0"],
            self.io_dtype,
            [None, num_e],
        )

        masked_name = f"{base}/router/ScatterElements"
        self.make_scatter_elements(
            masked_name,
            [f"{fill_name}/output_0", indices, f"{log_name}/output_0"],
            self.io_dtype,
            [None, num_e],
            axis=-1,
        )
        return f"{masked_name}/output_0"

    def make_topk_moe(
        self, layer_id: int, mlp, root_input: str, op_type: str, weight_type: str
    ) -> str:
        """Build the routed MoE block with learned top-k routing."""
        base = f"/model/layers.{layer_id}/moe"
        num_e = self.moe_attrs["num_experts"]
        top_k = self.moe_attrs["top_k"]

        scores_flat = self.make_moe_router_scores(layer_id, mlp.gate.weight.data, root_input, base)

        # e_score_correction_bias (trained buffer) only perturbs which experts are
        # *selected*; the routing weight itself is still the unbiased score (see
        # DeepseekV4TopKRouter.forward: `weights = scores.gather(1, indices)`, not
        # `(scores + bias).gather(...)`).
        bias_name = f"model.layers.{layer_id}.moe.gate.e_score_correction_bias"
        self.make_initializer(mlp.gate.e_score_correction_bias.data, bias_name, to=self.io_dtype)
        biased_name = f"{base}/router/BiasAdd"
        self.make_add(biased_name, [scores_flat, bias_name], self.io_dtype, [None, num_e])

        _, indices_name = self.make_topk(
            f"{base}/router/TopK",
            [f"{biased_name}/output_0", f"/model/constants/INT64/[{top_k}]"],
            self.io_dtype,
            [None, top_k],
            [None, top_k],
            axis=-1,
            largest=1,
            sorted=0,
        )

        router_probs = self.make_moe_masked_router_logits(base, scores_flat, indices_name, top_k)
        return self.make_deepseek_moe_op(layer_id, mlp, root_input, router_probs, op_type, weight_type, base)

    def make_hash_moe(
        self, layer_id: int, mlp, root_input: str, op_type: str, weight_type: str
    ) -> str:
        """Build the hash-routed MoE block (frozen token→expert lookup)."""
        base = f"/model/layers.{layer_id}/moe"
        num_e = self.moe_attrs["num_experts"]
        top_k = self.moe_attrs["top_k"]
        gate_weight_name = f"model.layers.{layer_id}.moe.gate.weight"
        tid2eid_name = f"model.layers.{layer_id}.moe.gate.tid2eid"
        self.make_initializer(mlp.gate.weight.data, gate_weight_name, to=self.io_dtype)
        self.make_initializer(mlp.gate.tid2eid.data, tid2eid_name, to=ir.DataType.INT64)

        hash_name = f"{base}/HashRouter"
        hash_outputs = [f"{hash_name}/output_{index}" for index in range(3)]
        self.make_node(
            "HashRouter",
            [root_input, self.input_names["input_ids"], gate_weight_name, tid2eid_name],
            hash_outputs,
            name=hash_name,
            domain="com.microsoft",
            score_function=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        self.make_value(hash_outputs[0], self.io_dtype, ["batch_size", "sequence_length", num_e])
        self.make_value(hash_outputs[1], self.io_dtype, ["batch_size", "sequence_length", top_k])
        self.make_value(hash_outputs[2], ir.DataType.INT64, ["batch_size", "sequence_length", top_k])

        logits_flat_name = f"{base}/hash_router/logits/Reshape"
        self.make_reshape(
            logits_flat_name,
            [hash_outputs[0], f"/model/constants/INT64/[-1, {num_e}]"],
            self.io_dtype,
            [None, num_e],
        )
        weights_flat_name = f"{base}/hash_router/weights/Reshape"
        indices_flat_name = f"{base}/hash_router/indices/Reshape"
        self.make_reshape(
            weights_flat_name,
            [hash_outputs[1], f"/model/constants/INT64/[-1, {top_k}]"],
            self.io_dtype,
            [None, top_k],
        )
        self.make_reshape(
            indices_flat_name,
            [hash_outputs[2], f"/model/constants/INT64/[-1, {top_k}]"],
            ir.DataType.INT64,
            [None, top_k],
        )

        shape_name = f"{base}/hash_router/Shape"
        fill_name = f"{base}/hash_router/NegFill"
        self.make_shape(shape_name, f"{logits_flat_name}/output_0", shape=[2])
        self.make_expand(
            fill_name,
            [f"/model/constants/{self.to_str_dtype(self.io_dtype)}/-10000.0", f"{shape_name}/output_0"],
            self.io_dtype,
            [None, num_e],
        )
        router_probs_name = f"{base}/hash_router/router_probs/ScatterElements"
        router_weights_name = f"{base}/hash_router/router_weights/ScatterElements"
        self.make_scatter_elements(
            router_probs_name,
            [f"{fill_name}/output_0", f"{indices_flat_name}/output_0", f"{weights_flat_name}/output_0"],
            self.io_dtype,
            [None, num_e],
            axis=-1,
        )
        zeros_name = f"{base}/hash_router/ZeroFill"
        self.make_constant_of_shape(
            zeros_name,
            f"{shape_name}/output_0",
            ir.tensor([0.0], dtype=self.io_dtype),
            self.io_dtype,
            [None, num_e],
        )
        self.make_scatter_elements(
            router_weights_name,
            [f"{zeros_name}/output_0", f"{indices_flat_name}/output_0", f"{weights_flat_name}/output_0"],
            self.io_dtype,
            [None, num_e],
            axis=-1,
            reduction="add",
        )
        return self.make_deepseek_moe_op(
            layer_id,
            mlp,
            root_input,
            f"{router_probs_name}/output_0",
            op_type,
            weight_type,
            base,
            router_weights=f"{router_weights_name}/output_0",
        )

    def make_deepseek_moe_op(
        self,
        layer_id: int,
        mlp,
        root_input: str,
        router_probs: str,
        op_type: str,
        weight_type: str,
        base: str,
        router_weights: str = "",
    ) -> str:
        """Emit MoE/QMoE op and return output value name."""
        num_e = self.moe_attrs["num_experts"]

        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Repack [gate|up] → interleaved [g0,u0,g1,u1,...] for swiglu_fusion=1
        raw_gu = mlp.experts.gate_up_proj
        half = raw_gu.shape[1] // 2
        interleaved = torch.stack(
            [raw_gu[:, :half, :], raw_gu[:, half:, :]], dim=2
        ).reshape_as(raw_gu)

        if op_type == "MoE":
            self.make_initializer(interleaved, gate_up_proj_weight, to=self.io_dtype)
            self.make_initializer(mlp.experts.down_proj, down_proj_weight, to=self.io_dtype)
        else:
            gu_qw, gu_sc, down_qw, down_sc = [], [], [], []
            for i in range(num_e):
                qw1, sc1 = self.make_qmoe_weights(interleaved[i])
                gu_qw.append(qw1)
                gu_sc.append(sc1)
                qw2, sc2 = self.make_qmoe_weights(mlp.experts.down_proj[i])
                down_qw.append(qw2)
                down_sc.append(sc2)
            self.make_initializer(torch.stack(gu_qw).to(torch.uint8), gate_up_proj_weight)
            self.make_initializer(torch.stack(down_qw).to(torch.uint8), down_proj_weight)
            self.make_initializer(torch.stack(gu_sc), gate_up_proj_scales, to=self.io_dtype)
            self.make_initializer(torch.stack(down_sc), down_proj_scales, to=self.io_dtype)

        self.make_initializer(
            torch.zeros(num_e, 2 * self.moe_intermediate_size),
            gate_up_proj_bias, to=self.io_dtype
        )
        self.make_initializer(
            torch.zeros(num_e, self.hidden_size),
            down_proj_bias, to=self.io_dtype
        )

        moe_name = f"{base}/{op_type}"
        # Override MoE attrs for V4.
        # normalize_routing_weights=True is required (not merely config-dependent): it makes
        # the op renormalize the weights among only the selected top-k experts, which is what
        # turns the log(selected_score)/-inf router_probs built above into exactly
        # `selected_score / sum(selected_scores)` — see make_moe_masked_router_logits. The
        # reference always performs this renormalization unconditionally.
        saved_top_k = self.moe_attrs["top_k"]
        saved_swiglu = self.moe_attrs["swiglu_fusion"]
        saved_limit = self.moe_attrs["swiglu_limit"]
        saved_norm = self.moe_attrs["normalize_routing_weights"]
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["swiglu_limit"] = self.swiglu_limit
        self.moe_attrs["normalize_routing_weights"] = not router_weights

        self.make_moe_op(
            moe_name,
            root_input=root_input,
            router_probs=router_probs,
            weight1=gate_up_proj_weight,
            scales1=gate_up_proj_scales if op_type == "QMoE" else "",
            bias1=gate_up_proj_bias,
            weight2=down_proj_weight,
            scales2=down_proj_scales if op_type == "QMoE" else "",
            bias2=down_proj_bias,
            router_weights=router_weights,
        )

        # Restore
        self.moe_attrs["top_k"] = saved_top_k
        self.moe_attrs["swiglu_fusion"] = saved_swiglu
        self.moe_attrs["swiglu_limit"] = saved_limit
        self.moe_attrs["normalize_routing_weights"] = saved_norm

        if router_weights:
            return f"{moe_name}/output_0"

        # `routed_scaling_factor` scales each selected expert's weight before it is combined
        # with that expert's FFN output (see DeepseekV4{TopK,Hash}Router.forward); since the
        # combine is linear in the weights, this is equivalent to scaling the MoE op's output.
        scale_name = f"model.layers.{layer_id}.moe.routed_scaling_factor"
        self.make_initializer(
            torch.tensor(self.routed_scaling_factor, dtype=to_torch_dtype(self.io_dtype)),
            scale_name, to=self.io_dtype,
        )
        scaled_name = f"{base}/routed_scale/Mul"
        self.make_mul(
            scaled_name,
            [f"{moe_name}/output_0", scale_name],
            self.io_dtype,
            self.hidden_state_shape(),
        )

        return f"{scaled_name}/output_0"

    def make_shared_expert(
        self, layer_id: int, shared_expert, root_input: str
    ) -> str:
        """Emit the always-active shared SwiGLU-MLP.  Returns output [B, S, D]."""
        base = f"/model/layers.{layer_id}/shared_expert"
        inter = self.moe_intermediate_size

        gate_name = self.make_matmul(shared_expert.gate_proj, f"{base}/gate_proj/MatMul", root_input)
        up_name = self.make_matmul(shared_expert.up_proj, f"{base}/up_proj/MatMul", root_input)

        # SiLU on gate (with clamp at swiglu_limit)
        clamp_g_max_name = f"model.layers.{layer_id}.shared_expert.gate_limit_max"
        self.make_initializer(
            torch.tensor(self.swiglu_limit, dtype=to_torch_dtype(self.io_dtype)),
            clamp_g_max_name, to=self.io_dtype
        )
        gate_clamped_name = f"{base}/gate/Clip"
        self.make_clip(gate_clamped_name,
                       [f"{gate_name}/output_0", "", clamp_g_max_name],
                       self.io_dtype,
                       ["batch_size", "sequence_length", inter])

        silu_sig_name = f"{base}/silu/Sigmoid"
        self.make_sigmoid(
            silu_sig_name,
            f"{gate_clamped_name}/output_0",
            self.io_dtype,
            ["batch_size", "sequence_length", inter],
        )

        silu_name = f"{base}/silu/Mul"
        self.make_mul(silu_name,
                      [f"{gate_clamped_name}/output_0", f"{silu_sig_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", "sequence_length", inter])

        # Clamp up (both sides)
        clamp_u_min_name = f"model.layers.{layer_id}.shared_expert.up_limit_min"
        clamp_u_max_name = f"model.layers.{layer_id}.shared_expert.up_limit_max"
        self.make_initializer(
            torch.tensor(-self.swiglu_limit, dtype=to_torch_dtype(self.io_dtype)),
            clamp_u_min_name, to=self.io_dtype
        )
        self.make_initializer(
            torch.tensor(self.swiglu_limit, dtype=to_torch_dtype(self.io_dtype)),
            clamp_u_max_name, to=self.io_dtype
        )
        up_clamped_name = f"{base}/up/Clip"
        self.make_clip(up_clamped_name,
                       [f"{up_name}/output_0", clamp_u_min_name, clamp_u_max_name],
                       self.io_dtype,
                       ["batch_size", "sequence_length", inter])

        gate_up_mul_name = f"{base}/gate_up/Mul"
        self.make_mul(gate_up_mul_name,
                      [f"{silu_name}/output_0", f"{up_clamped_name}/output_0"],
                      self.io_dtype,
                      ["batch_size", "sequence_length", inter])

        down_name = self.make_matmul(shared_expert.down_proj, f"{base}/down_proj/MatMul",
                                     f"{gate_up_mul_name}/output_0")
        return f"{down_name}/output_0"

    # ------------------------------------------------------------------ #
    # HyperHead (collapse HC streams to [B, S, D])
    # ------------------------------------------------------------------ #

    def make_hc_head_decomposed(self, hc_head, hc_streams: str) -> str:
        """Emit the HyperHead subgraph.  Returns [B, S, D] value name."""
        base = "/model/hc_head"
        hc = self.hc_mult
        d = self.hidden_size

        # Flatten: [B, S, hc_mult, D] → [B, S, hc_mult * D]
        flat_name = f"{base}/flatten/Reshape"
        self.make_reshape(
            flat_name,
            [hc_streams, f"/model/constants/INT64/[0, 0, {hc * d}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", hc * d],
        )

        # UnweightedRMSNorm in fp32
        normed_name = self.make_unweighted_rms_norm(
            f"{base}/InputNorm",
            f"{flat_name}/output_0",
            self.hc_eps,
            ["batch_size", "sequence_length", hc * d],
        )

        # Linear: [B, S, hc*D] @ hc_fn.T → [B, S, hc_mult]
        hc_fn_w_name = "model.hc_head.hc_fn"
        # hc_head.hc_fn: [hc_mult, hc_mult * D] → store transposed
        self.make_initializer(hc_head.hc_fn.data.T.float(), hc_fn_w_name, to=ir.DataType.FLOAT)

        fn_mm_name = f"{base}/fn/MatMul"
        fn_mm_output = f"{fn_mm_name}/output_0"
        self.make_node("MatMul", inputs=[normed_name, hc_fn_w_name], outputs=[fn_mm_output], name=fn_mm_name)
        self.make_value(fn_mm_output, ir.DataType.FLOAT, shape=["batch_size", "sequence_length", hc])

        # Scale and bias
        hc_scale_name = "model.hc_head.hc_scale"
        hc_base_name = "model.hc_head.hc_base"
        self.make_initializer(hc_head.hc_scale.data.float(), hc_scale_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_head.hc_base.data.float(), hc_base_name, to=ir.DataType.FLOAT)

        scaled_name = f"{base}/scaled/Mul"
        self.make_mul(scaled_name,
                      [f"{fn_mm_name}/output_0", hc_scale_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        biased_name = f"{base}/biased/Add"
        self.make_add(biased_name,
                      [f"{scaled_name}/output_0", hc_base_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # Sigmoid + eps
        sig_name = f"{base}/Sigmoid"
        self.make_sigmoid(sig_name, f"{biased_name}/output_0",
                          ir.DataType.FLOAT,
                          ["batch_size", "sequence_length", hc])

        hchead_eps_name = "model.hc_head.eps"
        self.make_initializer(torch.tensor(self.hc_eps, dtype=torch.float32),
                              hchead_eps_name, to=ir.DataType.FLOAT)

        pre_name = f"{base}/pre/Add"
        self.make_add(pre_name,
                      [f"{sig_name}/output_0", hchead_eps_name],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc])

        # Unsqueeze pre: [B, S, hc] → [B, S, hc, 1]
        pre_u_name = f"{base}/pre/Unsqueeze"
        self.make_unsqueeze(
            pre_u_name,
            [f"{pre_name}/output_0", "/model/constants/INT64/[3]"],
            ir.DataType.FLOAT,
            ["batch_size", "sequence_length", hc, 1],
        )

        # Cast hc_streams to fp32 for weighted sum
        hc_fp32_name = f"{base}/hc_fp32/Cast"
        self.make_cast(hc_fp32_name, hc_streams, ir.DataType.FLOAT,
                       ["batch_size", "sequence_length", hc, d])

        # Weighted: [B, S, hc, 1] * [B, S, hc, D] → [B, S, hc, D]
        weighted_name = f"{base}/weighted/Mul"
        self.make_mul(weighted_name,
                      [f"{pre_u_name}/output_0", f"{hc_fp32_name}/output_0"],
                      ir.DataType.FLOAT,
                      ["batch_size", "sequence_length", hc, d])

        # Sum over hc axis
        out_fp32_name = f"{base}/out_fp32/ReduceSum"
        self.make_reduce_sum(out_fp32_name,
                             [f"{weighted_name}/output_0", "/model/constants/INT64/[2]"],
                             ir.DataType.FLOAT,
                             ["batch_size", "sequence_length", d], keepdims=False)

        # Cast back to IO dtype
        out_name = f"{base}/out/Cast"
        self.make_cast(out_name, f"{out_fp32_name}/output_0", self.io_dtype,
                       ["batch_size", "sequence_length", d])
        return f"{out_name}/output_0"

    def make_hc_head(self, hc_head, hc_streams: str) -> str:
        name = "/model/hc_head/HyperHead"
        weight_name = "model.hc_head.hc_fn"
        bias_name = "model.hc_head.hc_base"
        scale_name = "model.hc_head.hc_scale"
        self.make_initializer(hc_head.hc_fn.data.float(), weight_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_head.hc_base.data.float(), bias_name, to=ir.DataType.FLOAT)
        self.make_initializer(hc_head.hc_scale.data.float(), scale_name, to=ir.DataType.FLOAT)
        output = f"{name}/output_0"
        self.make_node(
            "HyperHead",
            [hc_streams, weight_name, bias_name, scale_name],
            [output],
            name=name,
            domain="com.microsoft",
            epsilon=self.hc_eps,
        )
        self.make_value(output, self.io_dtype, ["batch_size", "sequence_length", self.hidden_size])
        return output

    # ------------------------------------------------------------------ #
    # Decoder layer
    # ------------------------------------------------------------------ #

    def make_layer(
        self,
        layer_id: int,
        layer,
        post_attn: str,
        comb_attn: str,
        attn_ln_output: str,
        next_layer=None,
    ) -> tuple[str, str, str] | None:
        """Build one DeepSeek V4 decoder block.

        Updates ``self.hc_streams`` and ``self.layernorm_attrs``.
        """
        # ---- Attention ----
        attn_out = self.make_deepseek_attention(
            layer_id, layer.self_attn, attn_ln_output
        )

        # ---- Attention residual update + FFN HC + post-attention norm ----
        self.hc_streams, post_ffn, comb_ffn, ffn_ln_output = self.make_hyper_connection_mix(
            layer_id,
            "ffn",
            layer.ffn_hc,
            layer.post_attention_layernorm,
            attn_out,
            self.hc_streams,
            post_attn,
            comb_attn,
        )

        # ---- MoE / FFN ----
        moe_out = self.make_deepseek_moe(
            layer_id, layer.mlp, ffn_ln_output
        )

        if next_layer is not None:
            # ---- FFN residual update + next attention HC + next input norm ----
            self.hc_streams, next_post, next_comb, next_attn_input = self.make_hyper_connection_mix(
                layer_id + 1,
                "attn",
                next_layer.attn_hc,
                next_layer.input_layernorm,
                moe_out,
                self.hc_streams,
                post_ffn,
                comb_ffn,
            )
            next_state = (next_post, next_comb, next_attn_input)
        else:
            post_ffn = self.cast_hyper_mix_state(
                post_ffn, ["batch_size", "sequence_length", self.hc_mult], self.io_dtype
            )
            comb_ffn = self.cast_hyper_mix_state(
                comb_ffn,
                ["batch_size", "sequence_length", self.hc_mult, self.hc_mult],
                self.io_dtype,
            )
            self.hc_streams = self.make_hc_combine(
                layer_id, "ffn", post_ffn, comb_ffn, self.hc_streams, moe_out
            )
            next_state = None

        # Mark as not the first layernorm anymore (for skip tracking)
        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True
        return next_state

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #

    def is_layer(self, module) -> bool:  # type: ignore[override]
        return module.__class__.__name__ == "DeepseekV4DecoderLayer"

    def has_final_norm(self, module, orig_model) -> bool:  # type: ignore[override]
        if orig_model.__class__.__name__.startswith("Peft"):
            model = orig_model.base_model.model
        else:
            model = orig_model
        return (
            hasattr(model, "model")
            and hasattr(model.model, "norm")
            and module == model.model.norm
        )

    def make_model(self, input_path: str) -> None:
        """Override make_model to handle V4's unique module structure.

        Flow:
          embed_tokens → (expand to 4-D) → 43 decoder layers →
          hc_head → norm → lm_head
        """
        self.make_inputs_and_outputs()
        if self.kv_cache_quant_type != "none":
            self.make_kv_cache_scale_initializers()

        self.weights = self.load_weights(input_path)
        self.make_preprocessing_nodes()

        model_inner = self.weights.model

        # ---- 1. Embedding ----
        print("Reading embedding layer")
        self.make_embedding(model_inner.embed_tokens.weight)
        embed_out = self.layernorm_attrs["root_input"]

        # ---- 2. Expand embedding [B, S, D] → [B, S, hc_mult, D] ----
        hc = self.hc_mult
        d = self.hidden_size

        expand_u_name = "/model/hc_expand/Unsqueeze"
        self.make_unsqueeze(
            expand_u_name,
            [embed_out, "/model/constants/INT64/[2]"],
            self.io_dtype,
            ["batch_size", "sequence_length", 1, d],
        )

        expand_tile_name = "/model/hc_expand/Tile"
        self.make_tile(
            expand_tile_name,
            [f"{expand_u_name}/output_0",
             f"/model/constants/INT64/[1, 1, {hc}, 1]"],
            self.io_dtype,
            ["batch_size", "sequence_length", hc, d],
        )
        self.hc_streams = f"{expand_tile_name}/output_0"

        # ---- 3. Decoder layers ----
        self.layer_id = 0
        self.layernorm_attrs["first_layernorm"] = True
        self.layernorm_attrs["last_layernorm"] = False
        layers = model_inner.layers[: self.num_layers]
        post_attn, comb_attn, collapsed_attn = self.make_hyper_connection(
            0, "attn", layers[0].attn_hc, self.hc_streams
        )
        post_attn = self.cast_hyper_mix_state(
            post_attn, ["batch_size", "sequence_length", hc], ir.DataType.FLOAT
        )
        comb_attn = self.cast_hyper_mix_state(
            comb_attn, ["batch_size", "sequence_length", hc, hc], ir.DataType.FLOAT
        )
        attn_ln_output = self.make_layernorm_no_skip(0, layers[0].input_layernorm, collapsed_attn, "input")
        attn_state = (post_attn, comb_attn, attn_ln_output)
        for layer_index, layer in enumerate(layers):
            print(f"Reading layer {self.layer_id}")
            next_layer = layers[layer_index + 1] if layer_index + 1 < len(layers) else None
            attn_state = self.make_layer(self.layer_id, layer, *attn_state, next_layer=next_layer)
            self.layer_id += 1

        # ---- 4. HC Head ----
        print("Reading HC head")
        hc_head_out = self.make_hc_head(model_inner.hc_head, self.hc_streams)

        # ---- 5. Final norm (SkipSimplifiedLayerNorm style, but we treat as plain
        #         SimplifiedLayerNorm since HC head already collapsed) ----
        print("Reading final norm")
        final_norm_output = self.make_layernorm_no_skip(self.num_layers, model_inner.norm, hc_head_out, "final")

        # Update layernorm tracking for LM head
        self.layernorm_attrs["output_0"] = final_norm_output

        # ---- 6. LM head ----
        print("Reading LM head")
        self.make_lm_head(self.weights.lm_head)

        self.make_postprocessing_nodes()
        del self.weights
