# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
#
# Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# --------------------------------------------------------------------------
"""
Run the model builder to create the desired ONNX model.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import textwrap
from typing import Any, Literal, Sequence

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import to_torch_dtype, TorchTensor
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    QuantFormat,
    RTNWeightOnlyQuantConfig,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


class Model:
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.context_length = config.seq_length if hasattr(config, "seq_length") else config.max_position_embeddings
        self.original_context_length = config.original_max_position_embeddings if hasattr(config, "original_max_position_embeddings") else config.rope_scaling["original_max_position_embeddings"] if hasattr(config, "rope_scaling") and hasattr(config.rope_scaling, "original_max_position_embeddings") else self.context_length
        self.window_size = config.sliding_window if hasattr(config, "sliding_window") else -1  # default is -1 in GroupQueryAttention kernel
        self.intermediate_size = config.ffn_hidden_size if hasattr(config, "ffn_hidden_size") else config.intermediate_size
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.multi_query_group_num if hasattr(config, "multi_query_group_num") else config.num_attention_heads
        self.num_attn_heads = config.num_attention_heads
        self.head_size = config.head_dim if hasattr(config, "head_dim") and config.head_dim is not None else config.hidden_size // config.num_attention_heads
        self.num_layers = int(extra_options["num_hidden_layers"]) if "num_hidden_layers" in extra_options else config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.num_layers
        self.vocab_size = config.vocab_size
        self.activation = config.hidden_activation if hasattr(config, "hidden_activation") and config.hidden_activation is not None else config.hidden_act

        self.model_name_or_path = config._name_or_path
        self.model_type = config.architectures[0]
        self.io_dtype = ir.DataType(io_dtype)
        self.onnx_dtype = ir.DataType(onnx_dtype)
        self.quant_type = config.quantization_config["quant_method"] if hasattr(config, "quantization_config") else None
        self.adapter_path = extra_options.get("adapter_path", None)

        self.cache_dir = cache_dir
        self.filename = extra_options.get("filename", "model.onnx")
        self.hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
        self.hf_remote = extra_options.get("hf_remote", True)
        self.extra_options = extra_options

        # States for building the model
        self.graph = ir.Graph(
            inputs=(),
            outputs=(),
            nodes=(),
            opset_imports={"": 21, "com.microsoft": 1},
            name="main_graph",
        )
        self.model = ir.Model(self.graph, ir_version=10, producer_name="onnxruntime-genai")
        self.values = {}

        # EP-specific variables
        self.ep = ep
        self.ep_attrs = {
            "cpu": {},
            "cuda": {
                "enable_cuda_graph": "1" if extra_options.get("enable_cuda_graph", False) else "0",              # "1" if the model is able to enable cuda graph, "0" otherwise
                "enable_skip_layer_norm_strict_mode": "1"
            },
            "dml": {},
            # TODO: Enable graph capture for webgpu once supported both in onnxruntime-genai and onnxruntime.
            "webgpu": {},
            "trt-rtx": {"enable_cuda_graph": "1"}
        }

        # Map input names to their types and shapes
        self.input_names = ["input_ids", "attention_mask", "position_ids"]
        self.input_types = {
            "input_ids": ir.DataType.INT64,                                                                      # For standard models
            "attention_mask": ir.DataType.INT64,                                                                 # For standard models
            "position_ids": ir.DataType.INT64,                                                                   # For standard models
            "inputs_embeds": self.io_dtype,                                                                      # For standard models where you want to remove the embedding layer from the model (note that `inputs_embeds` is written this way to match Hugging Face format)
            "past_key_values.key": self.io_dtype,                                                                # For standard models (note that `past_key_values.key` is written this way to match Hugging Face format)
            "past_key_values.value": self.io_dtype,                                                              # For standard models (note that `past_key_values.value` is written this way to match Hugging Face format)
        }
        self.input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],                                                      # For standard models
            "attention_mask": ["batch_size", "total_sequence_length"],                                           # For standard models
            "position_ids": ["batch_size", "sequence_length"],                                                   # For standard models
            "inputs_embeds": ["batch_size", "sequence_length", self.hidden_size],                                # For standard models where you want to remove the embedding layer from the model (note that `inputs_embeds` is written this way to match Hugging Face format)
            "past_key_values.key": ["batch_size", self.num_kv_heads, "past_sequence_length", self.head_size],    # For standard models (note that `past_key_values.key` is written this way to match Hugging Face format)
            "past_key_values.value": ["batch_size", self.num_kv_heads, "past_sequence_length", self.head_size],  # For standard models (note that `past_key_values.value` is written this way to match Hugging Face format)
        }
        self.exclude_embeds = extra_options.get("exclude_embeds", False)
        if self.exclude_embeds:
            self.input_names = [name.replace("input_ids", "inputs_embeds") for name in self.input_names]

        # Map output names to their types and shapes
        self.output_names = ["logits"]
        self.output_types = {
            "hidden_states": self.io_dtype,                                                                      # For standard models where you want to remove the language modeling head from the model (note that `hidden_states` is written this way to match Hugging Face format)
            "logits": self.io_dtype,                                                                             # For standard models
            "present.key": self.io_dtype,                                                                        # For standard models (note that `present.key` is written this way to match Hugging Face format)
            "present.value": self.io_dtype,                                                                      # For standard models (note that `present.value` is written this way to match Hugging Face format)
        }
        self.output_shapes = {
            "hidden_states": ["batch_size", "sequence_length", self.hidden_size],                                # For standard models where you want to remove the language modeling head from the model (note that `hidden_states` is written this way to match Hugging Face format)
            "logits": ["batch_size", "sequence_length", self.vocab_size],                                        # For standard models
            "present.key": ["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size],           # For standard models (note that `present.key` is written this way to match Hugging Face format)
            "present.value": ["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size],         # For standard models (note that `present.value` is written this way to match Hugging Face format)
        }
        self.make_outputs_init()

        # Store names of nodes already created
        self.node_names = set()

        # Mask-specific variables
        # TODO: Reconcile differences between `seqlens_k` and `key_total_seq_lens` in the GroupQueryAttention and SparseAttention implementations. Ideally the same subgraph can be shared for both.
        self.mask_attrs = {
            "mask_name": "",            # Name of node that outputs 4D causal attention mask (used as add_qk in MultiHeadAttention)
            "seqlens_k": "",            # Sum of each row in attention mask - 1 (used as input to GroupQueryAttention)
            "total_seq_len": "",        # Size of total sequence length in attention mask (used as input to GroupQueryAttention and SparseAttention)
            "block_row_indices": "",    # Row indices of CSR format of block mask (used as input to SparseAttention)
            "block_col_indices": "",    # Col indices of CSR format of block mask (used as input to SparseAttention)
            "key_total_seq_lens": "",   # Sum of each row in attention mask (used as input to SparseAttention)
        }

        # Embedding-specific variables
        self.embed_attrs = {
            "scale": 1,                 # Scale value to multiply output of Embedding layer by
        }

        # LayerNorm-specific variables
        epsilon = config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-06
        self.layernorm_attrs = {
            "simple": True,             # Use SimplifiedLayerNorm/SkipSimplifiedLayerNorm vs. LayerNorm/SkipLayerNorm
            "first_layernorm": True,    # 1st LayerNorm = LayerNorm, then SkipLayerNorm for all subsequent LayerNorms
            "last_layernorm": False,    # Last LayerNorm = SkipLayerNorm with only output 0 (no output 3)
            "root_input": "",           # Root input from parent node for LayerNorm and SkipLayerNorm
            "skip_input": "",           # Skip input from parent node for SkipLayerNorm
            "output_0": "",             # Output 0 for LayerNorm and SkipLayerNorm
            "output_3": "",             # Output 3 for SkipLayerNorm
            "add_offset": 0,            # Offset value for LayerNorm weight
            "epsilon": epsilon,         # Epsilon value to avoid `sqrt(0)` in LayerNorm
            "cast": {                   # Casting LayerNorm-specific variables
                "use_fp32": False,      # Use float32 precision to compute LayerNorm
                "root_input": False,    # Cast root_input
                "skip_input": False,    # Cast skip_input
                "output_0": False,      # Cast output_0
                "output_3": False,      # Cast output_3
            }
        }

        # MatMul-specific variables
        is_lora = hasattr(config, "peft_type") and config.peft_type == "LORA"
        self.matmul_attrs = {
            "use_lora": is_lora,        # Use LoRA/QLoRA format
        }

        # RotaryEmbedding-specific variables
        position_scale = config.rope_position_scale if hasattr(config, "rope_position_scale") else 1
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        rotemb_dim = int(self.head_size * partial_rotary_factor) if partial_rotary_factor != 1.0 else 0
        rope_theta = config.rope_theta if hasattr(config, "rope_theta") else config.rope_embedding_base if hasattr(config, "rope_embedding_base") else 10000
        self.rope_attrs = {
            "create_caches": True,                           # Create cos/sin caches for rotary embeddings
            "save_caches": True,                             # Auto-save cos/sin caches for rotary embeddings after creation
            "cache_length": self.context_length,             # Cache length to use when creating cos/sin caches for rotary embeddings
            "theta": rope_theta,                             # Base value if calculating cos/sin caches from scratch
            "partial_rotary_factor": partial_rotary_factor,  # Factor for partial rotary embeddings
            "interleaved": 0,                                # Interleave the rotary embeddings (e.g. [0, 0, 0, 1, 1, 1] to [0, 1, 0, 1, 0, 1], RotaryEmbedding kernel expects a default value of 0)
            "rotary_embedding_dim": rotemb_dim,              # For partial rotary embeddings (RotaryEmbedding kernel expects a default value of 0)
            "rescale_factors": 1,                            # Rescale factors when calculating `inv_freq` in rotary embeddings
            "t_dtype": torch.int64,                          # Torch dtype when calculating `t` in rotary embeddings
            "position_scale": position_scale,                # Scale value when calculating `t` in rotary embeddings
            "mscale": 1,                                     # Magnitude scaling factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
            "mscale_policy": "",                             # Magnitude scaling policy when scaling `emb.cos()/emb.sin()` in rotary embeddings
        }
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.make_rope_init(config)

        # Attention-specific variables (MHA, GQA, GQA + Rot.Emb., etc.)
        attn_softcap = config.attn_logit_softcapping if hasattr(config, "attn_logit_softcapping") and config.attn_logit_softcapping is not None else 0.0  # default is 0.0 in GroupQueryAttention kernel

        # Block-sparse attention-specific variables
        sparse_block_size = config.blocksparse_block_size if hasattr(config, "blocksparse_block_size") else 0
        kernel_block_size = config.blocksparse_triton_kernel_block_size if hasattr(config, "blocksparse_triton_kernel_block_size") else 0
        local_blocks = config.blocksparse_num_local_blocks if hasattr(config, "blocksparse_num_local_blocks") else 0
        vert_block_stride = config.blocksparse_vert_stride if hasattr(config, "blocksparse_vert_stride") else 0
        homo_head = config.blocksparse_homo_head_pattern if hasattr(config, "blocksparse_homo_head_pattern") else False
        self.attention_attrs = {
            "q_path": "",                                    # Q path to attention
            "k_path": "",                                    # K path to attention
            "v_path": "",                                    # V path to attention
            "op_type": "MultiHeadAttention",                 # Attention op to use
            "scale": 1 / np.sqrt(self.head_size),            # Scale value after calculating Q x K' in attention
            "softcap": attn_softcap,                         # Softcap value to prevent values from exploding in attention
            "use_rope_in_attn": False,                       # Use rotary embeddings within attention (instead of a separate RotaryEmbedding op)
            "use_packed_matmul": False,                      # Use packed MatMul (instead of 3 separate MatMuls for Q/K/V)
            "block_sparse": {                                # Block-sparse attention-specific variables
                "sparse_block_size": sparse_block_size,      # Sparse block size for SparseAttention op
                "kernel_block_size": kernel_block_size,      # Kernel block size for sparse attention
                "local_blocks": local_blocks,                # Number of local blocks for sparse attention
                "vert_stride": vert_block_stride,            # Vertical stride to use for sparse attention
                "homo_head": homo_head,                      # Use homo head pattern for sparse attention
            },
            "q_norm": False,                                 # LayerNorm after MatMul in Q path
            "k_norm": False,                                 # LayerNorm after MatMul in K path
            "sinks": False,                                  # Sink values for softmax in attention
        }
        self.make_attention_init()

        # MLP-specific variables
        self.mlp_attrs = {
            "use_proj": True,           # Use projection style for MLP (GateProj/UpProj/DownProj)
            "use_fc": False,            # Use fully-connected style for MLP (FC1/FC2)
            "output_0": "",             # Output 0 for MLP layer
        }

        # MoE-specific variables
        moe_op_type = "QMoE" if self.onnx_dtype == ir.DataType.INT4 else "MoE"
        num_experts = config.num_local_experts if hasattr(config, "num_local_experts") else 0
        top_k_experts = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 0
        expert_weight_bits = 8 if extra_options.get("use_8bits_moe", False) else 4
        swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
        self.moe_attrs = {
            "op_type": moe_op_type,                           # MoE op to use
            "num_experts": num_experts,                       # Number of experts in MoE layer
            "top_k": top_k_experts,                           # Number of experts to select in MoE layer
            "activation_alpha": 1.0,                          # Alpha parameter used in activation function
            "activation_beta": 0.0,                           # Beta parameter used in activation function
            "activation_type": self.activation,               # Activation function for MoE layer
            "expert_weight_bits": expert_weight_bits,         # Number of bits used in quantized MoE weights (only INT4 or INT8 are supported).
            "normalize_routing_weights": False,               # Normalize routing weights in MoE layer
            "swiglu_fusion": 0,                               # Fusion level for SwiGLU activation function
            "swiglu_limit": swiglu_limit,                     # Value used to clamp results into a certain range in SwiGLU activation function
            "use_sparse_mixer": False,                        # Use SparseMixer in MoE layer (used in Phi-3.5 MoE)
        }

        # LM head-specific variables
        lm_head_softcap = config.final_logit_softcapping if hasattr(config, "final_logit_softcapping") and config.final_logit_softcapping is not None else 0.0  # default is 0.0 in GroupQueryAttention kernel
        self.lm_head_attrs = {
            "scale": 1,                  # Scale value to multiply output of LM head by
            "mask": None,                # LM head mask for tokens in the vocabulary
            "softcap": lm_head_softcap,  # Softcap value to prevent values from exploding in LM head
        }
        if hasattr(config, "dummy_token_indices"):
            # Create LM head mask for tokens in the vocabulary
            dummy_tokens_mask = torch.zeros(self.vocab_size).bool()
            dummy_tokens_mask[config.dummy_token_indices] = True
            self.lm_head_attrs["mask"] = dummy_tokens_mask

        # Quantization-specific variables (INT4, INT8, etc.)
        int4_algo_config = self.make_int4_algo_config(extra_options.get("int4_algo_config", "default"))
        self.int4_block_size = extra_options.get("int4_block_size", 32)
        self.quant_attrs = {
            "int4": {
                "accuracy_level": int(extra_options.get("int4_accuracy_level", 4 if self.ep in ["cpu", "webgpu"] else 0)),
                "block_size": int(self.int4_block_size),
                "is_symmetric": extra_options.get("int4_is_symmetric", True),
                "op_types_to_quantize": extra_options.get("int4_op_types_to_quantize", ("MatMul", )),
                "nodes_to_exclude": extra_options.get("int4_nodes_to_exclude", []),
                "algo_config": int4_algo_config,
            },
            "use_qdq": extra_options.get("use_qdq", False),
        }
        if self.quant_type is not None:
            # Create quantized attributes from quantization config
            self.quant_attrs["config"] = config.quantization_config
            self.quant_attrs["use_g_idx"] = config.quantization_config["desc_act"] if "desc_act" in config.quantization_config else False

        self.int4_tied_embeddings = config.tie_word_embeddings if hasattr(config, "tie_word_embeddings") and config.tie_word_embeddings is not None else False
        self.int4_tied_embeddings = extra_options.get("int4_tied_embeddings", self.int4_tied_embeddings)
        self.int8_lm_head = extra_options.get("int4_algo_config", "default") in {"k_quant_mixed", "k_quant_last"}
        if not self.int8_lm_head:
            # matmul_nbits_quantizer.py has a different naming for default quantization, so lm_head.MatMul.weight_Q{}G{} does not match.
            self.int4_tied_embeddings = False

    def to_str_dtype(self, dtype: ir.DataType) -> str:
        return dtype.name

    def make_outputs_init(self):
        # Always use float32 logits to improve accuracy in the case of bf16 models.
        if self.io_dtype == ir.DataType.BFLOAT16:
            self.output_types["logits"] = ir.DataType.FLOAT

        self.exclude_lm_head = self.extra_options.get("exclude_lm_head", False)
        self.include_hidden_states = self.extra_options.get("include_hidden_states", False)

        if self.exclude_lm_head:
            self.output_names = [name.replace("logits", "hidden_states") for name in self.output_names]
        elif self.include_hidden_states:
            self.output_names = ["hidden_states"] + self.output_names

    def make_rope_init(self, config):
        if "short_factor" in config.rope_scaling:
            # For models with multiple rotary embedding caches (e.g. Phi-3 mini 128K)
            self.rope_attrs["mscale_policy"] = config.rope_scaling["type"]
            short_factor = torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32)
            long_factor = torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32)

            short_mscale = config.rope_scaling["short_mscale"] if "short_mscale" in config.rope_scaling else 0
            long_mscale = config.rope_scaling["long_mscale"] if "long_mscale" in config.rope_scaling else 0
            short_mscale = short_mscale if short_mscale > 0 else self.make_mscale(self.context_length / self.original_context_length)
            long_mscale = long_mscale if long_mscale > 0 else self.make_mscale(self.context_length / self.original_context_length)

            self.rope_attrs["multi_cache"] = {
                "short_factor": short_factor,                # Short factor when calculating `inv_freq` in rotary embeddings
                "long_factor": long_factor,                  # Long factor when calculating `inv_freq` in rotary embeddings
                "short_mscale": short_mscale,                # Magnitude scaling for short factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
                "long_mscale": long_mscale,                  # Magnitude scaling for long factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
            }

        elif "low_freq_factor" in config.rope_scaling:
            # For models that rescale `inv_freq` using `low_freq_factor` and `high_freq_factor` (e.g. LLaMA-3.1)
            factor = config.rope_scaling["factor"] if "factor" in config.rope_scaling else 0
            low_freq_factor = config.rope_scaling["low_freq_factor"] if "low_freq_factor" in config.rope_scaling else 0
            high_freq_factor = config.rope_scaling["high_freq_factor"] if "high_freq_factor" in config.rope_scaling else 0
            
            self.rope_attrs["rescale_inv_freq"] = {
                "factor": factor,                            # Scale factor when calculating `new_freq` in rotary embeddings
                "low_freq_factor": low_freq_factor,          # Low freq factor when calculating `low_freq_wavelen` in rotary embeddings
                "high_freq_factor": high_freq_factor,        # High freq factor when calculating `high_freq_wavelen` in rotary embeddings
            }

        elif "beta_fast" in config.rope_scaling:
            # For models that use YARN (e.g. OpenAI OS-minier)
            factor = config.rope_scaling["factor"] if "factor" in config.rope_scaling else 0
            beta_slow = config.rope_scaling["beta_slow"] if "beta_slow" in config.rope_scaling else 0
            beta_fast = config.rope_scaling["beta_fast"] if "beta_fast" in config.rope_scaling else 0

            self.rope_attrs["mscale_policy"] = config.rope_scaling["rope_type"]
            self.rope_attrs["mscale"] = self.make_mscale(config.rope_scaling["factor"])
            self.rope_attrs["rescale_inv_freq"] = {
                "factor": factor,
                "ntk_alpha": beta_slow,
                "ntk_beta": beta_fast,
            }

    def make_attention_init(self):
        valid_gqa_configurations = {
            ("cpu", ir.DataType.FLOAT),
            ("cuda", ir.DataType.FLOAT16),
            ("cuda", ir.DataType.BFLOAT16),
            ("dml", ir.DataType.FLOAT16),
            ("webgpu", ir.DataType.FLOAT16),
            ("webgpu", ir.DataType.FLOAT),
            ("trt-rtx", ir.DataType.FLOAT16),
        }
        if (self.ep, self.io_dtype) in valid_gqa_configurations:
            # Change model settings for GroupQueryAttention
            self.attention_attrs["op_type"] = "GroupQueryAttention"
            print("GroupQueryAttention (GQA) is used in this model.")

            # Some EPs don't support packed Q/K/V for GQA yet
            # Packed MatMul with LoRA/QLoRA is not currently supported
            self.attention_attrs["use_packed_matmul"] = (
                self.ep not in ["dml"]
                and not self.matmul_attrs["use_lora"]
                and not self.attention_attrs["q_norm"]
                and not self.attention_attrs["k_norm"]
            )

            # Some EPs don't support fusing rotary embeddings inside GQA yet
            self.attention_attrs["use_rope_in_attn"] = self.ep not in ["dml"]
            if self.attention_attrs["use_rope_in_attn"]:
                # GQA + Rot.Emb. does not require `position_ids` as input
                self.input_names.remove("position_ids")

        self.past_present_share_buffer = self.attention_attrs["op_type"] == "GroupQueryAttention"

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Create config with attributes from config.json and generation_config.json (if latter file exists)
        config = AutoConfig.from_pretrained(model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)
        try:
            # Override search attributes in config based on values in generation_config.json
            gen_config = GenerationConfig.from_pretrained(model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)
            defaults = {
                "bos_token_id": None,
                "do_sample": False,
                "eos_token_id": None,
                "pad_token_id": None,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
            }
            for key, default_val in defaults.items():
                val = getattr(gen_config, key)
                if val != default_val:
                    setattr(config, key, getattr(gen_config, key))
        except:
            pass

        inputs = dict(zip(self.input_names, self.input_names))
        inputs.update({
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        })
        outputs = dict(zip(self.output_names, self.output_names))
        outputs.update({
            "present_key_names": "present.%d.key",
            "present_value_names": "present.%d.value",
        })
        if "hidden_states" in outputs:
            # Remove 'hidden_states' from 'outputs' entry in config since ORT GenAI doesn't use it
            del outputs["hidden_states"]

        bos_token_id = config.bos_token_id if hasattr(config, "bos_token_id") and config.bos_token_id is not None else 1
        eos_token_id = config.eos_token_id
        pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        genai_config = {
            "model": {
                "bos_token_id": bos_token_id,
                "context_length": self.context_length,
                "decoder": {
                    "session_options" : {
                        "log_id": "onnxruntime-genai",
                        "provider_options" : [],
                    },
                    "filename": self.filename,
                    "head_size": self.head_size,
                    "hidden_size": self.hidden_size,
                    "inputs": inputs,
                    "outputs": outputs,
                    "num_attention_heads": self.num_attn_heads,
                    "num_hidden_layers": self.num_layers,
                    "num_key_value_heads": self.num_kv_heads,
                },
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "type": self.model_type[ : self.model_type.find("For") if "For" in self.model_type else len(self.model_type)].lower(),
                "vocab_size": self.vocab_size,
            },
            "search": {
                "diversity_penalty": config.diversity_penalty if hasattr(config, "diversity_penalty") else 0.0,
                "do_sample": config.do_sample if hasattr(config, "do_sample") else False,
                "early_stopping": True,
                "length_penalty": config.length_penalty if hasattr(config, "length_penalty") else 1.0,
                "max_length": self.context_length,
                "min_length": 0,
                "no_repeat_ngram_size": config.no_repeat_ngram_size if hasattr(config, "no_repeat_ngram_size") else 0,
                "num_beams": config.num_beams if hasattr(config, "num_beams") else 1,
                "num_return_sequences": config.num_return_sequences if hasattr(config, "num_return_sequences") else 1,
                "past_present_share_buffer": False if "config_only" in self.extra_options else self.past_present_share_buffer,
                "repetition_penalty": config.repetition_penalty if hasattr(config, "repetition_penalty") else 1.0,
                "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
                "top_k": config.top_k if hasattr(config, "top_k") else 50,
                "top_p": config.top_p if hasattr(config, "top_p") else 1.0,
            },
        }

        if self.ep == "trt-rtx" and self.window_size is not None and self.window_size > 0:
            # Compute layer indices that use sliding window attention
            layer_idxs = [layer_id for layer_id in range(self.num_layers) if hasattr(self, "is_local") and self.is_local(layer_id)]
            
            genai_config["model"]["decoder"]["sliding_window"] = {
                "window_size": self.window_size,
                "slide_key_value_cache": False,
                "slide_inputs": False,
                "layers": layer_idxs
            }

        if self.ep != "cpu":
            ep_name = self.ep.replace("trt-rtx", "NvTensorRtRtx")
            ep_options = { ep_name : self.ep_attrs[self.ep] }
            genai_config["model"]["decoder"]["session_options"]["provider_options"].append(ep_options)

        print(f"Saving GenAI config in {out_dir}")
        with open(os.path.join(out_dir,"genai_config.json"), "w") as f:
            json.dump(genai_config, f, indent=4)

    def make_key_value_cache_shape(self, layer_id, shape):
        """
        Modifies KV cache shape dimension names for models with alternating attention patterns.
        For TensorRT EP with sliding window layers, replaces 'sequence' with 'sliding' in dimension name.
        """
        if self.ep == "trt-rtx" and hasattr(self, "is_local") and self.is_local(layer_id):
            return [shape[0], shape[1], shape[2].replace("sequence", "sliding"), shape[3]]
        return shape

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)
        print(f"Saving processing files in {out_dir} for GenAI")
        tokenizer.save_pretrained(out_dir)

    def make_int4_algo_config(self, quant_method: str):
        customized_weight_config = {}
        int4_algo_config = None

        if quant_method == "rtn":
            int4_algo_config = RTNWeightOnlyQuantConfig()

        elif quant_method in {"k_quant_mixed", "k_quant_last"}:
            from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

            if quant_method == "k_quant_mixed":
                # k_quant_mixed is from llama.cpp.
                # Reference: https://github.com/ggml-org/llama.cpp/blob/36667c8edcded08063ed51c7d57e9e086bbfc903/src/llama-quant.cpp#L136
                # We also consider some MatMuls are more senstive to quantization than other MatMuls.
                layers_to_exclude = [
                    i
                    for i in range(self.num_layers)
                    if i < self.num_layers / 8 or i >= 7 * self.num_layers / 8 or (i - (round)(self.num_layers / 8)) % 3 == 2
                ]
                for i in layers_to_exclude:
                    customized_weight_config["/model/layers." + str(i) + "/attn/qkv_proj/MatMul"] = {"bits": 8}
                    customized_weight_config["/model/layers." + str(i) + "/attn/v_proj/MatMul"] = {"bits": 8}
                    customized_weight_config["/model/layers." + str(i) + "/mlp/down_proj/MatMul"] = {"bits": 8}

            customized_weight_config["/lm_head/MatMul"] = {"bits": 8}
            int4_algo_config = KQuantWeightOnlyQuantConfig(customized_weight_config=customized_weight_config)

        return int4_algo_config

    def to_int4(self) -> ir.Model:
        quant = MatMulNBitsQuantizer(
            model=ir.to_proto(self.model),
            block_size=self.quant_attrs["int4"]["block_size"],
            is_symmetric=self.quant_attrs["int4"]["is_symmetric"],
            accuracy_level=self.quant_attrs["int4"]["accuracy_level"],
            nodes_to_exclude=self.quant_attrs["int4"]["nodes_to_exclude"],
            quant_format=QuantFormat.QDQ if self.quant_attrs["use_qdq"] else QuantFormat.QOperator,
            op_types_to_quantize=self.quant_attrs["int4"]["op_types_to_quantize"],
            algo_config=self.quant_attrs["int4"]["algo_config"],
        )
        quant.process()
        return ir.from_proto(quant.model.model)

    def save_model(self, out_dir):
        print(f"Saving ONNX model in {out_dir}")

        already_quantized_in_qdq_format = self.quant_type is not None and self.quant_attrs["use_qdq"]  # Skip quantizing `MatMul` in `DequantizeLinear --> Transpose --> MatMul` path
        if self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4} and not already_quantized_in_qdq_format:
            model = self.to_int4()
        else:
            model = self.model

        # Make sure all nodes are topologically sorted
        model.graph.sort()

        # Save ONNX model with only one external data file and delete any existing duplicate copies
        out_path = os.path.join(out_dir, self.filename)
        data_path = os.path.join(out_dir, os.path.basename(out_path) + ".data")
        if os.path.exists(out_path):
            print(f"Overwriting {out_path}")
            os.remove(out_path)
        if os.path.exists(data_path):
            print(f"Overwriting {data_path}")
            os.remove(data_path)

        with tqdm() as pbar:
            total_set = False

            def callback(tensor: ir.TensorProtocol, metadata: dict):
                nonlocal total_set
                if not total_set:
                    pbar.total = metadata.total
                    total_set = True

                pbar.update()
                pbar.set_description(f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})")

            ir.save(
                model,
                out_path,
                external_data=os.path.basename(data_path),
                size_threshold_bytes=0,
                callback=callback,
            )

        # Delete temporary cache dir if empty
        if not os.listdir(self.cache_dir):
            os.rmdir(self.cache_dir)

    def make_initializer(self, tensor: torch.Tensor | np.ndarray | ir.TensorProtocol, /, name: str, to: ir.DataType | None = None):
        if to is not None:
            # Cast the tensor lazily if `to` is provided
            def tensor_func():
                nonlocal tensor
                tensor = tensor.to(to_torch_dtype(to))
                return TorchTensor(tensor, name=name)

            ir_tensor = ir.LazyTensor(
                tensor_func, dtype=to, shape=ir.Shape(tensor.shape), name=name
            )
        elif isinstance(tensor, torch.nn.parameter.Parameter):
            ir_tensor = TorchTensor(tensor, name=name)
        else:
            ir_tensor = ir.tensor(tensor, name=name)
        value = self.make_value(name, ir_tensor.dtype, ir_tensor.shape)
        value.const_value = ir_tensor
        self.model.graph.register_initializer(value)

    def make_node(self, op_type, inputs: Sequence[str], outputs: Sequence[str], *, name: str, domain="", **kwargs):
        assert name, "Node name must be provided"
        if name in self.node_names:
            # Note:
            #
            # This approach allows functions that make similar subgraphs with the same naming schema
            # to share existing nodes without needing to know whether the nodes already exist or not
            # (e.g. attention mask subgraphs).
            #
            # This means that the nodes can be created in those functions regardless of their actual
            # status in the graph. This checks can then decide whether the proposed node actually
            # needs to be added into the graph or not.
            return

        # Save any constants as nodes
        for input_name in inputs:
            if input_name.startswith("/model/constants") and input_name not in self.node_names:
                self.make_constant(input_name)

        # Resolve values from names
        input_values = [self.make_value(name) for name in inputs]
        output_values = [self.make_value(name) for name in outputs]
        node = ir.node(op_type, inputs=input_values, attributes=kwargs, domain=domain, outputs=output_values, name=name)
        self.model.graph.append(node)
        self.node_names.add(name)

    def make_value(self, name, dtype: ir.DataType | int| None = None, shape: Sequence[int | str] | ir.Shape | None = None) -> ir.Value:
        """Obtain or create an IR value by value name.

        If the value does not exist a new one is created.
        If dtype or shape is provided, it will be set on the value.

        Args:
            name: The name of the value.
            output: Whether the value is an output value.
        """
        if name == "":
            # None value
            return ir.Value(name="")
        value = self.values.setdefault(name, ir.Value(name=name))
        if dtype is not None:
            value.dtype = ir.DataType(dtype)
        if shape is not None:
            value.shape = ir.Shape(shape)
        return value

    def make_inputs_and_outputs(self):
        # Add model-specific inputs to list of model inputs
        inputs = self.model.graph.inputs
        for name in self.input_names:
            dtype = self.input_types[name]
            shape = self.input_shapes[name]
            inputs.append(self.make_value(name, dtype=dtype, shape=shape))

        # Add model-specific outputs to list of model outputs
        outputs = self.model.graph.outputs
        for name in self.output_names:
            dtype = self.output_types[name]
            shape = self.output_shapes[name]
            outputs.append(self.make_value(name, dtype=dtype, shape=shape))

        # Add KV cache to inputs and outputs
        for i in range(self.num_layers):
            # Add KV cache to inputs
            key_name = f"past_key_values.{i}.key"
            key_shape = self.make_key_value_cache_shape(i, self.input_shapes["past_key_values.key"])
            inputs.append(self.make_value(key_name, dtype=self.input_types["past_key_values.key"], shape=key_shape))

            value_name = f"past_key_values.{i}.value"
            value_shape = self.make_key_value_cache_shape(i, self.input_shapes["past_key_values.value"])
            inputs.append(self.make_value(value_name, dtype=self.input_types["past_key_values.value"], shape=value_shape))

            # Add KV cache to outputs
            key_name = f"present.{i}.key"
            key_shape = self.make_key_value_cache_shape(i, self.output_shapes["present.key"])
            outputs.append(self.make_value(key_name, dtype=self.output_types["present.key"], shape=key_shape))

            value_name = f"present.{i}.value"
            value_shape = self.make_key_value_cache_shape(i, self.output_shapes["present.value"])
            outputs.append(self.make_value(value_name, dtype=self.output_types["present.value"], shape=value_shape))

    def make_constant(self, name):
        # Make constant ops for 0, 1, 2, 3, etc.
        # Format of name is "/model/constants/{dtype}/{num}"

        path = name.split("/")
        onnx_dtype = ir.DataType[path[-2]]
        num = ast.literal_eval(path[-1])
        assert isinstance(num, (int, float, list, tuple)), f"Invalid constant value: {num}"
        tensor = ir.tensor(num, dtype=onnx_dtype, name=name)

        node_name = name.replace("constants", "constant_nodes")
        self.make_node("Constant", inputs=[], outputs=[name], name=node_name, value=tensor)
        self.make_value(name, onnx_dtype, shape=[])

    def make_gather(self, name, inputs, dtype, shape, axis):
        output = f"{name}/output_0"
        self.make_node("Gather", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_reshape(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Reshape", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_shape(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node("Shape", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, ir.DataType.INT64, shape=shape)

    def make_constant_of_shape(self, name, root_input, value, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("ConstantOfShape", inputs=[root_input], outputs=[output], name=name, value=value)
        self.make_value(output, dtype, shape=shape)

    def make_unsqueeze(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Unsqueeze", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_squeeze(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Squeeze", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_concat(self, name, inputs, dtype, shape, axis=0):
        output = f"{name}/output_0"
        self.make_node("Concat", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_tile(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Tile", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_equal(self, name, inputs, shape):
        output = f"{name}/output_0"
        self.make_node("Equal", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, ir.DataType.BOOL, shape=shape)

    def make_greater(self, name, inputs, shape):
        output = f"{name}/output_0"
        self.make_node("Greater", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, ir.DataType.BOOL, shape=shape)

    def make_greater_or_equal(self, name, inputs, shape):
        output = f"{name}/output_0"
        self.make_node("GreaterOrEqual", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, ir.DataType.BOOL, shape=shape)

    def make_isinf(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node("IsInf", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, ir.DataType.BOOL, shape=shape)

    def make_clip(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Clip", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_where(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Where", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_expand(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Expand", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_reduce_sum(self, name, inputs, dtype, shape, keepdims=True):
        output = f"{name}/output_0"
        self.make_node("ReduceSum", inputs=inputs, outputs=[output], name=name, keepdims=keepdims)
        self.make_value(output, dtype, shape=shape)

    def make_reduce_max(self, name, inputs, dtype, shape, keepdims=False):
        output = f"{name}/output_0"
        self.make_node("ReduceMax", inputs=inputs, outputs=[output], name=name, keepdims=keepdims)
        self.make_value(output, dtype, shape=shape)

    def make_reduce_mean(self, name, inputs, dtype, shape, keepdims=False):
        output = f"{name}/output_0"
        self.make_node("ReduceMean", inputs=inputs, outputs=[output], name=name, keepdims=keepdims)
        self.make_value(output, dtype, shape=shape)

    def make_sqrt(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sqrt", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_cast(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Cast", inputs=[root_input], outputs=[output], name=name, to=dtype)
        self.make_value(output, dtype, shape=shape)

    def make_add(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Add", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_sub(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sub", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_less(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Less", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, ir.DataType.BOOL, shape=None)

    def make_range(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Range", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, ir.DataType.INT64, shape=["unk"])

    def make_slice(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Slice", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_mul(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Mul", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_transpose(self, name, root_input, dtype, shape, perm):
        output = f"{name}/output_0"
        self.make_node("Transpose", inputs=[root_input], outputs=[output], name=name, perm=perm)
        self.make_value(output, dtype, shape=shape)

    def make_div(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Div", inputs=inputs, outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_tanh(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Tanh", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_softmax(self, name, root_input, dtype, shape, axis=-1):
        output = f"{name}/output_0"
        self.make_node("Softmax", inputs=[root_input], outputs=[output], name=name, axis=axis)
        self.make_value(output, dtype, shape=shape)

    def make_sigmoid(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sigmoid", inputs=[root_input], outputs=[output], name=name)
        self.make_value(output, dtype, shape=shape)

    def make_matmul(self, matmul, basename, root_input, **kwargs):
        if hasattr(matmul, "base_layer"):
            # For LoRA `MatMul`
            return self.make_matmul_lora(matmul, basename, root_input, **kwargs)
        else:
            # For regular `MatMul`
            return self.make_matmul_op(matmul, basename, root_input, **kwargs)

    def make_matmul_op(self, matmul, basename, root_input, **kwargs):
        if self.onnx_dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT}:
            return self.make_matmul_float(matmul, basename, root_input, **kwargs)
        elif self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            if self.quant_attrs["use_qdq"]:
                return self.make_matmul_int4_qdq(matmul, basename, root_input, **kwargs)
            else:
                return self.make_matmul_int4(matmul, basename, root_input, **kwargs)
        else:
            raise NotImplementedError(f"The {self.onnx_dtype} precision is not currently supported.")

    def make_matmul_float(self, matmul, name, root_input, **kwargs):
        weight = name[1:].replace("/", ".") + ".weight"
        self.make_initializer(matmul.weight.T, weight, to=self.io_dtype)

        last_dim = matmul.weight.shape[0]
        output = "logits" if kwargs.get("logits", False) else f"{name}/output_0"
        self.make_node("MatMul", inputs=[root_input, weight], outputs=[output], name=name)
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', last_dim])

        return name

    def make_matmul_int4(self, matmul, basename, root_input, **kwargs):
        if not hasattr(matmul, "qweight"):
            # TODO: quantize weights, then save new MatMul weights for onnx model
            # print(f"Quantizing to {self.onnx_dtype} on-the-fly is not currently supported.")
            # print(f"Saving as {self.io_dtype} on-the-fly and quantizing to {self.onnx_dtype} at the end.")
            return self.make_matmul_float(matmul, basename, root_input, **kwargs)

        name = f"{basename}NBits"

        # Input weights are quantized, save quantized MatMul weights for onnx model
        weight = name[1:].replace("/", ".") + ".qweight"
        self.make_initializer(matmul.qweight, weight)
        scales = name[1:].replace("/", ".") + ".scales"
        self.make_initializer(matmul.scales, scales, to=self.io_dtype)

        inputs = [root_input, weight, scales]

        if hasattr(matmul, "qzeros") and matmul.qzeros is not None:
            zeros = name[1:].replace("/", ".") + ".qzeros"
            self.make_initializer(matmul.qzeros, zeros)
            inputs.append(zeros)

        if hasattr(matmul, "g_idx") and matmul.g_idx is not None:
            g_idx = name[1:].replace("/", ".") + ".g_idx"
            self.make_initializer(matmul.g_idx, g_idx, to=ir.DataType.INT32)
            inputs.append(g_idx)

        output = "logits" if kwargs.get("logits", False) else f"{name}/output_0"
        self.make_node(
            "MatMulNBits", inputs=inputs, outputs=[output], name=name, domain="com.microsoft",
            accuracy_level=self.quant_attrs["int4"]["accuracy_level"],
            bits=matmul.bits, block_size=matmul.group_size, K=matmul.in_features, N=matmul.out_features,
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', matmul.out_features])

        return name

    def make_dequantize_linear(self, dequantize_name, quantized_op):
        # Input weights are quantized, save quantized MatMul weights for onnx model
        qweight = dequantize_name[1:].replace("/", ".") + ".qweight"
        qweight_shape = quantized_op.qweight.shape
        self.make_initializer(
            ir.PackedTensor(
                quantized_op.qweight,
                self.onnx_dtype,
                shape=[*qweight_shape[:-2], qweight_shape[-2] * qweight_shape[-1] * 2],
            ),
            qweight,
        )

        scales = dequantize_name[1:].replace("/", ".") + ".scales"
        scales_target_shape = [
            *qweight_shape[:-2],
            qweight_shape[-2] * qweight_shape[-1] * 2 // quantized_op.group_size,
        ]
        scales_pt = quantized_op.scales.to(to_torch_dtype(self.io_dtype))
        scales_pt = scales_pt.reshape(scales_target_shape)
        self.make_initializer(scales_pt, scales)

        dequantize_inputs = [qweight, scales]

        if hasattr(quantized_op, "qzeros") and quantized_op.qzeros is not None:
            zeros = dequantize_name[1:].replace("/", ".") + ".qzeros"
            self.make_initializer(
                ir.PackedTensor(
                    quantized_op.qzeros, self.onnx_dtype, shape=scales_target_shape
                ),
                zeros,
            )
            dequantize_inputs.append(zeros)

        dequantize_output = f"{dequantize_name}/output_0"
        self.make_node("DequantizeLinear", inputs=dequantize_inputs, outputs=[dequantize_output], name=dequantize_name, block_size=quantized_op.group_size, axis=-1)
        self.make_value(dequantize_output, self.io_dtype, shape=[*scales_pt.shape[:-1], scales_pt.shape[-1] * quantized_op.group_size])

        return dequantize_output

    def make_matmul_int4_qdq(self, matmul, matmul_name, root_input, **kwargs):
        if not hasattr(matmul, "qweight"):
            # TODO: quantize weights, then save new MatMul weights for onnx model
            # print(f"Quantizing to {self.onnx_dtype} on-the-fly is not currently supported.")
            # print(f"Saving as {self.io_dtype} on-the-fly and quantizing to {self.onnx_dtype} at the end.")
            return self.make_matmul_float(matmul, matmul_name, root_input, **kwargs)

        if matmul.bits != 4:
            # Code below assume 4 bits with hard coded shapes (* 2)
            raise NotImplementedError(f"{matmul.bits} bits precision is not currently supported in QDQ format.")

        dequantize_output = self.make_dequantize_linear(f"{matmul_name}/DequantizeLinear", matmul)

        # Add a transpose instead of transposing the weights offline. The reason for this is that it is more natural and usually more performant to
        # compute quantized matmul when the weights are transposed. In most implementations, the transpose should usually be converted to a "transposeB"
        # attribute on the MatMul itself. A more natural way to represent this would have been to use Gemm since it already supports a transB attribute,
        # but unfortunately Gemm doesn't support batches.
        qweight_shape = matmul.qweight.shape
        transposed_shape = [qweight_shape[1] * qweight_shape[2] * 2, qweight_shape[0]]
        transpose_name = f"{matmul_name}/Transpose"
        self.make_transpose(transpose_name, dequantize_output, self.io_dtype, transposed_shape, [1, 0])

        matmul_output = "logits" if kwargs.get("logits", False) else f"{matmul_name}/output_0"
        self.make_node("MatMul", inputs=[root_input, f"{transpose_name}/output_0"], outputs=[matmul_output], name=matmul_name)
        self.make_value(matmul_output, self.io_dtype, shape=['batch_size', 'sequence_length', matmul.out_features])

        return matmul_name

    def make_matmul_lora(self, matmul, basename, root_input, **kwargs):
        # Make nodes for the MatMul-LoRA subgraph
        #
        #            root_input
        #                |
        #         +------+------+
        #         |             |
        #   MatMul_LoRA_A     MatMul
        #         |             |
        #   MatMul_LoRA_B       |
        #         |             |
        #         +------+------+
        #                |
        #           Add_LoRA_Add

        basename_parts = basename.split("/")

        # Make LoRA MatMul path
        matmul_A_basename = "/".join(basename_parts[:-1] + ["lora_A"] + basename_parts[-1:])
        matmul_A_name = self.make_matmul_op(matmul.lora_A.default, matmul_A_basename, root_input=root_input)
        lora_A = f"{matmul_A_name}/output_0"

        matmul.lora_B.default.weight.requires_grad = False  # since a leaf variable is updated in-place
        matmul.lora_B.default.weight *= matmul.scaling["default"]
        matmul_B_basename = "/".join(basename_parts[:-1] + ["lora_B"] + basename_parts[-1:])
        matmul_B_name = self.make_matmul_op(matmul.lora_B.default, matmul_B_basename, root_input=lora_A)
        lora_B = f"{matmul_B_name}/output_0"

        # Make regular MatMul path
        last_dim = matmul.base_layer.weight.shape[0]
        matmul_name = self.make_matmul_op(matmul.base_layer, basename, root_input, **kwargs)

        # Make LoRA Add node
        add_name = "/".join(basename_parts[:-1] + ["lora", "Add"])
        add_inputs = [f"{matmul_name}/output_0", lora_B]
        add_shape = ["batch_size", "sequence_length", last_dim]
        self.make_add(add_name, add_inputs, dtype=self.io_dtype, shape=add_shape)

        return add_name

    def make_packed_matmul(self, q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs):
        if self.onnx_dtype in {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
            return self.make_packed_matmul_float(q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs)
        elif self.onnx_dtype in {ir.DataType.INT4, ir.DataType.UINT4}:
            return self.make_packed_matmul_int4(q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs)
        else:
            raise NotImplementedError(f"The {self.onnx_dtype} precision is not currently supported.")

    def make_packed_matmul_float(self, q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs):
        # N_q = num_attention_heads * head_size, N_kv = num_key_value_heads * head_size, H = hidden_size
        # Combine 3 MatMuls of shape N_q x H, N_kv x H, N_kv x H into 1 packed MatMul of shape (N_q+N_kv+N_kv)xH
        # Note: Packed MatMul is of shape (N_q+N_kv+N_kv)xH instead of Hx(N_q+N_kv+N_kv) because `make_matmul` will
        # apply a transpose before saving
        N_q, H = q_matmul.weight.shape
        N_kv, _ = k_matmul.weight.shape

        # Create dummy PackedMatMul class
        class PackedMatMul:
            def __init__(self):
                self.weight = torch.cat([q_matmul.weight, k_matmul.weight, v_matmul.weight], dim=0).reshape(N_q + N_kv + N_kv, H)
        matmul = PackedMatMul()
        new_name = self.make_matmul(matmul, basename, root_input, **kwargs)

        return new_name

    def make_packed_matmul_int4(self, q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs):
        if not hasattr(q_matmul, "qweight"):
            # TODO: quantize weights, then save new MatMul weights for onnx model
            # print(f"Quantizing to {self.onnx_dtype} on-the-fly is not currently supported.")
            # print(f"Saving as {self.io_dtype} on-the-fly and quantizing to {self.onnx_dtype} at the end.")
            return self.make_packed_matmul_float(q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs)

        # Create dummy PackedMatMul class
        class PackedMatMul:
            def __init__(self):
                if q_matmul.bits != k_matmul.bits or q_matmul.bits != v_matmul.bits:
                    raise ValueError("All MatMuls must have the same bits for packed MatMul.")
                if q_matmul.group_size != k_matmul.group_size or q_matmul.group_size != v_matmul.group_size:
                    raise ValueError("All MatMuls must have the same group size for packed MatMul.")
                self.qweight = torch.cat([q_matmul.qweight, k_matmul.qweight, v_matmul.qweight], dim=0)
                self.scales = torch.cat([q_matmul.scales, k_matmul.scales, v_matmul.scales], dim=0)
                self.qzeros = torch.cat([q_matmul.qzeros, k_matmul.qzeros, v_matmul.qzeros], dim=0)
                self.g_idx = q_matmul.g_idx

                self.in_features = q_matmul.in_features
                self.out_features = q_matmul.out_features + k_matmul.out_features + v_matmul.out_features
                self.bits = q_matmul.bits
                self.group_size = q_matmul.group_size
        matmul = PackedMatMul()
        new_name = self.make_matmul_int4(matmul, basename, root_input, **kwargs)

        return new_name

    def make_add_bias(self, add, name, root_input, **kwargs):
        bias = name[1:].replace("/", ".") + ".bias"
        self.make_initializer(add, bias, to=self.io_dtype)

        add_bias_inputs = [root_input, bias]
        shape = ['batch_size', 'sequence_length', add.shape[0]]

        if kwargs.get("logits", False):
            output = "logits"
            self.make_node("Add", inputs=add_bias_inputs, outputs=[output], name=name)
            self.make_value(output, dtype=self.io_dtype, shape=shape)
        else:
            self.make_add(name, add_bias_inputs, dtype=self.io_dtype, shape=shape)

    def make_packed_add(self, q_add, k_add, v_add, name, root_input, **kwargs):
        # Combine 3 Adds of shape N_q, N_kv, and N_kv into 1 packed Add of shape N_q + N_kv + N_kv
        add = torch.cat([q_add, k_add, v_add], dim=0).flatten()
        self.make_add_bias(add, name, root_input, **kwargs)

    def make_embedding(self, embedding):
        basename = "/model/embed_tokens"
        if self.int4_tied_embeddings:
            gather_name = f"{basename}/GatherBlockQuantized"
            gather_output = f"{gather_name}/output_0"

            weight_reshape_name = f"{basename}/Reshape"
            bits = 8 if self.int8_lm_head else 4
            weight_reshape_inputs = [f"lm_head.MatMul.weight_Q{bits}G{self.int4_block_size}", f"/model/constants/INT64/[{self.vocab_size}, {self.hidden_size}]"]
            weight_reshape_output = f"{weight_reshape_name}/output_0"
            # quantized weight dtype is uint8, see here
            # https://github.com/microsoft/onnxruntime/blob/0c9356cb986fd4cd2c5d510909d31186010ba226/onnxruntime/python/tools/quantization/neural_compressor/weight_only.py#L73
            self.make_reshape(weight_reshape_name, weight_reshape_inputs, dtype=ir.DataType.UINT8, shape=['vocab_size', 'hidden_size'])

            self.make_node('GatherBlockQuantized', inputs=[weight_reshape_output, 'input_ids', 'lm_head.MatMul.weight_scale', 'lm_head.MatMul.weight_zp'], outputs=[gather_output], name=gather_name, domain="com.microsoft", bits=bits, block_size=int(self.int4_block_size))
        else:
            weight = "model.embed_tokens.weight"
            self.make_initializer(embedding, weight, to=self.io_dtype)

            gather_name = f"{basename}/Gather"
            gather_output = f"{gather_name}/output_0"
            self.make_node('Gather', inputs=[weight, 'input_ids'], outputs=[gather_output], name=gather_name)

        self.make_value(gather_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        if self.embed_attrs["scale"] != 1:
            # Scale the embeddings
            mul_name = f"{basename}/Mul"
            mul_inputs = [gather_output, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.embed_attrs['scale']}"]
            mul_output = f"{mul_name}/output_0"
            self.make_node('Mul', inputs=mul_inputs, outputs=[mul_output], name=mul_name)
            self.make_value(mul_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

            layernorm_attrs_value = mul_output
        else:
            layernorm_attrs_value = gather_output

        if self.layernorm_attrs["cast"]["use_fp32"] and self.io_dtype != ir.DataType.FLOAT:
            # Insert output Cast node
            cast_name = f"{basename}/Cast"
            self.make_cast(cast_name, layernorm_attrs_value, ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.hidden_size])
            layernorm_attrs_value = f"{cast_name}/output_0"

        self.layernorm_attrs["root_input"] = layernorm_attrs_value
        self.layernorm_attrs["skip_input"] = layernorm_attrs_value

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if self.ep == "trt-rtx" and (skip or simple):
            # Fall back to primitive ops
            self._make_layernorm_op(layer_id, layernorm, skip, simple, location)
        else:
            self.make_layernorm_op(layer_id, layernorm, skip, simple, location)

    def make_layernorm_op(self, layer_id, layernorm, skip, simple, location):
        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        # Get precision types to use
        old_io_dtype = self.io_dtype
        new_io_dtype = ir.DataType.FLOAT if self.layernorm_attrs["cast"]["use_fp32"] else self.io_dtype
        cast = old_io_dtype != new_io_dtype

        # Create weight and bias tensors
        weight = f"model.layers.{layer_id}.{location}_layernorm.weight"
        self.make_initializer(
            layernorm.weight + self.layernorm_attrs["add_offset"],
            weight,
            to=new_io_dtype
        )
        bias = f"model.layers.{layer_id}.{location}_layernorm.bias"
        if not simple:
            self.make_initializer(layernorm.bias, bias, to=new_io_dtype)

        # Create input names for op
        inputs = [root_input, skip_input, weight] if skip else [root_input, weight]
        if not simple:
            inputs.append(bias)

        name = f"/model/layers.{layer_id}/{location}_layernorm/{'Skip' if skip else ''}LayerNorm"
        op_type = f"{'Skip' if skip else ''}{'Simplified' if simple else ''}LayerNormalization"
        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        if not skip:
            kwargs.update({"axis": -1, "stash_type": 1})

        # Create output names for op
        output_0 = f"/model/layers.{layer_id}/{location}_layernorm/output_0"
        output_3 = f"/model/layers.{layer_id}/{location}_layernorm/output_3"
        if self.layernorm_attrs["last_layernorm"] and (self.include_hidden_states or self.exclude_lm_head):
            output_0 = "hidden_states"
        outputs = [output_0, "", "", output_3] if skip and not self.layernorm_attrs["last_layernorm"] else [output_0]

        # Create Cast nodes for inputs and outputs if old_dtype != new_dtype
        if cast:
            inputs, outputs = self.make_layernorm_casts(name, inputs, outputs, old_io_dtype, new_io_dtype)

        # Make op and its shape
        self.make_node(op_type, inputs=inputs, outputs=outputs, name=name, domain=("com.microsoft" if skip else None), **kwargs)
        self.make_value(outputs[0], new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.make_value(outputs[3], new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        # Update LayerNorm attributes
        self.layernorm_attrs["output_0"] = output_0
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.layernorm_attrs["output_3"] = output_3

            # Assign output 3 of current SkipLayerNorm as root input to next SkipLayerNorm
            self.layernorm_attrs["root_input"] = output_3

    def _make_layernorm_op(self, layer_id, layernorm, skip, simple, location):
        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        # Get precision types to use
        old_io_dtype = self.io_dtype
        new_io_dtype = ir.DataType.FLOAT if self.layernorm_attrs["cast"]["use_fp32"] else self.io_dtype
        cast = old_io_dtype != new_io_dtype

        # Create weight and bias tensors
        weight = f"model.layers.{layer_id}.{location}_layernorm.weight"
        self.make_initializer(
            layernorm.weight + self.layernorm_attrs["add_offset"],
            weight,
            to=new_io_dtype
        )
        bias = f"model.layers.{layer_id}.{location}_layernorm.bias"
        if not simple:
            self.make_initializer(layernorm.bias, bias, to=new_io_dtype)

         # Create input names for op
        inputs = [root_input, skip_input, weight] if skip else [root_input, weight]
        if not simple:
            inputs.append(bias)

        name = f"/model/layers.{layer_id}/{location}_layernorm/{'Skip' if skip else ''}LayerNorm"
        op_type = f"{'Skip' if skip else ''}{'Simplified' if simple else ''}LayerNormalization"
        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        if not skip:
            kwargs.update({"axis": -1, "stash_type": 1})

        # Create output names for op
        output_0 = f"/model/layers.{layer_id}/{location}_layernorm/output_0"
        output_3 = f"/model/layers.{layer_id}/{location}_layernorm/output_3"
        if self.layernorm_attrs["last_layernorm"] and (self.include_hidden_states or self.exclude_lm_head):
            output_0 = "hidden_states"
        outputs = [output_0, "", "", output_3] if skip and not self.layernorm_attrs["last_layernorm"] else [output_0]

        # Create Cast nodes for inputs and outputs if old_dtype != new_dtype
        if cast:
            inputs, outputs = self.make_layernorm_casts(name, inputs, outputs, old_io_dtype, new_io_dtype)
            root_input = inputs[0]
            skip_input = inputs[1] if skip else None

        if op_type == "SimplifiedLayerNormalization":
            self._make_simplified_layer_norm(name, root_input, weight, outputs[0], new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        elif op_type == "SkipSimplifiedLayerNormalization":
            self._make_skip_simplified_layer_norm(name, root_input, skip_input, weight, outputs[0], output_3, new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        elif op_type == "SkipLayerNormalization":
            self._make_skip_layer_norm(name, root_input, skip_input, weight, bias, outputs[0], output_3, new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        else:
            raise ValueError(f"Invalid op_type: {op_type}")

        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.make_value(outputs[3], new_io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        # Update LayerNorm attributes
        self.layernorm_attrs["output_0"] = output_0
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.layernorm_attrs["output_3"] = output_3

            # Assign output 3 of current SkipLayerNorm as root input to next SkipLayerNorm
            self.layernorm_attrs["root_input"] = output_3

    def make_layernorm_casts(self, name, inputs, outputs, old_dtype, new_dtype):
        # Name = name of original LayerNorm op as if the cast nodes did not exist
        # Inputs = inputs into the original LayerNorm op as if the cast nodes did not exist
        # Outputs = outputs from the original LayerNorm op as if the cast nodes did not exist

        # Save original inputs and outputs
        skip = len(inputs) > 2  # [root_input, skip_input, weight] vs. [root_input, weight]
        root_input = inputs[0]
        skip_input = inputs[1] if skip else None
        output_0 = outputs[0]
        output_3 = outputs[3] if skip and not self.layernorm_attrs["last_layernorm"] else None

        root_input_shape = self.values[root_input].shape

        if self.layernorm_attrs["cast"]["root_input"] and self.values[root_input].dtype != new_dtype:
            # Cast root_input
            root_input_cast_name = f"{name}/root_input/Cast"
            root_input_cast_output = f"{root_input_cast_name}/output_0"
            self.make_node("Cast", inputs=[root_input], outputs=[root_input_cast_output], name=root_input_cast_name, to=new_dtype)
            self.make_value(root_input_cast_output, new_dtype, shape=root_input_shape)
            inputs[0] = root_input_cast_output

        if skip and self.layernorm_attrs["cast"]["skip_input"] and self.values[skip_input].dtype != new_dtype:
            # Cast skip_input
            assert skip_input is not None
            skip_input_cast_name = f"{name}/skip_input/Cast"
            skip_input_cast_output = f"{skip_input_cast_name}/output_0"
            self.make_node("Cast", inputs=[skip_input], outputs=[skip_input_cast_output], name=skip_input_cast_name, to=new_dtype)
            self.make_value(skip_input_cast_output, new_dtype, shape=self.values[skip_input].shape)
            inputs[1] = skip_input_cast_output

        if self.layernorm_attrs["cast"]["output_0"]:
            # Cast output_0
            output_0_cast_name = f"{name}/output_0/Cast"
            output_0_cast_output = f"{output_0_cast_name}/output_0"
            self.make_node("Cast", inputs=[output_0_cast_output], outputs=[output_0], name=output_0_cast_name, to=old_dtype)
            self.make_value(output_0, old_dtype, shape=root_input_shape)
            outputs[0] = output_0_cast_output

        if skip and not self.layernorm_attrs["last_layernorm"] and self.layernorm_attrs["cast"]["output_3"]:
            # Cast output_3
            assert output_3 is not None
            output_3_cast_name = f"{name}/output_3/Cast"
            output_3_cast_output = f"{output_3_cast_name}/output_3"
            self.make_node("Cast", inputs=[output_3_cast_output], outputs=[output_3], name=output_3_cast_name, to=old_dtype)
            self.make_value(output_3, old_dtype, shape=root_input_shape)
            outputs[3] = output_3_cast_output

        return inputs, outputs

    def make_mscale_su(self, mscale):
        if mscale <= 1.0:
            return 1.0
        return np.sqrt(1 + np.log(mscale) / np.log(self.original_context_length))

    def make_mscale_yarn(self, mscale):
        if mscale <= 1.0:
            return 1.0
        return 0.1 * np.log(mscale) + 1.0

    def make_mscale(self, mscale):
        if self.rope_attrs["mscale_policy"] in {"su", "longrope"}:
            return self.make_mscale_su(mscale)
        elif self.rope_attrs["mscale_policy"] == "yarn":
            return self.make_mscale_yarn(mscale)
        else:
            return float(mscale)

    def make_inv_freq_rescaled(self, inv_freq):
        if "low_freq_factor" in self.rope_attrs["rescale_inv_freq"]:
            return self.make_inv_freq_rescaled_with_freq_factors(inv_freq)
        elif "ntk_alpha" in self.rope_attrs["rescale_inv_freq"]:
            return self.make_inv_freq_rescaled_with_ntk(inv_freq)
        else:
            raise NotImplementedError(f"The method to rescale inv_freq could not be identified.")

    def make_inv_freq_rescaled_with_freq_factors(self, inv_freq):
        scale_factor = self.rope_attrs["rescale_inv_freq"]["factor"]
        low_freq_factor = self.rope_attrs["rescale_inv_freq"]["low_freq_factor"]
        high_freq_factor = self.rope_attrs["rescale_inv_freq"]["high_freq_factor"]
        old_context_len = self.original_context_length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * torch.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=inv_freq.dtype)

    def make_inv_freq_rescaled_with_ntk(self, inv_freq):
        d_half = self.head_size / 2
        # NTK by parts
        low = (
            d_half
            * np.log(self.original_context_length / (self.rope_attrs["rescale_inv_freq"]["ntk_beta"] * 2 * np.pi))
            / np.log(self.rope_attrs["theta"])
        )
        high = (
            d_half
            * np.log(self.original_context_length / (self.rope_attrs["rescale_inv_freq"]["ntk_alpha"] * 2 * np.pi))
            / np.log(self.rope_attrs["theta"])
        )
        assert 0 < low < high < d_half - 1

        interpolation = 1.0 / (self.rope_attrs["rescale_inv_freq"]["factor"] * inv_freq)
        extrapolation = 1.0 / inv_freq

        ramp = (
            torch.arange(d_half, dtype=torch.float32, device=inv_freq.device) - low
        ) / (high - low)
        mask = 1 - ramp.clamp(0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        return inv_freq

    def make_rotary_embedding_caches_from_scratch(self):
        dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        inv_freq = 1.0 / (self.rope_attrs["rescale_factors"] * (self.rope_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)))
        if "rescale_inv_freq" in self.rope_attrs:
            inv_freq = self.make_inv_freq_rescaled(inv_freq)

        position_scale = self.rope_attrs["position_scale"] if self.context_length == self.original_context_length else 1
        t = (torch.arange(self.rope_attrs["cache_length"], dtype=self.rope_attrs["t_dtype"]) * position_scale).type_as(inv_freq)

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cache, sin_cache = emb.cos() * self.rope_attrs["mscale"], emb.sin() * self.rope_attrs["mscale"]
        return cos_cache, sin_cache

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get("cos_cache_name", "cos_cache")
        sin_cache_name = kwargs.get("sin_cache_name", "sin_cache")

        if self.rope_attrs["create_caches"]:
            # Create cos/sin caches if not already created
            cos_cache, sin_cache = self.make_rotary_embedding_caches_from_scratch()

            # Remove any dims of size 1 and cast to target dtype
            cos_cache = cos_cache.squeeze().to(to_torch_dtype(self.io_dtype))
            sin_cache = sin_cache.squeeze().to(to_torch_dtype(self.io_dtype))

            # Slice cos/sin caches from (M, H) to (M, H/2) if hidden dim = head size (i.e. if cos/sin caches haven't been halved yet)
            hidden_dim = cos_cache.shape[-1]
            if hidden_dim == self.head_size:
                cos_cache = cos_cache[:, : (hidden_dim // 2)]
                sin_cache = sin_cache[:, : (hidden_dim // 2)]

            # Slice cos/sin caches from (M, H/2) to (M, R/2) if partial rotary embeddings are used
            if self.rope_attrs["partial_rotary_factor"] != 1.0:
                cos_cache = cos_cache[:, : (self.rope_attrs["rotary_embedding_dim"] // 2)]
                sin_cache = sin_cache[:, : (self.rope_attrs["rotary_embedding_dim"] // 2)]

            self.rope_attrs["create_caches"] = False

            if self.rope_attrs["save_caches"]:
                # Save cos/sin caches to disk
                self.make_initializer(cos_cache, cos_cache_name)
                self.make_initializer(sin_cache, sin_cache_name)
            else:
                # Return cos/sin caches since they will be custom-saved
                return cos_cache, sin_cache

        return cos_cache_name, sin_cache_name

    def make_padded_cache(self, small_cache, large_cache, pad_value=0.0):
        """Pad small cache to match large cache shape for uniform If node branches.

        This is used for TRT-RTX EP which requires uniform dimensions in both branches of If nodes.

        Args:
            small_cache: The smaller cache tensor to pad
            large_cache: The larger cache tensor (defines target shape)
            pad_value: Value to use for padding (1.0 for cos_cache, 0.0 for sin_cache)
        """
        target_shape = large_cache.shape
        if small_cache.shape == target_shape:
            return small_cache

        # Create padded tensor filled with pad_value
        padded_cache = torch.full(target_shape, pad_value, dtype=small_cache.dtype)
        # Copy original data to the beginning
        padded_cache[:small_cache.shape[0], :] = small_cache
        return padded_cache

    def _make_split_if_nodes_for_trt_rtx(self, basename, greater_name,
                                          cos_cache_name, sin_cache_name,
                                          cos_cache_large, sin_cache_large,
                                          cos_cache_small, sin_cache_small,
                                          cos_cache_large_name, sin_cache_large_name,
                                          cos_cache_small_name, sin_cache_small_name,
                                          small_cache_shape):
        """Create split If nodes for TRT-RTX to workaround trt-rtx multi-output bug.

        This is a TEMPORARY workaround for TRT-RTX bug where If nodes with
        multiple outputs

        Creates two separate If nodes instead of one:
        - {basename}/cos/If: Outputs cos_cache only
        - {basename}/sin/If: Outputs sin_cache only

        Both If nodes use the same condition and independently select their respective caches.
        """
        cos_if_name = f"{basename}/cos/If"

        cos_large_for_split = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=f"{cos_cache_large_name}_split", type=ir.TensorType(self.io_dtype), shape=ir.Shape(cos_cache_large.shape))
            ],
            name=f"/large/cos_cache/Constant_split_cos", attributes=dict(value=ir.tensor(cos_cache_large)))

        cos_small_for_split = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=f"{cos_cache_small_name}_split", type=ir.TensorType(self.io_dtype), shape=ir.Shape(small_cache_shape))
            ],
            name=f"/small/cos_cache/Constant_split_cos", attributes=dict(value=ir.tensor(cos_cache_small)))

        self.make_node(
            "If", inputs=[f"{greater_name}/output_0"], outputs=[cos_cache_name], name=cos_if_name,
            then_branch=ir.Graph(
                inputs=[],
                outputs=[cos_large_for_split.outputs[0]],
                nodes=[cos_large_for_split],
                name="large_cos_cache_graph",
            ),
            else_branch=ir.Graph(
                inputs=[],
                outputs=[cos_small_for_split.outputs[0]],
                nodes=[cos_small_for_split],
                name="small_cos_cache_graph",
            ),
        )

        # Create separate If node for sin_cache only
        sin_if_name = f"{basename}/sin/If"

        # Create unique constant nodes for sin to avoid tensor sharing
        sin_large_for_split = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=f"{sin_cache_large_name}_split", type=ir.TensorType(self.io_dtype), shape=ir.Shape(sin_cache_large.shape))
            ],
            name=f"/large/sin_cache/Constant_split_sin", attributes=dict(value=ir.tensor(sin_cache_large)))

        sin_small_for_split = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=f"{sin_cache_small_name}_split", type=ir.TensorType(self.io_dtype), shape=ir.Shape(small_cache_shape))
            ],
            name=f"/small/sin_cache/Constant_split_sin", attributes=dict(value=ir.tensor(sin_cache_small)))

        self.make_node(
            "If", inputs=[f"{greater_name}/output_0"], outputs=[sin_cache_name], name=sin_if_name,
            then_branch=ir.Graph(
                inputs=[],
                outputs=[sin_large_for_split.outputs[0]],
                nodes=[sin_large_for_split],
                name="large_sin_cache_graph",
            ),
            else_branch=ir.Graph(
                inputs=[],
                outputs=[sin_small_for_split.outputs[0]],
                nodes=[sin_small_for_split],
                name="small_sin_cache_graph",
            ),
        )

        # Create output values
        self.make_value(cos_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])
        self.make_value(sin_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])

    def make_rotary_embedding(self, name, root_input, **kwargs):
        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        num_heads = self.num_kv_heads if "k_rotary" in name else self.num_attn_heads

        inputs = [root_input, kwargs.pop("position_ids"), cos_cache_name, sin_cache_name]
        output = f"{name}/output_0"
        self.make_node(
            "RotaryEmbedding", inputs=inputs, outputs=[output], name=name, domain="com.microsoft",
            interleaved=self.rope_attrs["interleaved"], num_heads=(0 if self.rope_attrs["partial_rotary_factor"] == 1.0 else num_heads),  # default is 0 in RotaryEmbedding kernel
            rotary_embedding_dim=self.rope_attrs["rotary_embedding_dim"],
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * num_heads])

    def make_rotary_embedding_multi_cache(self, **kwargs):
        cos_cache_name = kwargs.get("cos_cache_name", "cos_cache")
        sin_cache_name = kwargs.get("sin_cache_name", "sin_cache")

        # Set cache attributes for when sequence_length > self.original_context_length
        self.rope_attrs["rescale_factors"] = self.rope_attrs["multi_cache"]["long_factor"]
        self.rope_attrs["cache_length"] = self.context_length
        self.rope_attrs["mscale"] = self.rope_attrs["multi_cache"]["long_mscale"]

        # Create caches for when sequence_length > self.original_context_length
        cos_cache_large_name, sin_cache_large_name = "cos_cache_large", "sin_cache_large"
        self.rope_attrs["save_caches"] = False
        cos_cache_large, sin_cache_large = self.make_rotary_embedding_caches(cos_cache_name=cos_cache_large_name, sin_cache_name=sin_cache_large_name)

        # Set cache attributes for when sequence_length <= self.original_context_length
        self.rope_attrs["rescale_factors"] = self.rope_attrs["multi_cache"]["short_factor"]
        self.rope_attrs["cache_length"] = self.original_context_length
        self.rope_attrs["mscale"] = self.rope_attrs["multi_cache"]["short_mscale"]
        self.rope_attrs["create_caches"] = True

        # Create caches for when sequence_length <= self.original_context_length
        cos_cache_small_name, sin_cache_small_name = "cos_cache_small", "sin_cache_small"
        self.rope_attrs["save_caches"] = False
        cos_cache_small, sin_cache_small = self.make_rotary_embedding_caches(cos_cache_name=cos_cache_small_name, sin_cache_name=sin_cache_small_name)

        # Determine which EPs don't support the If operator
        self.eps_without_if_support = ["dml"]
        if self.extra_options.get("enable_webgpu_graph", False):
            self.eps_without_if_support.append("webgpu")

        if self.ep in self.eps_without_if_support:
            # Concat small and large cos/sin caches for DML and WebGPU (when graph enabled) EPs
            # These EPs don't support the If operator
            cos_cache = torch.cat((cos_cache_small, cos_cache_large), dim=0)
            sin_cache = torch.cat((sin_cache_small, sin_cache_large), dim=0)
            # Save cos/sin caches to disk
            self.make_initializer(cos_cache, cos_cache_name)
            self.make_initializer(sin_cache, sin_cache_name)
            # Do NOT make the subgraph with the If node for DML EP.
            return

        # TRT-RTX: Apply padding and create split If nodes with early return
        if self.ep == "trt-rtx":
            # Pad small caches to match large cache dimensions
            # Pad cos_cache with 1s (cos(0)=1) and sin_cache with 0s (sin(0)=0)
            cos_cache_small = self.make_padded_cache(cos_cache_small, cos_cache_large, pad_value=1.0)
            sin_cache_small = self.make_padded_cache(sin_cache_small, sin_cache_large, pad_value=0.0)

            # Create Greater condition node for If nodes
            basename = "/model/rotemb_caches_subgraph"
            gather_name = ""
            if self.attention_attrs["op_type"] == "GroupQueryAttention":
                gather_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather"
            else:
                gather_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather_2"

            greater_name = f"{basename}/Greater"
            greater_inputs = [f"{gather_name}/output_0", f"/model/constants/INT64/{self.original_context_length}"]
            self.make_greater(greater_name, greater_inputs, shape=[])

            # Create split If nodes and return early
            self._make_split_if_nodes_for_trt_rtx(
                basename=basename,
                greater_name=greater_name,
                cos_cache_name=cos_cache_name,
                sin_cache_name=sin_cache_name,
                cos_cache_large=cos_cache_large,
                sin_cache_large=sin_cache_large,
                cos_cache_small=cos_cache_small,
                sin_cache_small=sin_cache_small,
                cos_cache_large_name=cos_cache_large_name,
                sin_cache_large_name=sin_cache_large_name,
                cos_cache_small_name=cos_cache_small_name,
                sin_cache_small_name=sin_cache_small_name,
                small_cache_shape=cos_cache_large.shape
            )
            return

        # For other EPs (CUDA, CPU, WebGPU), create regular If node with multiple outputs
        # Make the following subgraph to decide which cos/sin caches to use in the rotary embeddings
        #
        # attention_mask --> Shape --> Gather --> Greater --> If --> (cos_cache, sin_cache)
        #                             (idx=1)
        #

        basename = "/model/rotemb_caches_subgraph"
        gather_name = ""
        if self.attention_attrs["op_type"] == "GroupQueryAttention":
            gather_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather"
        else:
            gather_name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather_2"

        greater_name = f"{basename}/Greater"
        greater_inputs = [f"{gather_name}/output_0", f"/model/constants/INT64/{self.original_context_length}"]
        self.make_greater(greater_name, greater_inputs, shape=[])
        if_name = f"{basename}/If"

        cos_cache_large_node = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=cos_cache_large_name, type=ir.TensorType(self.io_dtype), shape=ir.Shape(cos_cache_large.shape))
            ],
            name="/large/cos_cache/Constant", attributes=dict(value=ir.tensor(cos_cache_large)))
        sin_cache_large_node = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=sin_cache_large_name, type=ir.TensorType(self.io_dtype), shape=ir.Shape(sin_cache_large.shape))
            ],
            name="/large/sin_cache/Constant", attributes=dict(value=ir.tensor(sin_cache_large)))
        cos_cache_small_node = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=cos_cache_small_name, type=ir.TensorType(self.io_dtype), shape=ir.Shape(cos_cache_small.shape))
            ],
            name="/small/cos_cache/Constant", attributes=dict(value=ir.tensor(cos_cache_small)))
        sin_cache_small_node = ir.node(
            "Constant", [], outputs=[
                ir.Value(name=sin_cache_small_name, type=ir.TensorType(self.io_dtype), shape=ir.Shape(sin_cache_small.shape))
            ],
            name="/small/sin_cache/Constant", attributes=dict(value=ir.tensor(sin_cache_small)))

        # Create single If node with multiple outputs
        self.make_node(
            "If", inputs=[f"{greater_name}/output_0"], outputs=[cos_cache_name, sin_cache_name], name=if_name,
            then_branch=ir.Graph(
                inputs=[],
                outputs=[
                    cos_cache_large_node.outputs[0],
                    sin_cache_large_node.outputs[0],
                ],
                nodes=[
                    cos_cache_large_node,
                    sin_cache_large_node,
                ],
                name="large_rotemb_caches_graph",
            ),
            else_branch=ir.Graph(
                inputs=[],
                outputs=[
                    cos_cache_small_node.outputs[0],
                    sin_cache_small_node.outputs[0],
                ],
                nodes=[
                    cos_cache_small_node,
                    sin_cache_small_node,
                ],
                name="small_rotemb_caches_graph",
            ),
        )
        self.make_value(cos_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])
        self.make_value(sin_cache_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])

    # This expansion of contrib-op can be updated / deprecated in future.
    def _make_skip_simplified_layer_norm(self, basename, root_input, skip_input, weight_name, output_0, output_3, io_dtype, shape):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      SimplifiedLayerNorm----> output (0)
        make_add_name = f"{basename}/Add"
        output_3 = f"{basename}/Add/output_0" if output_3 is None else output_3
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        make_simplified_layer_norm_name = f"{basename}/skip_simplified_layer_norm"
        self._make_simplified_layer_norm(make_simplified_layer_norm_name, output_3, weight_name, output_0, io_dtype, shape=shape)

    # This expansion contrib-op can be updated / deprecated in the future.
    def _make_skip_layer_norm(self, basename, root_input, skip_input, weight_name, bias_name, output_0, output_3, io_dtype, shape):
        #                          root_input         skip_input
        #                              |                  |
        #                              +------------------+
        #                              |
        #                             Add-------------> output (1)
        #                              |
        #                      LayerNormalization-----> output (0)
        output_3 = f"{basename}/Add/output_0" if output_3 is None else output_3
        make_add_name = f"{basename}/Add"
        self.make_node("Add", inputs=[root_input, skip_input], outputs=[output_3], name=make_add_name)
        self.make_value(output_3, io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        make_layer_norm_name = f"{basename}/LayerNormalization"
        inputs = [output_3, weight_name, bias_name]

        kwargs = {"epsilon": self.layernorm_attrs["epsilon"]}
        kwargs.update({"axis": -1, "stash_type": 1})

        self.make_node("LayerNormalization", inputs=inputs, outputs=[output_0], name=make_layer_norm_name, **kwargs)
        self.make_value(output_0, io_dtype, shape=shape)

    # This expansion contrib-op can be updated / deprecated in the future.
    def _make_simplified_layer_norm(self, basename, root_input, weight_name, output_0, io_dtype, shape):

        #                            Cast (float32) - most calc happens in higher precision
        #                              |
        #                      +-------+-------+
        #                      |               |
        #                     Pow              |
        #                      |               |
        #                  ReduceMean          |
        #                      |               |
        #                     Add              |
        #                      |               |
        #                    Sqrt              |
        #                      |               |
        #                     Div              |
        #                      |               |
        #                      +-------+-------+
        #                              |
        #                             Mul
        #                              |
        #                            Cast_1 (io_dtype - float16)
        #                              |
        #                            Mul_1

        make_cast_name = f"{basename}/Cast"
        self.make_cast(make_cast_name, root_input, ir.DataType.FLOAT, shape=shape)

        make_pow_name = f"{basename}/Pow"
        make_pow_inputs = [f"{make_cast_name}/output_0", "/model/constants/FLOAT/2"]

        self.make_node("Pow", inputs=make_pow_inputs, outputs=[f"{make_pow_name}/output_0"], name=make_pow_name, domain="")
        self.make_value(f"{make_pow_name}/output_0", ir.DataType.FLOAT, shape=shape)

        make_reducemean_name = f"{basename}/ReduceMean"
        make_reducemean_inputs = [f"{make_pow_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_reduce_mean(make_reducemean_name, make_reducemean_inputs, ir.DataType.FLOAT, keepdims=True, shape=shape)

        make_add_name = f"{basename}/Add"
        make_add_inputs = [f"{make_reducemean_name}/output_0", f"/model/constants/FLOAT/{self.layernorm_attrs['epsilon']}"]
        self.make_add(make_add_name, make_add_inputs, ir.DataType.FLOAT, shape=shape)

        make_sqrt_name = f"{basename}/Sqrt"
        make_sqrt_inputs = [f"{make_add_name}/output_0"]
        self.make_sqrt(make_sqrt_name, make_sqrt_inputs, ir.DataType.FLOAT, shape=shape)

        make_div_name = f"{basename}/Div"
        make_div_inputs = ["/model/constants/FLOAT/1", f"{make_sqrt_name}/output_0"]
        self.make_div(make_div_name, make_div_inputs, ir.DataType.FLOAT, shape=shape)

        make_mul_name = f"{basename}/Mul"
        make_mul_inputs = [f"{make_div_name}/output_0", f"{make_cast_name}/output_0"]
        self.make_mul(make_mul_name, make_mul_inputs, ir.DataType.FLOAT, shape=shape)

        make_cast_1_name = f"{basename}/Cast_1"
        self.make_cast(make_cast_1_name, f"{make_mul_name}/output_0", dtype=io_dtype, shape=shape)

        make_mul_1_name = f"{basename}/Mul_1"
        make_mul_1_inputs = [f"{make_cast_1_name}/output_0", weight_name]

        self.make_node("Mul", inputs=make_mul_1_inputs, outputs=[output_0], name=make_mul_1_name)
        self.make_value(output_0, dtype=io_dtype, shape=shape)


    def make_qk_norm(self, layer_id, attention):
        # Make subgraph to compute SimplifiedLayerNorm after Q and K MatMuls in attention:
        #
        #     root_input (BxSxD)
        #          |
        #       Reshape (BxSxNxH)
        #          |
        #  SimplifiedLayerNorm (BxSxNxH)
        #          |
        #       Reshape (BxSxD)

        # Save kwargs shared by LayerNorm ops and precision types to use
        layernorm_kwargs = {"epsilon": self.layernorm_attrs["epsilon"], "axis": -1, "stash_type": 1}
        old_io_dtype = self.io_dtype
        new_io_dtype = ir.DataType.FLOAT if self.layernorm_attrs["cast"]["use_fp32"] else self.io_dtype
        cast = old_io_dtype != new_io_dtype

        # Reshape Q MatMul from BxSxD to Bx(SxN)xH before LayerNorm
        q_reshape_1_name = f"/model/layers.{layer_id}/attn/q_norm/Reshape_1"
        q_reshape_1_inputs = [self.attention_attrs["q_path"], f"/model/constants/INT64/[0, -1, {self.head_size}]"]
        q_reshape_1_output = f"{q_reshape_1_name}/output_0"
        self.make_reshape(q_reshape_1_name, q_reshape_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length * num_attention_heads', self.head_size])

        # Make Q LayerNorm
        q_layernorm_name = f"/model/layers.{layer_id}/attn/q_norm/SimplifiedLayerNormalization"
        q_weight_name = f"model.layers.{layer_id}.attn.q_norm.layernorm.weight"
        q_layernorm_output = f"{q_layernorm_name}/output_0"
        self.make_initializer(
            attention.q_norm.weight + self.layernorm_attrs["add_offset"],
            q_weight_name,
            to=new_io_dtype
        )

        # Create Cast nodes for inputs and outputs if old_dtype != new_dtype
        q_layernorm_inputs = [q_reshape_1_output, q_weight_name]
        q_layernorm_outputs = [q_layernorm_output]
        if cast:
            q_layernorm_inputs, q_layernorm_outputs = self.make_layernorm_casts(q_layernorm_name, q_layernorm_inputs, q_layernorm_outputs, old_io_dtype, new_io_dtype)

        self.make_node("SimplifiedLayerNormalization", inputs=q_layernorm_inputs, outputs=q_layernorm_outputs, name=q_layernorm_name, **layernorm_kwargs)
        self.make_value(q_layernorm_outputs[0], dtype=new_io_dtype, shape=['batch_size', 'sequence_length * num_attention_heads', self.head_size])

        # Reshape Q path after LayerNorm from Bx(SxN)xH to BxSxD
        q_reshape_2_name = f"/model/layers.{layer_id}/attn/q_norm/Reshape_2"
        q_reshape_2_inputs = [q_layernorm_output, f"/model/constants/INT64/[0, -1, {self.num_attn_heads * self.head_size}]"]
        self.make_reshape(q_reshape_2_name, q_reshape_2_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_attn_heads * self.head_size])

        # Reshape K MatMul from BxSxD to Bx(SxN)xH before LayerNorm
        k_reshape_1_name = f"/model/layers.{layer_id}/attn/k_norm/Reshape_1"
        k_reshape_1_inputs = [self.attention_attrs["k_path"], f"/model/constants/INT64/[0, -1, {self.head_size}]"]
        k_reshape_1_output = f"{k_reshape_1_name}/output_0"
        self.make_reshape(k_reshape_1_name, k_reshape_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length * num_key_value_heads', self.head_size])

        # Make K LayerNorm
        k_layernorm_name = f"/model/layers.{layer_id}/attn/k_norm/SimplifiedLayerNormalization"
        k_weight_name = f"model.layers.{layer_id}.attn.k_norm.layernorm.weight"
        k_layernorm_output = f"{k_layernorm_name}/output_0"
        self.make_initializer(
            attention.k_norm.weight + self.layernorm_attrs["add_offset"],
            k_weight_name,
            to=new_io_dtype
        )

        # Create Cast nodes for inputs and outputs if old_dtype != new_dtype
        k_layernorm_inputs = [k_reshape_1_output, k_weight_name]
        k_layernorm_outputs = [k_layernorm_output]
        if cast:
            k_layernorm_inputs, k_layernorm_outputs = self.make_layernorm_casts(k_layernorm_name, k_layernorm_inputs, k_layernorm_outputs, old_io_dtype, new_io_dtype)

        self.make_node("SimplifiedLayerNormalization", inputs=k_layernorm_inputs, outputs=k_layernorm_outputs, name=k_layernorm_name, **layernorm_kwargs)
        self.make_value(k_layernorm_outputs[0], dtype=new_io_dtype, shape=['batch_size', 'sequence_length * num_key_value_heads', self.head_size])

        # Reshape K path after LayerNorm from Bx(SxN)xH to BxSxD
        k_reshape_2_name = f"/model/layers.{layer_id}/attn/k_norm/Reshape_2"
        k_reshape_2_inputs = [k_layernorm_output, f"/model/constants/INT64/[0, -1, {self.num_kv_heads * self.head_size}]"]
        self.make_reshape(k_reshape_2_name, k_reshape_2_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_kv_heads * self.head_size])

        # Update q_path and k_path now
        self.attention_attrs["q_path"] = f"{q_reshape_2_name}/output_0"
        self.attention_attrs["k_path"] = f"{k_reshape_2_name}/output_0"

    def make_repeat_kv(self, layer_id, root_input, past_kv, present_kv, **kwargs):
        # Make subgraph that repeats tensor of shape (batch_size, sequence_length, num_kv_heads, head_size)
        # to shape (batch_size, sequence_length, num_attn_heads, head_size) in an interleaved pattern
        # and updates the KV caches
        #
        #           root_input
        #                |
        #             Reshape
        #                |
        #            Transpose
        #                |
        #                |   past_kv
        #                |  /
        #             Concat
        #                |  \
        #                |   present_kv
        #                |
        #        +-------+---------+
        #        |                 |
        #        |               Shape
        #        |                 |
        #        |     +-----------+-----------+-----------+
        #        |     |           |           |           |
        #        |   Gather     Gather      Gather      Gather
        #        |   (idx=0)    (idx=1)     (idx=2)     (idx=3)
        #        |     |           |           |           |
        #        | Unsqueeze   Unsqueeze   Unsqueeze   Unsqueeze
        #        |     |           |           |           |
        #        |     +-----------+-----------+-----------+
        #        |                 |
        #        |                 +-----------------------+
        #        |                 |                       |
        #        |                 |                      Mul
        #        |                 |                       |
        #        |              Concat                   Concat
        #        |               (5D)                     (4D)
        #        |                 |                       |
        #        |              Reshape                    |
        #        |             /   |   \                   |
        #        |            /    |    \                  |
        #        |           /     |     \                /
        #        |          /      |      \              /
        #        |         /       |       \            /
        #        |        /      Shape      \          /
        #        |       /         |         \        /
        #        |      |   ConstantOfShape   \      /
        #        |       \         |       \   \    /
        #        |        \        |       Mul  |  /
        #        |         \       |        |  /  /
        #        |          \      |      Equal  /
        #        |           \     |       /    /
        #         \           \    |      /    /
        #          \           \   |     /    /
        #           \           \  |    /    /
        #            \           \ |   /    /
        #         Unsqueeze       Where    /
        #             \           /       /
        #              \         /       /
        #               \       /       /
        #                \     /       /
        #                 Expand      /
        #                    |       /
        #                    |      /
        #                    |     /
        #                    |    /
        #                    |   /
        #                 Reshape
        #                    |
        #                Transpose
        #                    |
        #                 Reshape
        basename = f"/model/layers.{layer_id}/attn/{'k_proj' if past_kv.endswith('key') else 'v_proj'}/repeat_kv"

        # Make the initial subgraph
        #
        #                                                       +------> Gather --> Unsqueeze -----+
        #                                                       |                                  |
        #                                         past_kv       +------> Gather --> Unsqueeze -----+---> Mul --> Concat (4D)
        #                                            |          |                                  |
        # root_input --> Reshape --> Transpose --> Concat --> Shape ---> Gather --> Unsqueeze -----+---> Concat (5D)
        #                                            |          |                                  |
        #                                        present_kv     +------> Gather --> Unsqueeze -----+
        reshape_1_name = f"{basename}/Reshape_1"
        reshape_1_inputs = [root_input, f"/model/constants/INT64/[0, 0, {self.num_kv_heads}, -1]"]
        self.make_reshape(reshape_1_name, reshape_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_kv_heads, self.head_size])
        transpose_1_name = f"{basename}/Transpose_1"
        transpose_1_input = f"{reshape_1_name}/output_0"
        self.make_transpose(transpose_1_name, transpose_1_input, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, 'sequence_length', self.head_size], perm=[0,2,1,3])
        concat_1_name = f"{basename}/Concat_1"
        concat_1_inputs = [past_kv, f"{transpose_1_name}/output_0"]
        self.make_node("Concat", inputs=concat_1_inputs, outputs=[present_kv], name=concat_1_name, axis=2)

        shape_1_name = f"{basename}/Shape_1"
        self.make_shape(shape_1_name, present_kv, shape=[4])
        gather_1_name = f"{basename}/Gather_1"
        gather_1_inputs = [f"{shape_1_name}/output_0", "/model/constants/INT64/0"]
        self.make_gather(gather_1_name, gather_1_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_1_name = f"{basename}/Unsqueeze_1"
        unsqueeze_1_inputs = [f"{gather_1_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_1_name, unsqueeze_1_inputs, dtype=ir.DataType.INT64, shape=[1])
        gather_2_name = f"{basename}/Gather_2"
        gather_2_inputs = [f"{shape_1_name}/output_0", "/model/constants/INT64/1"]
        self.make_gather(gather_2_name, gather_2_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_2_name = f"{basename}/Unsqueeze_2"
        unsqueeze_2_inputs = [f"{gather_2_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_2_name, unsqueeze_2_inputs, dtype=ir.DataType.INT64, shape=[1])
        gather_3_name = f"{basename}/Gather_3"
        gather_3_inputs = [f"{shape_1_name}/output_0", "/model/constants/INT64/2"]
        self.make_gather(gather_3_name, gather_3_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_3_name = f"{basename}/Unsqueeze_3"
        unsqueeze_3_inputs = [f"{gather_3_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=ir.DataType.INT64, shape=[1])
        gather_4_name = f"{basename}/Gather_4"
        gather_4_inputs = [f"{shape_1_name}/output_0", "/model/constants/INT64/3"]
        self.make_gather(gather_4_name, gather_4_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        unsqueeze_4_inputs = [f"{gather_4_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_4_inputs, dtype=ir.DataType.INT64, shape=[1])
        concat_2_name = f"{basename}/Concat_2"
        concat_2_inputs = [f"{unsqueeze_1_name}/output_0", f"{unsqueeze_2_name}/output_0", f"/model/constants/INT64/[{self.num_attn_heads // self.num_kv_heads}]", f"{unsqueeze_3_name}/output_0", f"{unsqueeze_4_name}/output_0"]
        self.make_concat(concat_2_name, concat_2_inputs, dtype=ir.DataType.INT64, shape=[5], axis=0)

        mul_1_name = f"{basename}/Mul_1"
        mul_1_inputs = [f"{unsqueeze_2_name}/output_0", f"/model/constants/INT64/{self.num_attn_heads // self.num_kv_heads}"]
        self.make_mul(mul_1_name, mul_1_inputs, dtype=ir.DataType.INT64, shape=None)
        concat_3_name = f"{basename}/Concat_3"
        concat_3_inputs = [f"{unsqueeze_1_name}/output_0", f"{mul_1_name}/output_0", f"{unsqueeze_3_name}/output_0", f"{unsqueeze_4_name}/output_0"]
        self.make_concat(concat_3_name, concat_3_inputs, dtype=ir.DataType.INT64, shape=[4], axis=0)

        # Make the subgraph that follows the initial subgraph
        #
        #                               Mul ---> Equal
        #                              /              \
        # Reshape --> Shape --> ConstantOfShape --> Where
        #    |                                        |
        #    +----------------------------------------+
        reshape_2_name = f"{basename}/Reshape_2"
        reshape_2_inputs = [f"{concat_2_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_reshape(reshape_2_name, reshape_2_inputs, dtype=ir.DataType.INT64, shape=None)
        shape_2_name = f"{basename}/Shape_2"
        self.make_shape(shape_2_name, f"{reshape_2_name}/output_0", shape=[1])
        constant_shape_name = f"{basename}/ConstantOfShape"
        constant_shape_value = ir.tensor([1], dtype=ir.DataType.INT64)
        self.make_constant_of_shape(constant_shape_name, f"{shape_2_name}/output_0", value=constant_shape_value, dtype=ir.DataType.INT64, shape=[5])
        mul_2_name = f"{basename}/Mul"
        mul_2_inputs = [f"{constant_shape_name}/output_0", "/model/constants/INT64/-1"]
        self.make_mul(mul_2_name, mul_2_inputs, dtype=ir.DataType.INT64, shape=[5])
        equal_name = f"{basename}/Equal"
        equal_inputs = [f"{reshape_2_name}/output_0", f"{mul_2_name}/output_0"]
        self.make_equal(equal_name, equal_inputs, shape=[5])
        where_name = f"{basename}/Where"
        where_inputs = [f"{equal_name}/output_0", f"{constant_shape_name}/output_0", f"{reshape_2_name}/output_0"]
        self.make_where(where_name, where_inputs, dtype=ir.DataType.INT64, shape=[5])

        # Make the final nodes
        #
        # Where (from above)  Concat (from above)
        #                   \           \
        # Unsqueeze --> Expand --> Reshape --> Transpose --> Reshape
        unsqueeze_5_name = f"{basename}/Unsqueeze_5"
        unsqueeze_5_inputs = [present_kv, "/model/constants/INT64/[2]"]
        self.make_unsqueeze(unsqueeze_5_name, unsqueeze_5_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, 1, 'sequence_length', self.head_size])
        expand_name = f"{basename}/Expand"
        expand_inputs = [f"{unsqueeze_5_name}/output_0", f"{where_name}/output_0"]
        self.make_expand(expand_name, expand_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, self.num_attn_heads // self.num_kv_heads, 'sequence_length', self.head_size])
        reshape_3_name = f"{basename}/Reshape_3"
        reshape_3_inputs = [f"{expand_name}/output_0", f"{concat_3_name}/output_0"]
        self.make_reshape(reshape_3_name, reshape_3_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_attn_heads, 'sequence_length', self.head_size])
        transpose_2_name = f"{basename}/Transpose_2"
        transpose_2_input = f"{reshape_3_name}/output_0"
        self.make_transpose(transpose_2_name, transpose_2_input, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_attn_heads, self.head_size], perm=[0,2,1,3])
        reshape_4_name = f"{basename}/Reshape_4"
        reshape_4_inputs = [f"{transpose_2_name}/output_0", f"/model/constants/INT64/[0, 0, {self.num_attn_heads * self.head_size}]"]
        self.make_reshape(reshape_4_name, reshape_4_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_attn_heads * self.head_size])

        input_to_attention = f"{reshape_4_name}/output_0"
        return input_to_attention

    def make_attention_op(self, name, **kwargs):
        op_type = self.attention_attrs["op_type"]

        if op_type == "MultiHeadAttention":
            self.make_multi_head_attention(name, add_qk=f"{self.mask_attrs['mask_name']}/output_0", **kwargs)
        elif op_type == "GroupQueryAttention":
            self.make_group_query_attention(name, seqlens_k=f"{self.mask_attrs['seqlens_k']}/output_0", total_seq_len=f"{self.mask_attrs['total_seq_len']}/output_0", **kwargs)
        elif op_type == "SparseAttention":
            self.make_sparse_attention(name, block_row_indices=self.mask_attrs['block_row_indices'], block_col_indices=self.mask_attrs['block_col_indices'], key_total_seq_lens=f"{self.mask_attrs['key_total_seq_lens']}/output_0", total_seq_len=f"{self.mask_attrs['total_seq_len']}/output_0", **kwargs)
        else:
            raise NotImplementedError(f"The {op_type} op is not currently supported.")

    def make_multi_head_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"], kwargs["k_path"], kwargs["v_path"], kwargs.get("bias", ""),
            kwargs.get("attn_mask", ""), kwargs.get("add_qk", ""),
            kwargs.get("past_k", ""), kwargs.get("past_v", ""),
        ]
        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        self.make_node(
            "MultiHeadAttention", inputs=inputs, outputs=outputs, name=name, domain="com.microsoft",
            num_heads=self.num_attn_heads, scale=self.attention_attrs["scale"],
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * self.num_attn_heads])

    def make_group_query_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"], kwargs["k_path"], kwargs["v_path"],
            kwargs.get("past_k", ""), kwargs.get("past_v", ""),
            kwargs.get("seqlens_k", ""), kwargs.get("total_seq_len", ""),
            kwargs.get("cos_cache", ""), kwargs.get("sin_cache", ""),
            "", "",  # position_ids, attention_bias
        ]
        sinks = kwargs.get("sinks", "")  # TODO: add to inputs list directly once ORT 1.23 is out (one-time exception)
        if sinks:
            inputs += [sinks]

        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        self.make_node(
            "GroupQueryAttention", inputs=inputs, outputs=outputs, name=name, domain="com.microsoft",
            num_heads=self.num_attn_heads, kv_num_heads=self.num_kv_heads, scale=self.attention_attrs["scale"], local_window_size=self.window_size,
            softcap=self.attention_attrs["softcap"], do_rotary=self.attention_attrs["use_rope_in_attn"], rotary_interleaved=self.rope_attrs["interleaved"],
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * self.num_attn_heads])

    def make_sparse_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"], kwargs["k_path"], kwargs["v_path"],
            kwargs.get("past_k"), kwargs.get("past_v"),
            kwargs.get("block_row_indices"), kwargs.get("block_col_indices"),
            kwargs.get("total_seq_len"), kwargs.get("key_total_seq_lens"),
            kwargs.get("cos_cache", ""), kwargs.get("sin_cache", ""),
        ]
        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        self.make_node(
            "SparseAttention", inputs=inputs, outputs=outputs, name=name, domain="com.microsoft",
            num_heads=self.num_attn_heads, kv_num_heads=self.num_kv_heads, scale=self.attention_attrs["scale"], sparse_block_size=self.attention_attrs["block_sparse"]["sparse_block_size"],
            do_rotary=self.attention_attrs["use_rope_in_attn"], rotary_interleaved=self.rope_attrs["interleaved"],
        )

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph
        #
        # MultiHeadAttention example:
        #
        #               root_input
        #              /     |     \
        #       Q_MatMul  K_MatMul  V_MatMul  4D causal mask  past_key  past_value
        #           |        |         |            |            |           |
        #         Q_Add    K_Add     V_Add          +------------+-----------+
        #           |        |         |                         |
        #       Q_Rotary  K_Rotary     |                         |
        #           \        |        /                          |
        #            MultiHeadAttention--------------------------+
        #                    |
        #                O_MatMul
        #                    |
        #                  O_Add
        #
        # GroupQueryAttention example:
        #
        #               root_input
        #              /     |     \
        #       Q_MatMul  K_MatMul  V_MatMul  seqlens_k  total_seq_len  past_key  past_value
        #           |        |         |          |            |           |          |
        #         Q_Add    K_Add     V_Add        +------------+-----------+----------+
        #           |        |         |                       |
        #       Q_Rotary  K_Rotary     |                       |
        #           \        |        /                        |
        #            GroupQueryAttention-----------------------+
        #                    |
        #                O_MatMul
        #                    |
        #                  O_Add

        # Unpack attention weights if needed
        self.make_attention_unpacked(layer_id, attention, root_input, **kwargs)
        
        # Get dtype used for MatMul ops
        q_dtype = getattr(attention.q_proj, "weight", getattr(attention.q_proj, "bits", None))
        k_dtype = getattr(attention.k_proj, "weight", getattr(attention.k_proj, "bits", None))
        v_dtype = getattr(attention.v_proj, "weight", getattr(attention.v_proj, "bits", None))
        qkv_dtype_equal = getattr(q_dtype, "dtype", q_dtype) == getattr(k_dtype, "dtype", k_dtype) == getattr(v_dtype, "dtype", v_dtype)

        # Make MatMul nodes
        if self.attention_attrs["use_packed_matmul"] and qkv_dtype_equal:
            # Combine 3 MatMuls into 1 packed MatMul
            qkv_matmul_basename = f"/model/layers.{layer_id}/attn/qkv_proj/MatMul"
            qkv_matmul_name = self.make_packed_matmul(attention.q_proj, attention.k_proj, attention.v_proj, qkv_matmul_basename, root_input)
            self.attention_attrs["q_path"] = f"{qkv_matmul_name}/output_0"
        else:
            q_matmul_basename = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
            q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
            self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"
            k_matmul_basename = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
            k_matmul_name = self.make_matmul(attention.k_proj, k_matmul_basename, root_input)
            self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"
            v_matmul_basename = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
            v_matmul_name = self.make_matmul(attention.v_proj, v_matmul_basename, root_input)
            self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # Make Add nodes (if bias exists)
        q_bias_exists = attention.q_proj.bias is not None and torch.count_nonzero(attention.q_proj.bias) > 0
        k_bias_exists = attention.k_proj.bias is not None and torch.count_nonzero(attention.k_proj.bias) > 0
        v_bias_exists = attention.v_proj.bias is not None and torch.count_nonzero(attention.v_proj.bias) > 0
        any_bias_exists = q_bias_exists or k_bias_exists or v_bias_exists

        if self.attention_attrs["use_packed_matmul"] and qkv_dtype_equal and any_bias_exists:
            # Combine 3 Adds into 1 packed Add
            qkv_add_name = f"/model/layers.{layer_id}/attn/qkv_proj/Add"
            self.make_packed_add(attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias, qkv_add_name, root_input=self.attention_attrs["q_path"])
            self.attention_attrs["q_path"] = f"{qkv_add_name}/output_0"
        else:
            if q_bias_exists:
                q_add_name = f"/model/layers.{layer_id}/attn/q_proj/Add"
                self.make_add_bias(attention.q_proj.bias, q_add_name, root_input=self.attention_attrs["q_path"])
                self.attention_attrs["q_path"] = f"{q_add_name}/output_0"
            if k_bias_exists:
                k_add_name = f"/model/layers.{layer_id}/attn/k_proj/Add"
                self.make_add_bias(attention.k_proj.bias, k_add_name, root_input=self.attention_attrs["k_path"])
                self.attention_attrs["k_path"] = f"{k_add_name}/output_0"
            if v_bias_exists:
                v_add_name = f"/model/layers.{layer_id}/attn/v_proj/Add"
                self.make_add_bias(attention.v_proj.bias, v_add_name, root_input=self.attention_attrs["v_path"])
                self.attention_attrs["v_path"] = f"{v_add_name}/output_0"

        # Make Q/K SimplifiedLayerNorm nodes
        if self.attention_attrs["q_norm"] and self.attention_attrs["k_norm"]:
            self.make_qk_norm(layer_id, attention)

        # Make RotaryEmbedding nodes
        cos_cache_name, sin_cache_name = "", ""
        if self.attention_attrs["use_rope_in_attn"]:
            cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        else:
            q_rotary_name = f"/model/layers.{layer_id}/attn/q_rotary/RotaryEmbedding"
            self.make_rotary_embedding(q_rotary_name, root_input=self.attention_attrs["q_path"], position_ids=kwargs.get("position_ids", "position_ids"))
            self.attention_attrs["q_path"] = f"{q_rotary_name}/output_0"
            k_rotary_name = f"/model/layers.{layer_id}/attn/k_rotary/RotaryEmbedding"
            self.make_rotary_embedding(k_rotary_name, root_input=self.attention_attrs["k_path"], position_ids=kwargs.get("position_ids", "position_ids"))
            self.attention_attrs["k_path"] = f"{k_rotary_name}/output_0"

        # Make repeat KV nodes (Note: `repeat_kv` needs to be kept since GroupQueryAttention isn't supported for FP32 CUDA)
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"
        if self.num_attn_heads != self.num_kv_heads and self.attention_attrs["op_type"] == "MultiHeadAttention":
            self.attention_attrs["k_path"] = self.make_repeat_kv(layer_id, root_input=self.attention_attrs["k_path"], past_kv=past_k, present_kv=present_k)
            self.attention_attrs["v_path"] = self.make_repeat_kv(layer_id, root_input=self.attention_attrs["v_path"], past_kv=past_v, present_kv=present_v)
            past_k, past_v, present_k, present_v = "", "", "", ""

        # Make sinks input
        sinks_name = ""
        if self.attention_attrs["sinks"]:
            sinks_name = f"model.layers.{layer_id}.attn.sinks"
            self.make_initializer(attention.sinks, sinks_name, to=self.io_dtype)

        # Make attention node (e.g. MultiHeadAttention, GroupQueryAttention, etc.)
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name, q_path=self.attention_attrs["q_path"], k_path=self.attention_attrs["k_path"], v_path=self.attention_attrs["v_path"],
            past_k=past_k, past_v=past_v, present_k=present_k, present_v=present_v,
            cos_cache=cos_cache_name, sin_cache=sin_cache_name, sinks=sinks_name, **kwargs,
        )

        # Make MatMul node (output projection weight node)
        o_proj = 'o_proj' if hasattr(attention, 'o_proj') else 'dense'
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_weight = getattr(attention, o_proj)
        o_matmul_name = self.make_matmul(o_weight, o_matmul_basename, f"{attn_name}/output_0")

        # Make Add node (output projection bias node if bias exists)
        o_bias_exists = getattr(attention, o_proj).bias is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            o_bias = getattr(attention, o_proj).bias
            self.make_add_bias(o_bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name if not o_bias_exists else o_add_name}/output_0"

    def make_attention_unpacked(self, layer_id, attention, root_input, **kwargs):
        qkv_linear = getattr(attention, "qkv_proj", None) or getattr(attention, "query_key_value", None)
        if qkv_linear is None:
            # Return early if there's nothing to unpack
            return

        if hasattr(qkv_linear, "base_layer"):
            # For LoRA packed `MatMul`
            self.make_attention_unpacked_lora(layer_id, attention, qkv_linear, root_input, **kwargs)
        else:
            # For regular packed `MatMul`
            self.make_attention_unpacked_regular(layer_id, attention, qkv_linear, root_input, **kwargs)

        # Delete original packed weights
        del qkv_linear

    def make_attention_unpacked_lora(self, layer_id, attention, qkv_linear, root_input, **kwargs):
        from peft.tuners.lora.layer import LoraLayer

        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        # Create Q/K/V base layers
        q_proj = torch.nn.Linear(in_features=q_size, out_features=q_size)
        q_proj.weight = torch.nn.Parameter(qkv_linear.weight[: q_size, :], requires_grad=False)
        q_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[: q_size], requires_grad=False)

        k_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        k_proj.weight = torch.nn.Parameter(qkv_linear.weight[q_size : q_size + kv_size, :], requires_grad=False)
        k_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[q_size : q_size + kv_size], requires_grad=False)

        v_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        v_proj.weight = torch.nn.Parameter(qkv_linear.weight[q_size + kv_size :, :], requires_grad=False)
        v_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[q_size + kv_size :], requires_grad=False)

        # Create Q/K/V lora_B layers
        lora_B = qkv_linear.lora_B.default

        q_lora_B = torch.nn.Linear(in_features=q_size, out_features=q_size)
        q_lora_B.weight = torch.nn.Parameter(lora_B.weight[: q_size, :], requires_grad=False)
        q_lora_B.bias = None if lora_B.bias is None else torch.nn.Parameter(lora_B.bias[: q_size], requires_grad=False)

        k_lora_B = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        k_lora_B.weight = torch.nn.Parameter(lora_B.weight[q_size : q_size + kv_size, :], requires_grad=False)
        k_lora_B.bias = None if lora_B.bias is None else torch.nn.Parameter(lora_B.bias[q_size : q_size + kv_size], requires_grad=False)

        v_lora_B = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        v_lora_B.weight = torch.nn.Parameter(lora_B.weight[q_size + kv_size :, :], requires_grad=False)
        v_lora_B.bias = None if lora_B.bias is None else torch.nn.Parameter(lora_B.bias[q_size + kv_size :], requires_grad=False)

        # Create Q/K/V LoRA layers
        attention.q_proj = LoraLayer(q_proj)
        attention.q_proj.lora_A.default = qkv_linear.lora_A.default
        attention.q_proj.lora_B.default = q_lora_B
        attention.q_proj.scaling = qkv_linear.scaling

        attention.k_proj = LoraLayer(k_proj)
        attention.k_proj.lora_A.default = qkv_linear.lora_A.default
        attention.k_proj.lora_B.default = k_lora_B
        attention.k_proj.scaling = qkv_linear.scaling

        attention.v_proj = LoraLayer(v_proj)
        attention.v_proj.lora_A.default = qkv_linear.lora_A.default
        attention.v_proj.lora_B.default = v_lora_B
        attention.v_proj.scaling = qkv_linear.scaling

    def make_attention_unpacked_regular(self, layer_id, attention, qkv_linear, root_input, **kwargs):
        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        attention.q_proj = torch.nn.Linear(in_features=q_size, out_features=q_size)
        attention.q_proj.weight = torch.nn.Parameter(qkv_linear.weight[: q_size, :], requires_grad=False)
        attention.q_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[: q_size], requires_grad=False)

        attention.k_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.k_proj.weight = torch.nn.Parameter(qkv_linear.weight[q_size : q_size + kv_size, :], requires_grad=False)
        attention.k_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[q_size : q_size + kv_size], requires_grad=False)

        attention.v_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.v_proj.weight = torch.nn.Parameter(qkv_linear.weight[q_size + kv_size :, :], requires_grad=False)
        attention.v_proj.bias = None if qkv_linear.bias is None else torch.nn.Parameter(qkv_linear.bias[q_size + kv_size :], requires_grad=False)

    def make_mlp(self, layer_id, mlp, root_input):
        # Unpack MLP weights if needed
        self.make_mlp_unpacked(layer_id, mlp, root_input)

        if self.mlp_attrs["use_proj"]:
            self.make_mlp_proj(layer_id, mlp, root_input)
        elif self.mlp_attrs["use_fc"]:
            self.make_mlp_fc(layer_id, mlp, root_input)
        else:
            raise NotImplementedError(f"The MLP layer type is not set.")

    def make_mlp_unpacked(self, layer_id, mlp, root_input):
        gate_up_linear = getattr(mlp, "gate_up_proj", None) or getattr(mlp, "dense_h_to_4h", None)
        if gate_up_linear is None:
            # Return early if there's nothing to unpack
            return

        if hasattr(gate_up_linear, "base_layer"):
            # For LoRA packed `MatMul`
            self.make_mlp_unpacked_lora(layer_id, mlp, gate_up_linear, root_input)
        else:
            # For regular packed `MatMul`
            self.make_mlp_unpacked_regular(layer_id, mlp, gate_up_linear, root_input)

        # Delete original packed weights
        del gate_up_linear

    def make_mlp_unpacked_lora(self, layer_id, mlp, gate_up_linear, root_input):
        from peft.tuners.lora.layer import LoraLayer

        # Create GateProj/UpProj base layers
        gate_proj = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        gate_proj.weight = torch.nn.Parameter(gate_up_linear.weight[ : self.intermediate_size, :], requires_grad=False)
        gate_proj.bias = None if gate_up_linear.bias is None else torch.nn.Parameter(gate_up_linear.bias[: self.intermediate_size], requires_grad=False)

        up_proj = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        up_proj.weight = torch.nn.Parameter(gate_up_linear.weight[self.intermediate_size :, :], requires_grad=False)
        up_proj.bias = None if gate_up_linear.bias is None else torch.nn.Parameter(gate_up_linear.bias[self.intermediate_size :], requires_grad=False)

        # Create GateProj/UpProj lora_B layers
        lora_B = gate_up_linear.lora_B.default

        gate_proj_lora_B = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        gate_proj_lora_B.weight = torch.nn.Parameter(lora_B.weight[ : self.intermediate_size, :], requires_grad=False)
        gate_proj_lora_B.bias = None if lora_B.bias is None else torch.nn.Parameter(lora_B.bias[: self.intermediate_size], requires_grad=False)

        up_proj_lora_B = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        up_proj_lora_B.weight = torch.nn.Parameter(lora_B.weight[self.intermediate_size :, :], requires_grad=False)
        up_proj_lora_B.bias = None if lora_B.bias is None else torch.nn.Parameter(lora_B.bias[self.intermediate_size :], requires_grad=False)

        # Create GateProj/UpProj LoRA layers
        mlp.gate_proj = LoraLayer(gate_proj)
        mlp.gate_proj.lora_A.default = gate_up_linear.lora_A.default
        mlp.gate_proj.lora_B.default = gate_proj_lora_B
        mlp.gate_proj.scaling = gate_up_linear.scaling

        mlp.up_proj = LoraLayer(up_proj)
        mlp.up_proj.lora_A.default = gate_up_linear.lora_A.default
        mlp.up_proj.lora_B.default = up_proj_lora_B
        mlp.up_proj.scaling = gate_up_linear.scaling

    def make_mlp_unpacked_regular(self, layer_id, mlp, gate_up_linear, root_input):
        mlp.gate_proj = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        mlp.gate_proj.weight = torch.nn.Parameter(gate_up_linear.weight[: self.intermediate_size, :], requires_grad=False)
        mlp.gate_proj.bias = None if gate_up_linear.bias is None else torch.nn.Parameter(gate_up_linear.bias[: self.intermediate_size], requires_grad=False)

        mlp.up_proj = torch.nn.Linear(in_features=self.hidden_size, out_features=self.intermediate_size)
        mlp.up_proj.weight = torch.nn.Parameter(gate_up_linear.weight[self.intermediate_size :, :])
        mlp.up_proj.bias = None if gate_up_linear.bias is None else torch.nn.Parameter(gate_up_linear.bias[self.intermediate_size :], requires_grad=False)

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #            root_input
        #           /          \
        #          /            \
        #   UpProjMatMul    GateProjMatMul
        #         |              |
        #     UpProjAdd     GateProjAdd
        #          \             |
        #           \         ActFunc
        #            \       /
        #             \     /
        #              \   /
        #               Mul
        #                |
        #          DownProjMatMul
        #                |
        #           DownProjAdd

        # Check if Add nodes need to be made (if bias exists)
        gate_bias_exists = mlp.gate_proj.bias is not None and torch.count_nonzero(mlp.gate_proj.bias) > 0
        up_bias_exists = mlp.up_proj.bias is not None and torch.count_nonzero(mlp.up_proj.bias) > 0
        down_bias_exists = mlp.down_proj.bias is not None and torch.count_nonzero(mlp.down_proj.bias) > 0

        # Make Gate proj nodes
        gate_matmul_basename = f"/model/layers.{layer_id}/mlp/gate_proj/MatMul"
        gate_matmul_name = self.make_matmul(mlp.gate_proj, gate_matmul_basename, root_input)
        gate_name = gate_matmul_name
        if gate_bias_exists:
            gate_add_name = f"/model/layers.{layer_id}/mlp/gate_proj/Add"
            self.make_add_bias(mlp.gate_proj.bias, gate_add_name, root_input=f"{gate_name}/output_0")
            gate_name = gate_add_name

        # Make Up proj nodes
        up_matmul_basename = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        up_matmul_name = self.make_matmul(mlp.up_proj, up_matmul_basename, root_input)
        up_name = up_matmul_name
        if up_bias_exists:
            up_add_name = f"/model/layers.{layer_id}/mlp/up_proj/Add"
            self.make_add_bias(mlp.up_proj.bias, up_add_name, root_input=f"{up_name}/output_0")
            up_name = up_add_name

        # Make activation node(s)
        act_fn_name = self.make_activation(layer_id, root_input=f"{gate_name}/output_0")

        # Make Mul node after activation
        mul_name = f"/model/layers.{layer_id}/mlp/Mul"
        mul_inputs = [f"{act_fn_name}/output_0", f"{up_name}/output_0"]
        self.make_mul(mul_name, mul_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make output MatMul node
        down_matmul_basename = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        down_matmul_name = self.make_matmul(mlp.down_proj, down_matmul_basename, f"{mul_name}/output_0")
        down_name = down_matmul_name
        if down_bias_exists:
            down_add_name = f"/model/layers.{layer_id}/mlp/down_proj/Add"
            self.make_add_bias(mlp.down_proj.bias, down_add_name, root_input=f"{down_name}/output_0")
            down_name = down_add_name

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"

    def make_mlp_fc(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #          root_input
        #              |
        #          FC1_MatMul
        #              |
        #           FC1_Add
        #              |
        #           ActFunc
        #              |
        #          FC2_MatMul
        #              |
        #           FC2_Add

        # Check if Add nodes need to be made (if bias exists)
        fc1_bias_exists = mlp.fc1.bias is not None and torch.count_nonzero(mlp.fc1.bias) > 0
        fc2_bias_exists = mlp.fc2.bias is not None and torch.count_nonzero(mlp.fc2.bias) > 0

        # Make first layer of fully connected nodes (FC1)
        fc1_matmul_basename = f"/model/layers.{layer_id}/mlp/fc1/MatMul"
        fc1_matmul_name = self.make_matmul(mlp.fc1, fc1_matmul_basename, root_input)
        fc1_name = fc1_matmul_name
        if fc1_bias_exists:
            fc1_add_name = f"/model/layers.{layer_id}/mlp/fc1/Add"
            self.make_add_bias(mlp.fc1.bias, fc1_add_name, root_input=f"{fc1_name}/output_0")
            fc1_name = fc1_add_name

        # Make activation function
        act_fn_name = self.make_activation(layer_id, root_input=f"{fc1_name}/output_0")

        # Make second layer of fully connected nodes (FC2)
        fc2_matmul_basename = f"/model/layers.{layer_id}/mlp/fc2/MatMul"
        fc2_matmul_name = self.make_matmul(mlp.fc2, fc2_matmul_basename, root_input=f"{act_fn_name}/output_0")
        fc2_name = fc2_matmul_name
        if fc2_bias_exists:
            fc2_add_name = f"/model/layers.{layer_id}/mlp/fc2/Add"
            self.make_add_bias(mlp.fc2.bias, fc2_add_name, root_input=f"{fc2_name}/output_0")
            fc2_name = fc2_add_name

        # Assign output 0 of MLP layer as output of last layer
        self.mlp_attrs["output_0"] = f"{fc2_name}/output_0"

    def make_moe_op(self, name, **kwargs):
        op_type = self.moe_attrs["op_type"]

        if op_type == "MoE":
            self.make_base_moe_op(name, **kwargs)
        elif op_type == "QMoE":
            self.make_qmoe_op(name, **kwargs)
        else:
            raise NotImplementedError(f"The {op_type} op is not currently supported.")

    def make_base_moe_op(self, name, **kwargs):
        inputs = [
            kwargs["root_input"], kwargs["router_probs"],
            kwargs["weight1"], kwargs.get("bias1", ""),
            kwargs["weight2"], kwargs.get("bias2", ""),
            kwargs.get("weight3", ""), kwargs.get("bias3", ""),
        ]
        output = f"{name}/output_0"

        extra_kwargs = {"swiglu_limit": self.moe_attrs["swiglu_limit"]} if self.moe_attrs["swiglu_limit"] is not None else {}
        self.make_node(
            "MoE", inputs=inputs, outputs=[output], name=name, domain="com.microsoft",
            activation_alpha=self.moe_attrs["activation_alpha"],
            activation_beta=self.moe_attrs["activation_beta"],
            activation_type=self.moe_attrs["activation_type"],
            k=self.moe_attrs["top_k"],
            normalize_routing_weights=self.moe_attrs["normalize_routing_weights"],
            swiglu_fusion=self.moe_attrs["swiglu_fusion"],
            use_sparse_mixer=self.moe_attrs["use_sparse_mixer"],
            **extra_kwargs,
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

    def make_qmoe_op(self, name, **kwargs):
        inputs = [
            kwargs["root_input"], kwargs["router_probs"],
            kwargs["weight1"], kwargs["scales1"], kwargs.get("bias1", ""),
            kwargs["weight2"], kwargs["scales2"], kwargs.get("bias2", ""),
            kwargs.get("weight3", ""), kwargs.get("scales3", ""), kwargs.get("bias3", ""),
        ]
        output = f"{name}/output_0"

        extra_kwargs = {"swiglu_limit": self.moe_attrs["swiglu_limit"]} if self.moe_attrs["swiglu_limit"] is not None else {}
        self.make_node(
            "QMoE", inputs=inputs, outputs=[output], name=name, domain="com.microsoft",
            activation_alpha=self.moe_attrs["activation_alpha"],
            activation_beta=self.moe_attrs["activation_beta"],
            activation_type=self.moe_attrs["activation_type"],
            expert_weight_bits=self.moe_attrs["expert_weight_bits"],
            k=self.moe_attrs["top_k"],
            normalize_routing_weights=self.moe_attrs["normalize_routing_weights"],
            swiglu_fusion=self.moe_attrs["swiglu_fusion"],
            use_sparse_mixer=self.moe_attrs["use_sparse_mixer"],
            block_size=self.moe_attrs["block_size"],
            **extra_kwargs,
        )
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

    def make_qmoe_weights(self, weights):
        dtype = torch.quint4x2 if self.moe_attrs["expert_weight_bits"] == 4 else torch.int8
        qweight, scales = None, None

        # Get block size from quantization attributes
        block_size = self.quant_attrs["int4"]["block_size"]

        # Use block-wise quantization if block_size > 0
        if block_size > 0:
            try:
                qweight, scales = self._symmetric_blockwise_quantize(weights, block_size)
                self.moe_attrs["block_size"] = block_size
                return qweight, scales.to(torch.float16)
            except Exception as e:
                raise RuntimeError(f"Block-wise quantization failed with block_size={block_size}: {e}")
        else:
            # block_size is 0, so we're using tensor-level quantization
            self.moe_attrs["block_size"] = 0

        # Existing tensor-level quantization implementation (fallback)
        unsuccessful = True
        try:
            import tensorrt_llm

            _, qweight, scales = (
                torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.detach().cpu().contiguous(), dtype)
            )
            unsuccessful = False
        except ImportError:
            print("WARNING: TensorRT-LLM is needed to use torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix().")
        except RuntimeError as r:
            print("WARNING: TensorRT-LLM failed to run torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix() successfully.")
            err = str(r)
            print(err[ : err.find('\n1')])  # omit internal traceback inside TensorRT-LLM
        finally:
            if unsuccessful:
                raise RuntimeError("Failed to quantize MoE weights with TensorRT-LLM. Please ensure TensorRT-LLM installs and runs successfully in your environment.")

        return qweight, scales.to(torch.float16)

    def _symmetric_blockwise_quantize(self, weights, block_size):
        # Ensure weights are on CPU for quantization
        weights = weights.cpu().contiguous()

        original_shape = weights.shape
        bits = self.moe_attrs["expert_weight_bits"]

        qmin, qmax = (-7, 7) if bits == 4 else (-127, 127)

        # Reshape weights to process the last dimension in blocks
        # weights shape: [..., hidden_size] -> [..., num_blocks, block_size]
        last_dim = original_shape[-1]
        num_blocks = (last_dim + block_size - 1) // block_size

        # Pad the last dimension if necessary
        pad_size = num_blocks * block_size - last_dim
        if pad_size > 0:
            pad_shape = list(original_shape)
            pad_shape[-1] = pad_size
            padding = torch.zeros(pad_shape, dtype=weights.dtype, device=weights.device)
            weights_padded = torch.cat([weights, padding], dim=-1)
        else:
            weights_padded = weights

        reshaped_weights = weights_padded.view(*original_shape[:-1], num_blocks, block_size)
        block_max_abs = torch.max(torch.abs(reshaped_weights), dim=-1)[0]
        scales = block_max_abs / qmax

        # Avoid division by zero - set minimum scale
        min_scale = 1e-8
        scales = torch.where(scales < min_scale, torch.tensor(min_scale, dtype=scales.dtype, device=scales.device), scales)

        # Expand scales for broadcasting: [..., num_blocks, 1]
        scales_expanded = scales.unsqueeze(-1)

        # Quantize: q = round(w / scale), then clamp to valid range
        quantized = torch.round(reshaped_weights / scales_expanded)
        quantized = torch.clamp(quantized, qmin, qmax)

        if bits == 4:
            quantized_int8 = quantized.to(torch.int8)

            quantized_flat = quantized_int8.view(*original_shape[:-1], num_blocks * block_size)

            if pad_size > 0:
                quantized_flat = quantized_flat[..., :-pad_size]

            quantized_uint4 = (quantized_flat + 8).to(torch.uint8)

            packed_shape = list(original_shape)
            packed_shape[-1] = (original_shape[-1] + 1) // 2
            qweight = torch.zeros(packed_shape, dtype=torch.uint8, device=weights.device)

            # Pack two 4-bit values per byte
            for i in range(0, quantized_uint4.shape[-1], 2):
                val1 = quantized_uint4[..., i]
                if i + 1 < quantized_uint4.shape[-1]:
                    val2 = quantized_uint4[..., i + 1]
                    packed_val = (val1 & 0xF) | ((val2 & 0xF) << 4)
                else:
                    # Odd number of values - pack only lower 4 bits
                    packed_val = val1 & 0xF
                qweight[..., i // 2] = packed_val

        else:  # 8-bit
            quantized_int8 = quantized.to(torch.int8)

            qweight = quantized_int8.view(*original_shape[:-1], num_blocks * block_size)
            if pad_size > 0:
                qweight = qweight[..., :-pad_size]
            else:
                qweight = qweight.view(original_shape)

        return qweight.cpu(), scales.cpu()

    def make_block_sparse_moe(self, layer_id, bsm, root_input):
        # Make nodes for the QMoE block-sparse subgraph
        #
        #                  root_input
        #                 /       \
        #         router_MatMul    |
        #             /     \      |
        #         Shape      |     |
        #           |        |     |
        #         Gather     |     |
        #           |        |     |
        #       Unsqueeze    |     |
        #           |        |    /
        #        Concat     /    /
        #             \    /    /
        #             Reshape  /
        #                 \   /
        #                  QMoE
        #                   |
        #                 output
        moe_name = f"/model/layers.{layer_id}/moe"
        gate_ops_base = f"{moe_name}/gate"

        # Make MoE nodes
        gate_name = f"{gate_ops_base}/MatMul"
        self.make_matmul(bsm.gate, gate_name, root_input)
        shape_name = f"{gate_ops_base}/Shape"
        self.make_shape(shape_name, f"{gate_name}/output_0", shape=[3])
        gather_name = f"{gate_ops_base}/Gather"
        self.make_gather(gather_name, [f"{shape_name}/output_0", "/model/constants/INT64/2"], dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_name = f"{gate_ops_base}/Unsqueeze"
        self.make_unsqueeze(unsqueeze_name, [f"{gather_name}/output_0", "/model/constants/INT64/[0]"], dtype=ir.DataType.INT64, shape=[1])
        concat_name = f"{gate_ops_base}/Concat"
        self.make_concat(concat_name, ["/model/constants/INT64/[-1]", f"{unsqueeze_name}/output_0"], dtype=ir.DataType.INT64, shape=[2], axis=0)
        gate_reshape_name = f"{gate_ops_base}/Reshape"
        self.make_reshape(gate_reshape_name, [f"{gate_name}/output_0", f"{concat_name}/output_0"], dtype=self.io_dtype, shape=['num_rows', self.moe_attrs["num_experts"]])

        w1_list = []
        w2_list = []
        w3_list = []
        w1_scale_list = []
        w2_scale_list = []
        w3_scale_list = []

        for i in range(self.moe_attrs["num_experts"]):
            # Quantize the weights with uint8
            pre_qweight1, w1_scale = self.make_qmoe_weights(bsm.experts[i].w1.weight.T)
            pre_qweight2, w2_scale = self.make_qmoe_weights(bsm.experts[i].w2.weight.T)
            pre_qweight3, w3_scale = self.make_qmoe_weights(bsm.experts[i].w3.weight.T)

            w1_list.append(pre_qweight1)
            w2_list.append(pre_qweight2)
            w3_list.append(pre_qweight3)

            w1_scale_list.append(w1_scale)
            w2_scale_list.append(w2_scale)
            w3_scale_list.append(w3_scale)

        moe_expert_weight_1_name = f"model.layers.{layer_id}.moe.weight_1"
        moe_expert_weight_2_name = f"model.layers.{layer_id}.moe.weight_2"
        moe_expert_weight_3_name = f"model.layers.{layer_id}.moe.weight_3"

        moe_expert_scales_1_name = f"model.layers.{layer_id}.moe.scales_1"
        moe_expert_scales_2_name = f"model.layers.{layer_id}.moe.scales_2"
        moe_expert_scales_3_name = f"model.layers.{layer_id}.moe.scales_3"

        def make_moe_initializer(w_list, moe_expert_name, dtype):
            moe_experts_weight = torch.stack(w_list, dim=0)
            self.make_initializer(
                moe_experts_weight,
                moe_expert_name,
                to=dtype,
            )

        make_moe_initializer(w1_list, moe_expert_weight_1_name, ir.DataType.UINT8)
        make_moe_initializer(w2_list, moe_expert_weight_2_name, ir.DataType.UINT8)
        make_moe_initializer(w3_list, moe_expert_weight_3_name, ir.DataType.UINT8)

        # Currently we don't expect QMoE to be used with distributed inference
        make_moe_initializer(w1_scale_list, moe_expert_scales_1_name, self.io_dtype)
        make_moe_initializer(w2_scale_list, moe_expert_scales_2_name, self.io_dtype)
        make_moe_initializer(w3_scale_list, moe_expert_scales_3_name, self.io_dtype)

        self.make_moe_op(
            moe_name, root_input=root_input, router_probs=f"{gate_reshape_name}/output_0",
            weight1=moe_expert_weight_1_name, scales1=moe_expert_scales_1_name,
            weight2=moe_expert_weight_2_name, scales2=moe_expert_scales_2_name,
            weight3=moe_expert_weight_3_name, scales3=moe_expert_scales_3_name,
        )

        # Assign output 0 of previous MoE as root input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = output

    def make_activation_with_mul(self, layer_id, root_input, activation, domain):
        # Make nodes for this activation subgraph
        #
        #       root_input (GateProjMatMul)
        #         /  |
        #   ActFunc  |
        #          \ |
        #           Mul
        act_name = f"/model/layers.{layer_id}/mlp/act_fn/{activation}"
        act_output = f"{act_name}/output_0"
        self.make_node(activation, inputs=[root_input], outputs=[act_output], name=act_name, domain=domain)
        self.make_value(act_output, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        mul_act_name = f"/model/layers.{layer_id}/mlp/act_fn/Mul"
        mul_act_inputs = [root_input, act_output]
        self.make_mul(mul_act_name, mul_act_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        return mul_act_name

    def make_gelu(self, layer_id, root_input, activation):
        # Make nodes for this activation subgraph
        #
        #       root_input (Add)
        #           |
        #        GeluAct
        gelu_name = f"/model/layers.{layer_id}/mlp/act_fn/{activation}"
        output = f"{gelu_name}/output_0"

        if activation == "Gelu":
            self.make_node("Gelu", inputs=[root_input], outputs=[output], name=gelu_name, approximate="none")
        elif activation == "FastGelu":
            self.make_node("Gelu", inputs=[root_input], outputs=[output], name=gelu_name, approximate="tanh")
        else:
            self.make_node(activation, inputs=[root_input], outputs=[output], name=gelu_name, domain="com.microsoft")

        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.intermediate_size])

        return gelu_name

    def make_relu(self, layer_id, root_input, activation):
        relu_name = f"/model/layers.{layer_id}/mlp/act_fn/{activation}"
        output = f"{relu_name}/output_0"
        self.make_node(activation, inputs=[root_input], outputs=[output], name=relu_name, domain="")
        self.make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.intermediate_size])
        return relu_name

    def make_relu_squared(self, layer_id, root_input, activation):
        relu_name = self.make_relu(layer_id, root_input, "Relu")
        basename = f"/model/layers.{layer_id}/mlp/square/{activation}"
        pow_name = f"{basename}/pow"
        pow_inputs = [f"{relu_name}/output_0", "/model/constants/INT32/[2]"]
        self.make_node("Pow", inputs=pow_inputs, outputs=[f"{pow_name}/output_0"], name=pow_name, domain="")
        self.make_value(f"{pow_name}/output_0", self.io_dtype, shape=['batch_size', 'sequence_length', self.intermediate_size])
        return pow_name

    def make_activation(self, layer_id, root_input):
        if self.activation in {"silu", "swish", "swiglu"}:
            output_name = self.make_activation_with_mul(layer_id, root_input, activation="Sigmoid", domain=None)
        elif self.activation in {"gelu_new", "gelu_fast", "gelu_pytorch_tanh"}:
            output_name = self.make_gelu(layer_id, root_input, activation="FastGelu")
        elif self.activation in {"gelu"}:
            output_name = self.make_gelu(layer_id, root_input, activation="Gelu")
        elif self.activation in {"gegelu", "geglu"}:
            output_name = self.make_gelu(layer_id, root_input, activation="QuickGelu")
        elif self.activation in {"relu"}:
            output_name = self.make_relu(layer_id, root_input, activation="Relu")
        elif self.activation in {"relu2"}:
            output_name = self.make_relu_squared(layer_id, root_input, activation="Relu2")
        else:
            raise NotImplementedError(f"The {self.activation} activation function is not currently supported.")
        return output_name

    def make_lm_head(self, lm_head):
        # Check if there are ops to insert after MatMul
        bias_exists = lm_head.bias is not None
        scale_exists = self.lm_head_attrs["scale"] != 1
        mask_exists = self.lm_head_attrs["mask"] is not None
        softcap_exists = self.lm_head_attrs["softcap"] != 0.0
        cast_exists = self.io_dtype != self.output_types["logits"]

        # List order matters here. It should match the order of the below if condition checks.
        # Add new checks to the end of the list and after the below if condition checks.
        exists_checks = [bias_exists, scale_exists, mask_exists, softcap_exists, cast_exists]

        matmul_basename = "/lm_head/MatMul"
        root_input = self.layernorm_attrs["output_0"]
        matmul_name = self.make_matmul(lm_head, matmul_basename, root_input, logits=not any(exists_checks))
        lm_name = matmul_name

        if bias_exists:
            add_name = "/lm_head/Add"
            self.make_add_bias(lm_head.bias, add_name, root_input=f"{lm_name}/output_0", logits=not any(exists_checks[1:]))
            lm_name = add_name

        if scale_exists:
            mul_name = "/lm_head/Mul"
            mul_inputs = [f"{lm_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.lm_head_attrs['scale']}"]
            mul_output = "logits" if not any(exists_checks[2:]) else f"{mul_name}/output_0"
            self.make_node('Mul', inputs=mul_inputs, outputs=[mul_output], name=mul_name)
            self.make_value(mul_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.vocab_size])
            lm_name = mul_name

        if mask_exists:
            # Save logits mask as initializer
            logits_mask_name = "logits_mask"
            self.make_initializer(self.lm_head_attrs["mask"], logits_mask_name)

            where_name = "/lm_head/Where"
            where_inputs = [logits_mask_name, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{torch.finfo(to_torch_dtype(self.io_dtype)).min}", f"{lm_name}/output_0"]
            where_output = "logits" if not any(exists_checks[3:]) else f"{where_name}/output_0"
            self.make_node('Where', inputs=where_inputs, outputs=[where_output], name=where_name)
            self.make_value(where_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.vocab_size])
            lm_name = where_name

        if softcap_exists:
            # Add final logit softcapping (Div --> Tanh --> Mul)
            div_name = "/lm_head/softcap/Div"
            div_inputs = [f"{lm_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.lm_head_attrs['softcap']}"]
            self.make_div(div_name, div_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.vocab_size])

            tanh_name = "/lm_head/softcap/Tanh"
            self.make_tanh(tanh_name, f"{div_name}/output_0", dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.vocab_size])

            mul_name = "/lm_head/softcap/Mul"
            mul_inputs = [f"{tanh_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.lm_head_attrs['softcap']}"]
            mul_output = "logits" if not any(exists_checks[4:]) else f"{mul_name}/output_0"
            self.make_node('Mul', inputs=mul_inputs, outputs=[mul_output], name=mul_name)
            self.make_value(mul_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.vocab_size])
            lm_name = mul_name

        if cast_exists:
            # Add final cast from io_dtype to logits_dtype
            cast_name = "/lm_head/Cast"
            cast_output = "logits"
            self.make_node('Cast', inputs=[f"{lm_name}/output_0"], outputs=[cast_output], name=cast_name, to=self.output_types['logits'])
            self.make_value(cast_output, self.output_types['logits'], shape=['batch_size', 'sequence_length', self.vocab_size])

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MLP
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_model(self, input_path):
        # Make inputs and outputs to ONNX model
        self.make_inputs_and_outputs()

        # Make pre-processing nodes
        self.make_preprocessing_nodes()

        # Load weights of original model
        if input_path.endswith(".gguf"):
            # Load GGUF model
            try:
                from gguf_model import GGUFModel
            except ImportError:
                from onnxruntime_genai.models.gguf_model import GGUFModel
            model = GGUFModel.from_pretrained(self.model_type, input_path, self.head_size, self.hidden_size, self.intermediate_size, self.num_attn_heads, self.num_kv_heads, self.vocab_size)
            self.layernorm_attrs["add_offset"] = 0  # add offset already done for GGUF models

        elif self.quant_type is not None:
            # Load quantized PyTorch model
            try:
                from quantized_model import QuantModel
            except ImportError:
                from onnxruntime_genai.models.quantized_model import QuantModel
            q_size = self.num_attn_heads * self.head_size
            kv_size = self.num_kv_heads * self.head_size
            model = QuantModel.from_pretrained(self.quant_type, input_path=input_path, quant_attrs=self.quant_attrs, q_size=q_size, kv_size=kv_size, intermediate_size=self.intermediate_size, num_layers=self.num_layers)

        else:
            # Load PyTorch model
            extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

        if "adapter_path" in self.extra_options:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.extra_options["adapter_path"], cache_dir=self.cache_dir, token=self.hf_token)

        # Loop through model and map each module to ONNX/ORT ops
        self.layer_id = 0
        for module in model.modules():
            if (isinstance(module, torch.nn.Embedding) and module.weight.shape[0] == self.vocab_size) or (hasattr(model, "embedding") and module == model.embedding):
                # Checks (Hugging Face logic) or (GGUF logic)
                if not self.exclude_embeds:
                    # Embedding layer
                    print("Reading embedding layer")
                    self.make_embedding(module.weight)
                else:
                    # Exclude embedding layer from model
                    self.layernorm_attrs["root_input"] = "inputs_embeds"
                    self.layernorm_attrs["skip_input"] = "inputs_embeds"

            elif (module.__class__.__name__.endswith("DecoderLayer") or module.__class__.__name__.endswith("GLMBlock")) and self.layer_id < self.num_layers:
                # Each decoder layer of model
                print(f"Reading decoder layer {self.layer_id}")
                self.make_layer(self.layer_id, module)
                self.layer_id += 1

            elif self.layer_id == self.num_layers and self.has_final_norm(module, model):
                # SkipLayerNorm after last decoder layer (MatMul --> SkipLayerNorm)
                print("Reading final norm")
                self.make_layernorm(self.layer_id, module, skip=True, simple=self.layernorm_attrs["simple"], location="final_norm")

            elif (isinstance(module, torch.nn.Linear) and module.out_features == self.vocab_size) or (hasattr(model, "lm_head") and module == model.lm_head):
                # Checks (Hugging Face logic) or (GGUF logic)
                if not self.exclude_lm_head:
                    # Language modeling head (SkipLayerNorm --> logits)
                    print("Reading LM head")
                    self.make_lm_head(module)

        del model

    def has_final_norm(self, module, orig_model):
        # Find where the language model is stored to check attributes. Some classes
        # store the language model in a different attribute than `model.model`.
        if orig_model.__class__.__name__.startswith("Peft"):
            # Model is from PEFT
            model = orig_model.base_model.model
        else:
            model = orig_model

        # Hugging Face names (all models loaded with AutoModelForCausalLM.from_pretrained)
        #
        # hf_norm:                        for most models
        # hf_final_layernorm:             for Phi-2
        # hf_transformer_final_layernorm: for ChatGLM-3
        # hf_language_model_norm:         for Gemma-3 multimodal (4B, 12B, 27B)
        hf_norm = hasattr(model, "model") and hasattr(model.model, "norm") and module == model.model.norm
        hf_final_layernorm = hasattr(model, "model") and hasattr(model.model, "final_layernorm") and module == model.model.final_layernorm
        hf_transformer_final_layernorm = hasattr(model, "transformer") and hasattr(model.transformer, "encoder") and hasattr(model.transformer.encoder, "final_layernorm") and module == model.transformer.encoder.final_layernorm
        hf_language_model_norm = hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "norm") and module == model.model.language_model.norm

        # GGUF names (all models loaded with GGUFModel.from_pretrained)
        gguf_final_norm = hasattr(model, "final_norm") and module == model.final_norm

        hf_names = [hf_norm, hf_final_layernorm, hf_transformer_final_layernorm, hf_language_model_norm]
        gguf_names = [gguf_final_norm]
        return any(hf_names + gguf_names)

    def make_preprocessing_nodes(self):
        self.make_attention_mask_reformatting()

    def make_attention_mask_reformatting(self):
        if self.extra_options.get("enable_cuda_graph", False) or self.extra_options.get("enable_webgpu_graph", False) or self.ep == "dml":
            # ORT does not allow nodes to be placed on mulitple execution providers
            # with graph capture enabled. We've only verified it works with GQA and with
            # past_present_share_buffer enabled(so the total_seq_len in GQA is hardcoded
            # to a fixed value by logic).
            # For other models, we need to check if it works and update the logic here.
            # This assertion is temporary.
            assert self.past_present_share_buffer

        if self.attention_attrs["op_type"] == "GroupQueryAttention":
            self.make_attention_mask_reformatting_for_gqa()
        elif self.attention_attrs["op_type"] == "MultiHeadAttention":
            # Make attention mask reformatting nodes
            #
            #           2D attention mask
            #                   |
            #    attention mask reformatting subgraph
            #                   |
            #         4D causal attention mask
            self.make_attention_mask_reformatting_for_mha()

        if self.attention_attrs["block_sparse"]["sparse_block_size"] != 0:
            self.make_attention_mask_reformatting_for_sparse_attn()

    def make_attention_mask_reformatting_for_mha(self):
        # Make nodes for the attention mask subgraphs that reformat the
        # 2D attention mask (B, S) to 4D causal attention mask (B, N, S, T)
        #
        #             input_ids       past_key_values.0.key
        #            /         \               |
        #         Shape       Shape          Shape
        #          |            |              |
        #        Gather       Gather         Gather
        #       (idx=0)       (idx=1)        (idx=2)
        #          |            |    |\      /
        #          |            |    | \    /
        #          |            |    |   Add                                      attention_mask--------+
        #          |            |    |    |                                       /           \         |
        #      Unsqueeze   Unsqueeze | Unsqueeze                                Shape       Shape       |
        #              \        |    |  /                                         |           |         |
        #               \       |    +-/--------+----------+----------+         Gather      Gather    Unsqueeze
        #                \      |     /         |          |          |        (idx=0)     (idx=1)      |
        #                 \     |    /          |          |          |           |           |         |
        #                  \    |   /       Unsqueeze  Unsqueeze  Unsqueeze   Unsqueeze   Unsqueeze   Unsqueeze
        #                   \   |  /                \ /                    \      |      /              |
        #                    Concat               Concat                    \     |     /               |
        #                   /   |   \                |                       \    |    /                |
        #                  /    |    \               |                        \   |   /                 |
        #                 /     |     \        ConstantOfShape                  Concat                  |
        #                /      |      \           /   \      \                /  |   \                 |
        #               /     Shape     \      Shape   Shape   |              /   |    \                |
        #              /        |        \       |       |     |             /    |     \               |
        #             /         |         \    Slice   Slice   |            /     |      \              |
        #             \   ConstantOfShape  |     |       |     |           /    Shape     \             |
        #              \        |     |    |  Squeeze  Squeeze |          /       |        \            |
        #               \      Mul    |    |     |       |     |         /        |         \           |
        #                \      |     |    | Unsqueeze  Range  |         \  ConstantOfShape  \         /
        #                 \     |     |    |     |       |  |  |          \       |      |    |       /
        #                  \    |     |    |   Concat  Add  |  |           \     Mul     |    |      /
        #                   \   |     |    |     |    /     |  |            \     |      |    |     /
        #                     Equal   |   /    Reshape      |  |             \    |      |    |    /
        #                          \  |  /       |          |  |              \   |      |    |   /
        #                           Where      Less---------+  |               \  |      |    |  /
        #                             |          |             |                 Equal   |   /  /
        #                             |        Where-----------+                      \  |  /  /
        #                             |          |                                     Where  /
        #                             |      Unsqueeze                                   |   /
        #                             |          |                                    Expand
        #                             |      Unsqueeze                                   |
        #                              \    /                                          Cast
        #                              Expand                                            |
        #                                 |                                             Sub
        #                                 |                                            / |
        #                                 |                                           / Cast
        #                                 |                                           |  |
        #                                 |                                           Where
        #                                 |                                             |
        #                                 +----------------------+----------------------+
        #                                                        |
        #                                                       Add
        #                                                        |
        #                                                      Concat

        basename = "/model/attn_mask_reformat"
        input_ids_basename = f"{basename}/input_ids_subgraph"
        past_key_basename = f"{basename}/past_key_subgraph"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Make past_key_values.0.key subgraph
        past_key_gather_name = self.make_past_key_subgraph(past_key_basename)

        # Make common attention mask subgraphs, one each for input_ids and attention_mask
        shared_unsqueeze_name, end_expand_name = self.make_input_ids_subgraph(input_ids_basename, past_key_gather_name)
        end_where_name = self.make_attention_mask_subgraph(attn_mask_basename, shared_unsqueeze_name)

        end_add_name = f"{basename}/Add"
        end_add_inputs = [f"{end_where_name}/output_0", f"{end_expand_name}/output_0"]
        end_add_shape = ["batch_size", 1, "source_sequence_length", "target_sequence_length"]
        self.make_add(end_add_name, end_add_inputs, dtype=self.io_dtype, shape=end_add_shape) # Shape of mask is now (B, 1, S, T)

        tile_name = f"{basename}/Tile"
        tile_inputs = [f"{end_add_name}/output_0", f"/model/constants/INT64/[1, {self.num_attn_heads}, 1, 1]"]
        tile_shape = ["batch_size", self.num_attn_heads, "source_sequence_length", "target_sequence_length"]
        self.make_tile(tile_name, tile_inputs, dtype=self.io_dtype, shape=tile_shape) # Shape of mask is now (B, N, S, T)

        self.mask_attrs["mask_name"] = tile_name

    def make_past_key_subgraph(self, basename):
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, "past_key_values.0.key", shape=[4])
        gather_name = f"{basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/INT64/2"]
        self.make_gather(gather_name, gather_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        return gather_name

    def make_input_ids_subgraph(self, basename, past_key_gather_name):
        # Make shared nodes between past_key_values.0.key (Gather with idx=2) and input_ids (Gather with idx=1) subgraphs
        #
        #       Gather          Gather
        #       (idx=1)         (idx=2)
        #              \       /
        #               \     /
        #                \   /
        #                 Add
        #                  |
        #              Unsqueeze
        shared_add_name = f"{basename}/Add_1"
        shared_add_inputs = [f"{basename}/Gather_2/output_0", f"{past_key_gather_name}/output_0"]
        self.make_add(shared_add_name, shared_add_inputs, dtype=ir.DataType.INT64, shape=[])
        unsqueeze_3_name = f"{basename}/Unsqueeze_3"  # shared unsqueeze for input_ids and past_key_values.0.key
        unsqueeze_3_inputs = [f"{shared_add_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=ir.DataType.INT64, shape=[1])

        # Make the additional subgraph for input_ids
        #
        #       Unsqueeze (unsqueeze_4)                   Shape --> Slice --> Squeeze --> Unsqueeze --> Concat
        #      /          \                              /                                                    \
        # Gather (idx=1)   --> Concat --> ConstantOfShape                                                      Reshape --> Less --> Where --> Unsqueeze --> Unsqueeze --> Expand
        #      \          /                              \                                                     |
        #       Unsqueeze (unsqueeze_5)                   Shape --> Slice --> Squeeze --> Range --> Add -------+
        unsqueeze_inputs = [f"{basename}/Gather_2/output_0", "/model/constants/INT64/[0]"]
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_inputs, dtype=ir.DataType.INT64, shape=[1])
        unsqueeze_5_name = f"{basename}/Unsqueeze_5"
        self.make_unsqueeze(unsqueeze_5_name, unsqueeze_inputs, dtype=ir.DataType.INT64, shape=[1])
        unsqueeze_6_name = f"{basename}/Unsqueeze_6"  # shared unsqueeze for input_ids and attention_mask
        self.make_unsqueeze(unsqueeze_6_name, unsqueeze_inputs, dtype=ir.DataType.INT64, shape=[1])
        concat_2_name = f"{basename}/Concat_2"
        concat_inputs = [f"{unsqueeze_4_name}/output_0", f"{unsqueeze_5_name}/output_0"]
        self.make_concat(concat_2_name, concat_inputs, dtype=ir.DataType.INT64, shape=[2], axis=0)
        constant_shape_name = f"{basename}/ConstantOfShape_2"
        constant_shape_torch_dtype = to_torch_dtype(self.io_dtype)
        constant_shape_value = ir.tensor(torch.tensor([torch.finfo(constant_shape_torch_dtype).min], dtype=constant_shape_torch_dtype), name="make_input_ids_subgraph_shape")
        self.make_constant_of_shape(constant_shape_name, f"{concat_2_name}/output_0", value=constant_shape_value, dtype=self.io_dtype, shape=['unk', 'unk'])

        # Top path
        shape_4_name = f"{basename}/Shape_4"
        self.make_shape(shape_4_name, f"{constant_shape_name}/output_0", shape=[2])
        slice_1_name = f"{basename}/Slice_1"
        slice_1_inputs = [f"{shape_4_name}/output_0", "/model/constants/INT64/[-1]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[0]"]
        self.make_slice(slice_1_name, slice_1_inputs, dtype=ir.DataType.INT64, shape=[1])
        squeeze_1_name = f"{basename}/Squeeze_1"
        squeeze_1_inputs = [f"{slice_1_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_squeeze(squeeze_1_name, squeeze_1_inputs, dtype=ir.DataType.INT64, shape=[])
        unsqueeze_7_name = f"{basename}/output_0"
        unsqueeze_7_inputs = [f"{squeeze_1_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_7_name, unsqueeze_7_inputs, dtype=ir.DataType.INT64, shape=[1])
        concat_3_name = f"{basename}/Concat_3"
        concat_3_inputs = [f"{unsqueeze_7_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_concat(concat_3_name, concat_3_inputs, dtype=ir.DataType.INT64, shape=[2], axis=0)

        # Bottom path
        shape_5_name = f"{basename}/Shape_5"
        self.make_shape(shape_5_name, f"{constant_shape_name}/output_0", shape=[2])
        slice_2_name = f"{basename}/Slice_2"
        slice_2_inputs = [f"{shape_5_name}/output_0", "/model/constants/INT64/[-1]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[0]"]
        self.make_slice(slice_2_name, slice_2_inputs, dtype=ir.DataType.INT64, shape=[1])
        squeeze_2_name = f"{basename}/Squeeze_2"
        squeeze_2_inputs = [f"{slice_2_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_squeeze(squeeze_2_name, squeeze_2_inputs, dtype=ir.DataType.INT64, shape=[])
        range_name = f"{basename}/Range"
        range_inputs = ["/model/constants/INT64/0", f"{squeeze_2_name}/output_0", "/model/constants/INT64/1"]
        self.make_range(range_name, range_inputs)
        add_2_name = f"{basename}/Add_2"
        add_inputs = [f"{range_name}/output_0", "/model/constants/INT64/1"]
        self.make_add(add_2_name, add_inputs, dtype=ir.DataType.INT64, shape=["unk"])

        # Merged path
        reshape_name = f"{basename}/Reshape"
        reshape_inputs = [f"{add_2_name}/output_0", f"{concat_3_name}/output_0"]
        self.make_reshape(reshape_name, reshape_inputs, dtype=ir.DataType.INT64, shape=None)
        less_name = f"{basename}/Less"
        less_inputs = [f"{range_name}/output_0", f"{reshape_name}/output_0"]
        self.make_less(less_name, less_inputs)
        where_2_name = f"{basename}/Where_2"
        where_2_inputs = [f"{less_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/0", f"{constant_shape_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=None)
        unsqueeze_8_name = f"{basename}/Unsqueeze_8"
        unsqueeze_8_inputs = [f"{where_2_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_8_name, unsqueeze_8_inputs, dtype=self.io_dtype, shape=None)
        unsqueeze_9_name = f"{basename}/Unsqueeze_9"
        unsqueeze_9_inputs = [f"{unsqueeze_8_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_unsqueeze(unsqueeze_9_name, unsqueeze_9_inputs, dtype=self.io_dtype, shape=None)

        expand_name = self.make_common_mask_reformat_subgraph(basename, root_input="input_ids" if not self.exclude_embeds else "inputs_embeds", unsqueeze_for_concat=unsqueeze_3_name, unsqueeze_for_expand=unsqueeze_9_name, input_ids_subgraph=True)
        return unsqueeze_6_name, expand_name

    def make_attention_mask_subgraph(self, basename, unsqueeze_for_concat):
        # Make the additional subgraph to join Expand:
        # attention_mask --> Unsqueeze --> Unsqueeze --> Expand
        attention_mask_shape = self.input_shapes["attention_mask"]

        unsqueeze_3_name = f"{basename}/Unsqueeze_3"
        unsqueeze_3_inputs = ["attention_mask", "/model/constants/INT64/[1]"]
        attention_mask_shape.insert(1, 1) # ['batch_size', 'total_sequence_length'] --> ['batch_size', 1, 'total_sequence_length']
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=ir.DataType.INT64, shape=attention_mask_shape)
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        unsqueeze_4_inputs = [f"{unsqueeze_3_name}/output_0", "/model/constants/INT64/[2]"]
        attention_mask_shape.insert(1, 1) # ['batch_size', 1, 'total_sequence_length'] --> ['batch_size', 1, 1, 'total_sequence_length']
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_4_inputs, dtype=ir.DataType.INT64, shape=attention_mask_shape)

        # Make the main subgraph
        expand_name = self.make_common_mask_reformat_subgraph(basename, root_input="attention_mask", unsqueeze_for_concat=unsqueeze_for_concat, unsqueeze_for_expand=unsqueeze_4_name)

        # Make the additional subgraph after Expand:
        #                      +-----------------+
        #                      |                 |
        # Expand --> Cast --> Sub --> Cast --> Where
        cast_1_name = f"{basename}/Cast_1"
        self.make_cast(cast_1_name, f"{expand_name}/output_0", dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])
        sub_name = f"{basename}/Sub"
        sub_inputs = [f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1", f"{cast_1_name}/output_0"]
        self.make_sub(sub_name, sub_inputs, dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])
        cast_2_name = f"{basename}/Cast_2"
        self.make_cast(cast_2_name, f"{sub_name}/output_0", dtype=ir.DataType.BOOL, shape=["unk", "unk", "unk", "unk"])
        where_2_name = f"{basename}/Where_2"
        where_2_inputs = [f"{cast_2_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{torch.finfo(to_torch_dtype(self.io_dtype)).min}", f"{sub_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])

        return where_2_name

    def make_common_mask_reformat_subgraph(self, basename, root_input, unsqueeze_for_concat, unsqueeze_for_expand, input_ids_subgraph=False):
        #             root_input
        #            /         \
        #         Shape       Shape
        #          |            |
        #        Gather       Gather
        #       (idx=0)       (idx=1)
        #          |            |
        #      Unsqueeze   Unsqueeze   Unsqueeze (unsqueeze_for_concat)
        #              \        |       /
        #               \       |      /
        #                \      |     /
        #                 \     |    /
        #                  \    |   /
        #                   \   |  /
        #                    Concat
        #                   /   |   \
        #                  /    |    \
        #                 /     |     \
        #                /      |      \
        #               /     Shape     \
        #              /        |        \
        #             /         |         \
        #             \   ConstantOfShape  |
        #              \        |     |    |
        #               \      Mul    |    |
        #                \      |     |    |
        #                 \     |     |    |
        #                  \    |     |    |
        #                   \   |     |    |
        #                     Equal   |   /
        #                          \  |  /
        #                           Where
        #                             |   Unsqueeze (unsqueeze_for_expand)
        #                              \    /
        #                              Expand

        shape_1_name = f"{basename}/Shape_1"
        self.make_shape(shape_1_name, root_input, shape=[3] if self.exclude_embeds and input_ids_subgraph else [2])
        shape_2_name = f"{basename}/Shape_2"
        self.make_shape(shape_2_name, root_input, shape=[3] if self.exclude_embeds and input_ids_subgraph else [2])
        gather_1_name = f"{basename}/Gather_1"
        gather_1_inputs = [f"{shape_1_name}/output_0", "/model/constants/INT64/0"]
        self.make_gather(gather_1_name, gather_1_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        gather_2_name = f"{basename}/Gather_2"
        gather_2_inputs = [f"{shape_2_name}/output_0", "/model/constants/INT64/1"]
        self.make_gather(gather_2_name, gather_2_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        unsqueeze_1_name = f"{basename}/Unsqueeze_1"
        unsqueeze_1_inputs = [f"{gather_1_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_1_name, unsqueeze_1_inputs, dtype=ir.DataType.INT64, shape=[1])
        unsqueeze_2_name = f"{basename}/Unsqueeze_2"
        unsqueeze_2_inputs = [f"{gather_2_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_unsqueeze(unsqueeze_2_name, unsqueeze_2_inputs, dtype=ir.DataType.INT64, shape=[1])

        concat_name = f"{basename}/Concat" if not input_ids_subgraph else f"{basename}/Concat_1"
        concat_first_two_inputs = [f"{unsqueeze_1_name}/output_0", "/model/constants/INT64/[1]"]
        concat_last_two_inputs = [f"{unsqueeze_for_concat}/output_0", f"{unsqueeze_2_name}/output_0"] if not input_ids_subgraph else [f"{unsqueeze_2_name}/output_0", f"{unsqueeze_for_concat}/output_0"]
        concat_inputs = concat_first_two_inputs + concat_last_two_inputs
        self.make_concat(concat_name, concat_inputs, dtype=ir.DataType.INT64, shape=[4], axis=0)
        shape_3_name = f"{basename}/Shape_3"
        self.make_shape(shape_3_name, f"{concat_name}/output_0", shape=[1])
        constant_shape_name = f"{basename}/ConstantOfShape" if not input_ids_subgraph else f"{basename}/ConstantOfShape_1"
        constant_shape_value = ir.tensor([1], dtype=ir.DataType.INT64)
        self.make_constant_of_shape(constant_shape_name, f"{shape_3_name}/output_0", value=constant_shape_value, dtype=ir.DataType.INT64, shape=["unk"])
        mul_name = f"{basename}/Mul"
        mul_inputs = [f"{constant_shape_name}/output_0", "/model/constants/INT64/-1"]
        self.make_mul(mul_name, mul_inputs, dtype=ir.DataType.INT64, shape=["unk"])
        equal_name = f"{basename}/Equal"
        equal_inputs = [f"{concat_name}/output_0", f"{mul_name}/output_0"]
        self.make_equal(equal_name, equal_inputs, shape=[4])

        where_name = f"{basename}/Where_1"
        where_inputs = [f"{equal_name}/output_0", f"{constant_shape_name}/output_0", f"{concat_name}/output_0"]
        self.make_where(where_name, where_inputs, dtype=ir.DataType.INT64, shape=[4])
        expand_name = f"{basename}/Expand"
        expand_inputs = [f"{unsqueeze_for_expand}/output_0", f"{where_name}/output_0"]
        expand_dtype = self.io_dtype if input_ids_subgraph else ir.DataType.INT64
        expand_shape = None if input_ids_subgraph else ["unk", "unk", "unk", "unk"]
        self.make_expand(expand_name, expand_inputs, dtype=expand_dtype, shape=expand_shape)

        return expand_name

    def make_attention_mask_graph_capture_reformatting_for_gqa(self, attn_mask_basename):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in GroupQueryAttention
        #
        # Key difference vs make_attention_mask_standard_reformatting_for_gqa:
        # - Standard mode: total_seq_len is calculated from Shape op (always runs on CPU)
        # - Graph capture mode: No Shape ops inserted to ensure all ops run on GPU (no CPU ops)
        #
        #          attention_mask
        #               |
        #         Cast to int32
        #               |
        #           ReduceSum
        #              /    \
        #             /      \
        #           Sub    Squeeze
        #            |        |
        #       seqlens_k  total_seq_len
        #         (1D)       (int)

        # Calculate ReduceSum from attention_mask
        cast_1_name = f"{attn_mask_basename}/Cast"
        self.make_cast(cast_1_name, "attention_mask", dtype=ir.DataType.INT32, shape=["batch_size", "total_sequence_length"])
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = [f"{cast_1_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT32, shape=["batch_size", 1])

        # Left branch: Calculate seqlens_k = ReduceSum - 1
        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT32/[1]"]
        self.make_sub(sub_name, sub_inputs, dtype=ir.DataType.INT32, shape=["batch_size", 1])

        # Right branch: Squeeze to get int value for total_seq_len
        squeeze_name = f"{attn_mask_basename}/Squeeze"
        squeeze_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT64/[0]"]
        self.make_squeeze(squeeze_name, squeeze_inputs, dtype=ir.DataType.INT32, shape=[])

        self.mask_attrs["seqlens_k"] = sub_name
        self.mask_attrs["total_seq_len"] = squeeze_name

    def make_attention_mask_standard_reformatting_for_gqa(self, attn_mask_basename):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in GroupQueryAttention
        #
        #                attention_mask
        #               /              \
        #          ReduceSum          Shape
        #              |                |
        #             Sub             Gather
        #              |                |
        #        Cast to int32    Cast to int32
        #              |                |
        #          seqlens_k      total_seq_len
        #            (1D)             (int)

        # Left path
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = ["attention_mask", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT64, shape=["batch_size", 1])
        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT64/[1]"]
        self.make_sub(sub_name, sub_inputs, dtype=ir.DataType.INT64, shape=["batch_size", 1])
        cast_1_name = f"{attn_mask_basename}/Sub/Cast"
        self.make_cast(cast_1_name, f"{sub_name}/output_0", dtype=ir.DataType.INT32, shape=["batch_size", 1])

        # Right path
        shape_name = f"{attn_mask_basename}/Shape"
        self.make_shape(shape_name, "attention_mask", shape=[2])
        gather_name = f"{attn_mask_basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/INT64/1"]
        self.make_gather(gather_name, gather_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        cast_2_name = f"{attn_mask_basename}/Gather/Cast"
        self.make_cast(cast_2_name, f"{gather_name}/output_0", dtype=ir.DataType.INT32, shape=None)

        self.mask_attrs["seqlens_k"] = cast_1_name
        self.mask_attrs["total_seq_len"] = cast_2_name

    def make_attention_mask_reformatting_for_gqa(self):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in GroupQueryAttention
        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        if self.extra_options.get("enable_webgpu_graph", False):
            self.make_attention_mask_graph_capture_reformatting_for_gqa(attn_mask_basename)
        else:
            self.make_attention_mask_standard_reformatting_for_gqa(attn_mask_basename)

    def make_attention_mask_reformatting_for_sparse_attn(self):
        # Make nodes for the attention mask subgraph that calculates
        # attributes about the 2D attention mask to use in SparseAttention
        #
        #                attention_mask
        #               /              \
        #          ReduceSum          Shape
        #              |                |
        #        Cast to int32        Gather
        #              |                |
        #      key_total_seq_lens  Cast to int32
        #            (1D)               |
        #                          total_seq_len
        #                             (int)

        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Left path
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = ["attention_mask", "/model/constants/INT64/[1]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.INT64, shape=["batch_size", 1])
        cast_1_name = f"{attn_mask_basename}/ReduceSum/Cast"
        self.make_cast(cast_1_name, f"{reduce_sum_name}/output_0", dtype=ir.DataType.INT32, shape=["batch_size", 1])

        # Right path
        shape_name = f"{attn_mask_basename}/Shape"
        self.make_shape(shape_name, "attention_mask", shape=[2])
        gather_name = f"{attn_mask_basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/INT64/1"]
        self.make_gather(gather_name, gather_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)
        cast_2_name = f"{attn_mask_basename}/Gather/Cast"
        self.make_cast(cast_2_name, f"{gather_name}/output_0", dtype=ir.DataType.INT32, shape=None)

        self.mask_attrs["key_total_seq_lens"] = cast_1_name
        self.mask_attrs["total_seq_len"] = cast_2_name

    def make_position_ids_reformatting(self):
        # For most cases, position_ids are already properly formatted as 2D tensors
        # with int64 values matching input_ids shape, so we can use them directly
        return "position_ids"


class LlamaModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class QwenModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()


class PhiModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def make_layer(self, layer_id, layer):
        # Each Phi decoder layer is defined as:
        # input_layernorm --> attention --> MLP --> residual_add
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        residual_add_name = f"/model/layers.{layer_id}/residual_add/Add"
        residual_add_inputs = [self.layernorm_attrs['skip_input'], self.mlp_attrs["output_0"]]
        self.make_add(residual_add_name, residual_add_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

        # Assign output 0 of residual Add as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_add_name}/output_0"


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = np.round(np.sqrt(self.hidden_size), decimals=2)
        self.layernorm_attrs["add_offset"] = 1


class Gemma2Model(GemmaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = False
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = False
        self.attention_attrs["scale"] = config.query_pre_attn_scalar ** -0.5
        self.is_local = lambda layer_id: layer_id % 2 == 1

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
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
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
        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=False, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(layer_id, layer.pre_feedforward_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="pre_feedforward")
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
        self.make_layernorm(layer_id, layer.post_feedforward_layernorm, skip=False, simple=self.layernorm_attrs["simple"], location="post_feedforward")
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = original_window_size if self.is_local(layer_id) else -1  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size


class Phi3MiniModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MiniLongRoPEModel(Phi3MiniModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()

        # Set position_ids_name based on whether position_ids is available as an input
        if "position_ids" in self.input_names:
            position_ids_result = self.make_position_ids_reformatting()
            self.position_ids_name = f"{position_ids_result}/output_0" if position_ids_result != "position_ids" else "position_ids"
        else:
            # When position_ids is not an input (use_rope_in_attn is True),
            # position_ids won't be used since rotary embeddings are handled in GQA
            self.position_ids_name = None

    def make_position_ids_reformatting(self):
        if self.ep not in self.eps_without_if_support:
            position_ids_input_to_rotemb = super().make_position_ids_reformatting()
            return position_ids_input_to_rotemb

        # Make Long RoPE scaling subgraph to adjust position_ids for sequences beyond original context length
        # For WebGPU: use int32 for all ops due to limited int64 support, then cast back to int64
        # For other EPs: use native int64 throughout
        #
        # WebGPU graph:                              Other EPs graph:
        #   position_ids                               position_ids
        #        |                                          |
        #   Cast (int32)                               ReduceMax (int64)
        #        |                                          |
        #   ReduceMax (int32)                          GreaterOrEqual
        #        |                                          |
        #   GreaterOrEqual                             Cast (int64)
        #        |                                          |
        #   Cast (int32)                               Mul (int64)
        #        |                                          |
        #   Mul (int32)                                Add (int64)
        #        |
        #   Add (int32)
        #        |
        #   Cast (int64)

        basename = "/model/pos_ids_reformat"
        proto_dtype = self.input_types["position_ids"]
        str_dtype = self.to_str_dtype(proto_dtype)

        # For WebGPU, use int32 for computation due to limited int64 ops support
        is_webgpu = self.extra_options.get("enable_webgpu_graph", False)
        compute_dtype = ir.DataType.INT32 if is_webgpu else proto_dtype
        compute_str_dtype = self.to_str_dtype(compute_dtype)

        # Cast position_ids to int32 for WebGPU
        input_tensor = "position_ids"
        if is_webgpu:
            cast_input_name = f"{basename}/Cast_input"
            self.make_cast(cast_input_name, input_tensor, dtype=ir.DataType.INT32, shape=["batch_size", "sequence_length"])
            input_tensor = f"{cast_input_name}/output_0"

        reduce_max_name = f"{basename}/ReduceMax"
        reduce_max_inputs = [input_tensor]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=compute_dtype, shape=[1])
        greater_or_equal_name = f"{basename}/GreaterOrEqual"
        greater_or_equal_inputs = [f"{reduce_max_name}/output_0", f"/model/constants/{compute_str_dtype}/{self.original_context_length}"]
        self.make_greater_or_equal(greater_or_equal_name, greater_or_equal_inputs, shape=[])
        cast_name = f"{basename}/Cast"
        self.make_cast(cast_name, f"{greater_or_equal_name}/output_0", dtype=compute_dtype, shape=None)
        mul_name = f"{basename}/Mul"
        mul_inputs = [f"{cast_name}/output_0", f"/model/constants/{compute_str_dtype}/{self.original_context_length}"]
        self.make_mul(mul_name, mul_inputs, dtype=compute_dtype, shape=None)
        add_1_name = f"{basename}/Add_1"
        add_1_inputs = [f"{mul_name}/output_0", input_tensor]
        self.make_add(add_1_name, add_1_inputs, dtype=compute_dtype, shape=["batch_size", "sequence_length"])

        # Cast back to int64 for WebGPU to maintain compatibility
        result_name = add_1_name
        if is_webgpu:
            cast_output_name = f"{basename}/Cast_output"
            self.make_cast(cast_output_name, f"{add_1_name}/output_0", dtype=ir.DataType.INT64, shape=["batch_size", "sequence_length"])
            result_name = cast_output_name

        return result_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        if self.position_ids_name is not None:
            super().make_attention(layer_id, attention, root_input, position_ids=self.position_ids_name, **kwargs)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)

class Phi3SmallModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.embed_attrs["scale"] = config.mup_embedding_multiplier
        self.rope_attrs["t_dtype"] = torch.float32
        self.lm_head_attrs["scale"] = 1 / config.mup_width_multiplier

        self.calculate_block_mask()
        self.dense_attention_every_n_layers = config.dense_attention_every_n_layers
        if config.mup_use_scaling:
            self.attention_attrs["scale"] = config.mup_attn_multiplier / self.head_size

        self.clamp_limit = config.gegelu_limit

    def calculate_cdiv(self, a, b):
        return -(a // -b)

    def calculate_block_mask(self):
        # Initialize parameters for calculating block dense mask
        n_heads = self.num_attn_heads
        q_len = self.context_length
        N_CTX = self.context_length
        BLOCK = self.attention_attrs["block_sparse"]["sparse_block_size"]
        local_blocks = self.attention_attrs["block_sparse"]["local_blocks"]
        vert_stride = self.attention_attrs["block_sparse"]["vert_stride"]
        homo_head = self.attention_attrs["block_sparse"]["homo_head"]

        N_BLOCK = self.calculate_cdiv(N_CTX, BLOCK)
        if homo_head:
            q_pos = torch.arange(N_BLOCK)[:, None]
            k_pos = torch.arange(N_BLOCK)[None]
            mask_vert_strided = (torch.arange(N_BLOCK) + 1) % vert_stride == 0
            block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided))
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[-N_BLOCK_Q:].to_sparse_csr()

            crows = block_mask_dense_output.crow_indices()
            cols = block_mask_dense_output.col_indices()

            crows = crows[None].expand(n_heads, crows.shape[0])
            cols = cols[None].expand(n_heads, cols.shape[0])
        else:
            q_pos = torch.arange(N_BLOCK)[None, :, None]
            k_pos = torch.arange(N_BLOCK)[None, None]
            head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
            mask_vert_strided = [(torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(n_heads)]
            mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
            block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided))
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[:, -N_BLOCK_Q:]

            # Dense to crow_col
            pad = -1
            dim = block_mask_dense_output.dim()
            assert dim in (2, 3)
            if dim == 2:
                block_mask_dense_output = block_mask_dense_output[None]
            block_mask_dense_output = [xi.to_sparse_csr() for xi in block_mask_dense_output]
            crows = torch.vstack([xi.crow_indices() for xi in block_mask_dense_output])
            cols = [xi.col_indices() for xi in block_mask_dense_output]
            max_cols = max(len(xi) for xi in cols)
            cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
            cols = torch.vstack(cols)
            if dim == 2:
                crows = crows[0]
                cols = cols[0]

        # Create tensors for row indices and col indices
        crows_name = "block_row_indices"
        self.make_initializer(crows, crows_name, to=ir.DataType.INT32)
        self.mask_attrs["block_row_indices"] = crows_name

        cols_name = "block_col_indices"
        self.make_initializer(cols, cols_name, to=ir.DataType.INT32)
        self.mask_attrs["block_col_indices"] = cols_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        dense_attention_op = self.attention_attrs["op_type"]
        sparse_attention_op = "SparseAttention"

        # Use dense attention every n layers and use sparse attention otherwise
        if (self.layer_id + 1) % self.dense_attention_every_n_layers != 0:
            # Use sparse attention
            self.attention_attrs["op_type"] = sparse_attention_op

        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        qkv_weight = attention.query_key_value.weight.T.view(self.hidden_size, self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size)
        qkv_bias = attention.query_key_value.bias.view(self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size)

        attention.q_proj = torch.nn.Linear(in_features=q_size, out_features=q_size)
        attention.q_proj.weight = torch.nn.Parameter(qkv_weight[:, :, :-2].reshape(q_size, q_size).T, requires_grad=False)
        attention.q_proj.bias = None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, :-2].flatten(), requires_grad=False)

        attention.k_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.k_proj.weight = torch.nn.Parameter(qkv_weight[:, :, [-2]].reshape(q_size, kv_size).T, requires_grad=False)
        attention.k_proj.bias = None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, [-2]].flatten(), requires_grad=False)

        attention.v_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.v_proj.weight = torch.nn.Parameter(qkv_weight[:, :, [-1]].reshape(q_size, kv_size).T, requires_grad=False)
        attention.v_proj.bias = None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, [-1]].flatten(), requires_grad=False)

        del qkv_weight
        del qkv_bias
        del attention.query_key_value

        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.attention_attrs["op_type"] = dense_attention_op

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #           root_input
        #               |
        #          UpProjMatMul
        #               |
        #           UpProjAdd
        #          /          \
        #         /            \
        #        /              \
        #      Slice             Slice
        #    (even idx)        (odd idx)
        #    /   |   \         /   |   \
        #  Cast  |    |      Cast  |    |
        #   |    |    |       |    |    |
        # IsInf  |   Clip   IsInf  |   Clip
        #   |    |    |       |    |    |
        #    \   |   /         \   |   /
        #     \  |  /           \  |  /
        #      Where             Where
        #        |                 |
        #    QuickGelu            Add
        #        |                 |
        #        +--------+--------+
        #                 |
        #                Mul
        #                 |
        #           DownProjMatMul
        #                 |
        #            DownProjAdd

        # Make input MatMul and Add nodes
        up_matmul_name = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        self.make_matmul(mlp.up_proj, up_matmul_name, root_input)
        up_add_name = f"/model/layers.{layer_id}/mlp/up_proj/Add"
        self.make_add_bias(mlp.up_proj.bias, up_add_name, f"{up_matmul_name}/output_0")

        # Left path
        slice_1_name = f"/model/layers.{layer_id}/mlp/gelu/Slice"
        slice_1_inputs = [f"{up_add_name}/output_0", "/model/constants/INT64/[0]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[-1]", "/model/constants/INT64/[2]"]
        self.make_slice(slice_1_name, slice_1_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        cast_1_name = f"/model/layers.{layer_id}/mlp/gelu/Cast"
        self.make_cast(cast_1_name, f"{slice_1_name}/output_0", dtype=ir.DataType.FLOAT, shape=["batch_size", "sequence_length", self.intermediate_size])
        isinf_1_name = f"/model/layers.{layer_id}/mlp/gelu/IsInf"
        self.make_isinf(isinf_1_name, f"{cast_1_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size])
        clip_1_name = f"/model/layers.{layer_id}/mlp/gelu/Clip"
        clip_1_inputs = [f"{slice_1_name}/output_0", "", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}"]
        self.make_clip(clip_1_name, clip_1_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        where_1_name = f"/model/layers.{layer_id}/mlp/gelu/Where"
        where_1_inputs = [f"{isinf_1_name}/output_0", f"{slice_1_name}/output_0", f"{clip_1_name}/output_0"]
        self.make_where(where_1_name, where_1_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        # Make activation
        act_fn_name = self.make_activation(layer_id, root_input=f"{where_1_name}/output_0")

        # Right path
        slice_2_name = f"/model/layers.{layer_id}/mlp/linear/Slice"
        slice_2_inputs = [f"{up_add_name}/output_0", "/model/constants/INT64/[1]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[-1]", "/model/constants/INT64/[2]"]
        self.make_slice(slice_2_name, slice_2_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        cast_2_name = f"/model/layers.{layer_id}/mlp/linear/Cast"
        self.make_cast(cast_2_name, f"{slice_2_name}/output_0", dtype=ir.DataType.FLOAT, shape=["batch_size", "sequence_length", self.intermediate_size])
        isinf_2_name = f"/model/layers.{layer_id}/mlp/linear/IsInf"
        self.make_isinf(isinf_2_name, f"{cast_2_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size])
        clip_2_name = f"/model/layers.{layer_id}/mlp/linear/Clip"
        clip_2_inputs = [f"{slice_2_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/-{self.clamp_limit}", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}"]
        self.make_clip(clip_2_name, clip_2_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        where_2_name = f"/model/layers.{layer_id}/mlp/linear/Where"
        where_2_inputs = [f"{isinf_2_name}/output_0", f"{slice_2_name}/output_0", f"{clip_2_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        add_name = f"/model/layers.{layer_id}/mlp/linear/Add"
        add_inputs = [f"{where_2_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1"]
        self.make_add(add_name, add_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make Mul node after activation
        mul_name = f"/model/layers.{layer_id}/mlp/Mul"
        mul_inputs = [f"{act_fn_name}/output_0", f"{add_name}/output_0"]
        self.make_mul(mul_name, mul_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make output MatMul and Add nodes
        down_matmul_name = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        self.make_matmul(mlp.down_proj, down_matmul_name, f"{mul_name}/output_0")
        down_add_name = f"/model/layers.{layer_id}/mlp/down_proj/Add"
        self.make_add_bias(mlp.down_proj.bias, down_add_name, f"{down_matmul_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_add_name}/output_0"


class Phi3SmallLongRoPEModel(Phi3SmallModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()


class Phi3VModel(Phi3MiniLongRoPEModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MoELongRoPEModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        assert io_dtype == ir.DataType.FLOAT16, "This model only supports float16 io type."
        self.layernorm_attrs["simple"] = False
        self.moe_attrs["use_sparse_mixer"] = True
        self.make_rotary_embedding_multi_cache()

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MoE
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.make_block_sparse_moe(layer_id, layer.block_sparse_moe, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True


class NemotronModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.layernorm_attrs["add_offset"] = 1

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #          root_input
        #              |
        #         UpProjMatMul
        #              |
        #           ActFunc
        #              |
        #         DownProjMatMul

        up_basename = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        up_name = self.make_matmul(mlp.up_proj, up_basename, root_input)

        act_fn_name = self.make_activation(layer_id, root_input=f"{up_name}/output_0")

        # Make output MatMul node
        down_basename = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        down_name = self.make_matmul(mlp.down_proj, down_basename, f"{act_fn_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"


class ChatGLMModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.rope_attrs["partial_rotary_factor"] = 0.5  # Line 755 of modeling_chatglm.py check self.rotary_pos_emb declaration
        self.rope_attrs["num_heads"] = self.num_attn_heads
        self.rope_attrs["rotary_embedding_dim"] = int(self.head_size * self.rope_attrs["partial_rotary_factor"])
        self.rope_attrs["interleaved"] = 1

    def make_mlp(self, layer_id, mlp, root_input):
        if not hasattr(mlp, 'down_proj'):
            # Attribute does not exist for original PyTorch model only
            mlp.down_proj = mlp.dense_4h_to_h
        super().make_mlp(layer_id, mlp, root_input)

    def make_layer(self, layer_id, layer):
        layer.self_attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.self_attention
        super().make_layer(layer_id, layer)


class OLMoModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        layernorm.weight = torch.ones(self.hidden_size)
        layernorm.bias = torch.zeros(self.hidden_size)
        super().make_layernorm(layer_id, layernorm, skip, simple, location)


class GraniteModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = config.embedding_multiplier
        self.attention_attrs["scale"] = config.attention_multiplier
        self.lm_head_attrs["scale"] = 1 / config.logits_scaling
        self.residual_scale = config.residual_multiplier

    def make_layer(self, layer_id, layer):
        # Each Granite decoder layer is defined as:
        # input_layernorm --> attention --> Mul --> output_layernorm --> MLP --> Mul
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        residual_mul_1_name = f"/model/layers.{layer_id}/residual_mul/Mul_1"
        residual_mul_1_inputs = [self.layernorm_attrs["skip_input"], f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.residual_scale}"]
        self.make_mul(residual_mul_1_name, residual_mul_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_mul_1_name}/output_0"

        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        residual_mul_2_name = f"/model/layers.{layer_id}/residual_mul/Mul_2"
        residual_mul_2_inputs = [self.layernorm_attrs["skip_input"], f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.residual_scale}"]
        self.make_mul(residual_mul_2_name, residual_mul_2_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_mul_2_name}/output_0"

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True


class Phi4MMModel(Phi3VModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.matmul_attrs["use_lora"] = True
        self.attention_attrs["use_packed_matmul"] = False

    def make_layer(self, layer_id, layer):
        layer.self_attn.qkv_proj.lora_A.default = layer.self_attn.qkv_proj.lora_A.vision
        layer.self_attn.qkv_proj.lora_B.default = layer.self_attn.qkv_proj.lora_B.vision
        layer.self_attn.qkv_proj.scaling["default"] = layer.self_attn.qkv_proj.scaling["vision"]
        layer.self_attn.o_proj.lora_A.default = layer.self_attn.o_proj.lora_A.vision
        layer.self_attn.o_proj.lora_B.default = layer.self_attn.o_proj.lora_B.vision
        layer.self_attn.o_proj.scaling["default"] = layer.self_attn.o_proj.scaling["vision"]

        layer.mlp.gate_up_proj.lora_A.default = layer.mlp.gate_up_proj.lora_A.vision
        layer.mlp.gate_up_proj.lora_B.default = layer.mlp.gate_up_proj.lora_B.vision
        layer.mlp.gate_up_proj.scaling["default"] = layer.mlp.gate_up_proj.scaling["vision"]
        layer.mlp.down_proj.lora_A.default = layer.mlp.down_proj.lora_A.vision
        layer.mlp.down_proj.lora_B.default = layer.mlp.down_proj.lora_B.vision
        layer.mlp.down_proj.scaling["default"] = layer.mlp.down_proj.scaling["vision"]

        super().make_layer(layer_id, layer)


class Gemma3Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.is_local = lambda layer_id: bool((layer_id + 1) % 6)
        self.rope_local_theta = config.rope_local_base_freq
        self.make_rotary_embedding_multi_cache()

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_rotary_embedding_multi_cache(self):
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        super().make_rotary_embedding_caches(cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name)

        # Create the new cos/sin caches for local attention layers with its own theta value
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super().make_rotary_embedding_caches(cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name)

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get("cos_cache_name", self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name)
        sin_cache_name = kwargs.get("sin_cache_name", self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name)
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)


class ErnieModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Ernie uses interleaved rotary position embeddings.
        self.rotemb_attrs["interleaved"] = 1

        # Ernie uses a `compression_ratio` for its RoPE scaling.
        # The original RoPE logic in ernie is: position_ids / compression_ratio,
        # which is equivalent to scaling the frequencies (inv_freq) by 1 / compression_ratio.
        if hasattr(config, "compression_ratio") and config.compression_ratio != 1.0:
            self.rotemb_attrs["rescale_factors"] = 1.0 / config.compression_ratio


class SmolLM3Model(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layer_types = config.layer_types
        self.no_rope_layers = config.no_rope_layers

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # SmolLM3 uses per-layer conditional RoPE and Sliding Window Attention.
        # So, we temporarily modify the model's attributes before calling the
        # base `make_attention` method, then restore them immediately after.
        original_use_rope = self.attention_attrs["use_rope_in_attn"]
        original_window_size = self.window_size

        # Enable/disable RoPE for the current layer.
        self.attention_attrs["use_rope_in_attn"] = bool(self.no_rope_layers[layer_id])

        # Set the sliding window size for the current layer.
        assert self.layer_types[layer_id] in {"sliding_attention", "full_attention"}
        if self.layer_types[layer_id] == "full_attention":
            self.window_size = -1

        # Call the original `make_attention` with the temporarily-modified settings.
        super().make_attention(layer_id, attention, root_input, **kwargs)

        # Restore original values
        self.attention_attrs["use_rope_in_attn"] = original_use_rope
        self.window_size = original_window_size


class GPTOSSModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.is_local = lambda layer_id: bool(layer_id % 2 == 0)
        self.attention_attrs["sinks"] = True

        self.moe_attrs["activation_alpha"] = 1.702
        self.moe_attrs["activation_beta"] = 1.0
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["normalize_routing_weights"] = True
        self.moe_attrs["swiglu_fusion"] = 1

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MoE
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.make_moe(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_rotary_embedding_caches_from_scratch(self):
        inv_freq = self.rope_attrs["theta"] ** (torch.arange(0, self.head_size, 2, dtype=torch.float) / self.head_size)
        inv_freq = self.make_inv_freq_rescaled(inv_freq)

        t = torch.arange(self.rope_attrs["cache_length"], dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos_cache, sin_cache = freqs.cos() * self.rope_attrs["mscale"], freqs.sin() * self.rope_attrs["mscale"]
        return cos_cache, sin_cache

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = original_window_size if self.is_local(layer_id) else -1  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size

    def make_moe(self, layer_id, mlp, root_input):
        if self.ep in {"cpu", "cuda"}:
            self.make_moe_fused(layer_id, mlp, root_input)            
        else:
            self.make_moe_decomposed(layer_id, mlp, root_input)

    def make_moe_decomposed(self, layer_id, mlp, root_input):
        # Make nodes for the MoE subgraph
        #
        #                                              root_input
        #                                                  |
        #          +---------------------------------------+
        #          |                                       |
        #      Unsqueeze                                 MatMul
        #       (dim=2)                                 (Router)
        #          |                                       |
        #       Expand                                    Add
        #          |                                       |
        #      Unsqueeze                                 Cast
        #      (dim=-1)                               (to = FP32)
        #          |                                       |
        #          |                                     TopK
        #          |                                       |
        #          |                                     Cast
        #          |                                 (to = io_dtype)
        #          |                                       |
        #          |           +-------------+-------------+-------------+-------------+
        #          |           |             |             |             |             |
        #          |         Gather        Gather        Gather        Gather       Softmax
        #          |      (MLP1 Weight)  (MLP1 Bias)  (MLP2 Weight)  (MLP2 Bias)   (Expert Weights)
        #          |           |             |             |             |             |
        #          +-----+-----+         Unsqueeze         |         Unsqueeze     Unsqueeze
        #                |               (dim=-1)          |         (dim=-1)      (dim=-1)
        #             MatMul                 |             |             |             |
        #          (gate_up_proj)            |             |             |         Unsqueeze
        #                |                   |             |             |         (dim=-1)
        #                +---------+---------+             |             |             |
        #                          |                       |             |           Cast
        #                         Add                      |             |        (to = FP32)
        #                          |                       |             |             |
        #                    +-----+-----+                 |             |             |
        #                    |           |                 |             |             |
        #                  Slice       Slice               |             |             |
        #                    |           |                 |             |             |
        #                  Clip        Clip                |             |             |
        #                    |  \        |                 |             |             |
        #                   Mul  \       |                 |             |             |
        #                    |    \      |                 |             |             |
        #                 Sigmoid  |     |                 |             |             |
        #                    |     |    Add                |             |             |
        #                    \    /      |                 |             |             |
        #                     \  /       |                 |             |             |
        #                      Mul       |                 |             |             |
        #                       |        |                 |             |             |
        #                       +----+---+                 |             |             |
        #                            |                     |             |             |
        #                           Mul                    |             |             |
        #                            |                     |             |             |
        #                            +----------+----------+             |             |
        #                                       |                        |             |
        #                                    MatMul                      |             |
        #                                  (down_proj)                   |             |
        #                                       |                        |             |
        #                                       +------------+-----------+             |
        #                                                    |                         |
        #                                                   Add                        |
        #                                                    |                         |
        #                                                   Cast                       |
        #                                               (to = FP32)                    |
        #                                                    |                         |
        #                                                    +------------+------------+
        #                                                                 |
        #                                                                Mul
        #                                                                 |
        #                                                             ReduceSum
        #                                                              (dim=2)
        #                                                                 |
        #                                                              Squeeze
        #                                                              (dim=-1)
        #                                                                 |
        #                                                                Cast
        #                                                           (to = io_dtype)
        basename = f"/model/layers.{layer_id}/moe"
        use_cast = self.io_dtype != ir.DataType.FLOAT

        # Make root_input expansion nodes (root_input --> Unsqueeze --> Expand --> Unsqueeze)
        expand_root_input_unsqueeze_1_name = f"{basename}/expand_root_input/Unsqueeze_1"
        expand_root_input_unsqueeze_1_inputs = [root_input, "/model/constants/INT64/[2]"]
        self.make_unsqueeze(expand_root_input_unsqueeze_1_name, expand_root_input_unsqueeze_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', 1, self.hidden_size])
        expand_name = f"{basename}/expand_root_input/Expand"
        expand_inputs = [f"{expand_root_input_unsqueeze_1_name}/output_0", f"/model/constants/INT64/[1, 1, {self.moe_attrs['top_k']}, 1]"]
        self.make_expand(expand_name, expand_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.hidden_size])
        expand_root_input_unsqueeze_2_name = f"{basename}/expand_root_input/Unsqueeze_2"
        expand_root_input_unsqueeze_2_inputs = [f"{expand_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(expand_root_input_unsqueeze_2_name, expand_root_input_unsqueeze_2_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.hidden_size, 1])

        # Make router nodes
        #                                          +--> Gather --> (MLP1 weight)
        #                                          |
        #                                          +--> Gather --> Unsqueeze --> (MLP1 bias)
        #                                          |
        # root_input --> MatMul --> Add --> TopK --+--> Gather --> (MLP2 weight)
        #                                          |
        #                                          +--> Gather --> Unsqueeze --> (MLP2 bias)
        #                                          |
        #                                          +--> Softmax --> Unsqueeze --> Unsqueeze --> Cast
        #
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.router, router_basename, root_input)
        router_add_name = f"{basename}/router/Add"
        self.make_add_bias(mlp.router.bias, router_add_name, root_input=f"{router_matmul_name}/output_0")

        if use_cast:
            topk_fp32_name = f"{basename}/topk_fp32/Cast"
            self.make_cast(topk_fp32_name, f"{router_add_name}/output_0", ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.moe_attrs["num_experts"]])
        topk_name = f"{basename}/TopK"
        topk_inputs = [f"{topk_fp32_name if use_cast else router_add_name}/output_0", f"/model/constants/INT64/[{self.moe_attrs['top_k']}]"]
        topk_outputs = [f"{topk_name}/output_0", f"{topk_name}/output_1"]
        self.make_node("TopK", inputs=topk_inputs, outputs=topk_outputs, name=topk_name, axis=-1, largest=True, sorted=True)
        self.make_value(topk_outputs[0], ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"]])
        self.make_value(topk_outputs[1], ir.DataType.INT64, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"]])
        if use_cast:
            topk_io_name = f"{basename}/topk_io/Cast"
            self.make_cast(topk_io_name, topk_outputs[0], self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"]])

        # Save initializers to use with Gather nodes
        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.weight"
        self.make_initializer(mlp.experts.gate_up_proj, gate_up_proj_weight, to=self.io_dtype)
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        self.make_initializer(mlp.experts.gate_up_proj_bias, gate_up_proj_bias, to=self.io_dtype)
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.weight"
        self.make_initializer(mlp.experts.down_proj, down_proj_weight, to=self.io_dtype)
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"
        self.make_initializer(mlp.experts.down_proj_bias, down_proj_bias, to=self.io_dtype)

        # Make Gather nodes + Unsqueeze nodes for biases
        mlp1_weight_gather_name = f"{basename}/mlp1/weight/Gather"
        mlp1_weight_gather_inputs = [gate_up_proj_weight, f"{topk_name}/output_1"]
        self.make_gather(mlp1_weight_gather_name, mlp1_weight_gather_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], 2 * self.intermediate_size, self.hidden_size], axis=0)
        mlp1_bias_gather_name = f"{basename}/mlp1/bias/Gather"
        mlp1_bias_gather_inputs = [gate_up_proj_bias, f"{topk_name}/output_1"]
        self.make_gather(mlp1_bias_gather_name, mlp1_bias_gather_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], 2 * self.intermediate_size], axis=0)
        mlp1_bias_unsqueeze_name = f"{basename}/mlp1/bias/Unsqueeze"
        mlp1_bias_unsqueeze_inputs = [f"{mlp1_bias_gather_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(mlp1_bias_unsqueeze_name, mlp1_bias_unsqueeze_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], 2 * self.intermediate_size, 1])
        mlp2_weight_gather_name = f"{basename}/mlp2/weight/Gather"
        mlp2_weight_gather_inputs = [down_proj_weight, f"{topk_name}/output_1"]
        self.make_gather(mlp2_weight_gather_name, mlp2_weight_gather_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.hidden_size, self.intermediate_size], axis=0)
        mlp2_bias_gather_name = f"{basename}/mlp2/bias/Gather"
        mlp2_bias_gather_inputs = [down_proj_bias, f"{topk_name}/output_1"]
        self.make_gather(mlp2_bias_gather_name, mlp2_bias_gather_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.hidden_size], axis=0)
        mlp2_bias_unsqueeze_name = f"{basename}/mlp2/bias/Unsqueeze"
        mlp2_bias_unsqueeze_inputs = [f"{mlp2_bias_gather_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(mlp2_bias_unsqueeze_name, mlp2_bias_unsqueeze_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.hidden_size, 1])

        # Make expert_weights path (Softmax --> Unsqueeze --> Unsqueeze --> Cast)
        softmax_name = f"{basename}/expert_weights/Softmax"
        self.make_softmax(softmax_name, f"{topk_io_name if use_cast else topk_name}/output_0", self.io_dtype, ['batch_size', 'sequence_length', 'num_experts_per_token'])
        expert_weights_unsqueeze_1_name = f"{basename}/expert_weights/Unsqueeze_1"
        expert_weights_unsqueeze_1_inputs = [f"{softmax_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(expert_weights_unsqueeze_1_name, expert_weights_unsqueeze_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', 'num_experts_per_token', 1])
        expert_weights_unsqueeze_2_name = f"{basename}/expert_weights/Unsqueeze_2"
        expert_weights_unsqueeze_2_inputs = [f"{expert_weights_unsqueeze_1_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_unsqueeze(expert_weights_unsqueeze_2_name, expert_weights_unsqueeze_2_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', 'num_experts_per_token', 1, 1])
        if use_cast:
            expert_weights_cast_name = f"{basename}/expert_weights/Cast"
            self.make_cast(expert_weights_cast_name, f"{expert_weights_unsqueeze_2_name}/output_0", ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', 'num_experts_per_token', 1, 1])

        # Make Gate/Up proj nodes (MatMul --> Add)
        gate_up_proj_weight_name = f"{basename}/gate_up_proj/MatMul"
        gate_up_proj_weight_output = f"{gate_up_proj_weight_name}/output_0"
        self.make_node("MatMul", inputs=[f"{mlp1_weight_gather_name}/output_0", f"{expand_root_input_unsqueeze_2_name}/output_0"], outputs=[gate_up_proj_weight_output], name=gate_up_proj_weight_name)
        self.make_value(gate_up_proj_weight_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], 2 * self.intermediate_size, 1])
        gate_up_proj_bias_name = f"{basename}/gate_up_proj/Add"
        self.make_add(gate_up_proj_bias_name, [gate_up_proj_weight_output, f"{mlp1_bias_unsqueeze_name}/output_0"], dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], 2 * self.intermediate_size, 1])

        # Make activation nodes
        #
        #         +--> Slice --> Clamp --> Mul --> Sigmoid --+
        #         |      |                                   |
        #         |      |                                  Mul --+
        # Add ----+      |                                   |    |
        #         |      +-----------------------------------+    +--> Mul
        #         |                                               |
        #         +---> Slice --> Clamp --> Add ------------------+
        glu_slice_name = f"{basename}/act_fn/Slice_1"
        glu_slice_inputs = [f"{gate_up_proj_bias_name}/output_0", "/model/constants/INT64/[0]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[3]", "/model/constants/INT64/[2]"]
        self.make_slice(glu_slice_name, glu_slice_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        glu_clip_name = f"{basename}/act_fn/Clip_1"
        glu_clip_inputs = [f"{glu_slice_name}/output_0", "", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.moe_attrs['swiglu_limit']}"]
        self.make_clip(glu_clip_name, glu_clip_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        linear_slice_name = f"{basename}/act_fn/Slice_2"
        linear_slice_inputs = [f"{gate_up_proj_bias_name}/output_0", "/model/constants/INT64/[1]", f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]", "/model/constants/INT64/[3]", "/model/constants/INT64/[2]"]
        self.make_slice(linear_slice_name, linear_slice_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        linear_clip_name = f"{basename}/act_fn/Clip_2"
        linear_clip_inputs = [f"{linear_slice_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{-self.moe_attrs['swiglu_limit']}", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.moe_attrs['swiglu_limit']}"]
        self.make_clip(linear_clip_name, linear_clip_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])

        # Make Mul node after activation
        act_fn_mul_1_name = f"{basename}/act_fn/Mul_1"
        act_fn_mul_1_inputs = [f"{glu_clip_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1.703125"]
        self.make_mul(act_fn_mul_1_name, act_fn_mul_1_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1])
        sigmoid_name = f"{basename}/act_fn/Sigmoid"
        self.make_sigmoid(sigmoid_name, f"{act_fn_mul_1_name}/output_0", dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1])
        act_fn_mul_2_name = f"{basename}/act_fn/Mul_2"
        act_fn_mul_2_inputs = [f"{glu_clip_name}/output_0", f"{sigmoid_name}/output_0"]
        self.make_mul(act_fn_mul_2_name, act_fn_mul_2_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1])
        act_fn_add_name = f"{basename}/act_fn/Add"
        self.make_add(act_fn_add_name, [f"{linear_clip_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1"], dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        act_fn_mul_3_name = f"{basename}/act_fn/Mul_3"
        act_fn_mul_3_inputs = [f"{act_fn_mul_2_name}/output_0", f"{act_fn_add_name}/output_0"]
        self.make_mul(act_fn_mul_3_name, act_fn_mul_3_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.moe_attrs["top_k"], self.intermediate_size, 1])

        # Make Down proj nodes (MatMul --> Add --> Cast)
        down_proj_weight_name = f"{basename}/down_proj/MatMul"
        down_proj_weight_output = f"{down_proj_weight_name}/output_0"
        self.make_node("MatMul", inputs=[f"{mlp2_weight_gather_name}/output_0", f"{act_fn_mul_3_name}/output_0"], outputs=[down_proj_weight_output], name=down_proj_weight_name)
        self.make_value(down_proj_weight_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        down_proj_bias_name = f"{basename}/down_proj/Add"
        self.make_add(down_proj_bias_name, [down_proj_weight_output, f"{mlp2_bias_unsqueeze_name}/output_0"], dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        if use_cast:
            down_proj_cast_name = f"{basename}/down_proj/Cast"
            self.make_cast(down_proj_cast_name, f"{down_proj_bias_name}/output_0", ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])

        # Make weighted sum nodes
        #
        # Cast (from Down proj) ------->
        #                               \
        #                                Mul --> ReduceSum --> Squeeze --> Cast (created in LayerNorm)
        #                               /
        # Cast (from expert weights) -->
        weighted_sum_mul_name = f"{basename}/weighted_sum/Mul"
        weighted_sum_mul_inputs = [f"{down_proj_cast_name if use_cast else down_proj_bias_name}/output_0", f"{expert_weights_cast_name if use_cast else expert_weights_unsqueeze_2_name}/output_0"]
        self.make_mul(weighted_sum_mul_name, weighted_sum_mul_inputs, dtype=ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.moe_attrs["top_k"], self.intermediate_size, 1])
        reduce_sum_name = f"{basename}/weighted_sum/ReduceSum"
        reduce_sum_inputs = [f"{weighted_sum_mul_name}/output_0", "/model/constants/INT64/[2]"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.intermediate_size, 1], keepdims=False)
        weighted_sum_squeeze_name = f"{basename}/weighted_sum/Squeeze"
        weighted_sum_squeeze_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/INT64/[-1]"]
        self.make_squeeze(weighted_sum_squeeze_name, weighted_sum_squeeze_inputs, dtype=ir.DataType.FLOAT, shape=['batch_size', 'sequence_length', self.intermediate_size])

        # Assign output 0 of previous MoE as root input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{weighted_sum_squeeze_name}/output_0"

    def make_moe_fused(self, layer_id, mlp, root_input):
        # Make nodes for the fused MoE subgraph
        #
        #               root_input
        #               /        \
        #             MatMul      |
        #            (router)     |
        #               |         |
        #              Add        |
        #            (router)     |
        #               |         |
        #            Reshape      |
        #               |         |
        #               +----+----+
        #                    |
        #                 MoE/QMoE
        basename = f"/model/layers.{layer_id}/moe"
        op_type = self.moe_attrs["op_type"]
        moe_weight_type = f"{'q' if op_type == 'QMoE' else ''}weight"

        # Make router nodes
        router_basename = f"{basename}/router/MatMul"
        router_matmul_name = self.make_matmul(mlp.router, router_basename, root_input)
        router_add_name = f"{basename}/router/Add"
        self.make_add_bias(mlp.router.bias, router_add_name, root_input=f"{router_matmul_name}/output_0")
        router_reshape_name = f"{basename}/router/Reshape"
        router_reshape_inputs = [f"{router_add_name}/output_0", f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}"]
        self.make_reshape(router_reshape_name, router_reshape_inputs, dtype=self.io_dtype, shape=['batch_size * sequence_length', self.moe_attrs['num_experts']])

        gate_up_proj_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.{moe_weight_type}"
        gate_up_proj_scales = f"model.layers.{layer_id}.moe.experts.gate_up_proj.scales"
        gate_up_proj_bias = f"model.layers.{layer_id}.moe.experts.gate_up_proj.bias"
        down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.{moe_weight_type}"
        down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"
        down_proj_bias = f"model.layers.{layer_id}.moe.experts.down_proj.bias"

        # Apply transpose once for both branches
        gate_up_proj_transposed = mlp.experts.gate_up_proj.transpose(-1, -2)
        down_proj_transposed = mlp.experts.down_proj.transpose(-1, -2)

        if op_type == "MoE":
            # Save non-quantized MoE weights as initializers
            self.make_initializer(gate_up_proj_transposed.view(self.moe_attrs["num_experts"], -1, self.hidden_size), gate_up_proj_weight, to=self.io_dtype)
            self.make_initializer(down_proj_transposed.view(self.moe_attrs["num_experts"], self.hidden_size, self.intermediate_size), down_proj_weight, to=self.io_dtype)
        else:
            # Create and save quantized MoE weights as initializers
            gate_up_proj_qweight_list, gate_up_proj_scales_list = [], []
            down_proj_qweight_list, down_proj_scales_list = [], []

            for i in range(self.moe_attrs["num_experts"]):
                qweight1, scales1 = self.make_qmoe_weights(gate_up_proj_transposed[i])
                gate_up_proj_qweight_list.append(qweight1)
                gate_up_proj_scales_list.append(scales1)
                qweight2, scales2 = self.make_qmoe_weights(down_proj_transposed[i])
                down_proj_qweight_list.append(qweight2)
                down_proj_scales_list.append(scales2)

            gate_up_proj_qweight_tensor = torch.stack(gate_up_proj_qweight_list, dim=0).to(torch.uint8)
            gate_up_proj_scales_tensor = torch.stack(gate_up_proj_scales_list, dim=0)
            down_proj_qweight_tensor = torch.stack(down_proj_qweight_list, dim=0).to(torch.uint8)
            down_proj_scales_tensor = torch.stack(down_proj_scales_list, dim=0)

            # qweight tensors always use the same shape regardless of quantization method
            pack_size = 8 // self.moe_attrs["expert_weight_bits"]
            self.make_initializer(gate_up_proj_qweight_tensor.view(self.moe_attrs["num_experts"], -1, self.hidden_size // pack_size), gate_up_proj_weight)
            self.make_initializer(down_proj_qweight_tensor.view(self.moe_attrs["num_experts"], self.hidden_size, self.intermediate_size // pack_size), down_proj_weight)
            
            # scales tensors have different shapes depending on quantization method
            self.make_initializer(gate_up_proj_scales_tensor, gate_up_proj_scales, to=self.io_dtype)
            self.make_initializer(down_proj_scales_tensor, down_proj_scales, to=self.io_dtype)

        # Save MoE biases as initializers
        self.make_initializer(mlp.experts.gate_up_proj_bias, gate_up_proj_bias, to=self.io_dtype)
        self.make_initializer(mlp.experts.down_proj_bias, down_proj_bias, to=self.io_dtype)

        moe_name = f"{basename}/{op_type}"
        self.make_moe_op(
            moe_name, root_input=root_input, router_probs=f"{router_reshape_name}/output_0",
            weight1=gate_up_proj_weight, scales1=gate_up_proj_scales, bias1=gate_up_proj_bias,
            weight2=down_proj_weight, scales2=down_proj_scales, bias2=down_proj_bias,
        )

        # Assign output 0 of previous MoE as root input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{moe_name}/output_0"


def check_extra_options(kv_pairs, execution_provider):
    """
    Check key-value pairs and set values correctly
    """
    bools = [
        "int4_is_symmetric", "exclude_embeds", "exclude_lm_head", "include_hidden_states", "enable_cuda_graph", "enable_webgpu_graph",
        "use_8bits_moe", "use_qdq", "use_webgpu_fp32", "use_cuda_bf16", "int4_tied_embeddings", "hf_remote",
    ]
    for key in bools:
        if key in kv_pairs:
            if kv_pairs[key] in {"false", "False", "0"}:
                kv_pairs[key] = False
            elif kv_pairs[key] in {"true", "True", "1"}:
                kv_pairs[key] = True
            else:
                raise ValueError(f"{key} must be false/False/0 or true/True/1.")

    if "int4_op_types_to_quantize" in kv_pairs:
        op_types_to_quantize = ()
        for op_type in kv_pairs["int4_op_types_to_quantize"].split("/"):
            op_types_to_quantize += (op_type, )
        kv_pairs["int4_op_types_to_quantize"] = op_types_to_quantize

    if "int4_nodes_to_exclude" in kv_pairs:
        nodes_to_exclude = []
        for node in kv_pairs["int4_nodes_to_exclude"].split(","):
            nodes_to_exclude.append(node)
        kv_pairs["int4_nodes_to_exclude"] = nodes_to_exclude

    if "exclude_lm_head" in kv_pairs and "include_hidden_states" in kv_pairs:
        # 'exclude_lm_head' is for when 'hidden_states' are outputted and 'logits' are not outputted
        # 'include_hidden_states' is for when 'hidden_states' are outputted and 'logits' are outputted
        raise ValueError("Both 'exclude_lm_head' and 'include_hidden_states' cannot be used together. Please use only one of them at once.")

    if kv_pairs.get("enable_webgpu_graph", False) and execution_provider != "webgpu":
        print("WARNING: enable_webgpu_graph is only supported with WebGPU execution provider. Disabling enable_webgpu_graph.")
        kv_pairs["enable_webgpu_graph"] = False


def parse_extra_options(kv_items, execution_provider):
    """
    Parse key-value pairs that are separated by '='
    """
    kv_pairs = {}

    if kv_items:
        for kv_str in kv_items:
            kv = kv_str.split('=')
            kv_pairs[kv[0].strip()] = kv[1].strip()

    print(f"Extra options: {kv_pairs}")
    check_extra_options(kv_pairs, execution_provider)
    return kv_pairs


def parse_hf_token(hf_token):
    """
    Returns the authentication token needed for Hugging Face.
    Token is obtained either from the user or the environment.
    """
    if hf_token.lower() in {"false", "0"}:
        # Default is `None` for disabling authentication
        return None

    if hf_token.lower() in {"true", "1"}:
        # Return token in environment
        return True

    # Return user-provided token as string
    return hf_token


def set_io_dtype(precision, execution_provider, extra_options) -> ir.DataType:
    int4_cpu = precision == "int4" and execution_provider == "cpu"
    fp32_webgpu = execution_provider == "webgpu" and extra_options.get("use_webgpu_fp32", False)
    bf16_cuda = precision == "int4" and execution_provider == "cuda" and extra_options.get("use_cuda_bf16", False)

    if precision in {"int8", "fp32"} or int4_cpu or fp32_webgpu:
        # FP32 precision
        return ir.DataType.FLOAT

    if precision == "bf16" or bf16_cuda:
        # BF16 precision
        return ir.DataType.BFLOAT16

    # FP16 precision
    return ir.DataType.FLOAT16


def set_onnx_dtype(precision: str, extra_options: dict[str, Any]) -> ir.DataType:
    if precision == "int4":
        return ir.DataType.INT4 if extra_options.get("int4_is_symmetric", True) else ir.DataType.UINT4

    to_onnx_dtype = {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "bf16": ir.DataType.BFLOAT16,
    }
    return to_onnx_dtype[precision]


@torch.no_grad
def create_model(model_name, input_path, output_dir, precision, execution_provider, cache_dir, **extra_options):
    if execution_provider == "NvTensorRtRtx":
        execution_provider = "trt-rtx"
        extra_options["use_qdq"] = True

    # Create cache and output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model config
    extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": cache_dir}
    hf_name = input_path if os.path.isdir(input_path) else model_name
    hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
    hf_remote = extra_options.get("hf_remote", True)

    config = AutoConfig.from_pretrained(hf_name, token=hf_token, trust_remote_code=hf_remote, **extra_kwargs)
    if "adapter_path" in extra_options:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(extra_options["adapter_path"], token=hf_token, trust_remote_code=hf_remote, **extra_kwargs)
        config.update(peft_config.__dict__)

    # Set input/output precision of ONNX model
    io_dtype = set_io_dtype(precision, execution_provider, extra_options)
    onnx_dtype = set_onnx_dtype(precision, extra_options)
    config_only = "config_only" in extra_options
    
    # List architecture options in alphabetical order
    if config.architectures[0] == "ChatGLMForConditionalGeneration" or config.architectures[0] == "ChatGLMModel":
        # Quantized ChatGLM model has ChatGLMForConditionalGeneration as architecture whereas HF model as the latter
        config.bos_token_id = 1
        config.hidden_act = "swiglu"
        onnx_model = ChatGLMModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
        onnx_model.model_type = "chatglm"
    elif config.architectures[0] == "Ernie4_5_ForCausalLM":
        onnx_model = ErnieModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GemmaForCausalLM":
        onnx_model = GemmaModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Gemma2ForCausalLM":
        print("WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default.")
        onnx_model = Gemma2Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Gemma3ForCausalLM":
        print("WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default.")
        onnx_model = Gemma3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
        onnx_model.model_type = "gemma3_text"
    elif config.architectures[0] == "Gemma3ForConditionalGeneration":
        text_config = config.text_config
        for key in text_config:
            if not hasattr(config, key):
                setattr(config, key, getattr(text_config, key))
        print("WARNING: This model loses accuracy with float16 precision. It is recommended to set `--precision bf16` or `--precision int4 --extra_options use_cuda_bf16=true` by default.")
        print("WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default.")
        extra_options["exclude_embeds"] = True
        onnx_model = Gemma3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GptOssForCausalLM":
        print("WARNING: This model only supports symmetric quantization for `QMoE`.")
        delattr(config, "quantization_config")
        onnx_model = GPTOSSModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "GraniteForCausalLM":
        onnx_model = GraniteModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "LlamaForCausalLM":
        onnx_model = LlamaModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "MistralForCausalLM":
        onnx_model = MistralModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "NemotronForCausalLM":
        onnx_model = NemotronModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "OlmoForCausalLM":
        onnx_model = OLMoModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "PhiForCausalLM":
        onnx_model = PhiModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3ForCausalLM" and config.max_position_embeddings == config.original_max_position_embeddings:
        onnx_model = Phi3MiniModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3ForCausalLM" and config.max_position_embeddings != config.original_max_position_embeddings:
        onnx_model = Phi3MiniLongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "PhiMoEForCausalLM" and config.max_position_embeddings != config.original_max_position_embeddings:
        print("WARNING: This model only works for CUDA currently because `MoE` is only supported for CUDA in ONNX Runtime. Setting `--execution_provider cuda` by default.")
        print("WARNING: This model currently only supports the quantized version. Setting `--precision int4` by default.")
        execution_provider = "cuda"
        onnx_dtype = set_onnx_dtype("int4", extra_options)
        onnx_model = Phi3MoELongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3SmallForCausalLM" and config.max_position_embeddings == config.original_max_position_embeddings:
        onnx_model = Phi3SmallModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3SmallForCausalLM" and config.max_position_embeddings != config.original_max_position_embeddings:
        onnx_model = Phi3SmallLongRoPEModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi3VForCausalLM":
        print("WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default.")
        extra_options["exclude_embeds"] = True
        onnx_model = Phi3VModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Phi4MMForCausalLM":
        print("WARNING: This is only generating the text component of the model. Setting `--extra_options exclude_embeds=true` by default.")
        extra_options["exclude_embeds"] = True
        onnx_model = Phi4MMModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Qwen2ForCausalLM":
        onnx_model = QwenModel(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "Qwen3ForCausalLM":
        onnx_model = Qwen3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config.architectures[0] == "SmolLM3ForCausalLM":
        onnx_model = SmolLM3Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    elif config_only:
        # Create base Model class to guess model attributes
        onnx_model = Model(config, io_dtype, onnx_dtype, execution_provider, cache_dir, extra_options)
    else:
        raise NotImplementedError(f"The {hf_name} model is not currently supported.")

    if not config_only:
        # Make ONNX model
        onnx_model.make_model(input_path)

        # Save ONNX model
        onnx_model.save_model(output_dir)

    # Make GenAI config
    onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)

    # Copy Hugging Face processing files to output folder
    onnx_model.save_processing(hf_name, extra_kwargs, output_dir)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        default=None,
        help="Model name in Hugging Face. Do not use if providing an input path to a Hugging Face directory in -i/--input.",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default="",
        help=textwrap.dedent("""\
            Input model source. Currently supported options are:
                hf_path: Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.
                gguf_path: Path to float16/float32 GGUF file on disk containing the GGUF model
            """),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["int4", "bf16", "fp16", "fp32"],
        help="Precision of model",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml", "webgpu", "NvTensorRtRtx"],
        help="Execution provider to target with precision of model (e.g. FP16 CUDA, INT4 CPU, INT4 WebGPU)",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join('.', 'cache_dir'),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )

    parser.add_argument(
        "--extra_options",
        required=False,
        metavar="KEY=VALUE",
        nargs='+',
        help=textwrap.dedent("""\
            Key value pairs for various options. Currently supports:
                int4_accuracy_level = 1/2/3/4: Specify the minimum accuracy level for activation of MatMul in int4 quantization.
                    4 is int8, which means input A of int4 quantized MatMul is quantized to int8 and input B is upcasted to int8 for computation.
                    3 is bf16.
                    2 is fp16.
                    1 is fp32.
                    Default is 4 for the CPU EP and 0 for non-CPU EPs.
                int4_block_size = 16/32/64/128/256: Specify the block size for int4 quantization.
                    Default value is 32.
                int4_is_symmetric = Quantize the weights symmetrically. Default is true.
                    If true, quantization is done to int4. If false, quantization is done to uint4.
                int4_op_types_to_quantize = MatMul/Gather: Specify op types to target for int4 quantization.
                    Use this option when you want to quantize specific ops.
                    Separate the op types with a '/' when passing them here (e.g. int4_op_types_to_quantize=MatMul/Gather)
                int4_nodes_to_exclude = Specify nodes to exclude from int4 quantization.
                    Use this option when you want to exclude certain nodes from being quantized.
                    Separate the node names with a ',' when passing them here (e.g. int4_nodes_to_exclude=/lm_head/MatMul,/model/embed_tokens/Gather)
                int4_algo_config = Method for int4 quantization. Default is 'default'.
                    Currently supported options are: 'default', 'rtn', 'k_quant_mixed', 'k_quant_last'.
                    k_quant_mixed = k_quant algorithm with mixed precision (int4 + int8).
                    k_quant_last = k_quant algorithm where only the last MatMul (/lm_head/MatMul) is quantized as int8. Other MatMuls are quantized as int4.
                int4_tied_embeddings = Enable weight sharing for quantization. Default is false.
                    Use this option when you want to share the weights in the embedding and unembedding.
                num_hidden_layers = Manually specify the number of layers in your ONNX model.
                    Used for unit testing purposes.
                filename = Filename for ONNX model (default is 'model.onnx').
                    For models with multiple components, each component is exported to its own ONNX model.
                    The filename for each component will be '<filename>_<component-name>.onnx' (ex: '<filename>_encoder.onnx', '<filename>_decoder.onnx').
                config_only = Generate config and pre/post processing files only.
                    Use this option when you already have your optimized and/or quantized ONNX model.
                hf_token = false/token: Use this to manage authentication with Hugging Face.
                    Default behavior is to use the authentication token stored by `huggingface-cli login`.
                    If false, authentication with Hugging Face will be disabled.
                    If token, you can provide a custom authentication token that differs from the one stored in your environment.
                    If you have already authenticated via `huggingface-cli login`, you do not need to use this flag because Hugging Face has already stored your authentication token for you.
                hf_remote = Use this to manage trusting remote code in Hugging Face repos.
                    Default behavior is set to true. If false, remote code stored in Hugging Face repos will not be used.
                exclude_embeds = Remove embedding layer from your ONNX model.
                    Use this option when you want to remove the embedding layer from within your ONNX model.
                    Instead of `input_ids`, you will have `inputs_embeds` as the input to your ONNX model.
                exclude_lm_head = Remove language modeling head from your ONNX model.
                    Use this option when you want to remove the language modeling head from within your ONNX model.
                    Instead of `logits`, you will have `hidden_states` as the output to your ONNX model.
                include_hidden_states = Include hidden states as output from your ONNX model.
                    Use this option when you want to have the hidden states as an output from your ONNX model.
                    In addition to `logits`, you will have `hidden_states` as an output to your ONNX model.
                enable_cuda_graph = Enable CUDA graph capture during inference. Default is false.
                    If enabled, all nodes being placed on the CUDA EP is the prerequisite for the CUDA graph to be used correctly.
                    It is not guaranteed that CUDA graph be enabled as it depends on the model and the graph structure.
                enable_webgpu_graph = Enable WebGPU graph capture during inference. Default is false.
                    If enabled, the model structure will be optimized for WebGPU graph execution.
                    This affects attention mask reformatting and position IDs handling.
                use_8bits_moe = Use 8-bit quantization for MoE layers. Default is false.
                    If true, the QMoE op will use 8-bit quantization. If false, the QMoE op will use 4-bit quantization.
                use_qdq = Use the QDQ decomposition for ops.
                    Use this option when you want to use quantize-dequantize ops. For example, you will have a quantized MatMul op instead of the MatMulNBits op.
                use_webgpu_fp32 = Use FP32 I/O precision for WebGPU EP.
                    Use this option to enable GPUs that do not support FP16 on WebGPU (e.g. GTX 10xx).
                use_cuda_bf16 = Use BF16 I/O precision in quantized ONNX models for CUDA EP.
                    Use this option to create quantized ONNX models that use BF16 precision.
                adapter_path = Path to folder on disk containing the adapter files (adapter_config.json and adapter model weights).
                    Use this option for LoRA models.
            """),
    )

    args = parser.parse_args()
    print("Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, FP16 DML, BF16 CUDA, FP16 TRT-RTX, INT4 CPU, INT4 CUDA, INT4 DML, INT4 WebGPU")
    return args

if __name__ == '__main__':
    args = get_args()
    extra_options = parse_extra_options(args.extra_options, args.execution_provider)
    create_model(args.model_name, args.input, args.output, args.precision, args.execution_provider, args.cache_dir, **extra_options)
