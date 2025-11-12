# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
"""
A set of Python classes to unpack the quantized weights and repack them in ONNX Runtime's
standard format.

The goal is for `QuantModel` to repack the quantized weights into a standard format
so that the original Hugging Face --> ONNX code can store the quantized weights as
ONNX Runtime's format no matter where the quantized weights actually come from.
"""

from safetensors.torch import load_file
import torch

import os
import re


class QuantizedTensorModule:
    def __init__(self):
        self.qweight = None
        self.scales = None
        self.qzeros = None
        self.g_idx = None
        self.bias = None

        self.in_features = 0
        self.out_features = 0
        self.bits = None
        self._group_size = None

    @property
    def group_size(self):
        return self._group_size if self._group_size != -1 else self.in_features

    @group_size.setter
    def group_size(self, value):
        self._group_size = value

    def __str__(self):
        qweight = f"qweight = {self.qweight.shape}, {self.qweight}\n"
        scales = f"scales = {self.scales.shape}, {self.scales}\n"
        qzeros = "" if self.qzeros is None else f"qzeros = {self.qzeros.shape}, {self.qzeros}\n"
        g_idx = "" if self.g_idx is None else f"g_idx = {self.g_idx.shape}, {self.g_idx}\n"

        in_feats = f"in_features = {self.in_features}, "
        out_feats = f"out_features = {self.out_features}, "
        bits = f"bits = {self.bits}, "
        group_size = f"group_size = {self.group_size}, "

        return qweight + qzeros + scales + g_idx + in_feats + out_feats + bits + group_size


class TensorModule:
    def __init__(self):
        self.weight = None
        self.bias = None

class QuantizedAttention:
    def __init__(self):
        self.q_proj = QuantizedTensorModule()
        self.k_proj = QuantizedTensorModule()
        self.v_proj = QuantizedTensorModule()
        self.o_proj = QuantizedTensorModule()
        self.rotary_emb = TensorModule()
        self.k_norm = TensorModule()
        self.q_norm = TensorModule()

class QuantizedMLP:
    def __init__(self):
        self.gate_proj = QuantizedTensorModule()
        self.up_proj = QuantizedTensorModule()
        self.down_proj = QuantizedTensorModule()
        self.fc1 = QuantizedTensorModule()
        self.fc2 = QuantizedTensorModule()


class QuantizedDecoderLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.input_layernorm = TensorModule()
        self.self_attn = QuantizedAttention()
        self.post_attention_layernorm = TensorModule()
        self.pre_feedforward_layernorm = TensorModule()
        self.post_feedforward_layernorm = TensorModule()
        self.mlp = QuantizedMLP()

    def is_empty(self):
        return self.input_layernorm.weight is None


class QuantizedModel:
    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        self.quant_type = quant_type
        self.embedding = TensorModule()
        self.final_norm = TensorModule()
        self.lm_head = TensorModule()
        self.layers = {}
        self.num_layers = num_layers
        self._quant_attrs = quant_attrs
        self._load_quant_config(quant_attrs)  # codeql[py/init-calls-subclass]

        for weight_file in os.listdir(input_path):
            if weight_file.endswith(".safetensors"):
                weights = load_file(os.path.join(input_path, weight_file))

                # Map weights to modules
                for name, tensor in weights.items():

                    # Per-layer quantization support
                    local_bits = self.get_layer_bits(name)  # codeql[py/init-calls-subclass]
                    local_group_size = self.get_layer_group_size(name)  # codeql[py/init-calls-subclass]

                    if name == "model.embed_tokens.weight" or name == "transformer.embedding.word_embeddings.weight":
                        self.embedding.weight = tensor
                    elif name == "model.norm.weight" or name == "transformer.encoder.final_layernorm.weight":
                        self.final_norm.weight = tensor
                    elif name == "model.norm.bias" or name == "transformer.encoder.final_layernorm.bias":
                        self.final_norm.bias = tensor
                    elif name == "lm_head.weight" or name == "transformer.output_layer.weight":
                        self.lm_head.weight = tensor
                    elif name == "lm_head.bias" or name == "transformer.output_layer.bias":
                        self.lm_head.bias = tensor
                    elif name == "transformer.rotary_pos_emb.inv_freq":
                        # transformer.rotary_pos_emb.inv_freq in ChatGLM3.
                        # Skip rotary embedding weights since they can be re-calculated when looping through the model
                        continue
                    elif name == "lm_head.qweight" or name == "transformer.output_layer.qweight":
                        self._initialize_quantized_lm_head(local_bits, local_group_size)
                        self.lm_head.qweight = tensor
                    elif name in {"lm_head.qzeros", "lm_head.weight_zero_point", "transformer.output_layer.qzeros"}:
                        self._initialize_quantized_lm_head(local_bits, local_group_size)
                        self.lm_head.qzeros = tensor
                    elif name in {"lm_head.scales", "lm_head.weight_scale", "transformer.output_layer.scales"}:
                        self._initialize_quantized_lm_head(local_bits, local_group_size)
                        self.lm_head.scales = tensor
                    elif name == "lm_head.g_idx" or name == "transformer.output_layer.g_idx":
                        self._initialize_quantized_lm_head(local_bits, local_group_size)
                        self.lm_head.g_idx = tensor
                    else:
                        if name.startswith("transformer.encoder"):
                            # Chatglm3, e.g., transformer.encoder.layers.0.input_layernorm.weight
                            name = name.replace("transformer.encoder", "model")
                        layer_id = int(name.split(".")[2])
                        module = self.layers.setdefault(layer_id, QuantizedDecoderLayer(layer_id))

                        # Map weights and biases of norm, attention, and feed-forward network
                        # Graph order is input_layernorm --> q_proj/k_proj/v_proj --> o_proj --> post_attention_layernorm --> gate_proj/up_proj --> down_proj
                        # If model uses q_norm and k_norm, graph order is input_layernorm --> q_norm/q_proj/k_norm/k_proj/v_proj --> o_proj --> post_attention_layernorm --> gate_proj/up_proj --> down_proj
                        tensor_map = {}
                        if bool(re.match(r"^model.layers\.\d+\.input_layernorm\.weight$", name)):
                            # model.layers.layer_id.input_layernorm.weight
                            tensor_map["input_layernorm.weight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.input_layernorm\.bias$", name)):
                            # model.layers.layer_id.input_layernorm.bias
                            tensor_map["input_layernorm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.rotary_emb\.inv_freq$", name)):
                            # model.layers.layer_id.self_attn.rotary_emb.inv_freq
                            # Skip rotary embedding weights since they can be re-calculated when looping through the model
                            continue
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.q?weight$", name)):
                            # model.layers.layer_id.self_attn.q_proj.weight
                            # model.layers.layer_id.self_attn.q_proj.qweight
                            tensor_map["self_attn.q_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.self_attn.q_proj.scales
                            # model.layers.layer_id.self_attn.q_proj.weight_scale
                            tensor_map["self_attn.q_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.self_attn.q_proj.qzeros
                            # model.layers.layer_id.self_attn.q_proj.weight_zero_point
                            tensor_map["self_attn.q_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.q_proj.g_idx
                            tensor_map["self_attn.q_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.q_proj.bias
                            tensor_map["self_attn.q_proj.bias"] = tensor
                        elif bool(re.match(r"^model\.layers\.\d+\.self_attn\.q_norm\.weight$", name)):
                            # model.layers.layer_id.self_attn.q_norm.weight
                            tensor_map["self_attn.q_norm.weight"] = tensor
                        elif bool(re.match(r"^model\.layers\.\d+\.self_attn\.q_norm\.bias$", name)):
                            # model.layers.layer_id.self_attn.q_norm.bias
                            tensor_map["self_attn.q_norm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.q?weight$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qweight
                            # model.layers.layer_id.self_attn.k_proj.weight
                            tensor_map["self_attn.k_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.self_attn.k_proj.scales
                            # model.layers.layer_id.self_attn.k_proj.weight_scale
                            tensor_map["self_attn.k_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qzeros
                            # model.layers.layer_id.self_attn.k_proj.weight_zero_point
                            tensor_map["self_attn.k_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.k_proj.g_idx
                            tensor_map["self_attn.k_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.k_proj.bias
                            tensor_map["self_attn.k_proj.bias"] = tensor
                        elif bool(re.match(r"^model\.layers\.\d+\.self_attn\.k_norm\.weight$", name)):
                            # model.layers.layer_id.self_attn.k_norm.weight
                            tensor_map["self_attn.k_norm.weight"] = tensor
                        elif bool(re.match(r"^model\.layers\.\d+\.self_attn\.k_norm\.bias$", name)):
                            # model.layers.layer_id.self_attn.k_norm.bias
                            tensor_map["self_attn.k_norm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.q?weight$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qweight
                            # model.layers.layer_id.self_attn.v_proj.weight
                            tensor_map["self_attn.v_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.self_attn.v_proj.scales
                            # model.layers.layer_id.self_attn.v_proj.weight_scale
                            tensor_map["self_attn.v_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qzeros
                            # model.layers.layer_id.self_attn.v_proj.weight_zero_point
                            tensor_map["self_attn.v_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.v_proj.g_idx
                            tensor_map["self_attn.v_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.v_proj.bias
                            tensor_map["self_attn.v_proj.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.q?weight$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qweight
                            # model.layers.layer_id.self_attention.dense.qweight
                            tensor_map["self_attn.o_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.self_attn.o_proj.scales
                            # model.layers.layer_id.self_attention.dense.scales
                            # model.layers.layer_id.self_attn.o_proj.weight_scale
                            # model.layers.layer_id.self_attention.dense.weight_scale
                            tensor_map["self_attn.o_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qzeros
                            # model.layers.layer_id.self_attention.dense.qzeros
                            # model.layers.layer_id.self_attn.o_proj.weight_zero_point
                            # model.layers.layer_id.self_attention.dense.weight_zero_point
                            tensor_map["self_attn.o_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.o_proj.g_idx
                            # model.layers.layer_id.self_attention.dense.g_idx
                            tensor_map["self_attn.o_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.bias$", name)):
                            # model.layers.layer_id.self_attn.o_proj.bias
                            # model.layers.layer_id.self_attention.dense.bias
                            tensor_map["self_attn.o_proj.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.weight$", name)):
                            # model.layers.layer_id.post_attention_layernorm.weight
                            tensor_map["post_attention_layernorm.weight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.bias$", name)):
                            # model.layers.layer_id.post_attention_layernorm.bias
                            tensor_map["post_attention_layernorm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.pre_feedforward_layernorm\.weight$", name)):
                            # model.layers.layer_id.pre_feedforward_layernorm.weight
                            tensor_map["pre_feedforward_layernorm.weight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.pre_feedforward_layernorm\.bias$", name)):
                            # model.layers.layer_id.pre_feedforward_layernorm.bias
                            tensor_map["pre_feedforward_layernorm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_feedforward_layernorm\.weight$", name)):
                            # model.layers.layer_id.post_feedforward_layernorm.weight
                            tensor_map["post_feedforward_layernorm.weight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_feedforward_layernorm\.bias$", name)):
                            # model.layers.layer_id.post_feedforward_layernorm.bias
                            tensor_map["post_feedforward_layernorm.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.q?weight$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qweight
                            # model.layers.layer_id.mlp.gate_proj.weight
                            tensor_map["mlp.gate_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.mlp.gate_proj.scales
                            # model.layers.layer_id.mlp.gate_proj.weight_scale
                            tensor_map["mlp.gate_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qzeros
                            # model.layers.layer_id.mlp.gate_proj.weight_zero_point
                            tensor_map["mlp.gate_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.gate_proj.g_idx
                            tensor_map["mlp.gate_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.gate_proj.bias
                            tensor_map["mlp.gate_proj.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.q?weight$", name)):
                            # model.layers.layer_id.mlp.up_proj.qweight
                            # model.layers.layer_id.mlp.up_proj.weight
                            tensor_map["mlp.up_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.mlp.up_proj.scales
                            # model.layers.layer_id.mlp.up_proj.weight_scale
                            tensor_map["mlp.up_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.mlp.up_proj.qzeros
                            # model.layers.layer_id.mlp.up_proj.weight_zero_point
                            tensor_map["mlp.up_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.up_proj.g_idx
                            tensor_map["mlp.up_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.up_proj.bias
                            tensor_map["mlp.up_proj.bias"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.q?weight$", name)):
                            # model.layers.layer_id.mlp.down_proj.qweight
                            # model.layers.layer_id.mlp.dense_4h_to_h.qweight
                            # model.layers.layer_id.mlp.down_proj.weight
                            # model.layers.layer_id.mlp.dense_4h_to_h.weight
                            tensor_map["mlp.down_proj.qweight"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.mlp.down_proj.scales
                            # model.layers.layer_id.mlp.dense_4h_to_h.scales
                            # model.layers.layer_id.mlp.down_proj.weight_scale
                            # model.layers.layer_id.mlp.dense_4h_to_h.weight_scale
                            tensor_map["mlp.down_proj.scales"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.mlp.down_proj.qzeros
                            # model.layers.layer_id.mlp.dense_4h_to_h.qzeros
                            # model.layers.layer_id.mlp.down_proj.weight_zero_point
                            # model.layers.layer_id.mlp.dense_4h_to_h.weight_zero_point
                            tensor_map["mlp.down_proj.qzeros"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.g_idx$", name)):
                            # model.layers.layer_id.mlp.down_proj.g_idx
                            # model.layers.layer_id.mlp.dense_4h_to_h.g_idx
                            tensor_map["mlp.down_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.bias$", name)):
                            # model.layers.layer_id.mlp.down_proj.bias
                            # model.layers.layer_id.mlp.dense_4h_to_h.bias
                            tensor_map["mlp.down_proj.bias"] = tensor
                        # Match against fused layers
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.q?weight$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.qweight
                            # model.layers.layer_id.self_attention.query_key_value.qweight
                            # model.layers.layer_id.self_attn.qkv_proj.weight
                            # model.layers.layer_id.self_attention.query_key_value.weight
                            q_dim = q_size // (32 // local_bits) if quant_type in {"awq", "quark"} else q_size
                            kv_dim = kv_size // (32 // local_bits) if quant_type in {"awq", "quark"} else kv_size
                            tensor_map["self_attn.q_proj.qweight"] = tensor[:, : q_dim]
                            tensor_map["self_attn.k_proj.qweight"] = tensor[:, q_dim : q_dim + kv_dim]
                            tensor_map["self_attn.v_proj.qweight"] = tensor[:, q_dim + kv_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.scales
                            # model.layers.layer_id.self_attention.query_key_value.scales
                            # model.layers.layer_id.self_attn.qkv_proj.weight_scale
                            # model.layers.layer_id.self_attention.query_key_value.weight_scale
                            tensor_map["self_attn.q_proj.scales"] = tensor[:, : q_size]
                            tensor_map["self_attn.k_proj.scales"] = tensor[:, q_size : q_size + kv_size]
                            tensor_map["self_attn.v_proj.scales"] = tensor[:, q_size + kv_size :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.qzeros
                            # model.layers.layer_id.self_attention.query_key_value.qzeros
                            # model.layers.layer_id.self_attn.qkv_proj.weight_zero_point
                            # model.layers.layer_id.self_attention.query_key_value.weight_zero_point
                            q_dim = q_size // (32 // local_bits) if quant_type in {"awq", "gptq", "olive", "quark"} else q_size
                            kv_dim = kv_size // (32 // local_bits) if quant_type in {"awq", "gptq", "olive", "quark"} else kv_size
                            tensor_map["self_attn.q_proj.qzeros"] = tensor[:, : q_dim]
                            tensor_map["self_attn.k_proj.qzeros"] = tensor[:, q_dim : q_dim + kv_dim]
                            tensor_map["self_attn.v_proj.qzeros"] = tensor[:, q_dim + kv_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.g_idx
                            # model.layers.layer_id.self_attention.query_key_value.g_idx
                            tensor_map["self_attn.q_proj.g_idx"] = tensor
                            tensor_map["self_attn.k_proj.g_idx"] = tensor
                            tensor_map["self_attn.v_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.bias$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.bias
                            # model.layers.layer_id.self_attention.query_key_value.bias
                            tensor_map["self_attn.q_proj.bias"] = tensor[: q_size]
                            tensor_map["self_attn.k_proj.bias"] = tensor[q_size : q_size + kv_size]
                            tensor_map["self_attn.v_proj.bias"] = tensor[q_size + kv_size : ]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h|gate_proj)\.q?weight$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.qweight
                            # model.layers.layer_id.mlp.dense_h_to_4h.qweight
                            # model.layers.layer_id.mlp.gate_up_proj.weight
                            # model.layers.layer_id.mlp.dense_h_to_4h.weight
                            intermediate_dim = intermediate_size // (32 // local_bits) if quant_type in {"awq", "quark"} else intermediate_size
                            tensor_map["mlp.gate_proj.qweight"] = tensor[:, : intermediate_dim]
                            tensor_map["mlp.up_proj.qweight"] = tensor[:, intermediate_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h|gate_proj)\.(scales|weight_scale)$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.scales
                            # model.layers.layer_id.mlp.dense_h_to_4h.scales
                            # model.layers.layer_id.mlp.gate_up_proj.weight_scale
                            # model.layers.layer_id.mlp.dense_h_to_4h.weight_scale
                            tensor_map["mlp.gate_proj.scales"] = tensor[:, : intermediate_size]
                            tensor_map["mlp.up_proj.scales"] = tensor[:, intermediate_size :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h|gate_proj)\.(qzeros|weight_zero_point)$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.qzeros
                            # model.layers.layer_id.mlp.dense_h_to_4h.qzeros
                            # model.layers.layer_id.mlp.gate_up_proj.weight_zero_point
                            # model.layers.layer_id.mlp.dense_h_to_4h.weight_zero_point
                            intermediate_dim = intermediate_size // (32 // local_bits) if quant_type in {"awq", "gptq", "quark", "olive"} else intermediate_size
                            tensor_map["mlp.gate_proj.qzeros"] = tensor[:, : intermediate_dim]
                            tensor_map["mlp.up_proj.qzeros"] = tensor[:, intermediate_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.g_idx$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.g_idx
                            # model.layers.layer_id.mlp.dense_h_to_4h.g_idx
                            tensor_map["mlp.gate_proj.g_idx"] = tensor
                            tensor_map["mlp.up_proj.g_idx"] = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.bias$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.bias
                            # model.layers.layer_id.mlp.dense_h_to_4h.bias
                            tensor_map["mlp.gate_proj.bias"] = tensor[: intermediate_size]
                            tensor_map["mlp.up_proj.bias"] = tensor[intermediate_size: ]
                        else:
                            raise NotImplementedError(f"{name} in your quantized model is not recognized.")

                        for tensor_name, tensor_value in tensor_map.items():
                            submodule = module
                            for sub_name in tensor_name.split(".")[:-1]:
                                submodule = getattr(submodule, sub_name)
                            if isinstance(submodule, QuantizedTensorModule):
                                for q_attr, q_value in [("bits", local_bits), ("_group_size", local_group_size)]:
                                    if getattr(submodule, q_attr) is not None and getattr(submodule, q_attr) != q_value:
                                        raise ValueError(f"Quantization {q_attr} mismatch for {name}: expected {getattr(submodule, q_attr)}, got {q_value}.")
                                    setattr(submodule, q_attr, q_value)
                            setattr(submodule, tensor_name.split(".")[-1], tensor_value)

        # Set LM head weights + biases if not already set
        if isinstance(self.lm_head, TensorModule) and self.lm_head.weight is None:
            # Embedding and LM head share same weights + biases (lm_head.weight == embedding.weight and lm_head.bias == embedding.bias)
            self.lm_head.weight = self.embedding.weight
            if self.lm_head.bias is not None:
                self.lm_head.bias = self.embedding.bias

        # Sort list of layers by layer id
        self.layers = list(self.layers.values())
        self.layers.sort(key=lambda m: m.layer_id)

        # Set properties of each layer based on quantization type
        self.set_properties()

    def _load_quant_config(self, quant_attrs):
        self.global_group_size = quant_attrs["config"]["group_size"]
        self.global_bits = quant_attrs["config"]["bits"]

    def get_layer_bits(self, layer_name):
        # 'bits' is globally defined for all layers
        return self.global_bits
    
    def get_layer_group_size(self, layer_name):
        # 'group_size' is globally defined for all layers
        return self.global_group_size

    def _initialize_quantized_lm_head(self, bits, group_size):
        """
        Initialize `QuantizedTensorModule` for LM head if not already set
        """
        if not isinstance(self.lm_head, QuantizedTensorModule):
            q_lm_head = QuantizedTensorModule()
            q_lm_head.qweight = self.lm_head.weight
            q_lm_head.bias = self.lm_head.bias
            q_lm_head.bits = bits
            q_lm_head.group_size = group_size
            self.lm_head = q_lm_head

    def set_properties(self):
        """
        Set in_features, out_features, and g_idx based on quantization type
        """
        if isinstance(self.lm_head, QuantizedTensorModule):
            if self.quant_type == "awq" or self.quant_type == "quark":
                self.lm_head.out_features = self.lm_head.scales.shape[1]
                self.lm_head.in_features = self.lm_head.qweight.shape[0]
                # Set g_idx if not already set
                self.lm_head.g_idx = self.lm_head.g_idx if self.lm_head.g_idx is not None else torch.tensor([i // self.lm_head.group_size for i in range(self.lm_head.in_features)], dtype=torch.int32)
            elif self.quant_type == "gptq":
                self.lm_head.out_features = self.lm_head.qweight.shape[1]
                self.lm_head.in_features = self.lm_head.g_idx.shape[0]
            elif self.quant_type == "olive":
                self.lm_head.out_features = self.lm_head.qweight.shape[1]
                # expects in_features to be divisible by the packing factor (32 // bits)
                # not a new assumption since no code here accounts for padded packed weights
                self.lm_head.in_features = self.lm_head.qweight.shape[0] * 32 // self.lm_head.bits
            else:
                raise NotImplementedError(f"The {self.quant_type} quantization method is not recognized.")
        for module in self.layers:
            if self.quant_type == "awq" or self.quant_type == "quark":
                # Set in_features and out_features
                module.self_attn.q_proj.out_features = module.self_attn.q_proj.scales.shape[1]
                module.self_attn.q_proj.in_features = module.self_attn.q_proj.qweight.shape[0]
                module.self_attn.k_proj.out_features = module.self_attn.k_proj.scales.shape[1]
                module.self_attn.k_proj.in_features = module.self_attn.k_proj.qweight.shape[0]
                module.self_attn.v_proj.out_features = module.self_attn.v_proj.scales.shape[1]
                module.self_attn.v_proj.in_features = module.self_attn.v_proj.qweight.shape[0]
                module.self_attn.o_proj.out_features = module.self_attn.o_proj.scales.shape[1]
                module.self_attn.o_proj.in_features = module.self_attn.o_proj.qweight.shape[0]
                module.mlp.gate_proj.out_features = module.mlp.gate_proj.scales.shape[1]
                module.mlp.gate_proj.in_features = module.mlp.gate_proj.qweight.shape[0]
                module.mlp.up_proj.out_features = module.mlp.up_proj.scales.shape[1]
                module.mlp.up_proj.in_features = module.mlp.up_proj.qweight.shape[0]
                module.mlp.down_proj.out_features = module.mlp.down_proj.scales.shape[1]
                module.mlp.down_proj.in_features = module.mlp.down_proj.qweight.shape[0]

                # Set g_idx if not already set
                module.self_attn.q_proj.g_idx = module.self_attn.q_proj.g_idx if module.self_attn.q_proj.g_idx is not None else torch.tensor([i // module.self_attn.q_proj.group_size for i in range(module.self_attn.q_proj.in_features)], dtype=torch.int32)
                module.self_attn.k_proj.g_idx = module.self_attn.k_proj.g_idx if module.self_attn.k_proj.g_idx is not None else torch.tensor([i // module.self_attn.k_proj.group_size for i in range(module.self_attn.k_proj.in_features)], dtype=torch.int32)
                module.self_attn.v_proj.g_idx = module.self_attn.v_proj.g_idx if module.self_attn.v_proj.g_idx is not None else torch.tensor([i // module.self_attn.v_proj.group_size for i in range(module.self_attn.v_proj.in_features)], dtype=torch.int32)
                module.self_attn.o_proj.g_idx = module.self_attn.o_proj.g_idx if module.self_attn.o_proj.g_idx is not None else torch.tensor([i // module.self_attn.o_proj.group_size for i in range(module.self_attn.o_proj.in_features)], dtype=torch.int32)
                module.mlp.gate_proj.g_idx = module.mlp.gate_proj.g_idx if module.mlp.gate_proj.g_idx is not None else torch.tensor([i // module.mlp.gate_proj.group_size for i in range(module.mlp.gate_proj.in_features)], dtype=torch.int32)
                module.mlp.up_proj.g_idx = module.mlp.up_proj.g_idx if module.mlp.up_proj.g_idx is not None else torch.tensor([i // module.mlp.up_proj.group_size for i in range(module.mlp.up_proj.in_features)], dtype=torch.int32)
                module.mlp.down_proj.g_idx = module.mlp.down_proj.g_idx if module.mlp.down_proj.g_idx is not None else torch.tensor([i // module.mlp.down_proj.group_size for i in range(module.mlp.down_proj.in_features)], dtype=torch.int32)

            elif self.quant_type == "gptq":
                # Set in_features and out_features
                module.self_attn.q_proj.out_features = module.self_attn.q_proj.qweight.shape[1]
                module.self_attn.q_proj.in_features = module.self_attn.q_proj.g_idx.shape[0]
                module.self_attn.k_proj.out_features = module.self_attn.k_proj.qweight.shape[1]
                module.self_attn.k_proj.in_features = module.self_attn.k_proj.g_idx.shape[0]
                module.self_attn.v_proj.out_features = module.self_attn.v_proj.qweight.shape[1]
                module.self_attn.v_proj.in_features = module.self_attn.v_proj.g_idx.shape[0]
                module.self_attn.o_proj.out_features = module.self_attn.o_proj.qweight.shape[1]
                module.self_attn.o_proj.in_features = module.self_attn.o_proj.g_idx.shape[0]
                module.mlp.gate_proj.out_features = module.mlp.gate_proj.qweight.shape[1]
                module.mlp.gate_proj.in_features = module.mlp.gate_proj.g_idx.shape[0]
                module.mlp.up_proj.out_features = module.mlp.up_proj.qweight.shape[1]
                module.mlp.up_proj.in_features = module.mlp.up_proj.g_idx.shape[0]
                module.mlp.down_proj.out_features = module.mlp.down_proj.qweight.shape[1]
                module.mlp.down_proj.in_features = module.mlp.down_proj.g_idx.shape[0]

            elif self.quant_type == "olive":
                # Set in_features and out_features
                module.self_attn.q_proj.out_features = module.self_attn.q_proj.qweight.shape[1]
                module.self_attn.q_proj.in_features = module.self_attn.q_proj.qweight.shape[0] * 32 // module.self_attn.q_proj.bits
                module.self_attn.k_proj.out_features = module.self_attn.k_proj.qweight.shape[1]
                module.self_attn.k_proj.in_features = module.self_attn.k_proj.qweight.shape[0] * 32 // module.self_attn.k_proj.bits
                module.self_attn.v_proj.out_features = module.self_attn.v_proj.qweight.shape[1]
                module.self_attn.v_proj.in_features = module.self_attn.v_proj.qweight.shape[0] * 32 // module.self_attn.v_proj.bits
                module.self_attn.o_proj.out_features = module.self_attn.o_proj.qweight.shape[1]
                module.self_attn.o_proj.in_features = module.self_attn.o_proj.qweight.shape[0] * 32 // module.self_attn.o_proj.bits
                module.mlp.gate_proj.out_features = module.mlp.gate_proj.qweight.shape[1]
                module.mlp.gate_proj.in_features = module.mlp.gate_proj.qweight.shape[0] * 32 // module.mlp.gate_proj.bits
                module.mlp.up_proj.out_features = module.mlp.up_proj.qweight.shape[1]
                module.mlp.up_proj.in_features = module.mlp.up_proj.qweight.shape[0] * 32 // module.mlp.up_proj.bits
                module.mlp.down_proj.out_features = module.mlp.down_proj.qweight.shape[1]
                module.mlp.down_proj.in_features = module.mlp.down_proj.qweight.shape[0] * 32 // module.mlp.down_proj.bits

            else:
                raise NotImplementedError(f"The {self.quant_type} quantization method is not recognized.")

    def modules(self):
        """
        Return list of modules in quantized model in order of appearance in the model
        """
        return [self.embedding] + self.layers + [self.final_norm, self.lm_head]

    def unpack(self, module):
        """
        Unpack `qzeros` and `qweight` to standard format
        """
        self.unpack_qzeros(module)
        self.unpack_qweight(module)
        self.dequant_weight(module)

    def repack(self, module):
        """
        Repack `scales`, `qzeros` and `qweight` to ORT format
        """
        intweight = self.quant_weight(module)
        self.pack_ort_format(module, intweight)

    def unpack_qzeros(self, module):
        """
        Unpack `qzeros` to standard format
        """
        if module.qzeros is None:
            return
        expected_shape = (module.in_features // module.group_size, module.out_features)
        transpose = module.qzeros.shape[0] != expected_shape[0]
        module.qzeros = self.unpack_on_row(module.qzeros, module.bits, transpose)

    def unpack_qweight(self, module):
        """
        Unpack `qweight` to standard format
        """
        expected_shape = (module.in_features, module.qweight.shape[1])
        transpose = module.qweight.shape[0] != expected_shape[0]
        module.qweight = self.unpack_on_row(module.qweight, module.bits, transpose)

    def pack_qzeros(self, module):
        """
        Pack `qzeros` to quantized format
        """
        expected_shape = (module.in_features // module.group_size, module.out_features)
        transpose = module.qzeros.shape[0] != expected_shape[0]
        module.qzeros = self.pack_on_row(module.qzeros, module.bits, transpose)

    def unpack_on_row_for_2_4_8_bits(self, tensor, bits, transpose):
        """
        Perform general-purpose unpacking on 2-bit, 4-bit, or 8-bit tensor
        """
        pack_tensor = tensor.T if transpose else tensor
        wf = torch.arange(0, 32, bits, device=pack_tensor.device).unsqueeze(0).unsqueeze(0)
        out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf)
        out = out.reshape(pack_tensor.shape[0], -1)
        out = torch.bitwise_and(out, (2 ** bits) - 1)
        return out.T if transpose else out

    def unpack_on_row(self, tensor, bits, transpose):
        """
        Unpack tensor by row. Packed datatype is assumed to be int32.
        """
        if bits in {2, 4, 8}:
            return self.unpack_on_row_for_2_4_8_bits(tensor, bits, transpose)
        else:
            raise NotImplementedError(f"Unpacking for {bits}-bit quantization is not currently supported.")

    def pack_on_row_for_2_4_8_bits(self, tensor, bits, transpose, packed_dtype=torch.int32):
        """
        Perform general-purpose packing on 2-bit, 4-bit, or 8-bit tensor
        """
        packed_bitwidth = torch.iinfo(packed_dtype).bits
        values_per_pack = packed_bitwidth // bits

        orig_tensor = tensor.T if transpose else tensor

        original_cols = orig_tensor.shape[1]
        pad_len = (values_per_pack - (original_cols % values_per_pack)) % values_per_pack
        if pad_len > 0:
            orig_tensor = torch.nn.functional.pad(orig_tensor, (0, pad_len), "constant", 0)

        wf = torch.arange(0, bits).view(1, 1, -1)
        out = torch.bitwise_right_shift(orig_tensor.unsqueeze(-1), wf)
        out = torch.bitwise_and(out, 1)

        out = out.reshape(orig_tensor.shape[0], -1, values_per_pack * bits)
        wf1 = torch.arange(0, values_per_pack * bits, 1).view(1, 1, -1)
        out = torch.bitwise_left_shift(out, wf1)
        out = out.sum(dim=-1).to(packed_dtype)
        return out.T if transpose else out

    def pack_on_row(self, tensor, bits, transpose, packed_dtype=torch.int32):
        """
        Pack tensor by row
        """
        if bits in {2, 4, 8}:
            return self.pack_on_row_for_2_4_8_bits(tensor, bits, transpose, packed_dtype)
        else:
            raise NotImplementedError(f"Packing for {bits}-bit quantization is not currently supported.")

    def dequant_weight(self, module):
        """
        De-quantize `qweight` to higher precision (float16)
        """
        # Note: `qweight` and `qzeros` have already been unpacked and stored in those variables respectively
        intweight = module.qweight
        zeros = module.qzeros
        scales = module.scales
        g_idx = module.g_idx

        # De-quantize weight to higher precision
        scale_zeros = zeros * scales
        if g_idx is not None:
            scales = scales[g_idx]
            scale_zeros = scale_zeros[g_idx]
        elif module.group_size != module.in_features:
            scales = scales.repeat_interleave(module.group_size, 0)
            scale_zeros = scale_zeros.repeat_interleave(module.group_size, 0)
        qdq_weight_T = intweight * scales - scale_zeros.half()

        # Store unpacked result in `qweight`
        module.qweight = qdq_weight_T.T

    def quant_weight(self, module):
        """
        Calculate integer weight to quantize `qweight` with
        """
        weight = module.qweight.T
        zeros = module.qzeros
        scales = module.scales
        g_idx = module.g_idx

        scale_zeros = zeros * scales
        if g_idx is not None:
            scales = scales[g_idx]
            scale_zeros = scale_zeros[g_idx]
        elif module.group_size != module.in_features:
            scales = scales.repeat_interleave(module.group_size, 0)
            scale_zeros = scale_zeros.repeat_interleave(module.group_size, 0)
        intweight_T = torch.round((weight + scale_zeros) / scales).to(torch.int)

        return intweight_T

    def pack_ort_format(self, module, intweight):
        """
        Pack `scales`, `qzeros`, and `qweight` to ORT format
        """
        if module.bits not in [2, 4, 8]:
            raise NotImplementedError(f"{module.bits}-bit quantization in ORT is not currently supported by this tool.")

        intzeros_pt = module.qzeros.T if module.qzeros.dtype == module.scales.dtype else module.qzeros.T.byte()
        intweight_pt = intweight.byte()
        kpack = 8 // module.bits
        block_size = module.group_size

        rows, cols = intweight_pt.shape
        blob_size = (block_size + kpack - 1) // kpack
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(intweight_pt, (0, 0, 0, pad_len), "constant", 0)

        if module.qzeros.dtype != module.scales.dtype:
            intzeros_pt = self.pack_on_row(intzeros_pt, module.bits, transpose=False, packed_dtype=torch.uint8)
            intzeros_pt = intzeros_pt.reshape(-1)

        intweight_pt_T = intweight.T
        intweight_pt_T = self.pack_on_row(intweight_pt_T, module.bits, transpose=False, packed_dtype=torch.uint8)
        intweight_pt_T = intweight_pt_T.reshape(cols, k_blocks, blob_size)

        scales_pt = module.scales.T.reshape(-1)

        module.scales = scales_pt.contiguous()
        module.qweight = intweight_pt_T.contiguous().byte()
        if module.qzeros.dtype != module.scales.dtype:
            module.qzeros = intzeros_pt.contiguous().byte()
        else:
            module.qzeros = intzeros_pt.contiguous()


class AWQModel(QuantizedModel):
    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        super().__init__(quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break
            print(f"Unpacking and repacking layer {i}")

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            self_attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            for _, q_tensors in self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

            # Unpack and repack all `QuantizedTensorModule` classes in MLP
            for _, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

        if isinstance(self.lm_head, QuantizedTensorModule) and self.lm_head.qweight is not None:
            self.unpack(self.lm_head)
            self.repack(self.lm_head)

            # Set `g_idx` to None since it's not used in `MatMulNBits`
            self.lm_head.g_idx = None

    def unpack_qweight(self, module):
        """
        Unpack `qweight` to standard format
        """
        expected_shape = (module.qweight.shape[0], module.out_features)
        transpose = module.qweight.shape != expected_shape
        module.qweight = self.unpack_on_row(module.qweight.T, module.bits, transpose)
        module.qweight = self.reverse_reorder_tensor(module.qweight.T, module.bits)

    def unpack_qzeros(self, module):
        """
        Unpack `qzeros` to standard format
        """
        super().unpack_qzeros(module)
        module.qzeros = self.reverse_reorder_tensor(module.qzeros, module.bits)

    def reverse_reorder_tensor(self, tensor, bits):
        """
        Re-arrange tensor data in a new order
        """
        compress_ratio = 32 // bits
        assert tensor.shape[-1] % compress_ratio == 0

        if bits == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError(f"Unpacking for {bits}-bit quantization is not currently supported.")

        order_tensor = torch.tensor(order_map, dtype=torch.int32).reshape(1, -1)
        order_tensor = order_tensor.repeat(tensor.shape[1] // compress_ratio, 1)
        order_tensor = order_tensor + torch.arange(0, tensor.shape[1], compress_ratio, dtype=torch.int32).reshape(-1, 1)
        order_tensor = order_tensor.reshape(-1)

        reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
        reverse_order_tensor = reverse_order_tensor[order_tensor]
        int_tensor = tensor[:, reverse_order_tensor]
        return int_tensor


class GPTQModel(QuantizedModel):
    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        super().__init__(quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break
            print(f"Unpacking and repacking layer {i}")

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            for _, q_tensors in layer.self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.handle_qzeros(q_tensors)
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    if not quant_attrs["use_g_idx"]:
                        # Set `g_idx` to None since it's not used in `MatMulNBits`
                        q_tensors.g_idx = None

            # Unpack and repack all `QuantizedTensorModule` classes in MLP
            for _, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.handle_qzeros(q_tensors)
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    if not quant_attrs["use_g_idx"]:
                        # Set `g_idx` to None since it's not used in `MatMulNBits`
                        q_tensors.g_idx = None

        if isinstance(self.lm_head, QuantizedTensorModule) and self.lm_head.qweight is not None:
            self.handle_qzeros(self.lm_head)
            self.unpack(self.lm_head)
            self.repack(self.lm_head)

            if not quant_attrs["use_g_idx"]:
                # Set `g_idx` to None since it's not used in `MatMulNBits`
                self.lm_head.g_idx = None

    def handle_qzeros(self, module):
        """
        Re-pack `qzeros` to handle extra `-1`s
        """
        if module.qzeros is None or module.qzeros.numel() == 0:
            return

        class TempModule:
            def __init__(self, module):
                self.in_features = module.in_features
                self.out_features = module.out_features
                self.group_size = module.group_size
                self.bits = module.bits
                self.qzeros = module.qzeros

        temp_module = TempModule(module)
        self.unpack_qzeros(temp_module)

        temp_module.qzeros += 1
        temp_module.qzeros = torch.bitwise_and(temp_module.qzeros, (2 ** temp_module.bits) - 1)

        self.pack_qzeros(temp_module)
        module.qzeros = temp_module.qzeros

    def _load_quant_config(self, quant_attrs):
        super()._load_quant_config(quant_attrs)
        self.overrides = quant_attrs["config"].get("dynamic", {})

    def get_overrides(self, layer_name):
        for pattern, overrides in self.overrides.items():
            if re.match(pattern.removeprefix("+:"), layer_name):
                return overrides
        return {}

    def get_layer_bits(self, layer_name):
        return self.get_overrides(layer_name).get("bits", self.global_bits)

    def get_layer_group_size(self, layer_name):
        return self.get_overrides(layer_name).get("group_size", self.global_group_size)

class QuarkModel(QuantizedModel):
    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        super().__init__(quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            self_attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            for _, q_tensors in self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

            # Unpack and repack all `QuantizedTensorModule` classes in MLP
            for _, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

        if isinstance(self.lm_head, QuantizedTensorModule) and self.lm_head.qweight is not None:
            self.unpack(self.lm_head)
            self.repack(self.lm_head)

            # Set `g_idx` to None since it's not used in `MatMulNBits`
            self.lm_head.g_idx = None

    def _load_quant_config(self, quant_attrs):
        self.global_quant_config = quant_attrs["config"]["global_quant_config"]["weight"]
        self.global_group_size = self.global_quant_config["group_size"]
        global_dtype = self.global_quant_config["dtype"]

        dtype_bits_maps = {
            "uint4": 4,
            "int4": 4,
        }

        if global_dtype not in dtype_bits_maps:
            raise ValueError(f"Unexpected dtype: {global_dtype}.")
        self.global_bits = dtype_bits_maps[global_dtype]

    def get_layer_bits(self, layer_name):
        name = layer_name.split(".")[0]
        if name in self._quant_attrs["config"]["layer_quant_config"]:
            layer_quant_config = self._quant_attrs["config"]["layer_quant_config"][name]["weight"]
            local_dtype = layer_quant_config["dtype"]

            dtype_bits_maps = {
                "uint4": 4,
                "int4": 4,
            }
            if local_dtype not in dtype_bits_maps:
                raise ValueError(f"Unexpected dtype: {local_dtype}.")
            return dtype_bits_maps[local_dtype]
        return self.global_bits
    
    def get_layer_group_size(self, layer_name):
        name = layer_name.split(".")[0]
        if name in self._quant_attrs["config"]["layer_quant_config"]:
            layer_quant_config = self._quant_attrs["config"]["layer_quant_config"][name]["weight"]
            return layer_quant_config["group_size"]
        return self.global_group_size

    def unpack_qweight(self, module):
        """
        Unpack `qweight` to standard format
        """
        expected_shape = (module.qweight.shape[0], module.out_features)
        transpose = module.qweight.shape != expected_shape
        module.qweight = self.unpack_on_row(module.qweight.T, module.bits, transpose)
        module.qweight = self.reverse_reorder_tensor(module.qweight.T, module.bits)
        # Padding might happen on the last dimension.
        module.qweight = module.qweight[:, : module.out_features]

    def unpack_qzeros(self, module):
        """
        Unpack `qzeros` to standard format
        """
        super().unpack_qzeros(module)
        module.qzeros = self.reverse_reorder_tensor(module.qzeros, module.bits)
        # Padding might happen on the last dimension.
        module.qzeros = module.qzeros[:, : module.out_features]

    def reverse_reorder_tensor(self, tensor, bits):
        """
        Re-arrange tensor data in a new order
        """
        compress_ratio = 32 // bits
        assert tensor.shape[-1] % compress_ratio == 0

        if bits == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError(f"Unpacking for {bits}-bit quantization is not currently supported.")

        order_tensor = torch.tensor(order_map, dtype=torch.int32).reshape(1, -1)
        order_tensor = order_tensor.repeat(tensor.shape[1] // compress_ratio, 1)
        order_tensor = order_tensor + torch.arange(0, tensor.shape[1], compress_ratio, dtype=torch.int32).reshape(-1, 1)
        order_tensor = order_tensor.reshape(-1)

        reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
        reverse_order_tensor = reverse_order_tensor[order_tensor]
        int_tensor = tensor[:, reverse_order_tensor]
        return int_tensor

class OliveModel(GPTQModel):
    def _load_quant_config(self, quant_attrs):
        super()._load_quant_config(quant_attrs)
        self.overrides = quant_attrs["config"]["overrides"] or {}

    def get_layer_bits(self, layer_name):
        name = ".".join(layer_name.split(".")[:-1])
        return self.overrides.get(name, {}).get("bits", self.global_bits)

    def get_layer_group_size(self, layer_name):
        name = ".".join(layer_name.split(".")[:-1])
        return self.overrides.get(name, {}).get("group_size", self.global_group_size)

class QuantModel:
    @staticmethod
    def from_pretrained(quant_type, **kwargs):
        """
        Unpack quantized weights in PyTorch models, store them in a standard format, and repack them
        into ONNX Runtime's format. Also performs any pre-processing and post-processing when unpacking
        the quantized weights.
        """
        if quant_type == "awq":
            model = AWQModel(quant_type, **kwargs)
        elif quant_type == "gptq":
            model = GPTQModel(quant_type, **kwargs)
        elif quant_type == "olive":
            model = OliveModel(quant_type, **kwargs)
        elif quant_type == "quark":
            model = QuarkModel(quant_type, **kwargs)
        else:
            raise NotImplementedError(f"The {quant_type} quantized model is not currently supported.")

        return model
