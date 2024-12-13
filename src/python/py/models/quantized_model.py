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
    def __init__(self, bits, group_size):
        self.qweight = None
        self.scales = None
        self.qzeros = None
        self.g_idx = None
        self.bias = None

        self.in_features = 0
        self.out_features = 0
        self.bits = bits
        self.group_size = group_size

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
    def __init__(self, bits, group_size):
        self.q_proj = QuantizedTensorModule(bits, group_size)
        self.k_proj = QuantizedTensorModule(bits, group_size)
        self.v_proj = QuantizedTensorModule(bits, group_size)
        self.o_proj = QuantizedTensorModule(bits, group_size)
        self.rotary_emb = TensorModule()


class QuantizedMLP:
    def __init__(self, bits, group_size):
        self.gate_proj = QuantizedTensorModule(bits, group_size)
        self.up_proj = QuantizedTensorModule(bits, group_size)
        self.down_proj = QuantizedTensorModule(bits, group_size)
        self.fc1 = QuantizedTensorModule(bits, group_size)
        self.fc2 = QuantizedTensorModule(bits, group_size)


class QuantizedDecoderLayer:
    def __init__(self, layer_id, bits, group_size):
        self.layer_id = layer_id
        self.input_layernorm = TensorModule()
        self.self_attn = QuantizedAttention(bits, group_size)
        self.post_attention_layernorm = TensorModule()
        self.mlp = QuantizedMLP(bits, group_size)

    def is_empty(self):
        return self.input_layernorm.weight is None


class QuantizedModel:
    def __init__(self, quant_type, input_path, bits, group_size, q_size, kv_size, intermediate_size, num_layers):
        self.quant_type = quant_type
        self.embedding = TensorModule()
        self.final_norm = TensorModule()
        self.lm_head = TensorModule()
        self.layers = {}
        self.num_layers = num_layers

        layer_id = 0
        for weight_file in os.listdir(input_path):
            if weight_file.endswith(".safetensors"):
                module = self.layers.setdefault(layer_id, QuantizedDecoderLayer(layer_id, bits, group_size))
                weights = load_file(os.path.join(input_path, weight_file))

                # Map weights to modules
                for name, tensor in weights.items():
                    if tensor.dtype == torch.bfloat16:
                        # Cast bfloat16 to float32 since NumPy does not support bfloat16
                        tensor = tensor.to(torch.float32)
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
                        self._initialize_quantized_lm_head(bits, group_size)
                        self.lm_head.qweight = tensor
                    elif name == "lm_head.qzeros" or name == "transformer.output_layer.qzeros":
                        self._initialize_quantized_lm_head(bits, group_size)
                        self.lm_head.qzeros = tensor
                    elif name == "lm_head.scales" or name == "transformer.output_layer.scales":
                        self._initialize_quantized_lm_head(bits, group_size)
                        self.lm_head.scales = tensor
                    elif name == "lm_head.g_idx" or name == "transformer.output_layer.g_idx":
                        self._initialize_quantized_lm_head(bits, group_size)
                        self.lm_head.g_idx = tensor
                    else:
                        if name.startswith("transformer.encoder"):
                            # Chatglm3, e.g., transformer.encoder.layers.0.input_layernorm.weight
                            name = name.replace("transformer.encoder", "model")
                        curr_layer_id = int(name.split(".")[2])
                        if curr_layer_id != layer_id:
                            # Switch layer module used
                            layer_id = curr_layer_id
                            module = self.layers.setdefault(layer_id, QuantizedDecoderLayer(layer_id, bits, group_size))

                        # Map weights and biases of norm, attention, and feed-forward network
                        # Graph order is input_layernorm --> q_proj/k_proj/v_proj --> o_proj --> post_attention_layernorm --> gate_proj/up_proj --> down_proj
                        if bool(re.match(r"^model.layers\.\d+\.input_layernorm\.weight$", name)):
                            # model.layers.layer_id.input_layernorm.weight
                            module.input_layernorm.weight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.input_layernorm\.bias$", name)):
                            # model.layers.layer_id.input_layernorm.bias
                            module.input_layernorm.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.rotary_emb\.inv_freq$", name)):
                            # model.layers.layer_id.self_attn.rotary_emb.inv_freq
                            # Skip rotary embedding weights since they can be re-calculated when looping through the model
                            continue
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.q_proj.qweight
                            module.self_attn.q_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.q_proj.scales
                            module.self_attn.q_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.q_proj.qzeros
                            module.self_attn.q_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.q_proj.g_idx
                            module.self_attn.q_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.q_proj.bias
                            module.self_attn.q_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qweight
                            module.self_attn.k_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.k_proj.scales
                            module.self_attn.k_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qzeros
                            module.self_attn.k_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.k_proj.g_idx
                            module.self_attn.k_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.k_proj.bias
                            module.self_attn.k_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qweight
                            module.self_attn.v_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.v_proj.scales
                            module.self_attn.v_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qzeros
                            module.self_attn.v_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.v_proj.g_idx
                            module.self_attn.v_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.v_proj.bias
                            module.self_attn.v_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.qweight$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qweight
                            # model.layers.layer_id.self_attention.dense.qweight
                            module.self_attn.o_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.scales$", name)):
                            # model.layers.layer_id.self_attn.o_proj.scales
                            # model.layers.layer_id.self_attention.dense.scales
                            module.self_attn.o_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qzeros
                            # model.layers.layer_id.self_attention.dense.qzeros
                            module.self_attn.o_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.o_proj.g_idx
                            # model.layers.layer_id.self_attention.dense.g_idx
                            module.self_attn.o_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.o_proj|self_attention.dense)\.bias$", name)):
                            # model.layers.layer_id.self_attn.o_proj.bias
                            # model.layers.layer_id.self_attention.dense.bias
                            module.self_attn.o_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.weight$", name)):
                            # model.layers.layer_id.post_attention_layernorm.weight
                            module.post_attention_layernorm.weight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.bias$", name)):
                            # model.layers.layer_id.post_attention_layernorm.bias
                            module.post_attention_layernorm.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.qweight$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qweight
                            module.mlp.gate_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.scales$", name)):
                            # model.layers.layer_id.mlp.gate_proj.scales
                            module.mlp.gate_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.qzeros$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qzeros
                            module.mlp.gate_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.gate_proj.g_idx
                            module.mlp.gate_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.gate_proj.bias
                            module.mlp.gate_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.qweight$", name)):
                            # model.layers.layer_id.mlp.up_proj.qweight
                            module.mlp.up_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.scales$", name)):
                            # model.layers.layer_id.mlp.up_proj.scales
                            module.mlp.up_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.qzeros$", name)):
                            # model.layers.layer_id.mlp.up_proj.qzeros
                            module.mlp.up_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.up_proj.g_idx
                            module.mlp.up_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.up_proj.bias
                            module.mlp.up_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.qweight$", name)):
                            # model.layers.layer_id.mlp.down_proj.qweight
                            # model.layers.layer_id.mlp.dense_4h_to_h.qweight
                            module.mlp.down_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.scales$", name)):
                            # model.layers.layer_id.mlp.down_proj.scales
                            # model.layers.layer_id.mlp.dense_4h_to_h.scales
                            module.mlp.down_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.qzeros$", name)):
                            # model.layers.layer_id.mlp.down_proj.qzeros
                            # model.layers.layer_id.mlp.dense_4h_to_h.qzeros
                            module.mlp.down_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.g_idx$", name)):
                            # model.layers.layer_id.mlp.down_proj.g_idx
                            # model.layers.layer_id.mlp.dense_4h_to_h.g_idx
                            module.mlp.down_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(down_proj|dense_4h_to_h)\.bias$", name)):
                            # model.layers.layer_id.mlp.down_proj.bias
                            # model.layers.layer_id.mlp.dense_4h_to_h.bias
                            module.mlp.down_proj.bias = tensor
                        # Match against fused layers
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.qweight$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.qweight
                            # model.layers.layer_id.self_attention.query_key_value.qweight
                            q_dim = q_size // (32 // bits) if quant_type == "awq" else q_size
                            kv_dim = kv_size // (32 // bits) if quant_type == "awq" else kv_size
                            module.self_attn.q_proj.qweight = tensor[:, : q_dim]
                            module.self_attn.k_proj.qweight = tensor[:, q_dim : q_dim + kv_dim]
                            module.self_attn.v_proj.qweight = tensor[:, q_dim + kv_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.scales$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.scales
                            # model.layers.layer_id.self_attention.query_key_value.scales
                            module.self_attn.q_proj.scales = tensor[:, : q_size]
                            module.self_attn.k_proj.scales = tensor[:, q_size : q_size + kv_size]
                            module.self_attn.v_proj.scales = tensor[:, q_size + kv_size :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.qzeros
                            # model.layers.layer_id.self_attention.query_key_value.qzeros
                            q_dim = q_size // (32 // bits) if quant_type in {"awq", "gptq"} else q_size
                            kv_dim = kv_size // (32 // bits) if quant_type in {"awq", "gptq"} else kv_size
                            module.self_attn.q_proj.qzeros = tensor[:, : q_dim]
                            module.self_attn.k_proj.qzeros = tensor[:, q_dim : q_dim + kv_dim]
                            module.self_attn.v_proj.qzeros = tensor[:, q_dim + kv_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.g_idx
                            # model.layers.layer_id.self_attention.query_key_value.g_idx
                            module.self_attn.q_proj.g_idx = tensor
                            module.self_attn.k_proj.g_idx = tensor
                            module.self_attn.v_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.(self_attn.qkv_proj|self_attention.query_key_value)\.bias$", name)):
                            # model.layers.layer_id.self_attn.qkv_proj.bias
                            # model.layers.layer_id.self_attention.query_key_value.bias
                            module.self_attn.q_proj.bias = tensor[: q_size]
                            module.self_attn.k_proj.bias = tensor[q_size : q_size + kv_size]
                            module.self_attn.v_proj.bias = tensor[q_size + kv_size : ]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.qweight$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.qweight
                            # model.layers.layer_id.mlp.dense_h_to_4h.qweight
                            intermediate_dim = intermediate_size // (32 // bits) if quant_type == "awq" else intermediate_size
                            module.mlp.gate_proj.qweight = tensor[:, : intermediate_dim]
                            module.mlp.up_proj.qweight = tensor[:, intermediate_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.scales$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.scales
                            # model.layers.layer_id.mlp.dense_h_to_4h.scales
                            module.mlp.gate_proj.scales = tensor[:, : intermediate_size]
                            module.mlp.up_proj.scales = tensor[:, intermediate_size :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.qzeros$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.qzeros
                            # model.layers.layer_id.mlp.dense_h_to_4h.qzeros
                            intermediate_dim = intermediate_size // (32 // bits) if quant_type in {"awq", "gptq"} else intermediate_size
                            module.mlp.gate_proj.qzeros = tensor[:, : intermediate_dim]
                            module.mlp.up_proj.qzeros = tensor[:, intermediate_dim :]
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.g_idx$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.g_idx
                            # model.layers.layer_id.mlp.dense_h_to_4h.g_idx
                            module.mlp.gate_proj.g_idx = tensor
                            module.mlp.up_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.(gate_up_proj|dense_h_to_4h)\.bias$", name)):
                            # model.layers.layer_id.mlp.gate_up_proj.bias
                            # model.layers.layer_id.mlp.dense_h_to_4h.bias
                            module.mlp.gate_proj.bias = tensor[: intermediate_size]
                            module.mlp.up_proj.bias = tensor[intermediate_size: ]
                        else:
                            raise NotImplementedError(f"{name} in your quantized model is not recognized.")

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

    def _initialize_quantized_lm_head(self, bits, group_size):
        """
        Initialize `QuantizedTensorModule` for LM head if not already set
        """
        if isinstance(self.lm_head, TensorModule):
            assert self.lm_head.weight is None
            assert self.lm_head.bias is None
        if not isinstance(self.lm_head, QuantizedTensorModule):
            self.lm_head = QuantizedTensorModule(bits, group_size)

    def set_properties(self):
        """
        Set in_features, out_features, and g_idx based on quantization type
        """
        if isinstance(self.lm_head, QuantizedTensorModule):
            if self.quant_type == "awq":
                self.lm_head.out_features = self.lm_head.scales.shape[1]
                self.lm_head.in_features = self.lm_head.qweight.shape[0]
                # Set g_idx if not already set
                self.lm_head.g_idx = self.lm_head.g_idx if self.lm_head.g_idx is not None else torch.tensor([i // self.lm_head.group_size for i in range(self.lm_head.in_features)], dtype=torch.int32)
            elif self.quant_type == "gptq":
                self.lm_head.out_features = self.lm_head.qweight.shape[1]
                self.lm_head.in_features = self.lm_head.g_idx.shape[0]
            else:
                raise NotImplementedError(f"The {self.quant_type} quantization method is not recognized.")
        for module in self.layers:
            if self.quant_type == "awq":
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
        Unpack tensor by row
        """
        if bits in {2, 4, 8}:
            return self.unpack_on_row_for_2_4_8_bits(tensor, bits, transpose)
        else:
            raise NotImplementedError(f"Unpacking for {bits}-bit quantization is not currently supported.")

    def pack_on_row_for_2_4_8_bits(self, tensor, bits, transpose):
        """
        Perform general-purpose packing on 2-bit, 4-bit, or 8-bit tensor
        """
        orig_tensor = tensor.T if transpose else tensor
        wf = torch.arange(0, bits).view(1, 1, -1)
        out = torch.bitwise_right_shift(orig_tensor.unsqueeze(-1), wf)
        out = torch.bitwise_and(out, 1)
        out = out.reshape(orig_tensor.shape[0], -1, 32)
        wf1 = torch.arange(0, 32, 1).view(1, 1, -1)
        out = torch.bitwise_left_shift(out, wf1)
        out = out.sum(dim=-1).int()
        return out.T if transpose else out

    def pack_on_row(self, tensor, bits, transpose):
        """
        Pack tensor by row
        """
        if bits in {2, 4, 8}:
            return self.pack_on_row_for_2_4_8_bits(tensor, bits, transpose)
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
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        qdq_weight_T = intweight * scale_mat - scale_zeros_mat.half()

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
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round((weight + scale_zeros_mat) / scale_mat).to(torch.int)

        return intweight_T

    def pack_ort_format(self, module, intweight):
        """
        Pack `scales`, `qzeros`, and `qweight` to ORT format
        """
        if module.bits != 4:
            raise NotImplementedError(f"{module.bits}-bit quantization in ORT is not currently supported by this tool.")

        intzeros_pt = module.qzeros.T if module.qzeros.dtype == module.scales.dtype else module.qzeros.T.byte()
        intweight_pt = intweight.byte()
        block_size = module.group_size

        rows, cols = intweight_pt.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(intweight_pt, (0, 0, 0, pad_len), "constant", 0)
        intzeros_pt = torch.nn.functional.pad(intzeros_pt, (0, intzeros_pt.shape[-1] & 1, 0, 0), "constant", 0)

        if module.qzeros.dtype != module.scales.dtype:
            intzeros_pt = (intzeros_pt[:, 0::2]) | (intzeros_pt[:, 1::2] << 4)
            intzeros_pt = intzeros_pt.reshape(-1)

        intweight_pt_T = intweight.T
        intweight_pt_T = (intweight_pt_T[:, 0::2]) | (intweight_pt_T[:, 1::2] << 4)
        intweight_pt_T = intweight_pt_T.reshape(cols, k_blocks, blob_size)

        scales_pt = module.scales.T.reshape(-1)

        module.scales = scales_pt.contiguous()
        module.qweight = intweight_pt_T.contiguous().byte()
        if module.qzeros.dtype != module.scales.dtype:
            module.qzeros = intzeros_pt.contiguous().byte()
        else:
            module.qzeros = intzeros_pt.contiguous()


class AWQModel(QuantizedModel):
    def __init__(self, quant_type, input_path, bits, group_size, q_size, kv_size, intermediate_size, num_layers):
        super().__init__(quant_type, input_path, bits, group_size, q_size, kv_size, intermediate_size, num_layers)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break
            print(f"Unpacking and repacking layer {i}")

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            self_attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            for name, q_tensors in self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

            # Unpack and repack all `Quantized TensorModule` classes in MLP
            for name, q_tensors in layer.mlp.__dict__.items():
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
    def __init__(self, quant_type, input_path, bits, group_size, use_g_idx, q_size, kv_size, intermediate_size, num_layers):
        super().__init__(quant_type, input_path, bits, group_size, q_size, kv_size, intermediate_size, num_layers)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break
            print(f"Unpacking and repacking layer {i}")

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            for name, q_tensors in layer.self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.handle_qzeros(q_tensors)
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    if not use_g_idx:
                        # Set `g_idx` to None since it's not used in `MatMulNBits`
                        q_tensors.g_idx = None

            # Unpack and repack all `QuantizedTensorModule` classes in MLP
            for name, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.handle_qzeros(q_tensors)
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

                    if not use_g_idx:
                        # Set `g_idx` to None since it's not used in `MatMulNBits`
                        q_tensors.g_idx = None

        if isinstance(self.lm_head, QuantizedTensorModule) and self.lm_head.qweight is not None:
            self.handle_qzeros(self.lm_head)
            self.unpack(self.lm_head)
            self.repack(self.lm_head)

            if not use_g_idx:
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


class QuantModel:
    @staticmethod
    def from_pretrained(quant_type, input_path, bits, group_size, use_g_idx, q_size, kv_size, intermediate_size, num_layers):
        """
        Unpack quantized weights in PyTorch models, store them in a standard format, and repack them
        into ONNX Runtime's format. Also performs any pre-processing and post-processing when unpacking
        the quantized weights.
        """
        if quant_type == "awq":
            model = AWQModel(quant_type, input_path, bits, group_size, q_size, kv_size, intermediate_size, num_layers)
        elif quant_type == "gptq":
            model = GPTQModel(quant_type, input_path, bits, group_size, use_g_idx, q_size, kv_size, intermediate_size, num_layers)
        else:
            raise NotImplementedError(f"The {quant_type} quantized model is not currently supported.")

        return model
