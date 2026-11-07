# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import math
import os
from types import SimpleNamespace

import onnx_ir as ir
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .base import QuantizedDecoderLayer, QuantizedExperts, TensorModule


class BailingModel:
    """Stream Bailing BF16, block-FP8, and packed-INT4 checkpoints into the common loader IR."""

    def __init__(self, input_path, config, target_is_int=False, num_layers=None):
        self.input_path = input_path
        self.config = config
        self.target_is_int = target_is_int
        self.handles = {}
        self.handle_keys = {}
        self.layers = [
            QuantizedDecoderLayer(i) for i in range(config.num_hidden_layers if num_layers is None else num_layers)
        ]

        index_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
            self.single_file = None
        else:
            self.weight_map = None
            candidates = sorted(f for f in os.listdir(input_path) if f.endswith(".safetensors"))
            if len(candidates) != 1:
                raise ValueError(f"Expected model.safetensors.index.json or a single .safetensors in {input_path}")
            self.single_file = candidates[0]

        try:
            embed_w = self.get_tensor("model.word_embeddings.weight")
            if embed_w is None:
                embed_w = self.get_tensor("model.embed_tokens.weight")
            self.embedding = TensorModule(weight=embed_w)
            self.final_norm = TensorModule(weight=self.get_tensor("model.norm.weight"))
            self.lm_head = self.make_linear_module("lm_head")
            if self.lm_head is None or self.lm_head.weight is None:
                if not getattr(config, "tie_word_embeddings", True):
                    raise ValueError("Untied Bailing checkpoints require lm_head.weight")
                self.lm_head = TensorModule(weight=self.embedding.weight)
        finally:
            self.close()

    @classmethod
    def from_pretrained(cls, input_path, config, *, cache_dir, token, revision, target_is_int=False, num_layers=None):
        if not os.path.isdir(input_path):
            input_path = snapshot_download(
                repo_id=input_path,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        return cls(input_path, config, target_is_int, num_layers)

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()
        self.handle_keys.clear()

    def __del__(self):
        if getattr(self, "handles", None):
            self.close()

    def modules(self):
        try:
            yield self.embedding
            for layer in self.layers:
                yield self.make_layer(layer.layer_id)
            yield self.final_norm
            yield self.lm_head
        finally:
            self.close()

    def get_tensor(self, name):
        if self.weight_map is not None:
            fname = self.weight_map.get(name)
            files = [fname] if fname is not None else []
        else:
            files = [self.single_file]
        for fname in files:
            handle = self.handles.get(fname)
            if handle is None:
                handle = safe_open(os.path.join(self.input_path, fname), framework="pt", device="cpu")
                self.handles[fname] = handle
                self.handle_keys[fname] = set(handle.keys())
            if name in self.handle_keys[fname]:
                return handle.get_tensor(name)
        return None

    def validate_fp8_blocks(self, weight, scale_inv, base):
        if weight.ndim != 2:
            raise ValueError(f"FP8 tensor '{base}.weight' must be a matrix")
        expected = tuple(math.ceil(size / 128) for size in weight.shape)
        if scale_inv is None or tuple(scale_inv.shape) != expected:
            raise ValueError(f"FP8 tensor '{base}.weight_scale_inv' must have shape {expected}")

    def dequantize_fp8_block(self, weight, scale_inv, block_size=128):
        n, k = weight.shape
        scales = scale_inv.float().repeat_interleave(block_size, dim=0)[:n]
        scales = scales.repeat_interleave(block_size, dim=1)[:, :k]
        return (weight.float() * scales).to(torch.bfloat16)

    def dequantize_int4_packed(self, weight_packed, weight_scale):
        if (
            weight_packed.dtype != torch.int32
            or weight_packed.ndim != 2
            or weight_packed.shape[1] == 0
            or weight_scale.ndim != 2
            or weight_scale.shape[0] != weight_packed.shape[0]
            or weight_scale.shape[1] == 0
            or weight_packed.shape[1] * 8 % weight_scale.shape[1] != 0
        ):
            raise ValueError("Bailing packed INT4 weights require int32 [N, K/8] weights and [N, K/group_size] scales")
        n, packed_k = weight_packed.shape
        k = packed_k * 8
        group_size = k // weight_scale.shape[1]
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        u4 = ((weight_packed.unsqueeze(-1) >> shifts) & 0xF).reshape(n, k)
        return ((u4.float() - 8.0) * weight_scale.float().repeat_interleave(group_size, dim=1)).to(torch.bfloat16)

    def make_linear_module(self, base, dense=False):
        weight = self.get_tensor(f"{base}.weight")
        scale_inv = self.get_tensor(f"{base}.weight_scale_inv")
        bias = self.get_tensor(f"{base}.bias")
        if weight is not None:
            if weight.dtype == torch.float8_e4m3fn:
                self.validate_fp8_blocks(weight, scale_inv, base)
                if dense or self.target_is_int:
                    weight = self.dequantize_fp8_block(weight, scale_inv)
                else:
                    module = TensorModule(weight=weight, bias=bias)
                    module.quant_type = "fp8"
                    module.block_size = 128
                    # Dense FP8 MatMul expects one scale row per output channel.
                    module.weight_scale = scale_inv.float().repeat_interleave(128, dim=0)[: weight.shape[0]].contiguous()
                    module.can_reuse_as_embedding = False
                    return module
            module = TensorModule(weight=weight, bias=bias)
            module.exclude_from_quantization = dense
            return module

        weight_packed = self.get_tensor(f"{base}.weight_packed")
        if weight_packed is None:
            return None
        weight_scale = self.get_tensor(f"{base}.weight_scale")
        if weight_scale is None:
            raise ValueError(f"Packed INT4 tensor '{base}.weight_packed' is missing '{base}.weight_scale'")
        module = TensorModule(weight=self.dequantize_int4_packed(weight_packed, weight_scale), bias=bias)
        module.exclude_from_quantization = dense
        return module

    def concat_linears(self, linears):
        first = linears[0]
        if any(module.quant_type != first.quant_type for module in linears):
            raise ValueError("Bailing fused projections must use the same checkpoint weight format")
        module = TensorModule(
            weight=torch.cat([linear.weight for linear in linears], dim=0).contiguous(),
            bias=torch.cat([linear.bias for linear in linears], dim=0).contiguous() if first.bias is not None else None,
        )
        if first.quant_type == "fp8":
            module.quant_type = "fp8"
            module.block_size = first.block_size
            module.weight_scale = torch.cat([linear.weight_scale for linear in linears], dim=0).contiguous()
            module.can_reuse_as_embedding = False
        return module

    def make_mla_attention(self, prefix):
        attention = SimpleNamespace()
        if getattr(self.config, "q_lora_rank", 256) is None:
            attention.q_proj = self.make_linear_module(f"{prefix}.q_proj")
        else:
            for name in ("q_a_proj", "q_b_proj"):
                setattr(attention, name, self.make_linear_module(f"{prefix}.{name}"))
            attention.q_a_layernorm = TensorModule(weight=self.get_tensor(f"{prefix}.q_a_layernorm.weight"))
        for name in ("kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "dense"):
            setattr(attention, name, self.make_linear_module(f"{prefix}.{name}"))
        attention.kv_a_layernorm = TensorModule(weight=self.get_tensor(f"{prefix}.kv_a_layernorm.weight"))
        return attention

    def make_kda_attention(self, prefix):
        attention = SimpleNamespace()
        attention.in_proj_qkv = self.concat_linears(
            [self.make_linear_module(f"{prefix}.{name}_proj") for name in ("q", "k", "v")]
        )
        attention.in_proj_a = self.make_linear_module(f"{prefix}.f_proj")
        attention.in_proj_b = self.make_linear_module(f"{prefix}.b_proj", dense=True)
        if self.get_tensor(f"{prefix}.g_proj.weight") is not None:
            attention.in_proj_z = self.make_linear_module(f"{prefix}.g_proj")
            attention.in_proj_z_a = None
            attention.in_proj_z_b = None
        else:
            attention.in_proj_z = None
            attention.in_proj_z_a = self.make_linear_module(f"{prefix}.g_proj_a")
            attention.in_proj_z_b = self.make_linear_module(f"{prefix}.g_proj_b")
        conv_weights = [self.get_tensor(f"{prefix}.{name}_conv1d.weight") for name in ("q", "k", "v")]
        attention.conv1d = TensorModule(weight=torch.cat(conv_weights, dim=0).contiguous())
        attention.A_log = self.get_tensor(f"{prefix}.A_log").float().view(-1)
        attention.dt_bias = self.get_tensor(f"{prefix}.dt_bias").float().view(
            self.config.linear_num_value_heads, self.config.linear_key_head_dim
        )
        attention.norm = TensorModule(weight=self.get_tensor(f"{prefix}.o_norm.weight"))
        attention.out_proj = self.make_linear_module(f"{prefix}.o_proj")
        return attention

    def make_dense_mlp(self, prefix):
        return SimpleNamespace(
            gate_proj=self.make_linear_module(f"{prefix}.gate_proj"),
            up_proj=self.make_linear_module(f"{prefix}.up_proj"),
            down_proj=self.make_linear_module(f"{prefix}.down_proj"),
        )

    def make_fp8_experts(self, prefix):
        prepared = QuantizedExperts()
        prepared.quant_type = "fp8"
        prepared.block_size = 128
        prepared.scale_dtype = ir.DataType.FLOAT
        gate_up_w, gate_up_s, down_w, down_s = [], [], [], []
        for expert_id in range(self.config.num_experts):
            expert_prefix = f"{prefix}.{expert_id}"
            projections = [
                self.make_linear_module(f"{expert_prefix}.{name}") for name in ("gate_proj", "up_proj", "down_proj")
            ]
            if any(module.quant_type != "fp8" for module in projections):
                raise ValueError("Bailing native FP8 experts must use FP8 weights for every projection")
            gate, up, down = projections
            gate_up_w.append(torch.cat([gate.weight, up.weight], dim=0))
            gate_up_s.append(
                torch.cat(
                    [self.get_tensor(f"{expert_prefix}.{name}.weight_scale_inv").float() for name in ("gate_proj", "up_proj")],
                    dim=0,
                )
            )
            down_w.append(down.weight)
            down_s.append(self.get_tensor(f"{expert_prefix}.down_proj.weight_scale_inv").float())
        prepared.gate_up_qweight = torch.stack(gate_up_w).contiguous()
        prepared.gate_up_scales = torch.stack(gate_up_s).contiguous()
        prepared.down_qweight = torch.stack(down_w).contiguous()
        prepared.down_scales = torch.stack(down_s).contiguous()
        return prepared

    def make_experts(self, prefix):
        first_weight = self.get_tensor(f"{prefix}.0.gate_proj.weight")
        if first_weight is not None and first_weight.dtype == torch.float8_e4m3fn and not self.target_is_int:
            return self.make_fp8_experts(prefix)
        gate_up_weights, down_weights = [], []
        for expert_id in range(self.config.num_experts):
            expert_prefix = f"{prefix}.{expert_id}"
            gate = self.make_linear_module(f"{expert_prefix}.gate_proj", dense=True)
            up = self.make_linear_module(f"{expert_prefix}.up_proj", dense=True)
            down = self.make_linear_module(f"{expert_prefix}.down_proj", dense=True)
            gate_up_weights.append(torch.cat([gate.weight, up.weight], dim=0))
            down_weights.append(down.weight)
        return SimpleNamespace(
            gate_up_proj=torch.stack(gate_up_weights).contiguous(),
            down_proj=torch.stack(down_weights).contiguous(),
        )

    def make_moe(self, prefix):
        return SimpleNamespace(
            gate=self.make_linear_module(f"{prefix}.gate", dense=True),
            expert_bias=self.get_tensor(f"{prefix}.gate.expert_bias").float(),
            shared_experts=self.make_dense_mlp(f"{prefix}.shared_experts"),
            experts=self.make_experts(f"{prefix}.experts"),
        )

    def make_layer(self, layer_id):
        prefix = f"model.layers.{layer_id}"
        layer = QuantizedDecoderLayer(layer_id)
        layer.input_layernorm = TensorModule(weight=self.get_tensor(f"{prefix}.input_layernorm.weight"))
        layer.post_attention_layernorm = TensorModule(weight=self.get_tensor(f"{prefix}.post_attention_layernorm.weight"))
        is_mla = self.config.layer_types[layer_id] == "full_attention"
        layer.self_attn = self.make_mla_attention(f"{prefix}.attention") if is_mla else None
        layer.linear_attn = None if is_mla else self.make_kda_attention(f"{prefix}.attention")
        layer.mlp = (
            self.make_dense_mlp(f"{prefix}.mlp")
            if layer_id < getattr(self.config, "first_k_dense_replace", 1)
            else self.make_moe(f"{prefix}.mlp")
        )
        return layer
