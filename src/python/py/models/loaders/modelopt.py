# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import json
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open

from .base import QuantizedDecoderLayer, QuantizedExperts, QuantizedModel, TensorModule


class ModeloptModel(QuantizedModel):
    """Loader for NVIDIA Model Optimizer NVFP4 + FP8 mixed-precision checkpoints.

    The base initializes the module surface the builder walks without running its
    eager integer-checkpoint loading path. This loader instead streams tensors on
    demand and prepacks routed NVFP4 experts for native QMoE export.
    """

    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        with open(os.path.join(input_path, "config.json")) as f:
            cfg = json.load(f)
        text_config = cfg.get("text_config", cfg)
        num_layers = num_layers or text_config.get("num_hidden_layers")
        self.num_experts = text_config.get("num_experts", text_config.get("num_local_experts"))
        super().__init__(
            quant_type,
            input_path,
            quant_attrs,
            q_size,
            kv_size,
            intermediate_size,
            num_layers,
            load_weights=False,
            lm_head=TensorModule(),
        )
        self.input_path = input_path
        self.handles = {}
        self.handle_keys = {}

        index_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
            self.single_file = None
        else:
            self.weight_map = None
            candidates = sorted(f for f in os.listdir(input_path) if f.endswith(".safetensors"))
            if len(candidates) != 1:
                raise ValueError(
                    f"'{input_path}' has no 'model.safetensors.index.json', so it must contain exactly one "
                    f".safetensors file, but found {len(candidates)}: {candidates}."
                )
            self.single_file = candidates[0]

        try:
            self.layers.extend(self.make_layer(layer_id) for layer_id in range(num_layers))

            # Globals: embeddings + final norm are BF16; lm_head retains its native NVFP4 tensors.
            self.embedding.weight = self.get_tensor("model.language_model.embed_tokens.weight")
            self.final_norm.weight = self.get_tensor("model.language_model.norm.weight")
            self.make_linear_module("lm_head", self.lm_head)
            self.mtp = self.make_mtp()
        finally:
            # Every tensor is materialized above; do not hold file descriptors open for
            # the rest of the export. get_tensor re-opens lazily if it is called again.
            self.close()

    def close(self):
        """Release the cached safetensors file handles."""
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()
        self.handle_keys.clear()

    def __del__(self):
        if getattr(self, "handles", None):
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

    def validate_positive_scalar(self, tensor, name):
        if tensor is None or tensor.numel() != 1:
            shape = None if tensor is None else tuple(tensor.shape)
            raise ValueError(f"ModelOpt tensor '{name}' must be a scalar, got shape {shape}.")
        value = tensor.float()
        if not torch.isfinite(value).item() or value.item() <= 0:
            raise ValueError(f"ModelOpt tensor '{name}' must be finite and positive, got {value.item()}.")

    def validate_linear(self, module, base):
        if module.weight_scale_2 is not None:
            if module.weight.dtype != torch.uint8 or module.weight.ndim != 2 or module.weight.shape[1] % 8 != 0:
                raise ValueError(
                    f"ModelOpt tensor '{base}.weight' must be packed uint8 [N, K/2] with K divisible by 16, "
                    f"got dtype={module.weight.dtype} shape={tuple(module.weight.shape)}."
                )
            expected_shape = (int(module.weight.shape[0]), int(module.weight.shape[1]) // 8)
            if module.weight_scale is None:
                raise ValueError(
                    f"NVFP4 tensor '{base}' has 'weight_scale_2' but no 'weight_scale' "
                    "(FP8-E4M3 block scales). The Model Optimizer checkpoint is incomplete."
                )
            if module.weight_scale.dtype not in {torch.uint8, torch.float8_e4m3fn}:
                raise ValueError(
                    f"ModelOpt tensor '{base}.weight_scale' must contain E4M3 bytes, got {module.weight_scale.dtype}."
                )
            if tuple(module.weight_scale.shape) != expected_shape:
                raise ValueError(
                    f"ModelOpt tensor '{base}.weight_scale' has shape {tuple(module.weight_scale.shape)}, "
                    f"expected {expected_shape}."
                )
            self.validate_positive_scalar(module.weight_scale_2, f"{base}.weight_scale_2")
            module.quant_type = "nvfp4"
            module.weight_scale = module.weight_scale.view(torch.uint8).contiguous()
        elif module.weight.dtype == torch.float8_e4m3fn:
            self.validate_positive_scalar(module.weight_scale, f"{base}.weight_scale")
            if module.input_scale is not None:
                self.validate_positive_scalar(module.input_scale, f"{base}.input_scale")
            module.quant_type = "fp8"

    def make_linear_module(self, base, module=None):
        weight = self.get_tensor(f"{base}.weight")
        if weight is None:
            return None
        module = module if module is not None else TensorModule()
        module.weight = weight
        module.weight_scale = self.get_tensor(f"{base}.weight_scale")
        module.weight_scale_2 = self.get_tensor(f"{base}.weight_scale_2")
        if ".self_attn." in base:
            module.input_scale = self.get_tensor(f"{base}.input_scale")
        self.validate_linear(module, base)
        bias = self.get_tensor(f"{base}.bias")
        if bias is not None:
            module.bias = bias
        return module

    def make_tensor_module(self, name):
        module = TensorModule()
        module.weight = self.get_tensor(name)
        return module

    def prepare_qmoe_experts(self, experts):
        def unpack(weight):
            low = weight & 0x0F
            high = weight >> 4
            return torch.stack((low, high), dim=-1).reshape(weight.shape[0], -1)

        def pack(codes):
            if codes.shape[0] % 2 != 0:
                raise ValueError(f"NVFP4 QMoE packing requires an even N={codes.shape[0]} for nibble packing.")
            codes = codes.T.contiguous()
            return ((codes[:, 1::2] << 4) | (codes[:, 0::2] & 0x0F)).contiguous()

        def scale_bytes(projection):
            return projection.weight_scale.view(torch.uint8).contiguous()

        gate_up_weights, gate_up_scales, gate_up_globals = [], [], []
        down_weights, down_scales, down_globals = [], [], []
        for expert_id, expert in enumerate(experts):
            gate_codes = unpack(expert.gate_proj.weight)
            up_codes = unpack(expert.up_proj.weight)
            if gate_codes.shape != up_codes.shape:
                raise ValueError(
                    f"ModelOpt expert {expert_id} gate/up weights must have matching shapes, "
                    f"got {tuple(gate_codes.shape)} and {tuple(up_codes.shape)}."
                )

            intermediate_size = gate_codes.shape[0]
            fused_codes = torch.stack((gate_codes, up_codes), dim=1).reshape(2 * intermediate_size, -1)
            gate_up_weights.append(pack(fused_codes))
            gate_up_scales.append(
                torch.stack((scale_bytes(expert.gate_proj), scale_bytes(expert.up_proj)), dim=1).reshape(
                    2 * intermediate_size, -1
                )
            )

            gate_global = expert.gate_proj.weight_scale_2.float().reshape(())
            up_global = expert.up_proj.weight_scale_2.float().reshape(())
            if gate_global.item() != up_global.item():
                raise ValueError(
                    f"ModelOpt expert {expert_id} gate/up global scales must match for fused QMoE, "
                    f"got {gate_global.item()} and {up_global.item()}."
                )
            gate_up_globals.append(gate_global)

            down_weights.append(pack(unpack(expert.down_proj.weight)))
            down_scales.append(scale_bytes(expert.down_proj))
            down_globals.append(expert.down_proj.weight_scale_2.float().reshape(()))

        prepared = QuantizedExperts()
        prepared.quant_type = "nvfp4"
        prepared.block_size = 16
        prepared.gate_up_qweight = torch.stack(gate_up_weights)
        prepared.gate_up_scales = torch.stack(gate_up_scales)
        prepared.gate_up_global_scales = torch.stack(gate_up_globals)
        prepared.down_qweight = torch.stack(down_weights)
        prepared.down_scales = torch.stack(down_scales)
        prepared.down_global_scales = torch.stack(down_globals)
        return prepared

    def make_layer(self, layer_id, prefix=None):
        prefix = prefix or f"model.language_model.layers.{layer_id}"
        layer = QuantizedDecoderLayer(layer_id)
        layer.input_layernorm.weight = self.get_tensor(f"{prefix}.input_layernorm.weight")
        layer.post_attention_layernorm.weight = self.get_tensor(f"{prefix}.post_attention_layernorm.weight")
        layer.self_attn = None
        layer.linear_attn = None

        if self.get_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight") is not None:
            la = SimpleNamespace()
            la.in_proj_qkv = self.make_linear_module(f"{prefix}.linear_attn.in_proj_qkv")
            la.in_proj_z = self.make_linear_module(f"{prefix}.linear_attn.in_proj_z")
            la.in_proj_a = self.make_linear_module(f"{prefix}.linear_attn.in_proj_a")
            la.in_proj_b = self.make_linear_module(f"{prefix}.linear_attn.in_proj_b")
            la.out_proj = self.make_linear_module(f"{prefix}.linear_attn.out_proj")
            la.conv1d = self.make_tensor_module(f"{prefix}.linear_attn.conv1d.weight")
            la.A_log = self.get_tensor(f"{prefix}.linear_attn.A_log")
            la.dt_bias = self.get_tensor(f"{prefix}.linear_attn.dt_bias")
            la.norm = self.make_tensor_module(f"{prefix}.linear_attn.norm.weight")
            layer.linear_attn = la
        elif self.get_tensor(f"{prefix}.self_attn.q_proj.weight") is not None:
            sa = SimpleNamespace()
            sa.q_proj = self.make_linear_module(f"{prefix}.self_attn.q_proj")
            sa.k_proj = self.make_linear_module(f"{prefix}.self_attn.k_proj")
            sa.v_proj = self.make_linear_module(f"{prefix}.self_attn.v_proj")
            sa.o_proj = self.make_linear_module(f"{prefix}.self_attn.o_proj")
            sa.q_norm = self.make_tensor_module(f"{prefix}.self_attn.q_norm.weight")
            sa.k_norm = self.make_tensor_module(f"{prefix}.self_attn.k_norm.weight")
            layer.self_attn = sa
        else:
            raise ValueError(
                f"Layer {layer_id} has neither '{prefix}.linear_attn.in_proj_qkv.weight' nor "
                f"'{prefix}.self_attn.q_proj.weight', so its attention variant cannot be determined. "
                "The Model Optimizer checkpoint is incomplete or uses unsupported weight names."
            )

        mlp = SimpleNamespace()
        mlp.gate = self.make_tensor_module(f"{prefix}.mlp.gate.weight")
        shared = SimpleNamespace()
        shared.gate_proj = self.make_linear_module(f"{prefix}.mlp.shared_expert.gate_proj")
        shared.up_proj = self.make_linear_module(f"{prefix}.mlp.shared_expert.up_proj")
        shared.down_proj = self.make_linear_module(f"{prefix}.mlp.shared_expert.down_proj")
        mlp.shared_expert = shared
        mlp.shared_expert_gate = self.make_tensor_module(f"{prefix}.mlp.shared_expert_gate.weight")
        mlp.experts = []
        for expert_id in range(self.num_experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert_id}"
            expert = SimpleNamespace()
            expert.gate_proj = self.make_linear_module(f"{expert_prefix}.gate_proj")
            expert.up_proj = self.make_linear_module(f"{expert_prefix}.up_proj")
            expert.down_proj = self.make_linear_module(f"{expert_prefix}.down_proj")
            mlp.experts.append(expert)
        mlp.experts = self.prepare_qmoe_experts(mlp.experts)
        layer.mlp = mlp
        return layer

    def make_mtp(self):
        if self.get_tensor("mtp.fc.weight") is None:
            return None
        tensor_names = self.weight_map if self.weight_map is not None else self.handle_keys[self.single_file]
        state = {name: self.get_tensor(name) for name in tensor_names if name.startswith("mtp.")}
        return SimpleNamespace(
            fc=self.make_linear_module("mtp.fc"),
            pre_fc_norm_embedding=self.make_tensor_module("mtp.pre_fc_norm_embedding.weight"),
            pre_fc_norm_hidden=self.make_tensor_module("mtp.pre_fc_norm_hidden.weight"),
            norm=self.make_tensor_module("mtp.norm.weight"),
            layers=[self.make_layer(0, "mtp.layers.0")],
            state=state,
        )

    def dequantize_tensor(self, weight, weight_scale, weight_scale_2, name):
        if weight_scale_2 is not None:
            if weight_scale is None:
                raise ValueError(f"ModelOpt NVFP4 tensor '{name}' is missing its weight_scale.")
            if weight_scale_2.numel() != 1:
                raise ValueError(f"ModelOpt NVFP4 tensor '{name}' weight_scale_2 must be a scalar.")
            if weight_scale.dtype == torch.uint8:
                weight_scale = weight_scale.view(torch.float8_e4m3fn)
            elif weight_scale.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"ModelOpt NVFP4 tensor '{name}' weight_scale must be float8_e4m3fn or uint8, "
                    f"got {weight_scale.dtype}."
                )
            low = weight.to(torch.uint8) & 0x0F
            high = weight.to(torch.uint8) >> 4
            codes = torch.stack((low, high), dim=-1).reshape(weight.shape[0], -1).long()
            magnitudes = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                dtype=torch.float32,
            )[codes & 0x7]
            values = torch.where((codes & 0x8) > 0, -magnitudes, magnitudes)
            block_scales = weight_scale.float()
            if (
                block_scales.ndim != 2
                or block_scales.shape[0] != values.shape[0]
                or block_scales.shape[1] == 0
                or values.shape[1] % block_scales.shape[1] != 0
            ):
                raise ValueError(
                    f"ModelOpt NVFP4 tensor '{name}' has incompatible weight/scale shapes "
                    f"{tuple(weight.shape)} and {tuple(weight_scale.shape)}."
                )
            block_scales = block_scales.repeat_interleave(values.shape[1] // block_scales.shape[1], dim=1)
            return (values * block_scales * float(weight_scale_2.float().item())).to(torch.bfloat16)
        if weight.dtype == torch.float8_e4m3fn:
            if weight_scale is None or weight_scale.numel() != 1:
                raise ValueError(f"ModelOpt FP8 tensor '{name}' must have a scalar weight_scale.")
            return (weight.float() * float(weight_scale.float().item())).to(torch.bfloat16)
        return weight

    def dequantize_state(self, state):
        result = {}
        metadata_suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")
        for name, tensor in state.items():
            if name.endswith(metadata_suffixes):
                continue
            if not name.endswith(".weight"):
                result[name] = tensor
                continue
            prefix = name.removesuffix(".weight")
            result[name] = self.dequantize_tensor(
                tensor,
                state.get(f"{prefix}.weight_scale"),
                state.get(f"{prefix}.weight_scale_2"),
                name,
            )
        return result
