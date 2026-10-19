import json
import math
import os
from types import SimpleNamespace

import torch

from .base import TensorModule
from .engram import Qwen38EngramLoader
from .modelopt import ModeloptModel
from .mtp import Qwen38MTPLoader
from .vision import Qwen38VisionLoader


class Qwen38ModeloptModel(Qwen38EngramLoader, Qwen38MTPLoader, Qwen38VisionLoader, ModeloptModel):
    """Load native Qwen3.8 NVFP4 or official block-FP8 tensors without HF quantizers."""

    def __init__(self, quant_type, input_path, **kwargs):
        with open(os.path.join(input_path, "config.json")) as config_file:
            config = json.load(config_file)
        self.text_config = config.get("text_config", config)
        required = {"num_hidden_layers", "eos_token_id", "num_experts"}
        missing = required - self.text_config.keys()
        if missing:
            raise ValueError(f"Qwen3.8 text configuration is missing required fields: {sorted(missing)}.")
        self.vision_config = config.get("vision_config")
        if quant_type == "fp8":
            quantization = config["quantization_config"]
            if quantization.get("weight_block_size") != [128, 128]:
                raise ValueError("Native Qwen3.8 FP8 export requires weight_block_size=[128, 128].")
            if quantization.get("activation_scheme") != "dynamic":
                raise ValueError("Native Qwen3.8 FP8 export requires dynamic activation quantization.")
        super().__init__(quant_type, input_path, **kwargs)
        try:
            language_model = SimpleNamespace(
                embed_tokens=self.embedding,
                layers=self.layers,
                hyper_connection_mixer=self.make_namespace("model.language_model.hyper_connection_mixer"),
            )
            self.model = SimpleNamespace(language_model=language_model)
        finally:
            self.close()

    def validate_linear(self, module, base):
        if self.quant_type != "fp8":
            return super().validate_linear(module, base)
        if module.weight.dtype == torch.float8_e4m3fn:
            module.weight_scale_inv = self.get_tensor(f"{base}.weight_scale_inv")
            self.validate_fp8_blocks(module)
            module.quant_type = "fp8_block"
            module.can_reuse_as_embedding = False
        elif module.weight_scale is not None or module.weight_scale_2 is not None:
            raise ValueError(f"Unexpected quantization metadata for native Qwen3.8 FP8 tensor '{base}'.")
        return None

    def make_linear_module(self, base, module=None):
        if self.quant_type == "fp8" and self.get_tensor(f"{base}.weight_global_scale") is not None:
            raise ValueError(f"Native Qwen3.8 FP8 tensor '{base}' must not require scale inversion.")
        return super().make_linear_module(base, module)

    def make_dense_linear_module(self, base):
        if self.quant_type == "fp8":
            return self.make_linear_module(base)
        return super().make_dense_linear_module(base)

    def make_gate_module(self, base, dense=False):
        module = super().make_gate_module(base, dense)
        module.exclude_from_quantization = False
        return module

    def dequantize_tensor(self, weight, weight_scale, weight_scale_2, name):
        if self.quant_type == "fp8":
            raise ValueError(f"Native Qwen3.8 FP8 export must not dequantize '{name}'.")
        return super().dequantize_tensor(weight, weight_scale, weight_scale_2, name)

    def prepare_qmoe_experts(self, experts):
        if self.quant_type != "fp8":
            return super().prepare_qmoe_experts(experts)
        prepared = SimpleNamespace(quant_type="fp8_block")
        for expert_id, expert in enumerate(experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                self.validate_fp8_blocks(getattr(expert, name))
            setattr(prepared, str(expert_id), expert)
        return prepared

    def tensor_names(self):
        if self.weight_map is not None:
            return self.weight_map
        self.get_tensor("lm_head.weight")
        return self.handle_keys[self.single_file]

    def make_namespace(self, prefix):
        root = SimpleNamespace()
        for name in self.tensor_names():
            if not name.startswith(f"{prefix}."):
                continue
            parts = name[len(prefix) + 1 :].split(".")
            owner = root
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    setattr(owner, part, SimpleNamespace())
                owner = getattr(owner, part)
            setattr(owner, parts[-1], self.get_tensor(name))

        def convert(module):
            for name, child in vars(module).copy().items():
                if isinstance(child, SimpleNamespace):
                    setattr(module, name, convert(child))
            if not hasattr(module, "weight"):
                return module
            result = TensorModule(module.weight, getattr(module, "bias", None))
            result.__dict__.update(vars(module))
            if result.weight.ndim == 2:
                result.out_features = result.weight.shape[0]
                result.in_features = result.weight.shape[1]
            return result

        return convert(root)

    def make_layer(self, layer_id, prefix=None):
        prefix = prefix or f"model.language_model.layers.{layer_id}"
        layer = super().make_layer(layer_id, prefix)
        layer.attn_hyper_connection = self.make_namespace(f"{prefix}.attn_hyper_connection")
        layer.mlp_hyper_connection = self.make_namespace(f"{prefix}.mlp_hyper_connection")
        if layer.self_attn is not None:
            layer.self_attn.indexer = self.make_namespace(f"{prefix}.self_attn.indexer")
        if layer_id + 1 in self.text_config.get("ple_layer_ids", []):
            layer.ple = self.make_ple(f"{prefix}.ple")
        return layer

    def make_mtp(self):
        return None

    def validate_fp8_blocks(self, projection):
        weight = projection.weight
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Qwen3.8 experts require native FP8 E4M3 weights.")
        scale = getattr(projection, "weight_scale_inv", None)
        expected = tuple(math.ceil(size / 128) for size in weight.shape)
        if scale is None or tuple(scale.shape) != expected:
            raise ValueError(f"Qwen3.8 FP8 scale must have shape {expected}.")
        if not torch.isfinite(scale.float()).all() or (scale <= 0).any():
            raise ValueError("Qwen3.8 FP8 scales must be finite and positive.")

    def modules(self):
        return [self.embedding, *self.layers, self.lm_head]
