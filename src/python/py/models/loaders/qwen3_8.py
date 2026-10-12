import json
import math
import os
from types import SimpleNamespace

import torch

from .base import QuantizedDecoderLayer, TensorModule
from .modelopt import ModeloptModel


class Qwen38ModeloptModel(ModeloptModel):
    """Load native Qwen3.8 NVFP4 or official block-FP8 tensors without HF quantizers."""

    def __init__(self, quant_type, input_path, **kwargs):
        with open(os.path.join(input_path, "config.json")) as config_file:
            config = json.load(config_file)
        self.text_config = config["text_config"]
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

    def make_linear_module(self, base, module=None):
        if self.quant_type == "fp8" and self.get_tensor(f"{base}.weight_global_scale") is not None:
            raise ValueError(f"Native Qwen3.8 FP8 tensor '{base}' must not require scale inversion.")
        return super().make_linear_module(base, module)

    def make_dense_linear_module(self, base):
        if self.quant_type == "fp8":
            return self.make_linear_module(base)
        return super().make_dense_linear_module(base)

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

    def make_ple(self, prefix):
        ple = self.make_namespace(prefix)
        embedding = ple.ple_embedding
        embedding.eos_token_id = self.text_config["eos_token_id"]
        table = embedding.ngram_embedding
        if not hasattr(table, "weight"):
            shard_names = [name for name in vars(table) if name.startswith("shard_")]
            shard_names.sort(key=lambda name: int(name.removeprefix("shard_")))
            expected = self.text_config.get("split_ngram_parts", len(shard_names))
            if shard_names != [f"shard_{index}" for index in range(expected)] or not shard_names:
                raise ValueError(f"Incomplete Qwen3.8 PLE embedding shards at '{prefix}'.")
            table.weight = torch.cat([getattr(table, name).weight for name in shard_names], dim=0)
            for name in shard_names:
                delattr(table, name)
        if table.weight.dtype == torch.float8_e4m3fn:
            self.validate_positive_scalar(getattr(table, "weight_scale", None), f"{prefix}.weight_scale")
        return ple

    def make_mtp(self):
        return None

    def load_mtp(self):
        try:
            mtp = self.make_namespace("mtp")
            if not hasattr(mtp, "layers"):
                raise ValueError("The Qwen3.8 checkpoint has no MTP head.")
            layers = []
            for layer_id in range(self.text_config.get("mtp_num_hidden_layers", 0)):
                source = getattr(mtp.layers, str(layer_id))
                layer = QuantizedDecoderLayer(layer_id)
                layer.__dict__.update(vars(source))
                for expert_id in range(self.num_experts):
                    expert = getattr(layer.mlp.experts, str(expert_id))
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        self.validate_fp8_blocks(getattr(expert, name))
                layer.mlp.experts.quant_type = "fp8_block"
                layers.append(layer)
            mtp.layers = layers
            mtp.embedding = self.embedding
            mtp.lm_head = self.lm_head
            return mtp
        finally:
            self.close()

    @staticmethod
    def validate_fp8_blocks(projection):
        weight = projection.weight
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Qwen3.8 experts require native FP8 E4M3 weights.")
        scale = getattr(projection, "weight_scale_inv", None)
        expected = tuple(math.ceil(size / 128) for size in weight.shape)
        if scale is None or tuple(scale.shape) != expected:
            raise ValueError(f"Qwen3.8 FP8 scale must have shape {expected}.")
        if not torch.isfinite(scale.float()).all() or (scale <= 0).any():
            raise ValueError("Qwen3.8 FP8 scales must be finite and positive.")

    def load_visual(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig  # noqa: PLC0415
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: PLC0415
            Qwen4ExpVisionModel,
            Qwen4ExpVisionRotaryEmbedding,
        )

        try:
            config = Qwen4ExpVisionConfig(**self.vision_config)
            with torch.device("meta"):
                visual = Qwen4ExpVisionModel(config)
            prefix = "model.visual."
            state = {
                name[len(prefix) :]: self.get_tensor(name) for name in self.tensor_names() if name.startswith(prefix)
            }
            visual.load_state_dict(state, strict=True, assign=True)
            visual.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(config)
            return visual
        finally:
            self.close()

    def modules(self):
        return [self.embedding, *self.layers, self.lm_head]
