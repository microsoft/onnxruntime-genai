# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------

import glob
import os
from types import SimpleNamespace

from .base import TensorModule


class QwenMTPModel:
    @classmethod
    def from_pretrained(
        cls,
        quant_type,
        input_path,
        model_dir,
        layer_config,
        preserve_quantization=False,
        load_quantized_model=None,
        is_moe=True,
    ):
        if quant_type in {"modelopt", "compressed-tensors"}:
            if load_quantized_model is None:
                raise ValueError("A quantized model loader is required for ModelOpt/compressed-tensors MTP weights.")
            model = load_quantized_model(input_path)
            return cls.from_modelopt(model, layer_config, preserve_quantization, is_moe)
        return cls.from_safetensors(model_dir, layer_config, is_moe)

    @classmethod
    def from_modelopt(cls, model, layer_config, preserve_quantization, is_moe=True):
        if model.mtp is None:
            raise ValueError("The ModelOpt checkpoint has no MTP head.")
        if preserve_quantization:
            return SimpleNamespace(
                embedding=model.embedding,
                lm_head=model.lm_head,
                fc=model.mtp.fc,
                pre_fc_norm_embedding=model.mtp.pre_fc_norm_embedding,
                pre_fc_norm_hidden=model.mtp.pre_fc_norm_hidden,
                norm=model.mtp.norm,
                layers=model.mtp.layers,
            )

        mtp_state = model.dequantize_state(model.mtp.state)
        lm_head_weight = model.dequantize_tensor(
            model.lm_head.weight,
            model.lm_head.weight_scale,
            model.lm_head.weight_scale_2,
            "lm_head.weight",
        )
        return cls.from_state(mtp_state, model.embedding.weight, lm_head_weight, layer_config, is_moe)

    @classmethod
    def from_safetensors(cls, model_dir, layer_config, is_moe=True):
        import safetensors.torch as safetensors_torch  # noqa: PLC0415

        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in '{model_dir}' for MTP weight loading.")

        mtp_state = {}
        embed_weight = None
        lm_head_weight = None
        embed_keys = {"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"}
        for shard in shards:
            with safetensors_torch.safe_open(shard, framework="pt") as safetensors_file:
                for key in safetensors_file.keys():
                    if key.startswith("mtp."):
                        mtp_state[key] = safetensors_file.get_tensor(key)
                    elif key in embed_keys:
                        embed_weight = safetensors_file.get_tensor(key)
                    elif key == "lm_head.weight":
                        lm_head_weight = safetensors_file.get_tensor(key)

        if not mtp_state:
            raise ValueError(f"No 'mtp.*' weights found in '{model_dir}'; this model has no MTP head.")
        if embed_weight is None:
            raise ValueError(
                "Could not find the token embedding weight "
                "('model.embed_tokens.weight' or 'model.language_model.embed_tokens.weight') "
                "for the MTP head embedding."
            )
        if lm_head_weight is None:
            raise ValueError("Could not find 'lm_head.weight' for the MTP head LM head.")
        return cls.from_state(mtp_state, embed_weight, lm_head_weight, layer_config, is_moe)

    @staticmethod
    def from_state(mtp_state, embed_weight, lm_head_weight, layer_config, is_moe=True):
        try:
            if is_moe:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: PLC0415
                    Qwen3_5MoeDecoderLayer as DecoderLayer,
                )
            else:
                from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: PLC0415
                    Qwen3_5DecoderLayer as DecoderLayer,
                )
        except ImportError as exc:
            model_name = "qwen3_5_moe" if is_moe else "qwen3_5"
            raise ImportError(
                f"Building the Qwen3.6 MTP head requires the '{model_name}' modeling code in transformers."
            ) from exc

        mtp_layer = DecoderLayer(layer_config, layer_idx=0)
        layer_state = {
            key[len("mtp.layers.0.") :]: value for key, value in mtp_state.items() if key.startswith("mtp.layers.0.")
        }
        missing, unexpected = mtp_layer.load_state_dict(layer_state, strict=False)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError("Invalid MTP decoder-layer weights: " + ", ".join(details))
        mtp_layer.eval()

        return SimpleNamespace(
            embedding=TensorModule(embed_weight),
            lm_head=TensorModule(lm_head_weight),
            fc=TensorModule(mtp_state["mtp.fc.weight"]),
            pre_fc_norm_embedding=TensorModule(mtp_state["mtp.pre_fc_norm_embedding.weight"]),
            pre_fc_norm_hidden=TensorModule(mtp_state["mtp.pre_fc_norm_hidden.weight"]),
            norm=TensorModule(mtp_state["mtp.norm.weight"]),
            layers=[mtp_layer],
        )
