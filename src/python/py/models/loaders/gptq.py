# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import re

import torch

from .base import QuantizedModel, QuantizedTensorModule


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
        temp_module.qzeros = torch.bitwise_and(temp_module.qzeros, (2**temp_module.bits) - 1)

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