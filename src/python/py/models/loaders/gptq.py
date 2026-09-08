# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import re

import torch

from .base import QuantizedModel


class GPTQModel(QuantizedModel):
    override_config_key = "dynamic"

    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        self.overrides = quant_attrs["config"].get(self.override_config_key, {}) or {}
        super().__init__(quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers)
        self.repack_quantized_tensors(clear_g_idx=not quant_attrs["use_g_idx"])

    def set_quantized_tensor_properties(self, module):
        module.out_features = module.qweight.shape[1]
        module.in_features = module.g_idx.shape[0]

    def prepare_quantized_tensor(self, module):
        self.handle_qzeros(module)

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

    def get_overrides(self, layer_name):
        for pattern, overrides in self.overrides.items():
            if re.match(pattern.removeprefix("+:"), layer_name):
                return overrides
        return {}

    def get_layer_bits(self, layer_name):
        return self.get_overrides(layer_name).get("bits", self.global_bits)

    def get_layer_group_size(self, layer_name):
        return self.get_overrides(layer_name).get("group_size", self.global_group_size)
