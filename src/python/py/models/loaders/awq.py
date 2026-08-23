# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch

from .base import QuantizedModel, QuantizedTensorModule


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