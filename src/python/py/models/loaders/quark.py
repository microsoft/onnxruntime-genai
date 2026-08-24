# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import torch

from .base import QuantizedModel


class QuarkModel(QuantizedModel):
    weight_name_replacements = (
        (".weight_quantizer.scale", ".weight_scale"),
        (".weight_quantizer.zero_point", ".weight_zero_point"),
    )

    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        self.global_quant_config = quant_attrs["config"]["global_quant_config"]["weight"]
        global_dtype = self.global_quant_config["dtype"]
        if global_dtype not in {"uint4", "int4"}:
            raise ValueError(f"Unexpected dtype: {global_dtype}.")
        super().__init__(
            quant_type,
            input_path,
            quant_attrs,
            q_size,
            kv_size,
            intermediate_size,
            num_layers,
            global_group_size=self.global_quant_config["group_size"],
            global_bits=4,
        )
        self.repack_quantized_tensors(clear_g_idx=True)

    def set_quantized_tensor_properties(self, module):
        module.out_features = module.scales.shape[1]
        module.in_features = module.qweight.shape[0]
        self.set_g_idx(module)

    def repack_experts(self, experts):
        """
        Unpacks weights from pre-quantized Quark experts
        """
        for expert in experts.values():
            # Process gate_proj
            if expert.gate_proj.qweight is not None:
                self.unpack_qzeros(expert.gate_proj)
                self.pack_zeros_ort_format(expert.gate_proj)
                self.unpack_qweight_quark(expert.gate_proj)
                expert.gate_proj.g_idx = None

            # Process up_proj
            if expert.up_proj.qweight is not None:
                self.unpack_qzeros(expert.up_proj)
                self.pack_zeros_ort_format(expert.up_proj)
                self.unpack_qweight_quark(expert.up_proj)
                expert.up_proj.g_idx = None

            # Process fused gate_up_proj
            if expert.gate_up_proj.qweight is not None:
                self.unpack_qzeros(expert.gate_up_proj)
                self.pack_zeros_ort_format(expert.gate_up_proj)
                self.unpack_qweight_quark(expert.gate_up_proj)
                expert.gate_up_proj.g_idx = None

            # Process down_proj
            if expert.down_proj.qweight is not None:
                self.unpack_qzeros(expert.down_proj)
                self.pack_zeros_ort_format(expert.down_proj)
                self.unpack_qweight_quark(expert.down_proj)
                expert.down_proj.g_idx = None

        """
        Repacks weights from pre-quantized Quark experts
        into the format expected by the QMoE operator.
        """
        self.repack_qmoe_weights(experts)
        self.finalize_packed_experts(experts)

    def finalize_packed_experts(self, experts):
        first_expert = experts[min(experts.keys())]
        first_projection = (
            first_expert.gate_up_proj
            if first_expert.gate_up_proj.qweight is not None
            else first_expert.gate_proj
        )
        experts.quant_type = "int"
        experts.block_size = first_projection.group_size
        experts.gate_up_qweight = experts.fc1_weights
        experts.gate_up_scales = experts.fc1_scales
        experts.gate_up_zero_points = experts.fc1_zero_points
        experts.down_qweight = experts.fc2_weights
        experts.down_scales = experts.fc2_scales
        experts.down_zero_points = experts.fc2_zero_points
        experts.gate_up_bias = self.combine_gate_up_biases(experts)
        experts.down_bias = self.combine_down_biases(experts)

    @staticmethod
    def combine_gate_up_biases(experts):
        combined_biases = []
        for expert_id in sorted(experts.keys()):
            expert = experts[expert_id]
            if expert.gate_up_proj.qweight is not None:
                bias = expert.gate_up_proj.bias
                if bias is None:
                    bias = torch.zeros(expert.gate_up_proj.qweight.shape[0])
            else:
                gate_bias = expert.gate_proj.bias
                if gate_bias is None:
                    gate_bias = torch.zeros(expert.gate_proj.qweight.shape[0])
                up_bias = expert.up_proj.bias
                if up_bias is None:
                    up_bias = torch.zeros(expert.up_proj.qweight.shape[0])
                bias = torch.empty(gate_bias.shape[0] + up_bias.shape[0], dtype=gate_bias.dtype)
                bias[::2] = gate_bias
                bias[1::2] = up_bias
            combined_biases.append(bias)
        return torch.stack(combined_biases)

    @staticmethod
    def combine_down_biases(experts):
        combined_biases = []
        for expert_id in sorted(experts.keys()):
            down_proj = experts[expert_id].down_proj
            bias = down_proj.bias
            if bias is None:
                bias = torch.zeros(down_proj.qweight.shape[0])
            combined_biases.append(bias)
        return torch.stack(combined_biases)

    def repack_qmoe_weights(self, experts):
        """
        Create quantized MoE weights from pre-quantized Quark experts.
        For gate_up projection, it interleaves gate_proj and up_proj tensors
        where even indices are for gate and odd indices are for up.
        """
        has_split_gate_up = all(
            expert.gate_proj is not None
            and expert.gate_proj.qweight is not None
            and expert.up_proj is not None
            and expert.up_proj.qweight is not None
            for expert in experts.values()
        )

        if has_split_gate_up:
            self.combine_and_repack_gate_up(experts)
            self.repack_projections(experts, ["down_proj"])
        else:
            self.repack_projections(experts, ["gate_up_proj", "down_proj"])

    def repack_projections(self, experts, projection_types):
        for proj_type in projection_types:
            qweight_list = []
            scales_list = []
            zero_points_list = []

            for expert_id in sorted(experts.keys()):
                expert = experts[expert_id]
                # Handle single projections like down_proj (or any other case)
                combined_qweight_parts = []
                combined_scales_parts = []
                combined_zp_parts = []

                proj_module = getattr(expert, proj_type)
                weights = proj_module.qweight
                weights = self.repack_qweight(weights, bits=proj_module.bits)
                combined_qweight_parts.append(weights)
                combined_scales_parts.append(proj_module.scales.T)
                combined_zp_parts.append(proj_module.qzeros)

                qweight_list.append(torch.cat(combined_qweight_parts, dim=0))
                scales_list.append(torch.cat(combined_scales_parts, dim=0))
                zero_points_list.append(torch.cat(combined_zp_parts, dim=0))

            # Stack all experts' weights and scales
            qweight = torch.stack(qweight_list, dim=0)
            scales = torch.stack(scales_list, dim=0)
            zero_points = torch.stack(zero_points_list, dim=0)

            if proj_type == "down_proj":
                experts.fc2_weights, experts.fc2_scales, experts.fc2_zero_points = (
                    qweight,
                    scales.to(torch.float16),
                    zero_points,
                )
            else:
                experts.fc1_weights, experts.fc1_scales, experts.fc1_zero_points = (
                    qweight,
                    scales.to(torch.float16),
                    zero_points,
                )

    def combine_and_repack_gate_up(self, experts):
        gate_experts_weights = []
        up_experts_weights = []
        gate_expert_scales = []
        up_expert_scales = []
        gate_expert_zero_points = []
        up_expert_zero_points = []

        # Collect weights and scales for all experts
        for expert_id in sorted(experts.keys()):
            expert = experts[expert_id]
            gate_proj = expert.gate_proj
            up_proj = expert.up_proj

            # qweight: [inter, hidden], scales: [inter, hidden // block_size]
            gate_experts_weights.append(gate_proj.qweight)
            up_experts_weights.append(up_proj.qweight)
            gate_expert_scales.append(gate_proj.scales)
            up_expert_scales.append(up_proj.scales)
            gate_expert_zero_points.append(gate_proj.qzeros)
            up_expert_zero_points.append(up_proj.qzeros)

        # Stack experts: [experts, inter, hidden]
        gate_weights = torch.stack(gate_experts_weights, axis=0)
        up_weights = torch.stack(up_experts_weights, axis=0)

        # Concatenate along last dim, then reshape to [experts, inter*2, hidden]
        fc1 = torch.concat([gate_weights, up_weights], axis=-1).view(
            up_weights.shape[0], up_weights.shape[1] * 2, up_weights.shape[2]
        )

        packed_weights = [
            self.repack_qweight(fc1[expert_id], bits=experts[expert_id].gate_proj.bits)
            for expert_id in sorted(experts.keys())
        ]
        # Stack into a 3D tensor: [num_experts, inter_size * 2, hidden_size // pack_size]
        final_fc1 = torch.stack(packed_weights, dim=0)

        # Stack scales: [experts, inter, hidden // block_size]
        gate_scales = torch.stack(gate_expert_scales, axis=0).transpose(-1, -2)  # [experts, inter, hidden // 32]
        up_scales = torch.stack(up_expert_scales, axis=0).transpose(-1, -2)  # [experts, inter, hidden // 32]
        fc1_scales = torch.concat([gate_scales, up_scales], axis=-1).view(
            up_scales.shape[0], up_scales.shape[1] * 2, up_scales.shape[2]
        )

        gate_zero_points = torch.stack(gate_expert_zero_points, axis=0)
        up_zero_points = torch.stack(up_expert_zero_points, axis=0)
        fc1_zero_points = torch.concat([gate_zero_points, up_zero_points], axis=1)

        experts.fc1_weights, experts.fc1_scales, experts.fc1_zero_points = (
            final_fc1,
            fc1_scales.to(torch.float16),
            fc1_zero_points,
        )

    def repack_qweight(self, weights, bits) -> torch.Tensor:
        """
        Repacks unpacked uint8 weights (representing 4-bit values) into a packed uint8 tensor.
        This mirrors the packing logic from builder.py's _symmetric_blockwise_quantize.
        """
        if bits != 4:
            raise NotImplementedError("This repacking function is specifically for 4-bit weights.")

        quantized_flat = weights.cpu()
        original_shape = quantized_flat.shape
        quantized_uint4 = quantized_flat.to(torch.uint8)

        packed_shape = list(original_shape)
        packed_shape[-1] = (original_shape[-1] + 1) // 2
        packed_weight = torch.zeros(packed_shape, dtype=torch.uint8, device=quantized_flat.device)

        # Pack two 4-bit values per byte
        for i in range(0, quantized_uint4.shape[-1], 2):
            val1 = quantized_uint4[..., i]
            if i + 1 < quantized_uint4.shape[-1]:
                val2 = quantized_uint4[..., i + 1]
                packed_val = (val1 & 0xF) | ((val2 & 0xF) << 4)
            else:
                # Odd number of values - pack only lower 4 bits
                packed_val = val1 & 0xF
            packed_weight[..., i // 2] = packed_val

        return packed_weight

    def get_layer_bits(self, layer_name):
        name = layer_name.split(".")[0]
        if name in self.quant_attrs["config"]["layer_quant_config"]:
            layer_quant_config = self.quant_attrs["config"]["layer_quant_config"][name]["weight"]
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
        if name in self.quant_attrs["config"]["layer_quant_config"]:
            layer_quant_config = self.quant_attrs["config"]["layer_quant_config"][name]["weight"]
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

    def unpack_qweight_quark(self, module, reorder=True, dtype="uint4"):
        """
        Unpack `qweight` to standard format and reorder for OGA.
        This is based on the packing logic from Quark's Pack_4_bits with reorder=True
        and the unpacking logic from ORT GenAI's _symmetric_blockwise_quantize.
        """
        to_unpack = module.qweight

        if to_unpack.ndim > 2:
            raise ValueError("Unpack: Only supports tensors with dimensions not greater than 2.")

        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=to_unpack.device)
        org_ndim = to_unpack.ndim

        if org_ndim == 1:
            to_unpack = to_unpack.unsqueeze(0)

        if to_unpack.ndim == 2:
            unpacked = (to_unpack.unsqueeze(-1) >> shifts.view(1, 1, -1)).view(to_unpack.shape[0], -1).to(torch.int8)
            if reorder:
                order = [0, 4, 1, 5, 2, 6, 3, 7]
                order_tensor = torch.arange(
                    unpacked.shape[-1],
                    dtype=torch.int32,
                    device=unpacked.device,
                )
                order_tensor = order_tensor.view(-1, 8)
                order_tensor = order_tensor[:, order].view(-1)
                unpacked = unpacked[:, order_tensor]
        elif to_unpack.ndim == 0:
            unpacked = to_unpack

        unpacked &= 0b1111

        if dtype == "int4":
            mask = (unpacked & 0x08).bool()
            unpacked[mask] = unpacked[mask] | 0xF0

        if org_ndim == 1:
            unpacked = unpacked.squeeze(0)

        module.qweight = unpacked.T.contiguous()
