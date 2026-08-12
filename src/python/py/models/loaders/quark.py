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

    # uint2/int2 pack 4 codes per byte (MSB-first); uint4/int4 pack 2 codes per byte.
    _DTYPE_BITS = {"uint4": 4, "int4": 4, "uint2": 2, "int2": 2}

    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        self.global_quant_config = quant_attrs["config"]["global_quant_config"]["weight"]
        global_dtype = self.global_quant_config["dtype"]
        if global_dtype not in self._DTYPE_BITS:
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
            global_bits=self._DTYPE_BITS[global_dtype],
        )
        self.repack_quantized_tensors(clear_g_idx=True)

    def set_quantized_tensor_properties(self, module):
        # bf16 outlier experts (Gemma4 uint2) carry a full-precision weight with no scales here;
        # they are re-quantized later in repack_experts and their in/out_features aren't needed.
        if module.scales is None:
            return
        module.out_features = module.scales.shape[1]
        module.in_features = module.qweight.shape[0]
        self.set_g_idx(module)

    def repack_experts(self, experts):
        """
        Unpacks weights from pre-quantized Quark experts, then repacks them into the
        format expected by the QMoE operator.
        """
        # Pre-quantized 2-bit split experts (Gemma4 factored uint2 checkpoint): consume the
        # group-wise uint2 weights/scales/float zero-points directly and re-fuse gate|up as a
        # CONCAT block. The per-expert prescale and per_expert_scale are folded in the builder's
        # make_moe_quark, so the native QMoE attrs (finalize_packed_experts) are not used here.
        first = next(iter(experts.values()))
        is_2bit = first.gate_proj is not None and first.gate_proj.bits == 2
        if is_2bit:
            self.unpack_repack_uint2_experts(experts)
            return

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

        self.repack_qmoe_weights(experts)
        self.finalize_packed_experts(experts)

    def unpack_repack_uint2_experts(self, experts):
        """Unpack + re-fuse pre-quantized 2-bit (uint2) Gemma4 split experts.

        Each expert projection is unpacked into int codes with per-group float scales/zeros
        (bf16 outlier experts are re-quantized into the same rotated+prescaled uint2 domain),
        then gate|up are concatenated and repacked into the ORT QMoE tensors.
        """
        # gate/up prescale is byte-identical across experts and gate==up; find any set one so
        # bf16 outliers (which lack their own prescale) can be transformed into the rotated domain.
        shared_prescale = None
        for expert in experts.values():
            if expert.gate_proj.input_prescale is not None:
                shared_prescale = expert.gate_proj.input_prescale
                break

        for expert in experts.values():
            self.unpack_expert_proj(expert.gate_proj, is_gate_up=True, prescale=shared_prescale)
            self.unpack_expert_proj(expert.up_proj, is_gate_up=True, prescale=shared_prescale)
            self.unpack_expert_proj(expert.gate_up_proj, is_gate_up=True, prescale=shared_prescale)
            self.unpack_expert_proj(expert.down_proj, is_gate_up=False, prescale=None)

        self.combine_and_repack_split_experts_concat(experts)

    def unpack_expert_proj(self, proj, is_gate_up, prescale):
        """Unpack one expert projection into int codes + per-group scales/zeros.

        For 2-bit Quark experts the weight is uint8 ``[in, out/4]`` packed MSB-first
        along the output axis and the zero_point is a per-group FLOAT tensor
        ``[n_groups, out]`` (dequant = (code - zero_point) * scale). We unpack to int
        codes ``[in, out]`` (matching the 4-bit `unpack_qweight_quark` output) and keep
        the float zeros as-is.

        A few experts are stored as UNQUANTIZED bf16 outliers (full-precision `[out, in]`,
        no scales/zeros/prescale). The fused QMoE op requires all experts uniform, so those
        are re-quantized here into the same rotated+prescaled (gate/up) or plain (down) uint2
        domain as their siblings. Non-2-bit checkpoints use the 4-bit AWQ path.
        """
        if proj.qweight is None:
            return
        if proj.bits == 2 and proj.qweight.dtype == torch.uint8 and proj.scales is not None:
            # Normal packed uint2: uint8 [in, out/4] MSB-first along out -> int codes [in, out].
            proj.qweight = self._unpack_uint2_msb_first(proj.qweight)
            # Float zero points are already per-group [n_groups, out]; keep them.
        elif proj.bits == 2:
            # bf16 outlier expert: re-quantize into the same uint2 domain as the siblings.
            self._requantize_float_expert(proj, is_gate_up, prescale)
        else:
            self.unpack_qzeros(proj)
            self.pack_zeros_ort_format(proj)
            self.unpack_qweight_quark(proj)
        proj.g_idx = None

    def _requantize_float_expert(self, proj, is_gate_up, prescale):
        """Re-quantize a full-precision (bf16) outlier expert projection to uint2.

        gate/up outliers are stored in the plain domain but the op applies a shared
        prescale+rotation to the MoE input, so transform into the rotated domain first:
            W_rot[in,out] = R^T @ (W_orig^T / prescale)   (matches op x_rot=(x*prescale)@R)
        down is plain (no transform). Then per-group asymmetric uint2 quant along `in`.
        """
        W = proj.qweight.to(torch.float32)      # stored as [out, in] (standard Linear)
        WT = W.t().contiguous()                 # [in, out]
        if is_gate_up:
            R = self.shared_input_rotations[WT.shape[0]].to(torch.float32)   # [in, in]
            WT = WT / prescale.to(torch.float32).unsqueeze(1)               # scale rows by 1/prescale
            WT = R.t() @ WT                                                  # [in, out]
        gs = self.global_group_size
        in_dim, out_dim = WT.shape
        ng = in_dim // gs
        w = WT.reshape(ng, gs, out_dim)
        wmin = w.min(dim=1).values              # [ng, out]
        wmax = w.max(dim=1).values
        scale = (wmax - wmin) / 3.0             # 2-bit -> 4 levels (0..3)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zp = -wmin / scale                      # float zp; dequant = (code - zp) * scale
        codes = torch.clamp(torch.round(w / scale.unsqueeze(1) + zp.unsqueeze(1)), 0, 3).to(torch.uint8)
        proj.qweight = codes.reshape(in_dim, out_dim)
        proj.scales = scale.to(torch.float16)
        proj.qzeros = zp.to(torch.float16)
        proj.group_size = gs

    def combine_and_repack_split_experts_concat(self, experts):
        """Re-fuse per-expert split uint2 gate/up/down into ORT QMoE tensors (CONCAT layout).

        Input per expert (int codes after `_unpack_uint2_msb_first`, before transpose):
          gate/up.qweight [in=hidden, out=inter]; down.qweight [in=inter, out=hidden]
          *.scales / *.qzeros are per-group FLOAT [n_groups, out].
        Output (native Gemma4 fused orientation, gate block then up block):
          fc1_weights (E, 2*inter, hidden/pack) packed uint8; fc1_scales/zp (E, 2*inter, n_groups_h)
          fc2_weights (E, hidden, inter/pack) packed uint8; fc2_scales/zp (E, hidden, n_groups_i)
        per_expert_scale is folded into fc2_scales later in the builder's make_moe_quark.
        """
        fc1_w, fc1_s, fc1_z, fc2_w, fc2_s, fc2_z = [], [], [], [], [], []
        for expert_id in sorted(experts.keys()):
            e = experts[expert_id]
            # [in, out] -> [out, in]
            gate_w = e.gate_proj.qweight.T.contiguous()
            up_w = e.up_proj.qweight.T.contiguous()
            down_w = e.down_proj.qweight.T.contiguous()
            # concat gate|up along out: [2*inter, hidden]
            fc1_codes = torch.cat([gate_w, up_w], dim=0)
            fc1_w.append(self.repack_qweight(fc1_codes, bits=2))
            fc2_w.append(self.repack_qweight(down_w, bits=2))
            # scales / zeros: [n_groups, out] -> [out, n_groups], concat gate|up along out
            fc1_s.append(torch.cat([e.gate_proj.scales.T, e.up_proj.scales.T], dim=0))
            fc1_z.append(torch.cat([e.gate_proj.qzeros.T, e.up_proj.qzeros.T], dim=0))
            fc2_s.append(e.down_proj.scales.T)
            fc2_z.append(e.down_proj.qzeros.T)

        experts.fc1_weights = torch.stack(fc1_w, dim=0)
        experts.fc1_scales = torch.stack(fc1_s, dim=0).to(torch.float16)
        experts.fc1_zero_points = torch.stack(fc1_z, dim=0).to(torch.float16)
        experts.fc2_weights = torch.stack(fc2_w, dim=0)
        experts.fc2_scales = torch.stack(fc2_s, dim=0).to(torch.float16)
        experts.fc2_zero_points = torch.stack(fc2_z, dim=0).to(torch.float16)

    @staticmethod
    def _unpack_uint2_msb_first(packed):
        """Unpack uint8 ``[rows, cols/4]`` (4x 2-bit codes per byte, MSB-first along
        columns) into int codes ``[rows, cols]``."""
        shifts = torch.tensor([6, 4, 2, 0], dtype=torch.int32, device=packed.device)
        codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0x3
        return codes.reshape(packed.shape[0], -1)

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

    def combine_gate_up_biases(self, experts):
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

    def combine_down_biases(self, experts):
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
        Repacks unpacked uint8 int-code weights into the packed uint8 tensor that
        MatMulNBits / QMoE consume. Codes are packed low-order-first along the last
        dim: 2 codes/byte for 4-bit, 4 codes/byte for 2-bit. Mirrors the packing in
        builder.py's `_symmetric_blockwise_quantize`.
        """
        if bits not in (2, 4):
            raise NotImplementedError("This repacking function supports 2-bit or 4-bit weights only.")

        quantized_flat = weights.cpu().to(torch.uint8)
        original_shape = quantized_flat.shape
        pack = 8 // bits            # codes per byte (4-bit -> 2, 2-bit -> 4)
        mask = (1 << bits) - 1      # 0xF for 4-bit, 0x3 for 2-bit

        packed_shape = list(original_shape)
        packed_shape[-1] = (original_shape[-1] + pack - 1) // pack
        packed_weight = torch.zeros(packed_shape, dtype=torch.uint8, device=quantized_flat.device)

        # Pack `pack` codes per byte, low-order code in the least-significant bits.
        n = quantized_flat.shape[-1]
        for out_idx in range(packed_shape[-1]):
            byte = torch.zeros(original_shape[:-1], dtype=torch.uint8, device=quantized_flat.device)
            for j in range(pack):
                src = out_idx * pack + j
                if src >= n:
                    break
                byte = byte | ((quantized_flat[..., src] & mask) << (bits * j))
            packed_weight[..., out_idx] = byte

        return packed_weight

    def get_layer_bits(self, layer_name):
        name = layer_name.split(".")[0]
        layer_quant_config = self.quant_attrs["config"].get("layer_quant_config", {})
        if name in layer_quant_config:
            local_dtype = layer_quant_config[name]["weight"]["dtype"]
            if local_dtype not in self._DTYPE_BITS:
                raise ValueError(f"Unexpected dtype: {local_dtype}.")
            return self._DTYPE_BITS[local_dtype]
        return self.global_bits

    def get_layer_group_size(self, layer_name):
        name = layer_name.split(".")[0]
        layer_quant_config = self.quant_attrs["config"].get("layer_quant_config", {})
        if name in layer_quant_config:
            return layer_quant_config[name]["weight"]["group_size"]
        return self.global_group_size

    def unpack(self, module):
        """
        Unpack a Quark ``QuantizedTensorModule`` into the standard int-code layout.

        Standard Quark native uint2 stores weights as uint8 ``[in, out/4]`` packed
        MSB-first along the output axis, with per-group float ``scales`` and a float
        ``zero_point`` (e.g. a constant 1.5). These are unpacked + dequantized here so
        the shared ``repack``/``pack_ort_format`` pipeline can emit MatMulNBits-ready
        tensors (same packing as the 4-bit path). Other bit-widths use the base
        (int32-packed, AWQ-reorder) path.
        """
        if module.bits == 2:
            module.qweight = self._unpack_uint2_msb_first(module.qweight)
            self.dequant_weight(module)
        else:
            super().unpack(module)

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
