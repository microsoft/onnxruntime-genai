# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
A set of Python classes to unpack the quantized weights and store them in a standard format.

The goal is for `QuantModel` to unpack the quantized weights into a standard format
so that the original Hugging Face --> ONNX code can re-pack the quantized weights into
ONNX Runtime's format no matter where the quantized weights actually come from.
"""

from safetensors.torch import load_file
import torch

import json
import os
import re


# class Tensor:
#     def __init__(self):
#         self.tensor = None

#     def detach(self):
#         """
#         No-op operation since numpy() will handle it
#         """
#         return self

#     def cpu(self):
#         """
#         No-op operation since numpy() will handle it
#         """
#         return self

#     def numpy(self):
#         """
#         Convert Torch tensor to NumPy tensor
#         """
#         return self.tensor.detach().cpu().numpy()


class QuantizedTensorModule:
    def __init__(self, bits, group_size):
        self.qweight = None #Tensor()
        self.scales = None #Tensor()
        self.qzeros = None
        self.g_idx = None
        self.bias = None

        self.in_features = 0
        self.out_features = 0
        self.bits = bits
        self.group_size = group_size

    # def add_bias(self):
    #     self.bias = Tensor()

    # def add_qzeros(self):
    #     self.qzeros = Tensor()

    # def add_g_idx(self):
    #     self.g_idx = Tensor()

    def __str__(self):
        qweight = f"qweight = {self.qweight.shape}, {self.qweight}\n"
        scales = f"scales = {self.scales.shape}, {self.scales}\n"
        qzeros = "" if self.qzeros is None else f"qzeros = {self.qzeros.shape}, {self.qzeros}\n"
        g_idx = "" if self.g_idx is None else f"g_idx = {self.g_idx.shape}, {self.g_idx}\n"

        in_feats = f"in_features = {self.in_features}, "
        out_feats = f"out_features = {self.out_features}, "
        bits = f"bits = {self.bits}, "
        group_size = f"group_size = {self.group_size}, "

        return qweight + qzeros + scales + g_idx + in_feats + out_feats + bits + group_size


class TensorModule:
    def __init__(self):
        self.weight = None #Tensor()
        self.bias = None

    def add_bias(self):
        self.bias = None #Tensor()


class QuantizedAttention:
    def __init__(self, bits, group_size):
        self.q_proj = QuantizedTensorModule(bits, group_size)
        self.k_proj = QuantizedTensorModule(bits, group_size)
        self.v_proj = QuantizedTensorModule(bits, group_size)
        self.o_proj = QuantizedTensorModule(bits, group_size)
        self.rotary_emb = TensorModule()


class QuantizedMLP:
    def __init__(self, bits, group_size):
        self.gate_proj = QuantizedTensorModule(bits, group_size)
        self.up_proj = QuantizedTensorModule(bits, group_size)
        self.down_proj = QuantizedTensorModule(bits, group_size)
        self.fc1 = QuantizedTensorModule(bits, group_size)
        self.fc2 = QuantizedTensorModule(bits, group_size)


class QuantizedDecoderLayer:
    def __init__(self, layer_id, bits, group_size):
        self.layer_id = layer_id
        self.input_layernorm = TensorModule()
        self.self_attn = QuantizedAttention(bits, group_size)
        self.post_attention_layernorm = TensorModule()
        self.mlp = QuantizedMLP(bits, group_size)


class QuantizedModel:
    def __init__(self, input_path, bits, group_size):
        # self.qmodel = safetensors.torch.load_file(os.path.join(input_path, "model.safetensors"))
        self.embedding = TensorModule()
        self.final_norm = TensorModule()
        self.lm_head = TensorModule()
        self.layers = []

        layer_id = 0
        module = QuantizedDecoderLayer(layer_id, bits, group_size)

        for weight_file in os.listdir(input_path):
            if weight_file.endswith(".safetensors"):
                weights = load_file(os.path.join(input_path, weight_file))

                # Map weights to modules
                for name, tensor in weights.items():
                    if name == "model.embed_tokens.weight":
                        self.embedding.weight = tensor
                    elif name == "model.norm.weight":
                        self.final_norm.weight = tensor
                    elif name == "model.norm.bias":
                        self.final_norm.add_bias()
                        self.final_norm.bias = tensor
                    elif name == "lm_head.weight":
                        self.lm_head.weight = tensor
                    elif name == "lm_head.bias":
                        self.lm_head.add_bias()
                        self.lm_head.bias = tensor
                    else:
                        curr_layer_id = int(name.split(".")[2])
                        if curr_layer_id != layer_id:
                            # Add layer to list of modules
                            self.layers.append(module)
                            layer_id = curr_layer_id
                            module = QuantizedDecoderLayer(layer_id, bits, group_size)

                        # Map weights and biases of norm, attention, and feed-forward network
                        # Graph order is input_layernorm --> q_proj/k_proj/v_proj --> o_proj --> post_attention_layernorm --> gate_proj/up_proj --> down_proj
                        if bool(re.match(r"^model.layers\.\d+\.input_layernorm\.weight$", name)):
                            # model.layers.layer_id.input_layernorm.weight
                            module.input_layernorm.weight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.input_layernorm\.bias$", name)):
                            # model.layers.layer_id.input_layernorm.bias
                            module.input_layernorm.add_bias()
                            module.input_layernorm.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.rotary_emb\.inv_freq$", name)):
                            # model.layers.layer_id.self_attn.rotary_emb.inv_freq
                            # Skip rotary embedding weights since they can be re-calculated when looping through the model
                            continue
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.q_proj.qweight
                            module.self_attn.q_proj.out_features = tensor.shape[1]
                            module.self_attn.q_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.q_proj.scales
                            module.self_attn.q_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.q_proj.qzeros
                            # module.self_attn.q_proj.add_qzeros()
                            module.self_attn.q_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.q_proj.g_idx
                            # module.self_attn.q_proj.add_g_idx()
                            module.self_attn.q_proj.in_features = tensor.shape[0]
                            module.self_attn.q_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.q_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.q_proj.bias
                            # module.self_attn.q_proj.add_bias()
                            module.self_attn.q_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qweight
                            module.self_attn.k_proj.out_features = tensor.shape[1]
                            module.self_attn.k_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.k_proj.scales
                            module.self_attn.k_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.k_proj.qzeros
                            # module.self_attn.k_proj.add_qzeros()
                            module.self_attn.k_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.k_proj.g_idx
                            # module.self_attn.k_proj.add_g_idx()
                            module.self_attn.k_proj.in_features = tensor.shape[0]
                            module.self_attn.k_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.k_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.k_proj.bias
                            # module.self_attn.k_proj.add_bias()
                            module.self_attn.k_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qweight
                            module.self_attn.v_proj.out_features = tensor.shape[1]
                            module.self_attn.v_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.v_proj.scales
                            module.self_attn.v_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.v_proj.qzeros
                            # module.self_attn.v_proj.add_qzeros()
                            module.self_attn.v_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.v_proj.g_idx
                            # module.self_attn.v_proj.add_g_idx()
                            module.self_attn.v_proj.in_features = tensor.shape[0]
                            module.self_attn.v_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.v_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.v_proj.bias
                            # module.self_attn.v_proj.add_bias()
                            module.self_attn.v_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.o_proj\.qweight$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qweight
                            module.self_attn.o_proj.out_features = tensor.shape[1]
                            module.self_attn.o_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.o_proj\.scales$", name)):
                            # model.layers.layer_id.self_attn.o_proj.scales
                            module.self_attn.o_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.o_proj\.qzeros$", name)):
                            # model.layers.layer_id.self_attn.o_proj.qzeros
                            # module.self_attn.o_proj.add_qzeros()
                            module.self_attn.o_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.o_proj\.g_idx$", name)):
                            # model.layers.layer_id.self_attn.o_proj.g_idx
                            # module.self_attn.o_proj.add_g_idx()
                            module.self_attn.o_proj.in_features = tensor.shape[0]
                            module.self_attn.o_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.self_attn.o_proj\.bias$", name)):
                            # model.layers.layer_id.self_attn.o_proj.bias
                            # module.self_attn.o_proj.add_bias()
                            module.self_attn.o_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.weight$", name)):
                            # model.layers.layer_id.post_attention_layernorm.weight
                            module.post_attention_layernorm.weight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.post_attention_layernorm\.bias$", name)):
                            # model.layers.layer_id.post_attention_layernorm.bias
                            module.post_attention_layernorm.add_bias()
                            module.post_attention_layernorm.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.qweight$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qweight
                            module.mlp.gate_proj.out_features = tensor.shape[1]
                            module.mlp.gate_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.scales$", name)):
                            # model.layers.layer_id.mlp.gate_proj.scales
                            module.mlp.gate_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.qzeros$", name)):
                            # model.layers.layer_id.mlp.gate_proj.qzeros
                            # module.mlp.gate_proj.add_qzeros()
                            module.mlp.gate_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.gate_proj.g_idx
                            # module.mlp.gate_proj.add_g_idx()
                            module.mlp.gate_proj.in_features = tensor.shape[0]
                            module.mlp.gate_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.gate_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.gate_proj.bias
                            # module.mlp.gate_proj.add_bias()
                            module.mlp.gate_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.qweight$", name)):
                            # model.layers.layer_id.mlp.up_proj.qweight
                            module.mlp.up_proj.out_features = tensor.shape[1]
                            module.mlp.up_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.scales$", name)):
                            # model.layers.layer_id.mlp.up_proj.scales
                            module.mlp.up_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.qzeros$", name)):
                            # model.layers.layer_id.mlp.up_proj.qzeros
                            # module.mlp.up_proj.add_qzeros()
                            module.mlp.up_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.up_proj.g_idx
                            # module.mlp.up_proj.add_g_idx()
                            module.mlp.up_proj.in_features = tensor.shape[0]
                            module.mlp.up_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.up_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.up_proj.bias
                            # module.mlp.up_proj.add_bias()
                            module.mlp.up_proj.bias = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.down_proj\.qweight$", name)):
                            # model.layers.layer_id.mlp.down_proj.qweight
                            module.mlp.down_proj.out_features = tensor.shape[1]
                            module.mlp.down_proj.qweight = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.down_proj\.scales$", name)):
                            # model.layers.layer_id.mlp.down_proj.scales
                            module.mlp.down_proj.scales = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.down_proj\.qzeros$", name)):
                            # model.layers.layer_id.mlp.down_proj.qzeros
                            # module.mlp.down_proj.add_qzeros()
                            module.mlp.down_proj.qzeros = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.down_proj\.g_idx$", name)):
                            # model.layers.layer_id.mlp.down_proj.g_idx
                            # module.mlp.down_proj.add_g_idx()
                            module.mlp.down_proj.in_features = tensor.shape[0]
                            module.mlp.down_proj.g_idx = tensor
                        elif bool(re.match(r"^model.layers\.\d+\.mlp.down_proj\.bias$", name)):
                            # model.layers.layer_id.mlp.down_proj.bias
                            # module.mlp.down_proj.add_bias()
                            module.mlp.down_proj.bias = tensor
                        else:
                            raise NotImplementedError(f"{name} in your quantized model is not recognized.")
                
                # Append final layer to list of layers
                self.layers.append(module)
        
        # Set LM head weights + biases if not already set
        if self.lm_head.weight is None:
            # Embedding and LM head share same weights + biases (lm_head.weight == embedding.weight and lm_head.bias == embedding.bias)
            self.lm_head.weight = self.embedding.weight
            if self.lm_head.bias is not None:
                self.lm_head.bias = self.embedding.bias
    
        # Sort list of layers by layer id
        self.layers.sort(key=lambda m: m.layer_id)

    def modules(self):
        """
        Return list of modules in quantized model in order of appearance in the model
        """
        return [self.embedding] + self.layers + [self.final_norm, self.lm_head]

    def unpack(self, module):
        """
        Unpack `qzeros` and `qweight` to standard format
        """
        self.unpack_qzeros(module)
        self.unpack_qweight(module)
        self.dequant_weight(module)

    def repack(self, module):
        """
        Repack `scales`, `qzeros` and `qweight` to ORT format
        """
        # Cast `scales` to .half()?
        # Transpose `scale` for final result?
        intweight = self.quant_weight(module)
        # print(f"intweight = {intweight}")
        self.pack_ort_format(module, intweight)

    def unpack_qzeros(self, module):
        """
        Unpack `qzeros` to standard format
        """
        if module.qzeros is None:
            return
        expected_shape = (module.in_features // module.group_size, module.out_features)
        transpose = module.qzeros.shape[0] != expected_shape[0]
        module.qzeros = self.unpack_on_row(module.qzeros, module.bits, transpose)

    def unpack_qweight(self, module):
        """
        Unpack `qweight` to standard format
        """
        expected_shape = (module.in_features, module.qweight.shape[1])  # (in_features, out_features) instead?
        transpose = module.qweight.shape[0] != expected_shape[0]
        module.qweight = self.unpack_on_row(module.qweight, module.bits, transpose)

    def pack_qzeros(self, module):
        """
        Pack `qzeros` to quantized format
        """
        expected_shape = (module.in_features // module.group_size, module.out_features)
        transpose = module.qzeros.shape[0] != expected_shape[0]
        module.qzeros = self.pack_on_row(module.qzeros, module.bits, transpose)

    def unpack_on_row_for_2_4_8_bits(self, tensor, bits, transpose):
        """
        Perform general-purpose unpacking on 2-bit, 4-bit, or 8-bit tensor
        """
        pack_tensor = tensor.T if transpose else tensor
        wf = torch.arange(0, 32, bits, device=pack_tensor.device).unsqueeze(0).unsqueeze(0)
        out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf)
        out = out.reshape(pack_tensor.shape[0], -1)
        out = torch.bitwise_and(out, (2 ** bits) - 1)
        return out.T if transpose else out

    def unpack_on_row(self, tensor, bits, transpose):
        """
        Unpack tensor by row
        """
        if bits in {2, 4, 8}:
            return self.unpack_on_row_for_2_4_8_bits(tensor, bits, transpose)
        else:
            raise NotImplementedError(f"Unpacking for {bits}-bit quantization is not currently supported.")

    def pack_on_row_for_2_4_8_bits(self, tensor, bits, transpose):
        """
        Perform general-purpose packing on 2-bit, 4-bit, or 8-bit tensor
        """
        orig_tensor = tensor.T if transpose else tensor
        wf = torch.arange(0, bits).view(1, 1, -1)
        out = torch.bitwise_right_shift(orig_tensor.unsqueeze(-1), wf)
        out = torch.bitwise_and(out, 1)
        out = out.reshape(orig_tensor.shape[0], -1, 32)
        wf1 = torch.arange(0, 32, 1).view(1, 1, -1)
        out = torch.bitwise_left_shift(out, wf1)
        out = out.sum(dim=-1).int()
        return out.T if transpose else out

    def pack_on_row(self, tensor, bits, transpose):
        """
        Pack tensor by row
        """
        if bits in {2, 4, 8}:
            return self.pack_on_row_for_2_4_8_bits(tensor, bits, transpose)
        else:
            raise NotImplementedError(f"Packing for {bits}-bit quantization is not currently supported.")

    def dequant_weight(self, module):
        """
        De-quantize `qweight` to higher precision (float16)
        """
        # Note: `qweight` and `qzeros` have already been unpacked and stored in those variables respectively
        intweight = module.qweight
        zeros = module.qzeros
        scales = module.scales
        g_idx = module.g_idx

        # De-quantize weight to higher precision
        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        qdq_weight_T = intweight * scale_mat - scale_zeros_mat.half()
        
        # Store unpacked result in `qweight`
        module.qweight = qdq_weight_T.T

    def quant_weight(self, module):
        """
        Calculate integer weight to quantize `qweight` with
        """
        weight = module.qweight.T
        zeros = module.qzeros
        scales = module.scales
        g_idx = module.g_idx

        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round((weight + scale_zeros_mat) / scale_mat).to(torch.int)

        return intweight_T

    def pack_ort_format(self, module, intweight):
        """
        Pack `scales`, `qzeros`, and `qweight` to ORT format
        """
        if module.bits != 4:
            raise NotImplementedError(f"{modue.bits}-bit quantization in ORT is not currently supported by this tool.")
        
        intzeros_pt = module.qzeros.T if module.qzeros.dtype == module.scales.dtype else module.qzeros.T.byte()
        intweight_pt = intweight.byte()
        block_size = module.group_size

        rows, cols = intweight_pt.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(intweight_pt, (0, 0, 0, pad_len), "constant", 0)
        intzeros_pt = torch.nn.functional.pad(intzeros_pt, (0, intzeros_pt.shape[-1] & 1, 0, 0), "constant", 0)

        if module.qzeros.dtype != module.scales.dtype:
            intzeros_pt = (intzeros_pt[:, 0::2]) | (intzeros_pt[:, 1::2] << 4)
            intzeros_pt = intzeros_pt.reshape(-1)

        intweight_pt_T = intweight.T
        intweight_pt_T = (intweight_pt_T[:, 0::2]) | (intweight_pt_T[:, 1::2] << 4)
        intweight_pt_T = intweight_pt_T.reshape(cols, k_blocks, blob_size)

        scales_pt = module.scales.T.reshape(-1)

        # assert module.qweight.shape == intweight_pt_T.shape
        # assert module.qzeros.shape == intzeros_pt.shape or module.qzeros.dtype != intzeros_pt.dtype

        module.scales = scales_pt.contiguous()
        module.qweight = intweight_pt_T.contiguous().byte()
        if module.qzeros.dtype != module.scales.dtype:
            module.qzeros = intzeros_pt.contiguous().byte()
        else:
            module.qzeros = intzeros_pt.contiguous()


class AWQModel(QuantizedModel):
    def __init__(self, input_path, bits, group_size):
        # from awq import AutoAWQForCausalLM
        # self.qmodel = AutoAWQForCausalLM.from_quantized(input_path)

        # for module in self.qmodel.modules():
        #     if module.__class__.__name__ == "WQLinear_GEMM":
        #         # self.qweight = torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32)
        #         # self.qzeros = torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32)
        #         # self.scales = torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16)
        #         # self.bias = torch.zeros((out_features), dtype=torch.float16)
        #         for k, v in module.q_tensors.items():
        #             module.q_tensors[k] = v.to(torch.device("cpu"))

        #         self.unpack(module)
        #         self.repack(module)

        super().__init__(input_path, bits, group_size)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            print(f"Unpacking and repacking layer {i}")

            # Unpack and repack all `QuantizedTensorModule` classes in attention
            for name, q_tensors in layer.self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

            # Unpack and repack all `Quantized TensorModule` classes in MLP
            for name, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.unpack(q_tensors)
                    self.repack(q_tensors)

    def unpack_qweight(self, module):
        """
        Unpack `qweight` to standard format
        """
        expected_shape = (module.qweight.shape[0], module.out_features)  # (infeatures, outfeatures) instead?
        transpose = module.qweight.shape != expected_shape
        module.qweight = self.unpack_on_row(module.qweight.T, module.bits, transpose)
        module.qweight = self.reverse_reorder_tensor(module.qweight.T)

    def unpack_qzeros(self, module):
        """
        Unpack `qzeros` to standard format
        """
        super().unpack_qzeros(module)
        module.qzeros = self.reverse_reorder_tensor(module.qzeros)

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


class GPTQModel(QuantizedModel):
    def __init__(self, input_path, bits, group_size):
        # from auto_gptq import AutoGPTQForCausalLM
        # self.qmodel = AutoGPTQForCausalLM.from_quantized(input_path)

        # for module in self.qmodel.modules():
        #     if module.__class__.__name__ == "QuantLinear":
        #         # self.qweight = torch.zeros((module.infeatures // 32 * module.bits, module.outfeatures), dtype=torch.int32)
        #         # self.qzeros = torch.zeros((torch.ceil(module.infeatures / module.group_size), module.outfeatures // 32 * module.bits), dtype=torch.int32)
        #         # self.scales = torch.zeros((torch.ceil(module.infeatures / module.group_size), module.outfeatures), dtype=torch.float16)
        #         # self.g_idx = torch.tensor([i // module.group_size for i in range(module.infeatures)], dtype=torch.int32)
        #         # self.bias = torch.zeros((module.outfeatures), dtype=torch.float16)

        #         for k, v in module.q_tensors.items():
        #             module.q_tensors[k] = v.to(torch.device("cpu"))
        #         self.handle_qzeros(module)

        #         self.unpack(module)
        #         self.repack(module)

        super().__init__(input_path, bits, group_size)

        # Unpack and repack all `QuantizedTensorModule` classes in model
        for i, layer in enumerate(self.layers):
            print(f"Unpacking and repacking layer {i}")
            # Unpack and repack all `QuantizedTensorModule` classes in attention

            for name, q_tensors in layer.self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    # print(f"{name}:\n{q_tensors}")
                    self.handle_qzeros(q_tensors)
                    # print(f"Before unpacked: module_name = {name}")
                    # print(f"{q_tensors}")
                    self.unpack(q_tensors)
                    # print(f"After unpacked: module_name = {name}")
                    # print(f"{q_tensors}")
                    self.repack(q_tensors)
                    # print(f"After repacked: module_name = {name}")
                    # print(f"{q_tensors}")

                    # Set `g_idx` to None since it's not used in `MatMulNBits`
                    q_tensors.g_idx = None

            # Unpack and repack all `Quantized TensorModule` classes in MLP
            for name, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    # print(f"{name} = {q_tensors}")
                    self.handle_qzeros(q_tensors)
                    self.unpack(q_tensors)
                    self.repack(q_tensors)
                    # print(f"After repacked: module_name = {name}")
                    # print(f"{q_tensors}")

                    q_tensors.g_idx = None
    
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
        temp_module.qzeros = torch.bitwise_and(temp_module.qzeros, (2 ** temp_module.bits) - 1)

        self.pack_qzeros(temp_module)
        module.qzeros = temp_module.qzeros


# class HQQModel(QuantizedModel):
#     def __init__(self, input_path, bits, group_size):
#         # from hqq.engine.hf import HQQModelForCausalLM
#         # self.qmodel = HQQModelForCausalLM.from_quantized(input_path)

#         # for module in self.qmodel.modules():
#         #     if module.__class__.__name__ == "HQQLinear":
#         #         # self.qweight = torch.zeros((module.infeatures // 32 * module.bits, module.outfeatures), dtype=torch.int32)
#         #         # self.qzeros = torch.zeros((torch.ceil(module.infeatures / module.group_size), module.outfeatures), dtype=torch.float16)
#         #         # self.scales = torch.zeros((torch.ceil(module.infeatures / module.group_size), module.outfeatures), dtype=torch.float16)
#         #         # self.bias = torch.zeros((module.outfeatures), dtype=torch.float16)
#         #         for k, v in module.q_tensors.items():
#         #             module.q_tensors[k] = v.to(torch.device("cpu"))

#         #         self.unpack(module)
#         #         self.repack(module)

#         super().__init__(input_path, bits, group_size)

#     def unpack_qzeros(self, module):
#         """
#         Unpack `qzeros` to standard format
#         """
#         # `qzeros` are already in standard format
#         return


class QuantModel:
    @staticmethod
    def from_pretrained(input_path):
        """
        Unpack quantized weights in PyTorch models and store them in a standard format.
        Also performs any pre-processing and post-processing when unpacking the quantized weights.
        """
        # Get quantization info from `config.json`
        config = json.load(open(os.path.join(input_path, "config.json")))
        quant_type = config["quantization_config"]["quant_method"]
        bits = config["quantization_config"]["bits"]
        group_size = config["quantization_config"]["group_size"]

        if quant_type == "awq":
            model = AWQModel(input_path, bits, group_size)
        elif quant_type == "gptq":
            model = GPTQModel(input_path, bits, group_size)
        # elif quant_type == "hqq":
        #     model = HQQModel(input_path, bits, group_size)
        else:
            raise NotImplementedError(f"The {quant_type} quantized model is not currently supported.")

        return model