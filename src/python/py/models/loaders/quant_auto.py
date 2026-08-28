# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import re

import torch
from safetensors.torch import load_file

from .base import QuantizedModel, QuantizedTensorModule


class QuantAutoModel(QuantizedModel):
    """
    quant_auto format (quant_method: quant_auto):
    - weight: (out_features, in_features) float16 containing integer values (no bitwise packing)
    - scales: (out_features * num_groups, 1) float16
    - zeros:  (out_features * num_groups, 1) float16 (renamed to qzeros by normalize_weight_name)
    - bits and group_size are not in config; inferred from tensor shapes
    """

    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        # quant_auto configs omit bits/group_size; supply defaults so the base class does not KeyError
        global_bits = quant_attrs["config"].get("bits", 4)
        global_group_size = quant_attrs["config"].get("group_size", -1)
        super().__init__(
            quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers,
            global_bits=global_bits, global_group_size=global_group_size,
        )

        # Dequantize embedding: base class stored raw int4 F16 values; load scales/zeros and dequant
        self.dequantize_embedding(input_path)

        for i, layer in enumerate(self.layers):
            if i >= self.num_layers:
                break
            print(f"Repacking layer {i}")
            self_attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            for _, q_tensors in self_attn.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.repack(q_tensors)
                    q_tensors.g_idx = None
            for _, q_tensors in layer.mlp.__dict__.items():
                if isinstance(q_tensors, QuantizedTensorModule) and q_tensors.qweight is not None:
                    self.repack(q_tensors)
                    q_tensors.g_idx = None

        if isinstance(self.lm_head, QuantizedTensorModule) and self.lm_head.qweight is not None:
            self.repack(self.lm_head)
            self.lm_head.g_idx = None

    def normalize_weight_name(self, name):
        """Map .zeros suffix to .qzeros so existing loading patterns match."""
        name = super().normalize_weight_name(name)
        if name is not None:
            name = re.sub(r"\.zeros$", ".qzeros", name)
        return name

    def dequantize_embedding(self, input_path):
        """Set up the tied embedding and lm_head from the QAT model's int4 tensors.

        The embedding (and lm_head) is quantized in the QAT model: `weight` holds int4
        values (as F16), with trained `scales` and asymmetric `zeros`. The base class
        skips the embedding's scales/zeros at load, so both must be read here directly.

        - Embedding Gather lookup: dequantized to fp16 (avoids raw int4 values in the
          embedding table).
        - lm_head (tied): kept as native asymmetric int4 using the model's OWN trained
          scales/zeros, so it exports as a MatMulNBits with a zero-point like every other
          linear layer. This prevents the builder's int4 pass from re-deriving a fresh
          symmetric RTN scheme (which the QAT model was never trained for and which
          suppresses the low-magnitude EOS token, causing no-EOS/repetition)."""
        for weight_file in os.listdir(input_path):
            if not weight_file.endswith(".safetensors"):
                continue
            weights = load_file(os.path.join(input_path, weight_file))
            if "model.embed_tokens.scales" not in weights:
                continue
            w  = weights["model.embed_tokens.weight"]   # (vocab, hidden) int4 values as F16
            sc = weights["model.embed_tokens.scales"]   # (vocab*ng, 1)
            zp = weights["model.embed_tokens.zeros"]    # (vocab*ng, 1)

            # Infer group size from tensor shapes rather than hard-coding
            ng = sc.numel() // w.shape[0]
            gs = w.shape[1] // ng

            # Embedding Gather uses dequantized fp16 weights
            w_dq = ((w.float().reshape(-1, gs) - zp.float()) * sc.float()).reshape(w.shape).half()
            self.embedding.weight = w_dq

            # lm_head (tied) uses the trained int4 quantization directly. Properties are
            # set here (set_properties already ran in the base __init__ before this point,
            # when lm_head was still a plain TensorModule) so the shared repack() applies.
            lm = QuantizedTensorModule()
            lm.qweight = w                 # (vocab, hidden) integer values stored as F16
            lm.scales = sc                 # (vocab*ng, 1)
            lm.qzeros = zp                 # (vocab*ng, 1)
            lm.out_features = w.shape[0]   # vocab
            lm.in_features = w.shape[1]    # hidden
            lm.bits = self.global_bits
            lm.group_size = gs
            self.lm_head = lm
            break

    def set_properties(self):
        """Derive in_features, out_features, bits, and group_size from tensor shapes.
        Weights are (out_features, in_features) F16 with 1 value per element — no packing factor."""
        def configure(proj):
            if proj.qweight is None:
                return
            proj.out_features = proj.qweight.shape[0]
            proj.in_features = proj.qweight.shape[1]
            if proj.bits is None:
                proj.bits = self.global_bits
            # Infer group_size from scales shape: scales is (out * n_groups, 1)
            n_groups = proj.scales.reshape(proj.out_features, -1).shape[1]
            proj.group_size = proj.in_features // n_groups
            self.set_g_idx(proj)

        if isinstance(self.lm_head, QuantizedTensorModule):
            configure(self.lm_head)

        for module in self.layers:
            for proj in [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
                module.self_attn.o_proj,
                module.mlp.gate_proj,
                module.mlp.up_proj,
                module.mlp.down_proj,
            ]:
                configure(proj)

    def repack(self, module):
        """Weights are already integer-valued F16; cast to int32 and repack to ORT format directly."""
        if module.qzeros is not None:
            # Normalize qzeros to (ng, out) so pack_zeros_ort_format's .T gives (out, ng)
            # and produces ORT's expected output-first packed layout.
            # Fused layers arrive as (out, ng); unfused as (out*ng, 1) — flatten then reshape.
            ng = module.qzeros.numel() // module.out_features
            module.qzeros = module.qzeros.reshape(module.out_features, ng).T.to(torch.uint8).contiguous()
        intweight = module.qweight.to(torch.int32)  # (out, in) int values stored as F16
        self.pack_ort_format(module, intweight.T)   # pack_ort_format expects (in_features, out_features)
