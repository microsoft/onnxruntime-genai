# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
from .base import Model
import onnx_ir as ir
import torch

class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

class Qwen25VLTextModel(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # We must extract the text_config for the text model's parameters
        text_config_dict = config.text_config.to_dict()
        
        # Update the main config with text-specific parameters
        # The base.Model class reads from the top-level config object
        config.hidden_size = text_config_dict["hidden_size"]
        config.intermediate_size = text_config_dict["intermediate_size"]
        config.num_attention_heads = text_config_dict["num_attention_heads"]
        config.num_hidden_layers = text_config_dict["num_hidden_layers"]
        config.num_key_value_heads = text_config_dict["num_key_value_heads"]
        config.rms_norm_eps = text_config_dict["rms_norm_eps"]
        config.sliding_window = text_config_dict["sliding_window"]
        config.rope_scaling = text_config_dict["rope_scaling"]
        # Need this for attention_scaling calculation
        if "original_max_position_embeddings" in text_config_dict:
            config.original_max_position_embeddings = text_config_dict["original_max_position_embeddings"]

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The HF model (Qwen2RMSNorm) *always* computes LayerNorm in float32.
        # By inheriting from `base.Model`, all `layernorm_attrs["cast"]` flags
        # are `False`. This causes two problems:
        # 1. Parity Error (FP32 model): The 47% mismatch you saw.
        # 2. Type Mismatch Error (BF16 model): The `(float)` vs `(bfloat16)` error.
        #
        # SOLUTION: Manually set all `cast` flags to `True`. This forces the
        # builder to cast bf16 inputs -> fp32, compute LN, and cast fp32
        # outputs -> bf16, matching the HF model and fixing both errors.
        #
        print("Forcing LayerNorm computation to float32 (and enabling all casts) for Qwen2.5-VL parity.")
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True
        #
        # Qwen2's RoPE *always* computes in float32.
        # We must replicate this behavior.
        print("Forcing RoPE computation to float32 for Qwen2.5-VL parity.")
        if "rope_cast" not in self.attention_attrs:
            self.attention_attrs["rope_cast"] = {}
        self.attention_attrs["rope_cast"]["use_fp32"] = True
        
        # The base.Model.make_outputs_init() *always* casts logits to float32
        # if the io_dtype is bfloat16. This is to improve accuracy in general.
        #
        # PROBLEM: The HF model (Qwen2_5_VL) *does not* do this. It computes
        # the lm_head MatMul in bfloat16 and returns bfloat16 logits.
        # This causes the parity test (which compares bf16 vs fp32) to fail.
        #
        # SOLUTION: We must override the base model's decision and set the
        # output logits type to match the io_dtype (bfloat16).
        #
        self.allow_bf16_logits = os.getenv("allow_bf16_logits") in ["1", "true", "True"]
        if self.allow_bf16_logits and self.io_dtype == ir.DataType.BFLOAT16:
            print("Fixing output logits precision. Setting output_types['logits'] to BFLOAT16 to match HF model.")
            self.output_types["logits"] = ir.DataType.BFLOAT16
        
        # Manually get the attention_scaling from the rope_config
        # This replicates the logic from transformers.models.rope_utils._config_to_init_values
        rope_type = "default"
        if config.rope_scaling and "type" in config.rope_scaling:
            # The config re-maps 'mrope' to 'default'
            if config.rope_scaling["type"] != "mrope":
                rope_type = config.rope_scaling["type"]
        
        if rope_type == "yarn":
            factor = config.rope_scaling.get("factor", 1.0)
            self.rope_attrs["attention_scaling"] = config.rope_scaling.get("attention_factor", (0.1 * torch.log(torch.tensor(factor)) + 1.0).item())
        elif rope_type == "longrope":
            factor = config.rope_scaling.get("factor", 1.0)
            orig_max_pos = config.original_max_position_embeddings
            self.rope_attrs["attention_scaling"] = config.rope_scaling.get("attention_factor", torch.sqrt(1 + torch.log(torch.tensor(factor)) / torch.log(torch.tensor(orig_max_pos))).item())
        else:
            self.rope_attrs["attention_scaling"] = 1.0
        
        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False
        
        # Your inheritance change fixed this, but this check is harmless and safe.
        if "position_ids" not in self.input_names:
            print("Re-adding 'position_ids' to self.input_names.")
            if "attention_mask" in self.input_names:
                idx = self.input_names.index("attention_mask")
                self.input_names.insert(idx + 1, "position_ids")
            else:
                self.input_names.append("position_ids")
        
        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")
        
        # The HF logic is `mrope_section * 2`, not `[s * 2 for s in mrope_section]`.
        # This results in [16, 24, 24, 16, 24, 24]
        self.mrope_splits = self.mrope_sections * 2
        
        if sum(self.mrope_splits) != self.head_size:
            # The sum (128) should now correctly match self.head_size (128)
            raise ValueError(f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) does not match head size ({self.head_size})")

        # Force GroupQueryAttention for fp32 cuda,
        # as base.py's make_attention_init doesn't include this combo.
        if self.ep == "cuda" and self.io_dtype == ir.DataType.FLOAT:
            self.attention_attrs["op_type"] = "GroupQueryAttention"
            print("Forcing GroupQueryAttention (GQA) for FP32 CUDA.")
        
        if self.attention_attrs["op_type"] != "GroupQueryAttention":
            raise ValueError(f"Qwen2.5-VL requires GroupQueryAttention, but op_type is {self.attention_attrs['op_type']}. This may be due to an unsupported EP/precision combo.")

        # Create and save the inv_freq tensor
        self.make_inv_freq_tensor()

    def make_inv_freq_tensor(self):
        """
        Calculates and saves the `inv_freq` tensor as an initializer.
        This is copied from base.py:make_rotary_embedding_caches_from_scratch
        """
        dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        inv_freq = 1.0 / (self.rope_attrs["rescale_factors"] * (self.rope_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)))
        
        # The HF model expects H/2, not R/2
        if dim != self.head_size:
            print(f"Warning: partial_rotary_factor ({self.rope_attrs['partial_rotary_factor']}) is not 1. This might be unsupported.")
            inv_freq = inv_freq[:(self.head_size // 2)]
        
        self.make_initializer(inv_freq, "model.inv_freq", to=ir.DataType.FLOAT)
        print("Created and saved 'model.inv_freq' initializer.")


    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
        
        # Call the base Model's make_inputs_and_outputs (skipping MistralModel's)
        super().make_inputs_and_outputs()

    def make_dynamic_rope_caches(self, layer_id, basename):
        """
        Re-implements Qwen2_5_VLRotaryEmbedding.forward using ONNX ops.
        Takes 3D position_ids and inv_freq and dynamically creates
        the cos/sin caches.
        """
        pos_ids_name = "position_ids" 
        inv_freq_name = "model.inv_freq"
        head_dim_half = self.head_size // 2

        # Get Batch Size from position_ids.shape[1]
        shape_pos_ids_name = f"{basename}/shape_pos_ids"
        shape_pos_ids_output = f"{shape_pos_ids_name}/output_0"
        self.make_shape(shape_pos_ids_name, pos_ids_name, [3])

        gather_batch_size_name = f"{basename}/gather_batch_size"
        gather_batch_size_output = f"{gather_batch_size_name}/output_0"
        self.make_gather(gather_batch_size_name, [shape_pos_ids_output, "/model/constants/INT64/[1]"], ir.DataType.INT64, [1], axis=0)
        
        # Expand inv_freq: [H/2] -> [1, 1, H/2, 1]
        unsqueeze_1_name = f"{basename}/inv_freq_unsqueeze_1"
        unsqueeze_1_output = f"{unsqueeze_1_name}/output_0"
        self.make_unsqueeze(unsqueeze_1_name, [inv_freq_name, "/model/constants/INT64/[0, 1, 3]"], ir.DataType.FLOAT, [1, 1, head_dim_half, 1])
        
        # Create target shape for Expand: [3, B, H/2, 1]
        concat_expand_shape_name = f"{basename}/concat_expand_shape"
        concat_expand_shape_output = f"{concat_expand_shape_name}/output_0"
        self.make_concat(
            concat_expand_shape_name,
            ["/model/constants/INT64/[3]", gather_batch_size_output, f"/model/constants/INT64/[{head_dim_half}, 1]"],
            ir.DataType.INT64,
            [4],
            axis=0
        )
        
        expand_name = f"{basename}/inv_freq_expand"
        expand_output = f"{expand_name}/output_0"
        self.make_expand(expand_name, [unsqueeze_1_output, concat_expand_shape_output], ir.DataType.FLOAT, [3, "batch_size", head_dim_half, 1])
        
        # Expand position_ids: [3, B, S] -> [3, B, 1, S]
        unsqueeze_2_name = f"{basename}/pos_ids_unsqueeze"
        unsqueeze_2_output = f"{unsqueeze_2_name}/output_0"
        self.make_unsqueeze(unsqueeze_2_name, [pos_ids_name, "/model/constants/INT64/[2]"], ir.DataType.INT64, [3, "batch_size", 1, "sequence_length"])
        
        # Cast position_ids to float
        cast_name = f"{basename}/pos_ids_cast"
        cast_output = f"{cast_name}/output_0"
        self.make_cast(cast_name, unsqueeze_2_output, ir.DataType.FLOAT, [3, "batch_size", 1, "sequence_length"])

        # MatMul: [3, B, H/2, 1] @ [3, B, 1, S] -> [3, B, H/2, S]
        matmul_name = f"{basename}/freqs_matmul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [expand_output, cast_output], [matmul_output], name=matmul_name)
        self.make_value(matmul_output, ir.DataType.FLOAT, [3, "batch_size", head_dim_half, "sequence_length"])

        # Transpose: [3, B, H/2, S] -> [3, B, S, H/2]
        transpose_name = f"{basename}/freqs_transpose"
        transpose_output = f"{transpose_name}/output_0"
        self.make_transpose(transpose_name, matmul_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", head_dim_half], perm=[0, 1, 3, 2])

        # Concat (freqs, freqs): [3, B, S, H/2] -> [3, B, S, H]
        concat_name = f"{basename}/emb_concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(concat_name, [transpose_output, transpose_output], ir.DataType.FLOAT, [3, "batch_size", "sequence_length", self.head_size], axis=-1)

        # Cos(emb) and Sin(emb)
        cos_name = f"{basename}/cos"
        cos_output = f"{cos_name}/output_0"
        self.make_node("Cos", [concat_output], [cos_output], name=cos_name)
        self.make_value(cos_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", self.head_size])
        
        sin_name = f"{basename}/sin"
        sin_output = f"{sin_name}/output_0"
        self.make_node("Sin", [concat_output], [sin_output], name=sin_name)
        self.make_value(sin_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", self.head_size])

        # Apply attention_scaling
        cos_final_output = cos_output
        sin_final_output = sin_output
        scale = self.rope_attrs.get("attention_scaling", 1.0) # Get from rope_attrs

        if scale != 1.0:
            scale_const_name = f"/model/constants/FLOAT/{scale}"
            
            cos_mul_name = f"{basename}/cos_mul_scale"
            cos_final_output = f"{cos_mul_name}/output_0"
            self.make_node("Mul", [cos_output, scale_const_name], [cos_final_output], name=cos_mul_name)
            self.make_value(cos_final_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", self.head_size])

            sin_mul_name = f"{basename}/sin_mul_scale"
            sin_final_output = f"{sin_mul_name}/output_0"
            self.make_node("Mul", [sin_output, scale_const_name], [sin_final_output], name=sin_mul_name)
            self.make_value(sin_final_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", self.head_size])

        return cos_final_output, sin_final_output
    
    def rotate_half(self, x_name, x_shape, basename, compute_dtype):
        """
        Builds ONNX nodes for rotate_half(x)
        x_shape is [B, N, S, H]
        """
        # Split: [B, N, S, H] -> [B, N, S, H/2], [B, N, S, H/2]
        split_name = f"{basename}/rotate_half/Split"
        split_output_0 = f"{split_name}/output_0"
        split_output_1 = f"{split_name}/output_1"
        self.make_node("Split", [x_name], [split_output_0, split_output_1], name=split_name, axis=-1, num_outputs=2)
        half_shape = x_shape[:-1] + [x_shape[-1] // 2]
        self.make_value(split_output_0, compute_dtype, half_shape)
        self.make_value(split_output_1, compute_dtype, half_shape)
        
        # Negate x2
        neg_name = f"{basename}/rotate_half/Neg"
        neg_output = f"{neg_name}/output_0"
        self.make_node("Neg", [split_output_1], [neg_output], name=neg_name)
        self.make_value(neg_output, compute_dtype, half_shape)
        
        # Concat (-x2, x1)
        concat_name = f"{basename}/rotate_half/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(concat_name, [neg_output, split_output_0], compute_dtype, x_shape, axis=-1)
        
        return concat_output

    def apply_mrope_rotation(self, layer_id, q_or_k_path, q_or_k_shape, dyn_cos, dyn_sin, num_heads, basename):
        """
        Re-implements apply_multimodal_rotary_pos_emb using ONNX ops.
        Takes Q/K tensor and the dynamically generated 3D caches
        and applies the rotation.
        """
        
        # --- Handle precision for RoPE ---
        # Check if we need to force float32 computation
        force_fp32 = self.attention_attrs.get("rope_cast", {}).get("use_fp32", False)
        
        # Set compute_dtype (precision for math) and output_dtype (final precision)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype
        output_dtype = self.io_dtype
        # --------------------------------

        # Create a Constant node for mrope_splits
        # This holds the correct splits, e.g., [16, 24, 24, 16, 24, 24]
        mrope_splits_node_name = f"{basename}/mrope_splits_node"
        mrope_splits_output_name = f"{basename}/mrope_splits"
        mrope_splits_tensor = ir.tensor(
            torch.tensor(self.mrope_splits, dtype=torch.int64),
            name=mrope_splits_output_name
        )
        self.make_node(
            "Constant",
            inputs=[],
            outputs=[mrope_splits_output_name],
            name=mrope_splits_node_name,
            value=mrope_splits_tensor
        )
        self.make_value(mrope_splits_output_name, ir.DataType.INT64, [len(self.mrope_splits)])
        
        # Split the dynamic caches [3, B, S, H] into 6 chunks on axis -1
        # Caches (dyn_cos, dyn_sin) are already in float32
        num_splits = len(self.mrope_splits)
        
        cos_split_name = f"{basename}/cos_split"
        cos_split_outputs = [f"{cos_split_name}/output_{i}" for i in range(num_splits)]
        self.make_node("Split", [dyn_cos, mrope_splits_output_name], cos_split_outputs, name=cos_split_name, axis=-1)

        sin_split_name = f"{basename}/sin_split"
        sin_split_outputs = [f"{sin_split_name}/output_{i}" for i in range(num_splits)]
        self.make_node("Split", [dyn_sin, mrope_splits_output_name], sin_split_outputs, name=sin_split_name, axis=-1)
        
        # Re-order the caches: [T, H, W, T, H, W]
        cos_reordered = []
        sin_reordered = []
        for i in range(num_splits):
            dim_chunk = self.mrope_splits[i]
            cache_dim_to_use = i % 3 # 0 for T, 1 for H, 2 for W
            
            # Gather from dim 0 of the split cache chunk
            # input is [3, B, S, H_chunk], indices is [0, 1, or 2]
            gather_cos_name = f"{basename}/cos_gather_{i}"
            gather_cos_output = f"{gather_cos_name}/output_0"
            self.make_node("Gather", [cos_split_outputs[i], f"/model/constants/INT64/[{cache_dim_to_use}]"], [gather_cos_output], name=gather_cos_name, axis=0)
            self.make_value(gather_cos_output, ir.DataType.FLOAT, [1, "batch_size", "sequence_length", dim_chunk]) # Shape [1, B, S, H_chunk]
            
            gather_sin_name = f"{basename}/sin_gather_{i}"
            gather_sin_output = f"{gather_sin_name}/output_0"
            self.make_node("Gather", [sin_split_outputs[i], f"/model/constants/INT64/[{cache_dim_to_use}]"], [gather_sin_output], name=gather_sin_name, axis=0)
            self.make_value(gather_sin_output, ir.DataType.FLOAT, [1, "batch_size", "sequence_length", dim_chunk]) # Shape [1, B, S, H_chunk]

            # FIX: Squeeze the gathered cache to [B, S, H_chunk]
            squeeze_cos_name = f"{basename}/cos_squeeze_{i}"
            squeeze_cos_output = f"{squeeze_cos_name}/output_0"
            self.make_squeeze(squeeze_cos_name, [gather_cos_output, "/model/constants/INT64/[0]"], ir.DataType.FLOAT, ["batch_size", "sequence_length", dim_chunk])

            squeeze_sin_name = f"{basename}/sin_squeeze_{i}"
            squeeze_sin_output = f"{squeeze_sin_name}/output_0"
            self.make_squeeze(squeeze_sin_name, [gather_sin_output, "/model/constants/INT64/[0]"], ir.DataType.FLOAT, ["batch_size", "sequence_length", dim_chunk])
            
            # Unsqueeze to add the NumHeads dim: [B, 1, S, H_chunk]
            unsqueeze_cos_name = f"{basename}/cos_unsqueeze_{i}"
            unsqueeze_cos_output = f"{unsqueeze_cos_name}/output_0"
            self.make_unsqueeze(unsqueeze_cos_name, [squeeze_cos_output, "/model/constants/INT64/[1]"], ir.DataType.FLOAT, ["batch_size", 1, "sequence_length", dim_chunk])
            cos_reordered.append(unsqueeze_cos_output)
            
            unsqueeze_sin_name = f"{basename}/sin_unsqueeze_{i}"
            unsqueeze_sin_output = f"{unsqueeze_sin_name}/output_0"
            self.make_unsqueeze(unsqueeze_sin_name, [squeeze_sin_output, "/model/constants/INT64/[1]"], ir.DataType.FLOAT, ["batch_size", 1, "sequence_length", dim_chunk])
            sin_reordered.append(unsqueeze_sin_output)

        # Concat re-ordered chunks back to [B, 1, S, H]
        final_cos_concat_name = f"{basename}/cos_final_concat"
        final_cos_concat_output = f"{final_cos_concat_name}/output_0"
        self.make_concat(final_cos_concat_name, cos_reordered, ir.DataType.FLOAT, ["batch_size", 1, "sequence_length", self.head_size], axis=-1)

        final_sin_concat_name = f"{basename}/sin_final_concat"
        final_sin_concat_output = f"{final_sin_concat_name}/output_0"
        self.make_concat(final_sin_concat_name, sin_reordered, ir.DataType.FLOAT, ["batch_size", 1, "sequence_length", self.head_size], axis=-1)

        # Caches (final_cos_concat_output, final_sin_concat_output) are now in float32
        
        # Reshape input Q/K: [B, S, N*H] -> [B, N, S, H]
        reshape_1_name = f"{basename}/q_or_k_reshape_1"
        reshape_1_output = f"{reshape_1_name}/output_0"
        reshape_1_target_shape_onnx = f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"
        reshape_1_target_shape_ort = ["batch_size", "sequence_length", num_heads, self.head_size]
        self.make_reshape(reshape_1_name, [q_or_k_path, reshape_1_target_shape_onnx], self.io_dtype, reshape_1_target_shape_ort)

        # Transpose Q/K: [B, S, N, H] -> [B, N, S, H]
        transpose_1_name = f"{basename}/q_or_k_transpose_1"
        transpose_1_output = f"{transpose_1_name}/output_0"
        transpose_1_target_shape = ["batch_size", num_heads, "sequence_length", self.head_size]
        self.make_transpose(transpose_1_name, reshape_1_output, self.io_dtype, transpose_1_target_shape, perm=[0, 2, 1, 3])

        # --- Start RoPE computation ---
        q_or_k_compute_input = transpose_1_output
        cos_cache_compute_input = final_cos_concat_output
        sin_cache_compute_input = final_sin_concat_output

        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast Q/K (self.io_dtype) up to float32
            q_or_k_cast_name = f"{basename}/q_or_k_cast_fp32"
            q_or_k_cast_output = f"{q_or_k_cast_name}/output_0"
            self.make_cast(q_or_k_cast_name, transpose_1_output, compute_dtype, transpose_1_target_shape)
            q_or_k_compute_input = q_or_k_cast_output
        elif not force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast Caches (float32) down to self.io_dtype
            cos_cache_cast_name = f"{basename}/cos_final_cast"
            cos_cache_cast_output = f"{cos_cache_cast_name}/output_0"
            self.make_cast(cos_cache_cast_name, final_cos_concat_output, compute_dtype, ["batch_size", 1, "sequence_length", self.head_size])
            cos_cache_compute_input = cos_cache_cast_output

            sin_cache_cast_name = f"{basename}/sin_final_cast"
            sin_cache_cast_output = f"{sin_cache_cast_name}/output_0"
            self.make_cast(sin_cache_cast_name, final_sin_concat_output, compute_dtype, ["batch_size", 1, "sequence_length", self.head_size])
            sin_cache_compute_input = sin_cache_cast_output

        # Apply rotation: (q * cos) + (rotate_half(q) * sin)
        
        # 1. (q * cos)
        mul_1_name = f"{basename}/mul_1"
        mul_1_output = f"{mul_1_name}/output_0"
        self.make_mul(mul_1_name, [q_or_k_compute_input, cos_cache_compute_input], compute_dtype, transpose_1_target_shape)
        
        # 2. rotate_half(q)
        rotated_half_q_name = self.rotate_half(q_or_k_compute_input, transpose_1_target_shape, basename, compute_dtype)
        
        # 3. (rotate_half(q) * sin)
        mul_2_name = f"{basename}/mul_2"
        mul_2_output = f"{mul_2_name}/output_0"
        self.make_mul(mul_2_name, [rotated_half_q_name, sin_cache_compute_input], compute_dtype, transpose_1_target_shape)

        # 4. (q * cos) + (rotate_half(q) * sin)
        add_name = f"{basename}/add"
        add_output = f"{add_name}/output_0"
        self.make_add(add_name, [mul_1_output, mul_2_output], compute_dtype, transpose_1_target_shape)
        
        # --- End RoPE computation ---

        add_output_final = add_output
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            # Cast result back down to self.io_dtype
            add_cast_name = f"{basename}/add_cast_output"
            add_cast_output = f"{add_cast_name}/output_0"
            self.make_cast(add_cast_name, add_output, output_dtype, transpose_1_target_shape)
            add_output_final = add_cast_output

        # Transpose back: [B, N, S, H] -> [B, S, N, H]
        transpose_2_name = f"{basename}/q_or_k_transpose_2"
        transpose_2_output = f"{transpose_2_name}/output_0"
        self.make_transpose(transpose_2_name, add_output_final, output_dtype, reshape_1_target_shape_ort, perm=[0, 2, 1, 3])
        
        # Reshape back: [B, S, N, H] -> [B, S, N*H]
        reshape_2_name = f"{basename}/q_or_k_reshape_2"
        reshape_2_output = f"{reshape_2_name}/output_0"
        self.make_reshape(reshape_2_name, [transpose_2_output, f"/model/constants/INT64/[0, 0, {num_heads * self.head_size}]"], output_dtype, q_or_k_shape)
        
        return reshape_2_output

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        
        # 1. Unpack QKV if necessary (e.g. qkv_proj)
        super().make_attention_unpacked(layer_id, attention, root_input, **kwargs)
        
        # 2. Build Q/K/V MatMul and Add nodes
        q_matmul_basename = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"
        q_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
        
        k_matmul_basename = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        k_matmul_name = self.make_matmul(attention.k_proj, k_matmul_basename, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"
        k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]
        
        v_matmul_basename = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        v_matmul_name = self.make_matmul(attention.v_proj, v_matmul_basename, root_input)
        self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # Handle biases
        q_bias_exists = attention.q_proj.bias is not None and torch.count_nonzero(attention.q_proj.bias) > 0
        k_bias_exists = attention.k_proj.bias is not None and torch.count_nonzero(attention.k_proj.bias) > 0
        v_bias_exists = attention.v_proj.bias is not None and torch.count_nonzero(attention.v_proj.bias) > 0

        if q_bias_exists:
            q_add_name = f"/model/layers.{layer_id}/attn/q_proj/Add"
            self.make_add_bias(attention.q_proj.bias, q_add_name, root_input=self.attention_attrs["q_path"])
            self.attention_attrs["q_path"] = f"{q_add_name}/output_0"
        if k_bias_exists:
            k_add_name = f"/model/layers.{layer_id}/attn/k_proj/Add"
            self.make_add_bias(attention.k_proj.bias, k_add_name, root_input=self.attention_attrs["k_path"])
            self.attention_attrs["k_path"] = f"{k_add_name}/output_0"
        if v_bias_exists:
            v_add_name = f"/model/layers.{layer_id}/attn/v_proj/Add"
            self.make_add_bias(attention.v_proj.bias, v_add_name, root_input=self.attention_attrs["v_path"])
            self.attention_attrs["v_path"] = f"{v_add_name}/output_0"

        # 3. Apply 3D RoPE (MRoPE)
        cos_dynamic, sin_dynamic = self.make_dynamic_rope_caches(layer_id, basename=f"/model/layers.{layer_id}/attn/mrope_dynamic_cache")
        
        # Apply rotation to Q
        self.attention_attrs["q_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["q_path"],
            q_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_attn_heads,
            basename=f"/model/layers.{layer_id}/attn/q_mrope"
        )
        
        # Apply rotation to K
        self.attention_attrs["k_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["k_path"],
            k_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_kv_heads,
            basename=f"/model/layers.{layer_id}/attn/k_mrope"
        )

        # 4. Call GroupQueryAttention op
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name, 
            q_path=self.attention_attrs["q_path"], 
            k_path=self.attention_attrs["k_path"], 
            v_path=self.attention_attrs["v_path"],
            past_k=past_k, 
            past_v=past_v, 
            present_k=present_k, 
            present_v=present_v,
            # Pass empty strings for fused caches since we applied RoPE manually
            cos_cache="", 
            sin_cache="", 
            **kwargs,
        )

        # 5. Build O-proj
        o_proj = 'o_proj' if hasattr(attention, 'o_proj') else 'dense'
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_weight = getattr(attention, o_proj)
        o_matmul_name = self.make_matmul(o_weight, o_matmul_basename, f"{attn_name}/output_0")

        o_bias_exists = getattr(attention, o_proj).bias is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            o_bias = getattr(attention, o_proj).bias
            self.make_add_bias(o_bias, o_add_name, root_input=f"{o_matmul_name}/output_0")
            self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"
        else:
            self.layernorm_attrs["skip_input"] = f"{o_matmul_name}/output_0"

    def make_model(self, input_path, config=None):
        
        # Make inputs and outputs to ONNX model
        self.make_inputs_and_outputs()

        # Make pre-processing nodes
        self.make_preprocessing_nodes()
        
        # Load the Hugging Face model
        from transformers import Qwen2_5_VLForConditionalGeneration
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, 
            config=config, 
            cache_dir=self.cache_dir, 
            token=self.hf_token, 
            trust_remote_code=self.hf_remote
        )
        
        # We only want to export the text model
        model = hf_model.language_model
        print(f"Isolated language_model ({model.__class__.__name__}) for ONNX export.")

        # Loop through model and map each module to ONNX/ORT ops
        self.layer_id = 0
        
        # The base.Model.make_model() loop expects modules from a standard causal LM,
        # so we replicate its logic here but point to the correct modules in the hf_model
        
        # Handle Embeddings
        if not self.exclude_embeds:
            print("Reading embedding layer")
            # The text model's embeddings are at model.embed_tokens
            self.make_embedding(model.embed_tokens.weight)
        else:
            # When excluding embeds, the input is `inputs_embeds`
            print("Skipping embedding layer, model will expect 'inputs_embeds'.")
            self.layernorm_attrs["root_input"] = "inputs_embeds"
            self.layernorm_attrs["skip_input"] = "inputs_embeds"
            
        # Handle Decoder Layers
        for layer in model.layers:
            if self.layer_id < self.num_layers:
                print(f"Reading decoder layer {self.layer_id}")
                self.make_layer(self.layer_id, layer)
                self.layer_id += 1
        
        # Handle Final Norm
        if self.layer_id == self.num_layers and hasattr(model, "norm"):
            print("Reading final norm")
            self.make_layernorm(self.layer_id, model.norm, skip=True, simple=self.layernorm_attrs["simple"], location="final_norm")
        
        # Handle LM Head
        if not self.exclude_lm_head:
            # The LM head is part of the parent Qwen2_5_VLForConditionalGeneration model
            print("Reading LM head")
            self.make_lm_head(hf_model.lm_head)
