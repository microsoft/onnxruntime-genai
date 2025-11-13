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

class Qwen25VLModel(QwenModel):
    """Qwen 2.5 VL Text Model with 3D MRoPE support."""
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

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        
        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False
        
        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")
        
        # The HF logic is `mrope_section * 2`, not `[s * 2 for s in mrope_section]`.
        # This results in [16, 24, 24, 16, 24, 24]
        self.mrope_splits = self.mrope_sections * 2
        
        if sum(self.mrope_splits) != self.head_size:
             # The sum (128) should now correctly match self.head_size (128)
             raise ValueError(f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) does not match head size ({self.head_size})")

        # Force GroupQueryAttention as base.py's make_attention_init doesn't include this combo.
        # The MHA op does not support GQA (different Q/K/V dims), but the GQA op does.
        self.attention_attrs["op_type"] = "GroupQueryAttention"
        print("Forcing GroupQueryAttention (GQA) for FP32 CUDA.")
        
    def make_inputs_and_outputs(self):
        # Qwen 2.5 VL uses 3D position IDs: [3, batch_size, sequence_length]
        # for temporal, height, and width dimensions.
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
        
        # Call the base Model's make_inputs_and_outputs (skipping MistralModel's)
        super(QwenModel, self).make_inputs_and_outputs()

    def apply_mrope(self, layer_id, root_input, cos_cache, sin_cache, num_heads, head_dim, basename):
        """
        Applies the 3D MRoPE logic to a Q or K tensor.
        This replicates the logic of `apply_multimodal_rotary_pos_emb`.
        """
        
        # root_input shape: [B, S, NumHeads * HeadDim]
        
        # Reshape to [B, S, NumHeads, HeadDim]
        reshape_1_name = f"{basename}/Reshape_1"
        reshape_1_output = f"{reshape_1_name}/output_0"
        self.make_reshape(reshape_1_name, 
                          [root_input, f"/model/constants/INT64/[0, 0, {num_heads}, {head_dim}]"],
                          dtype=self.io_dtype, 
                          shape=["batch_size", "sequence_length", num_heads, head_dim])

        # Transpose to [B, NumHeads, S, HeadDim] to match RotaryEmbedding op spec
        transpose_1_name = f"{basename}/Transpose_1"
        transpose_1_output = f"{transpose_1_name}/output_0"
        self.make_transpose(transpose_1_name, reshape_1_output, self.io_dtype, 
                            shape=["batch_size", num_heads, "sequence_length", head_dim], 
                            perm=[0, 2, 1, 3]) # Swap S (1) and N (2)

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
        
        # Split into 6 parts
        split_name = f"{basename}/Split"
        num_splits = len(self.mrope_splits) # This is 6
        split_outputs = [f"{split_name}/output_{i}" for i in range(num_splits)]
        split_inputs = [transpose_1_output, mrope_splits_output_name]
        self.make_node("Split", split_inputs, split_outputs, name=split_name, axis=-1)
        
        for i, shape_dim in enumerate(self.mrope_splits):
            self.make_value(split_outputs[i], self.io_dtype, ["batch_size", num_heads, "sequence_length", shape_dim])

        # Get the 3 position_ids (Temporal, Height, Width)
        # This list will have 3 elements: [pos_t, pos_h, pos_w]
        pos_id_names = []
        for i, dim_name in enumerate(["temporal", "height", "width"]):
            # Slice to get [1, B, S]
            slice_name = f"{basename}/position_ids_slice_{dim_name}"
            slice_output = f"{slice_name}/output_0"
            slice_inputs = [
                "position_ids",
                f"/model/constants/INT64/[{i}]",   # start
                f"/model/constants/INT64/[{i+1}]", # end
                "/model/constants/INT64/[0]"    # axes
            ]
            self.make_slice(slice_name, slice_inputs, dtype=ir.DataType.INT64, shape=[1, "batch_size", "sequence_length"])
            
            # Squeeze to [B, S]
            squeeze_name = f"{basename}/position_ids_squeeze_{dim_name}"
            squeeze_output = f"{squeeze_name}/output_0"
            self.make_squeeze(squeeze_name, [slice_output, "/model/constants/INT64/[0]"], dtype=ir.DataType.INT64, shape=["batch_size", "sequence_length"])
            pos_id_names.append(squeeze_output)

        # Apply RotaryEmbedding to each split using its corresponding position_id
        rotated_splits = []
        cache_offset = 0
        # Loop 6 times
        for i in range(num_splits):
            dim_chunk = self.mrope_splits[i]
            cache_dim_chunk = dim_chunk // 2
            
            # FIX: Slice the main cache to match the head_size of this chunk
            cache_start_offset = cache_offset
            cache_end_offset = cache_start_offset + cache_dim_chunk
            
            # Slice cos_cache
            cos_slice_name = f"{basename}/cos_cache_slice_{i}"
            cos_slice_output = f"{cos_slice_name}/output_0"
            cos_slice_inputs = [
                cos_cache,
                f"/model/constants/INT64/[{cache_start_offset}]", # start
                f"/model/constants/INT64/[{cache_end_offset}]",   # end
                "/model/constants/INT64/[1]"                      # axes (axis 1)
            ]
            self.make_slice(cos_slice_name, cos_slice_inputs, self.io_dtype, shape=["max_sequence_length", cache_dim_chunk])

            # Slice sin_cache
            sin_slice_name = f"{basename}/sin_cache_slice_{i}"
            sin_slice_output = f"{sin_slice_name}/output_0"
            sin_slice_inputs = [
                sin_cache,
                f"/model/constants/INT64/[{cache_start_offset}]", # start
                f"/model/constants/INT64/[{cache_end_offset}]",   # end
                "/model/constants/INT64/[1]"                      # axes (axis 1)
            ]
            self.make_slice(sin_slice_name, sin_slice_inputs, self.io_dtype, shape=["max_sequence_length", cache_dim_chunk])

            # Update offset for next iteration
            cache_offset = cache_end_offset
            
            rotary_name = f"{basename}/RotaryEmbedding_{i}"
            rotary_output = f"{rotary_name}/output_0"
            
            # Use i % 3 to cycle through the 3 position_id tensors (T, H, W)
            pos_id_to_use = pos_id_names[i % 3]
            
            rotary_inputs = [
                split_outputs[i],   # Input shape [B, N, S, H_split]
                pos_id_to_use,      # Position ID shape [B, S]
                cos_slice_output,   # Sliced cos_cache shape [MaxSeq, H_split/2]
                sin_slice_output    # Sliced sin_cache shape [MaxSeq, H_split/2]
            ]
            
            # rotary_embedding_dim=0 means apply to the full dimension of the split
            self.make_node(
                "RotaryEmbedding", 
                inputs=rotary_inputs, 
                outputs=[rotary_output], 
                name=rotary_name, 
                domain="com.microsoft",
                interleaved=self.rope_attrs["interleaved"],
                rotary_embedding_dim=0
            )
            self.make_value(rotary_output, self.io_dtype, ["batch_size", num_heads, "sequence_length", self.mrope_splits[i]])
            rotated_splits.append(rotary_output)

        # Concat back to [B, NumHeads, S, HeadDim]
        concat_name = f"{basename}/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(concat_name, rotated_splits, dtype=self.io_dtype, 
                         shape=["batch_size", num_heads, "sequence_length", head_dim], axis=-1)

        # Transpose back to [B, S, NumHeads, HeadDim]
        transpose_2_name = f"{basename}/Transpose_2"
        transpose_2_output = f"{transpose_2_name}/output_0"
        self.make_transpose(transpose_2_name, concat_output, self.io_dtype,
                            shape=["batch_size", "sequence_length", num_heads, head_dim],
                            perm=[0, 2, 1, 3]) # Swap N (1) and S (2)

        # Reshape back to [B, S, NumHeads * HeadDim]
        reshape_2_name = f"{basename}/Reshape_2"
        reshape_2_output = f"{reshape_2_name}/output_0"
        self.make_reshape(reshape_2_name, 
                          [transpose_2_output, f"/model/constants/INT64/[0, 0, {num_heads * head_dim}]"], # Input is now transpose_2_output
                          dtype=self.io_dtype, 
                          shape=["batch_size", "sequence_length", num_heads * head_dim])
        
        return reshape_2_output

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """
        Overrides the base make_attention to insert custom 3D RoPE logic.
        """
        
        # 1. Unpack QKV if necessary (e.g. qkv_proj)
        # This is handled by Model's make_attention, but we can call it directly.
        super().make_attention_unpacked(layer_id, attention, root_input, **kwargs)
        
        # 2. Build Q/K/V MatMul and Add nodes
        q_matmul_basename = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"
        
        k_matmul_basename = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        k_matmul_name = self.make_matmul(attention.k_proj, k_matmul_basename, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"
        
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
        # We don't need position_ids from kwargs, as we use the global 3D position_ids
        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        
        self.attention_attrs["q_path"] = self.apply_mrope(
            layer_id,
            self.attention_attrs["q_path"],
            cos_cache_name,
            sin_cache_name,
            self.num_attn_heads,
            self.head_size,
            basename=f"/model/layers.{layer_id}/attn/q_mrope"
        )
        
        self.attention_attrs["k_path"] = self.apply_mrope(
            layer_id,
            self.attention_attrs["k_path"],
            cos_cache_name,
            sin_cache_name,
            self.num_kv_heads,
            self.head_size,
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
        # Override make_model to correctly load the language_model part
        
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
        # so we iterate through the `model` (which is the Qwen2_5_VLTextModel)
        
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
        
        del model
        del hf_model