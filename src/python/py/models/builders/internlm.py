# -------------------------------------------------------------------------
# Copyright (C)  [2026]  Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI generated content
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .base import Model
import torch.nn as nn


class InternLM2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # InternLM2 is based on Llama architecture, so use 'llama' as model_type for GenAI compatibility
        self.model_type = "LlamaForCausalLM"

    def load_weights(self, input_path):
        """
        Load the InternLM2 model and adapt attribute names to match base class expectations.
        InternLM2 uses:
        - attention_norm instead of input_layernorm
        - ffn_norm instead of post_attention_layernorm  
        - feed_forward instead of mlp
        - wqkv (combined QKV) instead of separate q_proj, k_proj, v_proj
        - wo instead of o_proj
        """
        # Load the model using the parent class method
        model = super().load_weights(input_path)
        
        # Get config from the loaded model
        config = model.config
        
        # Adapt each decoder layer to match the expected attribute names
        for layer in model.model.layers:
            # Map attention_norm to input_layernorm
            if hasattr(layer, 'attention_norm') and not hasattr(layer, 'input_layernorm'):
                layer.input_layernorm = layer.attention_norm
            
            # Map ffn_norm to post_attention_layernorm
            if hasattr(layer, 'ffn_norm') and not hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm = layer.ffn_norm
            
            # Map feed_forward to mlp
            if hasattr(layer, 'feed_forward') and not hasattr(layer, 'mlp'):
                layer.mlp = layer.feed_forward
            
            # Map attention to self_attn
            if hasattr(layer, 'attention') and not hasattr(layer, 'self_attn'):
                layer.self_attn = layer.attention
            
            # Map MLP projections (w1/w2/w3 to gate_proj/down_proj/up_proj)
            if hasattr(layer.mlp, 'w1') and not hasattr(layer.mlp, 'gate_proj'):
                layer.mlp.gate_proj = layer.mlp.w1
            if hasattr(layer.mlp, 'w2') and not hasattr(layer.mlp, 'down_proj'):
                layer.mlp.down_proj = layer.mlp.w2
            if hasattr(layer.mlp, 'w3') and not hasattr(layer.mlp, 'up_proj'):
                layer.mlp.up_proj = layer.mlp.w3
            
            # Handle the combined wqkv projection in attention
            # InternLM2 uses a grouped/interleaved layout: [Q1, Q2, ..., Qn, K, V] per KV group
            # Layout: [batch, seq, num_kv_heads, (num_q_heads_per_kv_group + 2), head_dim]
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'wqkv'):
                attn = layer.self_attn
                wqkv_weight = attn.wqkv.weight  # Shape: [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
                wqkv_bias = attn.wqkv.bias if hasattr(attn.wqkv, 'bias') and attn.wqkv.bias is not None else None
                
                # Calculate dimensions
                num_q_heads = config.num_attention_heads
                num_kv_heads = config.num_key_value_heads
                num_kv_groups = num_q_heads // num_kv_heads  # How many Q heads per KV head
                head_dim = config.hidden_size // num_q_heads
                
                q_size = num_q_heads * head_dim
                kv_size = num_kv_heads * head_dim
                
                # InternLM2's wqkv is organized as interleaved groups:
                # For each KV head group: [Q_heads for this group (num_kv_groups heads), K for this group, V for this group]
                # We need to reshape and reorder to standard [all Q | all K | all V] layout
                
                # Reshape to grouped format: [num_kv_heads, (num_kv_groups + 2), head_dim, hidden_size]
                group_size = num_kv_groups + 2
                wqkv_grouped = wqkv_weight.reshape(num_kv_heads, group_size, head_dim, config.hidden_size)
                
                # Extract Q, K, V from grouped layout
                # Q heads: first num_kv_groups entries in each group
                q_weight = wqkv_grouped[:, :num_kv_groups, :, :].reshape(num_q_heads, head_dim, config.hidden_size)
                q_weight = q_weight.reshape(q_size, config.hidden_size)
                
                # K heads: second to last entry in each group
                k_weight = wqkv_grouped[:, -2, :, :].reshape(kv_size, config.hidden_size)
                
                # V heads: last entry in each group
                v_weight = wqkv_grouped[:, -1, :, :].reshape(kv_size, config.hidden_size)
                
                # Create separate projection layers
                attn.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.bias)
                attn.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.bias)
                attn.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.bias)
                
                # Copy weights (ensure proper copy and contiguous memory)
                attn.q_proj.weight.data.copy_(q_weight.contiguous())
                attn.k_proj.weight.data.copy_(k_weight.contiguous())
                attn.v_proj.weight.data.copy_(v_weight.contiguous())
                
                # Handle biases if they exist (same grouped layout)
                if wqkv_bias is not None:
                    bias_grouped = wqkv_bias.reshape(num_kv_heads, group_size, head_dim)
                    
                    q_bias = bias_grouped[:, :num_kv_groups, :].reshape(q_size)
                    k_bias = bias_grouped[:, -2, :].reshape(kv_size)
                    v_bias = bias_grouped[:, -1, :].reshape(kv_size)
                    
                    attn.q_proj.bias.data.copy_(q_bias.contiguous())
                    attn.k_proj.bias.data.copy_(k_bias.contiguous())
                    attn.v_proj.bias.data.copy_(v_bias.contiguous())
                
                # Map wo to o_proj
                if hasattr(attn, 'wo') and not hasattr(attn, 'o_proj'):
                    attn.o_proj = attn.wo
        
        return model
