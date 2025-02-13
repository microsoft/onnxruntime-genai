# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
A set of Python classes to mimic Hugging Face's PyTorch models with GGUF weights that are already in NumPy.

The goal is for `GGUFModel` and a Hugging Face model produced by `AutoModel.from_pretrained(...)`
to share the same attributes so that the original Hugging Face --> ONNX code remains the same
no matter where the weights actually come from.
"""

from functools import reduce
from gguf.gguf_reader import GGUFReader

import re
import torch


class GGUFTensorModule:
    def __init__(self):
        self.weight = None
        self.bias = None


class GGUFAttention:
    def __init__(self):
        self.q_proj = GGUFTensorModule()
        self.k_proj = GGUFTensorModule()
        self.v_proj = GGUFTensorModule()
        self.o_proj = GGUFTensorModule()
        self.rotary_emb = GGUFTensorModule()


class GGUFMLP:
    def __init__(self):
        self.gate_proj = GGUFTensorModule()
        self.up_proj = GGUFTensorModule()
        self.down_proj = GGUFTensorModule()
        self.fc1 = GGUFTensorModule()
        self.fc2 = GGUFTensorModule()


class GGUFDecoderLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.input_layernorm = GGUFTensorModule()
        self.self_attn = GGUFAttention()
        self.post_attention_layernorm = GGUFTensorModule()
        self.mlp = GGUFMLP()

        # For Gemma-2:
        self.pre_feedforward_layernorm = GGUFTensorModule()
        self.post_feedforward_layernorm = GGUFTensorModule()


class GGUFModel:
    def __init__(self, input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size):
        # Load GGUF model and read its info
        reader = GGUFReader(input_path)

        self.embedding = GGUFTensorModule()
        self.final_norm = GGUFTensorModule()
        self.lm_head = GGUFTensorModule()
        self.layers = {}

        layer_id = 0
        for tensor in sorted(reader.tensors, key=lambda t: t.name):
            name = tensor.name
            data = torch.tensor(tensor.data)
            module = self.layers.setdefault(layer_id, GGUFDecoderLayer(layer_id))

            if name == "token_embd.weight":
                # Remove tensor data's padding via `reduce` when GGUF model's vocab size is larger than the config's vocab size
                embedding_shape = [vocab_size, hidden_size]
                self.embedding.weight = data[ : reduce(lambda x, y: x*y, embedding_shape)].reshape(embedding_shape)
            elif name == "output_norm.weight":
                self.final_norm.weight = data
            elif name == "output_norm.bias":
                self.final_norm.bias = data
            elif name == "output.weight":
                lm_head_shape = [vocab_size, hidden_size]
                self.lm_head.weight = data.reshape(lm_head_shape)
            elif name == "output.bias":
                self.lm_head.bias = data
            elif name in {"rope_freqs.weight", "rope_factors_short.weight", "rope_factors_long.weight"}:
                # Skip rotary embedding weights since they can be re-calculated when looping through the model
                continue
            else:
                curr_layer_id = int(name.split(".")[1])
                if curr_layer_id != layer_id:
                    # Switch layer module used
                    layer_id = curr_layer_id
                    module = self.layers.setdefault(layer_id, GGUFDecoderLayer(layer_id))

                # Map weights and biases of norm, attention, and feed-forward network
                # Graph order is attn_norm --> attn_q/k/v --> attn_output --> ffn_norm --> ffn_gate/up --> >ffn_down
                if bool(re.match(r"^blk\.\d+\.attn_norm\.weight$", name)):
                    # blk.layer_id.attn_norm.weight
                    module.input_layernorm.weight = data
                elif bool(re.match(r"^blk\.\d+\.attn_norm\.bias$", name)):
                    # blk.layer_id.attn_norm.bias
                    module.input_layernorm.bias = data
                elif bool(re.match(r"^blk\.\d+\.attn_q\.weight$", name)):
                    # blk.layer_id.attn_q.weight
                    q_shape = [head_size * num_attn_heads, hidden_size]
                    module.self_attn.q_proj.weight = data.reshape(q_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_q\.bias$", name)):
                    # blk.layer_id.attn_q.bias
                    module.self_attn.q_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.attn_k\.weight$", name)):
                    # blk.layer_id.attn_k.weight
                    k_shape = [head_size * num_kv_heads, hidden_size]
                    module.self_attn.k_proj.weight = data.reshape(k_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_k\.bias$", name)):
                    # blk.layer_id.attn_k.bias
                    module.self_attn.k_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.attn_v\.weight$", name)):
                    # blk.layer_id.attn_v.weight
                    v_shape = [head_size * num_kv_heads, hidden_size]
                    module.self_attn.v_proj.weight = data.reshape(v_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_v\.bias$", name)):
                    # blk.layer_id.attn_v.bias
                    module.self_attn.v_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.attn_output\.weight$", name)):
                    # blk.layer_id.attn_output.weight
                    o_shape = [hidden_size, head_size * num_attn_heads]
                    module.self_attn.o_proj.weight = data.reshape(o_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_output\.bias$", name)):
                    # blk.layer_id.attn_output.bias
                    module.self_attn.o_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.ffn_norm\.weight$", name)):
                    # blk.layer_id.ffn_norm.weight
                    module.post_attention_layernorm.weight = data
                elif bool(re.match(r"^blk\.\d+\.ffn_norm\.bias$", name)):
                    # blk.layer_id.ffn_norm.bias
                    module.post_attention_layernorm.bias = data
                elif bool(re.match(r"^blk\.\d+\.ffn_gate\.weight$", name)):
                    # blk.layer_id.ffn_gate.weight
                    gate_shape = [intermediate_size, hidden_size]
                    module.mlp.gate_proj.weight = data.reshape(gate_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_gate\.bias$", name)):
                    # blk.layer_id.ffn_gate.bias
                    module.mlp.gate_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.weight$", name)) and data.shape[0] == intermediate_size:
                    # blk.layer_id.ffn_up.weight
                    up_shape = [intermediate_size, hidden_size]
                    module.mlp.up_proj.weight = data.reshape(up_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.bias$", name)) and data.shape[0] == intermediate_size:
                    # blk.layer_id.ffn_up.bias
                    module.mlp.up_proj.bias = data
                elif bool(re.match(r"^blk\.\d+\.ffn_down\.weight$", name)):
                    # blk.layer_id.ffn_down.weight
                    down_shape = [hidden_size, intermediate_size]
                    module.mlp.down_proj.weight = data.reshape(down_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_down\.bias$", name)):
                    # blk.layer_id.ffn_down.bias
                    module.mlp.down_proj.bias = data
                # Match against fused layers
                elif bool(re.match(r"^blk\.\d+\.attn_qkv\.weight$", name)):
                    # blk.layer_id.attn_qkv.weight
                    q_size = num_attn_heads * head_size
                    kv_size = num_kv_heads * head_size
                    qkv_shape = [q_size + kv_size + kv_size, hidden_size]
                    qkv = data.reshape(qkv_shape)

                    module.self_attn.q_proj.weight = qkv[: q_size, :]
                    module.self_attn.k_proj.weight = qkv[q_size : q_size + kv_size, :]
                    module.self_attn.v_proj.weight = qkv[q_size + kv_size :, :]
                elif bool(re.match(r"^blk\.\d+\.attn_qkv\.bias$", name)):
                    # blk.layer_id.attn_qkv.bias
                    q_size = num_attn_heads * head_size
                    kv_size = num_kv_heads * head_size

                    module.self_attn.q_proj.bias = data[: q_size]
                    module.self_attn.k_proj.bias = data[q_size : q_size + kv_size]
                    module.self_attn.v_proj.bias = data[q_size + kv_size :]
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.weight$", name)) and data.shape[0] != intermediate_size:
                    # blk.layer_id.ffn_up.weight (gate_up_proj.weight)
                    module.mlp.gate_proj.weight = data[: intermediate_size, :]
                    module.mlp.up_proj.weight = data[intermediate_size :, :]
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.bias$", name)) and data.shape[0] != intermediate_size:
                    # blk.layer_id.ffn_up.bias (gate_up_proj.bias)
                    module.mlp.gate_proj.bias = data[: intermediate_size]
                    module.mlp.up_proj.bias = data[intermediate_size :]
                # Match against non-standard attribute names
                elif bool(re.match(r"^blk\.\d+\.post_attention_norm\.weight$", name)):
                    # Note: This meaning of this name differs in Hugging Face vs GGUF.
                    # Hugging Face labels this as the 'pre_feedforward_layernorm' since there is already a 'post_attention_layernorm'.
                    # GGUF labels this as the 'post_attention_norm' since the first norm after attention is named as 'ffn_norm'.

                    # blk.layer_id.post_attention_norm.weight
                    module.pre_feedforward_layernorm.weight = data
                elif bool(re.match(r"^blk\.\d+\.post_attention_norm\.bias$", name)):
                    # Note: This meaning of this name differs in Hugging Face vs GGUF.
                    # Hugging Face labels this as the 'pre_feedforward_layernorm' since there is already a 'post_attention_layernorm'.
                    # GGUF labels this as the 'post_attention_norm' since the first norm after attention is named as 'ffn_norm'.

                    # blk.layer_id.post_attention_norm.bias
                    module.pre_feedforward_layernorm.bias = data
                elif bool(re.match(r"^blk\.\d+\.post_ffw_norm\.weight$", name)):
                    # Note: This meaning of this name differs in Hugging Face vs GGUF.
                    # Hugging Face labels this as the 'input_layernorm' since it is the start of a layer.
                    # GGUF labels this as the 'post_ffw_norm' since the first norm to start a layer is named as 'attn_norm'.

                    # blk.layer_id.post_ffw_norm.weight
                    module.post_feedforward_layernorm.weight = data
                elif bool(re.match(r"^blk\.\d+\.post_ffw_norm\.bias$", name)):
                    # Note: This meaning of this name differs in Hugging Face vs GGUF.
                    # Hugging Face labels this as the 'input_layernorm' since it is the start of a layer.
                    # GGUF labels this as the 'post_ffw_norm' since the first norm to start a layer is named as 'attn_norm'.

                    # blk.layer_id.post_ffw_norm.bias
                    module.post_feedforward_layernorm.bias = data
                else:
                    raise NotImplementedError(f"{name} in your GGUF model is not recognized")
        
        # Set LM head weights + biases if not already set
        if self.lm_head.weight is None:
            # Embedding and LM head share same weights + biases (lm_head.weight == embedding.weight and lm_head.bias == embedding.bias)
            self.lm_head.weight = self.embedding.weight
            if self.lm_head.bias is not None:
                self.lm_head.bias = self.embedding.bias
    
        # Sort list of layers by layer id
        self.layers = list(self.layers.values())
        self.layers.sort(key=lambda m: m.layer_id)

    def modules(self):
        """
        Return list of modules in GGUF model in order of appearance in the model
        """
        return [self.embedding] + self.layers + [self.final_norm, self.lm_head]

    def undo_permute(self, head_size, hidden_size, num_attn_heads, num_kv_heads):
        """
        Undo `permute` operation by GGUF to get Hugging Face format
        For GGUF models that contain a `permute()` call in `convert_hf_to_gguf.py` (e.g. Granite, LLaMA, Mistral, OLMo)
        """
        for module in self.layers:
            q_shape = [head_size * num_attn_heads, hidden_size]
            module.self_attn.q_proj.weight = module.self_attn.q_proj.weight.flatten().reshape(num_attn_heads, q_shape[0] // num_attn_heads // 2, 2, *q_shape[1:]).swapaxes(1, 2).reshape(q_shape)

            k_shape = [head_size * num_kv_heads, hidden_size]
            module.self_attn.k_proj.weight = module.self_attn.k_proj.weight.flatten().reshape(num_kv_heads, k_shape[0] // num_kv_heads // 2, 2, *k_shape[1:]).swapaxes(1, 2).reshape(k_shape)

    def swap_mlp_types(self):
        """
        Switch from using the default `up_proj`/`down_proj` attributes to the `fc1`/`fc2` attributes respectively
        For GGUF models such as Phi-2
        """
        # Convert ffn_up (up_proj in Hugging Face model) to fc1
        # Convert ffn_down (down_proj in Hugging Face model) to fc2
        for module in self.layers:
            module.mlp.fc1, module.mlp.up_proj = module.mlp.up_proj, module.mlp.fc1
            module.mlp.fc2, module.mlp.down_proj = module.mlp.down_proj, module.mlp.fc2

    def swap_norm_types(self):
        """
        Swap `post_attention_layernorm` and `pre_feedforward_layernorm` attributes
        For GGUF models such as Gemma-2

        Ex: Gemma-2
        Regular model's mapping of GGUF --> Hugging Face:
        - attn_norm --> input_layernorm
        - ffn_norm --> post_attention_layernorm

        Gemma-2 model's mapping of GGUF --> Hugging Face:
        - attn_norm --> input_layernorm
        - ffn_norm --> pre_feedforward_layernorm
        - post_attention_norm --> post_attention_layernorm
        - post_ffw_norm --> post_feedforward_layernorm
        """
        for module in self.layers:
            module.post_attention_layernorm, module.pre_feedforward_layernorm = module.pre_feedforward_layernorm, module.post_attention_layernorm

    @staticmethod
    def from_pretrained(model_type, input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size):
        """
        Create GGUF models with the same attribute structures as Hugging Face's PyTorch models.
        Also performs any pre-processing and post-processing to the GGUF models to ensure the
        weights are the same as the PyTorch models.
        """
        if model_type == "ChatGLMModel":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        elif model_type == "GemmaForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        elif model_type == "Gemma2ForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.swap_norm_types()
        elif model_type == "GraniteForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "LlamaForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "MistralForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "NemotronForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        elif model_type == "OlmoForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "PhiForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.swap_mlp_types()
        elif model_type == "Phi3ForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        elif model_type == "Qwen2ForCausalLM":
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        else:
            raise NotImplementedError(f"The {model_type} model is not currently supported.")

        return model
