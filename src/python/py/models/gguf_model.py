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


class GGUFTensor:
    def __init__(self):
        self.tensor = None

    def detach(self):
        """
        No-op operation since NumPy tensors are on CPU
        """
        return self

    def numpy(self):
        """
        Return tensor, which is already stored as a NumPy tensor
        """
        return self.tensor


class GGUFTensorModule:
    def __init__(self):
        self.weight = GGUFTensor()
        self.bias = None

    def add_bias(self):
        self.bias = GGUFTensor()


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


class GGUFModel:
    def __init__(self, input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size):
        # Load GGUF model and read its info
        reader = GGUFReader(input_path)

        self.embedding = GGUFTensorModule()
        self.final_norm = GGUFTensorModule()
        self.lm_head = GGUFTensorModule()
        self.layers = []

        layer_id = 0
        module = GGUFDecoderLayer(layer_id)        
        for tensor in sorted(reader.tensors, key=lambda t: t.name):
            name = tensor.name

            if name == "token_embd.weight":
                # Remove tensor data's padding via `reduce` when GGUF model's vocab size is larger than the config's vocab size
                embedding_shape = [vocab_size, hidden_size]
                self.embedding.weight.tensor = tensor.data[ : reduce(lambda x, y: x*y, embedding_shape)].reshape(embedding_shape)
            elif name == "output_norm.weight":
                self.final_norm.weight.tensor = tensor.data
            elif name == "output_norm.bias":
                self.final_norm.add_bias()
                self.final_norm.bias.tensor = tensor.data
            elif name == "output.weight":
                lm_head_shape = [vocab_size, hidden_size]
                self.lm_head.weight.tensor = tensor.data.reshape(lm_head_shape)
            elif name == "output.bias":
                self.lm_head.add_bias()
                self.lm_head.bias.tensor = tensor.data
            else:
                curr_layer_id = int(name.split(".")[1])
                if curr_layer_id != layer_id:
                    # Add layer to list of modules
                    self.layers.append(module)
                    layer_id = curr_layer_id
                    module = GGUFDecoderLayer(layer_id)

                # Map weights and biases of norm, attention, and feed-forward network
                # Graph order is attn_norm --> attn_q/k/v --> attn_output --> ffn_norm --> ffn_gate/up --> >ffn_down
                if bool(re.match(r"^blk\.\d+\.attn_norm\.weight$", name)):
                    # blk.layer_id.attn_norm.weight
                    module.input_layernorm.weight.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.attn_norm\.bias$", name)):
                    # blk.layer_id.attn_norm.bias
                    module.input_layernorm.add_bias()
                    module.input_layernorm.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.attn_q\.weight$", name)):
                    # blk.layer_id.attn_q.weight
                    q_shape = [head_size * num_attn_heads, hidden_size]
                    module.self_attn.q_proj.weight.tensor = tensor.data.reshape(q_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_q\.bias$", name)):
                    # blk.layer_id.attn_q.bias
                    module.self_attn.q_proj.add_bias()
                    module.self_attn.q_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.attn_k\.weight$", name)):
                    # blk.layer_id.attn_k.weight
                    k_shape = [head_size * num_kv_heads, hidden_size]
                    module.self_attn.k_proj.weight.tensor = tensor.data.reshape(k_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_k\.bias$", name)):
                    # blk.layer_id.attn_k.bias
                    module.self_attn.k_proj.add_bias()
                    module.self_attn.k_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.attn_v\.weight$", name)):
                    # blk.layer_id.attn_v.weight
                    v_shape = [head_size * num_kv_heads, hidden_size]
                    module.self_attn.v_proj.weight.tensor = tensor.data.reshape(v_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_v\.bias$", name)):
                    # blk.layer_id.attn_v.bias
                    module.self_attn.v_proj.add_bias()
                    module.self_attn.v_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.attn_output\.weight$", name)):
                    # blk.layer_id.attn_output.weight
                    o_shape = [hidden_size, head_size * num_attn_heads]
                    module.self_attn.o_proj.weight.tensor = tensor.data.reshape(o_shape)
                elif bool(re.match(r"^blk\.\d+\.attn_output\.bias$", name)):
                    # blk.layer_id.attn_output.bias
                    module.self_attn.o_proj.add_bias()
                    module.self_attn.o_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.ffn_norm\.weight$", name)):
                    # blk.layer_id.ffn_norm.weight
                    module.post_attention_layernorm.weight.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.ffn_norm\.bias$", name)):
                    # blk.layer_id.ffn_norm.bias
                    module.post_attention_layernorm.add_bias()
                    module.post_attention_layernorm.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.ffn_gate\.weight$", name)):
                    # blk.layer_id.ffn_gate.weight
                    gate_shape = [intermediate_size, hidden_size]
                    module.mlp.gate_proj.weight.tensor = tensor.data.reshape(gate_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_gate\.bias$", name)):
                    # blk.layer_id.ffn_gate.bias
                    module.mlp.gate_proj.add_bias()
                    module.mlp.gate_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.weight$", name)):
                    # blk.layer_id.ffn_up.weight
                    up_shape = [intermediate_size, hidden_size]
                    module.mlp.up_proj.weight.tensor = tensor.data.reshape(up_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_up\.bias$", name)):
                    # blk.layer_id.ffn_up.bias
                    module.mlp.up_proj.add_bias()
                    module.mlp.up_proj.bias.tensor = tensor.data
                elif bool(re.match(r"^blk\.\d+\.ffn_down\.weight$", name)):
                    # blk.layer_id.ffn_down.weight
                    down_shape = [hidden_size, intermediate_size]
                    module.mlp.down_proj.weight.tensor = tensor.data.reshape(down_shape)
                elif bool(re.match(r"^blk\.\d+\.ffn_down\.bias$", name)):
                    # blk.layer_id.ffn_down.bias
                    module.mlp.down_proj.add_bias()
                    module.mlp.down_proj.bias.tensor = tensor.data
                else:
                    raise NotImplementedError(f"{name} in your GGUF model is not recognized")
        
        # Append final layer to list of layers
        self.layers.append(module)
        
        # Set LM head weights + biases if not already set
        if self.lm_head.weight.tensor is None:
            # Embedding and LM head share same weights + biases (lm_head.weight == embedding.weight and lm_head.bias == embedding.bias)
            self.lm_head.weight.tensor = self.embedding.weight.tensor
            if self.lm_head.bias is not None:
                self.lm_head.bias.tensor = self.embedding.bias.tensor
    
        # Sort list of layers by layer id
        self.layers.sort(key=lambda m: m.layer_id)

    def modules(self):
        """
        Return list of modules in GGUF model in order of appearance in the model
        """
        return [self.embedding] + self.layers + [self.final_norm, self.lm_head]

    def undo_permute(self, head_size, hidden_size, num_attn_heads, num_kv_heads):
        """
        Undo `permute` operation by GGUF to get Hugging Face format
        For GGUF models produced by `convert.py` (e.g. LLaMA, Mistral)
        """
        for module in self.layers:
            q_shape = [head_size * num_attn_heads, hidden_size]
            module.self_attn.q_proj.weight.tensor = module.self_attn.q_proj.weight.tensor.flatten().reshape(num_attn_heads, q_shape[0] // num_attn_heads // 2, 2, *q_shape[1:]).swapaxes(1, 2).reshape(q_shape)

            k_shape = [head_size * num_kv_heads, hidden_size]
            module.self_attn.k_proj.weight.tensor = module.self_attn.k_proj.weight.tensor.flatten().reshape(num_kv_heads, k_shape[0] // num_kv_heads // 2, 2, *k_shape[1:]).swapaxes(1, 2).reshape(k_shape)

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

    @staticmethod
    def from_pretrained(model_type, input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size):
        """
        Create GGUF models with the same attribute structures as Hugging Face's PyTorch models.
        Also performs any pre-processing and post-processing to the GGUF models to ensure the
        weights are the same as the PyTorch models.
        """
        if model_type == "GemmaForCausalLM":
            # convert-hf-to-gguf.py
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
        elif model_type == "LlamaForCausalLM":
            # convert.py
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "MistralForCausalLM":
            # convert.py
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.undo_permute(head_size, hidden_size, num_attn_heads, num_kv_heads)
        elif model_type == "PhiForCausalLM":
            # convert-hf-to-gguf.py
            model = GGUFModel(input_path, head_size, hidden_size, intermediate_size, num_attn_heads, num_kv_heads, vocab_size)
            model.swap_mlp_types()
        else:
            raise NotImplementedError(f"The {model_type} model is not currently supported.")

        return model