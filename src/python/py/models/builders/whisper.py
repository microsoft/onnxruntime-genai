# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
from .base import Model

import copy
import json
import os

import onnx_ir as ir
import torch

class WhisperEncoder(Model):
    # Each Whisper encoder layer is typically defined as:
    # input_layernorm --> attention --> output_layernorm --> MLP

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Standardize config attributes for encoder layers to simplify layer building logic
        config.num_hidden_layers = config.encoder_layers
        config.num_key_value_heads = config.encoder_attention_heads
        config.num_heads = config.encoder_attention_heads
        config.head_size = config.d_model // config.num_heads
        config.hidden_size = config.d_model
        config.intermediate_size = config.encoder_ffn_dim
        config.hidden_act = "gelu"
        config.seq_length = config.max_source_positions

        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        extra_options["include_hidden_states"] = True  # Include hidden states as output
        extra_options["exclude_lm_head"] = True  # Exclude LM head since it's not used in the encoder
        extra_options["filename"] = "encoder.onnx"  # Label encoder ONNX model

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.layernorm_attrs["simple"] = False
        self.attention_attrs["rope"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def is_gqa_supported(self):
        # GQA is not supported in Whisper since there is no attention mask input
        return False

    def is_packed_attn_supported(self):
        # Packed Attention is not supported in the encoder
        return False

    def make_inputs_and_outputs(self):
        # Set input dicts
        self.input_names = {"audio_features": "audio_features"}
        self.input_types = {"audio_features": self.io_dtype}
        self.input_shapes = {"audio_features": ["batch_size", self.num_mel_bins, 3000]}  # ['batch_size', 'num_mels', 'num_frames']

        # Set output dicts
        self.output_names = {
            "hidden_states": "hidden_states",
            "present_key_cross": [f"present_key_cross_{i}" for i in range(self.num_layers)],
            "present_value_cross": [f"present_value_cross_{i}" for i in range(self.num_layers)],
        }
        self.output_types = {
            "hidden_states": self.io_dtype,
            "present_key_cross": self.io_dtype,
            "present_value_cross": self.io_dtype,
        }
        self.output_shapes = {
            "hidden_states": ["batch_size", self.max_source_positions, self.hidden_size],  # ['batch_size', 'num_frames / 2', 'hidden_size']
            "present_key_cross": ["batch_size", self.num_attn_heads, self.max_source_positions, self.head_size],  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
            "present_value_cross": ["batch_size", self.num_attn_heads, self.max_source_positions, self.head_size]  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
        }

        # Now set inputs and outputs
        super().make_inputs_and_outputs()

    def make_preprocessing_nodes(self):
        # Make the following subgraph:
        #
        # audio_features --> Conv --> Gelu --> Conv --> Gelu --> Transpose --> Add
        basename = "/model/preprocessing"

        conv_1_weight = "encoder.conv1.weight"
        conv_1_bias = "encoder.conv1.bias"
        self.make_initializer(self.weights.model.encoder.conv1.weight, conv_1_weight, to=self.io_dtype)
        self.make_initializer(self.weights.model.encoder.conv1.bias, conv_1_bias, to=self.io_dtype)

        conv_1_name = f"{basename}/Conv_1"
        conv_1_inputs = ["audio_features", conv_1_weight, conv_1_bias]
        self.make_conv(conv_1_name, conv_1_inputs, dtype=self.io_dtype, shape=["batch_size", self.hidden_size, 3000], dilations=[1], group=1, kernel_shape=[3], pads=[1, 1], strides=[1])

        gelu_1_name = f"{basename}/Gelu_1"
        gelu_1_output = f"{gelu_1_name}/output_0"
        self.make_node("Gelu", inputs=[f"{conv_1_name}/output_0"], outputs=[gelu_1_output], name=gelu_1_name, approximate="none")
        self.make_value(gelu_1_output, dtype=self.io_dtype, shape=["batch_size", self.hidden_size, 3000])

        conv_2_weight = "encoder.conv2.weight"
        conv_2_bias = "encoder.conv2.bias"
        self.make_initializer(self.weights.model.encoder.conv2.weight, conv_2_weight, to=self.io_dtype)
        self.make_initializer(self.weights.model.encoder.conv2.bias, conv_2_bias, to=self.io_dtype)

        conv_2_name = f"{basename}/Conv_2"
        conv_2_inputs = [f"{gelu_1_name}/output_0", conv_2_weight, conv_2_bias]
        self.make_conv(conv_2_name, conv_2_inputs, dtype=self.io_dtype, shape=["batch_size", self.hidden_size, self.max_source_positions], dilations=[1], group=1, kernel_shape=[3], pads=[1, 1], strides=[2])

        gelu_2_name = f"{basename}/Gelu_2"
        gelu_2_output = f"{gelu_2_name}/output_0"
        self.make_node("Gelu", inputs=[f"{conv_2_name}/output_0"], outputs=[gelu_2_output], name=gelu_2_name, approximate="none")
        self.make_value(gelu_2_output, dtype=self.io_dtype, shape=["batch_size", self.hidden_size, self.max_source_positions])

        transpose_name = f"{basename}/Transpose"
        self.make_transpose(transpose_name, root_input=gelu_2_output, dtype=self.io_dtype, shape=["batch_size", self.max_source_positions, self.hidden_size], perm=[0, 2, 1])

        position_embeds = "encoder.embed_positions.weight"
        self.make_initializer(self.weights.model.encoder.embed_positions.weight, position_embeds, to=self.io_dtype)

        add_name = f"{basename}/Add"
        self.make_add add_name, inputs=[f"{transpose_name}/output_0", position_embeds], dtype=self.io_dtype, shape=["batch_size", self.max_source_positions, self.hidden_size])

        self.layernorm_attrs["root_input"] = f"{add_name}/output_0"
        self.layernorm_attrs["skip_input"] = f"{add_name}/output_0"

    def make_embedding(self, embedding):
        # Don't include the embedding in the encoder
        pass

    def make_layer(self, layer_id, layer):
        layer.input_layernorm = layer.self_attn_layer_norm
        layer.self_attn.k_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.self_attn.q_proj.bias), requires_grad=False)
        layer.post_attention_layernorm = layer.final_layer_norm

        class WhisperEncoderMLP(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.fc1 = layer.fc1
                self.fc2 = layer.fc2
                self.activation = layer.activation_fn

        layer.mlp = WhisperEncoderMLP(layer)
        super().make_layer(layer_id, layer)

    def make_key_value_cache_names(self, layer_id):
        # Whisper encoder does not use any KV caches
        past_k, past_v, present_k, present_v = "", "", "", ""
        return past_k, past_v, present_k, present_v

    def make_postprocessing_nodes(self):
        # Make the following subgraphs to extract and transpose cross-attention KV cache outputs for each layer:
        #
        # encoder_hidden_states --> MatMul --> Reshape --> Transpose --> present_key_cross_i
        # encoder_hidden_states --> MatMul --> Add --> Reshape --> Transpose --> present_value_cross_i

        # Add cross-attention KV cache outputs
        for i in range(self.num_layers):
            for proj_type in ["k_proj", "v_proj"]:
                basename = f"/model/layers.{i}/attn/cross/{proj_type}"
                matmul_name = f"{basename}/MatMul"
                proj = getattr(self.weights.model.decoder.layers[i].encoder_attn, proj_type)
                self.make_matmul(proj, matmul_name, root_input="hidden_states", seq_dim=self.max_source_positions)

                if proj_type == "v_proj":
                    add_name = f"{basename}/Add"
                    self.make_add_bias(
                        self.weights.model.decoder.layers[i].encoder_attn.v_proj.bias,
                        add_name,
                        root_input=f"{matmul_name}/output_0",
                        seq_dim=self.max_source_positions,
                    )

                reshape_name = f"{basename}/Reshape"
                self.make_reshape(
                    reshape_name,
                    [f"{add_name if proj_type == 'v_proj' else matmul_name}/output_0", f"/model/constants/INT64/[-1, {self.max_source_positions}, {self.num_attn_heads}, {self.head_size}]"],
                    dtype=self.io_dtype,
                    shape=["batch_size", self.max_source_positions, self.num_attn_heads, self.head_size],
                )

                transpose_name = f"{basename}/Transpose"
                output_name = f"present_{'key' if proj_type == 'k_proj' else 'value'}_cross_{i}"
                self.make_node(
                    "Transpose",
                    inputs=[f"{reshape_name}/output_0"],
                    outputs=[output_name],
                    name=transpose_name,
                    perm=[0, 2, 1, 3],
                )

    def is_layer(self, module):
        return module.__class__.__name__.endswith("EncoderLayer")

    def has_final_norm(self, module, model):
        hf_norm = hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model.encoder, "layer_norm") and module == model.model.encoder.layer_norm
        return hf_norm


class WhisperDecoder(Model):
    # Each Whisper decoder layer is typically defined as:
    # self_attn_layernorm --> self-attention --> cross_attn_layernorm --> cross-attention --> output_layernorm --> MLP

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Standardize config attributes for decoder layers to simplify layer building logic
        config.num_hidden_layers = config.decoder_layers
        config.num_key_value_heads = config.decoder_attention_heads
        config.num_heads = config.decoder_attention_heads
        config.head_size = config.d_model // config.num_heads
        config.hidden_size = config.d_model
        config.intermediate_size = config.decoder_ffn_dim
        config.hidden_act = "gelu"
        config.seq_length = config.max_target_positions

        extra_options["filename"] = "decoder.onnx"  # Label decoder ONNX model

        self.max_source_positions = config.max_source_positions
        self.max_target_positions = config.max_target_positions

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.layernorm_attrs["simple"] = False
        self.attention_attrs["rope"] = False
        self.attention_attrs["unidirectional"] = True
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def is_gqa_supported(self):
        # GQA is not supported in the decoder since there is no attention mask input
        return False

    def is_packed_attn_supported(self):
        # Packed Attention is not supported in the decoder since we deprecated
        # combined KV cache inputs/outputs in ORT GenAI
        return False

    def make_inputs_and_outputs(self):
        # Set input dicts
        self.input_names = {
            "input_ids": "input_ids",
            "past_key_self": [f"past_key_self_{i}" for i in range(self.num_layers)],
            "past_value_self": [f"past_value_self_{i}" for i in range(self.num_layers)],
            "past_key_cross": [f"past_key_cross_{i}" for i in range(self.num_layers)],
            "past_value_cross": [f"past_value_cross_{i}" for i in range(self.num_layers)],
        }
        self.input_types = {
            "input_ids": ir.DataType.INT32,
            "past_key_self": self.io_dtype,
            "past_value_self": self.io_dtype,
            "past_key_cross": self.io_dtype,
            "past_value_cross": self.io_dtype,
        }
        self.input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],
            "past_key_self": ["batch_size", self.num_attn_heads, "past_sequence_length", self.head_size],  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
            "past_value_self": ["batch_size", self.num_attn_heads, "past_sequence_length", self.head_size],  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
            "past_key_cross": ["batch_size", self.num_attn_heads, self.max_source_positions, self.head_size],  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
            "past_value_cross": ["batch_size", self.num_attn_heads, self.max_source_positions, self.head_size],  # ['batch_size', 'num_heads', 'num_frames / 2', 'head_size']
        }

        # Set output dicts
        self.output_names = {
            "hidden_states": "hidden_states",
            "logits": "logits",
            "present_key_self": [f"present_key_self_{i}" for i in range(self.num_layers)],
            "present_value_self": [f"present_value_self_{i}" for i in range(self.num_layers)],
        }
        self.output_types = {
            "hidden_states": self.io_dtype,
            "logits": self.io_dtype,
            "present_key_self": self.io_dtype,
            "present_value_self": self.io_dtype,
        }
        self.output_shapes = {
            "hidden_states": ["batch_size", "sequence_length", self.hidden_size],
            "logits": ["batch_size", "sequence_length", self.vocab_size],
            "present_key_self": ["batch_size", self.num_attn_heads, "total_sequence_length", self.head_size],  # ['batch_size', 'num_heads', 'total_sequence_length', 'head_size']
            "present_value_self": ["batch_size", self.num_attn_heads, "total_sequence_length", self.head_size],  # ['batch_size', 'num_heads', 'total_sequence_length', 'head_size']
        }
        self.make_outputs_init()

        # Now set inputs and outputs
        super().make_inputs_and_outputs()

    def make_preprocessing_nodes(self):
        # Make the following subgraph (from OpenAI version):
        #
        #       input_ids          past_key_self_0
        #       |       |                 |
        #       |     Shape             Shape
        #       |       |                 |
        #       |     Gather            Gather
        #       |     (idx=1)           (idx=2)
        #       |       |                 |   |
        #       |       +--------+--------+  Unsqueeze
        #       |                |           /
        #       |               Add         /
        #       |                |         /
        #       |            Unsqueeze    /
        #       |                  \     /
        #     Gather                Slice
        #       |                     |
        #       +--------------------Add
        basename = "/model/preprocessing"
        input_ids_basename = f"{basename}/input_ids_subgraph"
        past_key_basename = f"{basename}/past_key_subgraph"

        shape_0_name = f"{input_ids_basename}/Shape"
        self.make_shape(shape_0_name, root_input="input_ids", shape=[2])
        gather_0_name = f"{input_ids_basename}/Gather"
        self.make_gather(gather_0_name, [f"{shape_0_name}/output_0", "/model/constants/INT64/1"], dtype=ir.DataType.INT64, shape=[], axis=0)

        # Make past key 0 subgraph
        past_key_shape_name = f"{past_key_basename}/Shape"
        self.make_shape(past_key_shape_name, "past_key_self_0", shape=[4])
        past_key_gather_name = f"{past_key_basename}/Gather"
        past_key_gather_inputs = [f"{past_key_shape_name}/output_0", "/model/constants/INT64/2"]
        self.make_gather(past_key_gather_name, past_key_gather_inputs, dtype=ir.DataType.INT64, shape=[], axis=0)

        add_1_name = f"{basename}/Add_1"
        self.make_add(add_1_name, [f"{gather_0_name}/output_0", f"{past_key_gather_name}/output_0"], dtype=ir.DataType.INT64, shape=[])
        unsqueeze_0_name = f"{basename}/Unsqueeze_0"
        self.make_unsqueeze(unsqueeze_0_name, [f"{add_1_name}/output_0", "/model/constants/INT64/[0]"], dtype=ir.DataType.INT64, shape=[1])

        unsqueeze_1_name = f"{past_key_basename}/Unsqueeze"
        self.make_unsqueeze(unsqueeze_1_name, [f"{past_key_gather_name}/output_0", "/model/constants/INT64/[0]"], dtype=ir.DataType.INT64, shape=[1])

        position_embeds = "decoder.embed_positions.weight"
        self.make_initializer(self.weights.model.decoder.embed_positions.weight, position_embeds, to=self.io_dtype)
        slice_name = f"{basename}/Slice"
        slice_inputs = [position_embeds, f"{unsqueeze_1_name}/output_0", f"{unsqueeze_0_name}/output_0", "/model/constants/INT64/[0]", "/model/constants/INT64/[1]"]
        self.make_slice(slice_name, slice_inputs, dtype=self.io_dtype, shape=["sequence_length", self.hidden_size])

        token_embeds = "decoder.embed_tokens.weight"
        self.make_initializer(self.weights.model.decoder.embed_tokens.weight, token_embeds, to=self.io_dtype)
        gather_1_name = f"{basename}/Gather_1"
        self.make_gather(gather_1_name, [token_embeds, "input_ids"], dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size], axis=0)
        add_2_name = f"{basename}/Add"
        self.make_add(add_2_name, [f"{gather_1_name}/output_0", f"{slice_name}/output_0"], dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        self.layernorm_attrs["root_input"] = f"{add_2_name}/output_0"
        self.layernorm_attrs["skip_input"] = f"{add_2_name}/output_0"

    def make_embedding(self, embedding):
        # Embedding is already created in preprocessing
        pass

    def make_layer(self, layer_id, layer):
        # Each Whisper decoder layer is typically defined as:
        # self_attn_layernorm --> self-attention --> cross_attn_layernorm --> cross-attention --> output_layernorm --> MLP
        self.make_layernorm(
            layer_id,
            layer.self_attn_layer_norm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="self_attn",
        )

        # Self-attention (unidirectional = True)
        layer.self_attn.k_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.self_attn.q_proj.bias), requires_grad=False)
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # Encoder layernorm
        self.make_layernorm(
            layer_id,
            layer.encoder_attn_layer_norm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="cross_attn",
        )

        # Cross-attention (unidirectional = False)
        self.attention_attrs["unidirectional"] = False
        self.make_cross_attention(layer_id, layer.encoder_attn, root_input=self.layernorm_attrs["output_0"])
        self.attention_attrs["unidirectional"] = True

        # Output layernorm
        self.make_layernorm(
            layer_id,
            layer.final_layer_norm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )

        # MLP
        class WhisperDecoderMLP(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.fc1 = layer.fc1
                self.fc2 = layer.fc2
                self.activation = layer.activation_fn

        layer.mlp = WhisperDecoderMLP(layer)
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_cross_attention(self, layer_id, attention, root_input, **kwargs):
        # Whisper decoder cross-attention:
        #
        #               root_input
        #                    |
        #           +--------+---------+
        #           |        |         |
        #       Q_MatMul  past_key  past_value
        #           |        |         |
        #         Q_Add      |         |
        #           \        |        /
        #            MultiHeadAttention
        #                    |
        #                O_MatMul
        #                    |
        #                  O_Add
        #
        q_matmul_basename = f"/model/layers.{layer_id}/cross_attn/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"

        q_add_name = f"/model/layers.{layer_id}/cross_attn/q_proj/Add"
        self.make_add_bias(attention.q_proj.bias, q_add_name, root_input=self.attention_attrs["q_path"])
        self.attention_attrs["q_path"] = f"{q_add_name}/output_0"

        self.attention_attrs["k_path"] = self.input_names["past_key_cross"][layer_id]
        self.attention_attrs["v_path"] = self.input_names["past_value_cross"][layer_id]

        # Make attention node (e.g. MultiHeadAttention, GroupQueryAttention, etc.)
        attn_name = f"/model/layers.{layer_id}/cross_attn/{self.attention_attrs['op_type']}"
        attn_output = f"{attn_name}/output_0"
        self.make_attention_op(
            attn_name,
            root_input=root_input,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k="",
            past_v="",
            present_k="",
            present_v="",
            **kwargs,
        )

        # Make output projection
        o_matmul_basename = f"/model/layers.{layer_id}/cross_attn/o_proj/MatMul"
        o_matmul_name = self.make_matmul(attention.out_proj, o_matmul_basename, attn_output)

        # Make Add node
        o_add_name = f"/model/layers.{layer_id}/cross_attn/o_proj/Add"
        self.make_add_bias(attention.out_proj.bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{o_add_name}/output_0"

    def make_key_value_cache_names(self, layer_id):
        past_k = self.input_names["past_key_self"][layer_id]
        past_v = self.input_names["past_value_self"][layer_id]
        present_k = self.output_names["present_key_self"][layer_id]
        present_v = self.output_names["present_value_self"][layer_id]
        return past_k, past_v, present_k, present_v

    def is_layer(self, module):
        return module.__class__.__name__.endswith("DecoderLayer")

    def has_final_norm(self, module, model):
        hf_norm = hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layer_norm") and module == model.model.decoder.layer_norm
        return hf_norm


class WhisperModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config.rms_norm_eps = 0.000009999999747378752  # default value is insufficient for accuracy
        self.encoder = WhisperEncoder(copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, copy.deepcopy(extra_options))
        self.decoder = WhisperDecoder(copy.deepcopy(config), io_dtype, onnx_dtype, ep, cache_dir, copy.deepcopy(extra_options))

        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hf_token = self.decoder.hf_token
        self.hf_remote = self.decoder.hf_remote
        self.context_length = self.decoder.context_length

    def is_gqa_supported(self):
        # GQA is not supported in Whisper since there is no attention mask input
        return False

    def make_model(self, input_path):
        self.encoder.make_model(input_path)
        self.decoder.make_model(input_path)

    def save_model(self, output_dir):
        self.encoder.save_model(output_dir)
        self.decoder.save_model(output_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        audio_processor_cfg = {
            "feature_extraction": {
                "sequence": [
                    {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
                    {
                        "operation": {
                            "name": "STFT",
                            "type": "STFTNorm",
                            "attrs": {
                                "n_fft": 400,
                                "frame_length": 400,
                                "hop_length": 160,
                                "_comment": [
                                    0.0,
                                    0.0000616908073425293,
                                    0.0002467334270477295,
                                    0.0005550682544708252,
                                    0.000986635684967041,
                                    0.0015413463115692139,
                                    0.0022190213203430176,
                                    0.0030195116996765137,
                                    0.003942638635635376,
                                    0.004988163709640503,
                                    0.006155818700790405,
                                    0.007445335388183594,
                                    0.008856385946273804,
                                    0.010388582944869995,
                                    0.012041628360748291,
                                    0.013815045356750488,
                                    0.01570841670036316,
                                    0.01772129535675049,
                                    0.019853144884109497,
                                    0.022103488445281982,
                                    0.02447172999382019,
                                    0.026957333087921143,
                                    0.029559612274169922,
                                    0.03227800130844116,
                                    0.03511175513267517,
                                    0.03806024789810181,
                                    0.0411226749420166,
                                    0.044298380613327026,
                                    0.04758647084236145,
                                    0.05098623037338257,
                                    0.05449673533439636,
                                    0.058117181062698364,
                                    0.06184667348861694,
                                    0.0656842589378357,
                                    0.06962898373603821,
                                    0.07367992401123047,
                                    0.0778360664844513,
                                    0.08209633827209473,
                                    0.08645972609519958,
                                    0.09092515707015991,
                                    0.09549149870872498,
                                    0.10015767812728882,
                                    0.10492250323295593,
                                    0.1097848117351532,
                                    0.11474338173866272,
                                    0.11979702115058899,
                                    0.12494447827339172,
                                    0.13018447160720825,
                                    0.1355157196521759,
                                    0.14093685150146484,
                                    0.1464466154575348,
                                    0.15204361081123352,
                                    0.1577264666557312,
                                    0.16349375247955322,
                                    0.16934409737586975,
                                    0.1752760112285614,
                                    0.18128803372383118,
                                    0.18737870454788208,
                                    0.19354650378227234,
                                    0.1997898817062378,
                                    0.20610737800598145,
                                    0.21249738335609436,
                                    0.21895831823349,
                                    0.2254886031150818,
                                    0.23208662867546082,
                                    0.23875075578689575,
                                    0.24547931551933289,
                                    0.2522706985473633,
                                    0.25912320613861084,
                                    0.26603513956069946,
                                    0.27300477027893066,
                                    0.2800304591655731,
                                    0.2871103882789612,
                                    0.29424285888671875,
                                    0.30142611265182495,
                                    0.30865830183029175,
                                    0.31593772768974304,
                                    0.3232625722885132,
                                    0.3306310474872589,
                                    0.3380413055419922,
                                    0.34549152851104736,
                                    0.352979838848114,
                                    0.3605044484138489,
                                    0.3680635094642639,
                                    0.37565508484840393,
                                    0.38327735662460327,
                                    0.3909284174442291,
                                    0.39860638976097107,
                                    0.4063093662261963,
                                    0.41403549909591675,
                                    0.42178282141685486,
                                    0.4295494258403778,
                                    0.43733343482017517,
                                    0.44513291120529175,
                                    0.45294591784477234,
                                    0.46077051758766174,
                                    0.46860480308532715,
                                    0.4764467775821686,
                                    0.4842946231365204,
                                    0.492146372795105,
                                    0.5,
                                    0.5078536868095398,
                                    0.515705406665802,
                                    0.5235532522201538,
                                    0.5313953161239624,
                                    0.5392295718193054,
                                    0.5470541715621948,
                                    0.5548672080039978,
                                    0.562666654586792,
                                    0.5704506635665894,
                                    0.5782172679901123,
                                    0.5859646201133728,
                                    0.5936906933784485,
                                    0.6013936996459961,
                                    0.609071671962738,
                                    0.6167227625846863,
                                    0.6243450045585632,
                                    0.6319366097450256,
                                    0.6394955515861511,
                                    0.6470202207565308,
                                    0.6545085310935974,
                                    0.6619587540626526,
                                    0.6693689823150635,
                                    0.6767374277114868,
                                    0.6840623021125793,
                                    0.691341757774353,
                                    0.6985740065574646,
                                    0.7057572603225708,
                                    0.7128896713256836,
                                    0.719969630241394,
                                    0.7269952893257141,
                                    0.7339649796485901,
                                    0.7408769130706787,
                                    0.7477294206619263,
                                    0.7545207738876343,
                                    0.761249303817749,
                                    0.7679134607315063,
                                    0.774511456489563,
                                    0.7810417413711548,
                                    0.7875027060508728,
                                    0.7938927412033081,
                                    0.800210177898407,
                                    0.8064535856246948,
                                    0.8126214146614075,
                                    0.8187121152877808,
                                    0.8247240781784058,
                                    0.8306560516357422,
                                    0.8365063667297363,
                                    0.8422735929489136,
                                    0.8479564785957336,
                                    0.8535534143447876,
                                    0.8590631484985352,
                                    0.8644843101501465,
                                    0.8698155879974365,
                                    0.8750555515289307,
                                    0.8802030086517334,
                                    0.8852566480636597,
                                    0.8902152180671692,
                                    0.8950775265693665,
                                    0.899842381477356,
                                    0.9045084714889526,
                                    0.9090749025344849,
                                    0.9135403037071228,
                                    0.9179036617279053,
                                    0.9221639633178711,
                                    0.9263200759887695,
                                    0.9303710460662842,
                                    0.9343158006668091,
                                    0.9381533861160278,
                                    0.941882848739624,
                                    0.945503294467926,
                                    0.9490138292312622,
                                    0.9524135589599609,
                                    0.9557017087936401,
                                    0.9588773250579834,
                                    0.961939811706543,
                                    0.9648882746696472,
                                    0.9677220582962036,
                                    0.9704403877258301,
                                    0.9730427265167236,
                                    0.9755282998085022,
                                    0.9778965711593628,
                                    0.9801468849182129,
                                    0.9822787046432495,
                                    0.9842916131019592,
                                    0.9861849546432495,
                                    0.9879584312438965,
                                    0.9896113872528076,
                                    0.9911436438560486,
                                    0.9925546646118164,
                                    0.9938441514968872,
                                    0.9950118064880371,
                                    0.996057391166687,
                                    0.9969804883003235,
                                    0.997780978679657,
                                    0.9984586238861084,
                                    0.999013364315033,
                                    0.9994449615478516,
                                    0.9997532367706299,
                                    0.9999383091926575,
                                    1,
                                    0.9999383091926575,
                                    0.9997532367706299,
                                    0.9994449615478516,
                                    0.999013364315033,
                                    0.9984586238861084,
                                    0.997780978679657,
                                    0.9969804286956787,
                                    0.9960573315620422,
                                    0.9950118064880371,
                                    0.9938441514968872,
                                    0.9925546646118164,
                                    0.9911435842514038,
                                    0.9896113872528076,
                                    0.9879583716392517,
                                    0.9861849546432495,
                                    0.9842915534973145,
                                    0.9822787046432495,
                                    0.9801468253135681,
                                    0.9778964519500732,
                                    0.9755282402038574,
                                    0.9730426073074341,
                                    0.9704403877258301,
                                    0.9677219390869141,
                                    0.9648882150650024,
                                    0.9619396924972534,
                                    0.9588772654533386,
                                    0.9557015895843506,
                                    0.9524134397506714,
                                    0.9490137100219727,
                                    0.9455032348632812,
                                    0.9418827295303345,
                                    0.9381532669067383,
                                    0.9343156814575195,
                                    0.9303709268569946,
                                    0.9263200759887695,
                                    0.9221639633178711,
                                    0.9179036617279053,
                                    0.913540244102478,
                                    0.9090747833251953,
                                    0.9045084714889526,
                                    0.8998422622680664,
                                    0.8950774669647217,
                                    0.8902151584625244,
                                    0.8852565884590149,
                                    0.8802029490470886,
                                    0.8750554919242859,
                                    0.869815468788147,
                                    0.8644842505455017,
                                    0.8590630888938904,
                                    0.853553295135498,
                                    0.8479562997817993,
                                    0.842273473739624,
                                    0.836506187915802,
                                    0.8306558728218079,
                                    0.8247239589691162,
                                    0.8187118768692017,
                                    0.8126212358474731,
                                    0.8064534664154053,
                                    0.8002099990844727,
                                    0.793892502784729,
                                    0.7875025272369385,
                                    0.7810416221618652,
                                    0.7745113372802734,
                                    0.767913281917572,
                                    0.7612491846084595,
                                    0.7545205950737,
                                    0.7477291822433472,
                                    0.7408767342567444,
                                    0.7339648008346558,
                                    0.7269951105117798,
                                    0.7199694514274597,
                                    0.7128894925117493,
                                    0.7057570219039917,
                                    0.6985738277435303,
                                    0.6913415789604187,
                                    0.684062123298645,
                                    0.6767372488975525,
                                    0.6693688035011292,
                                    0.6619585752487183,
                                    0.6545083522796631,
                                    0.6470199823379517,
                                    0.6394953727722168,
                                    0.6319363117218018,
                                    0.6243447661399841,
                                    0.6167224645614624,
                                    0.6090714335441589,
                                    0.601393461227417,
                                    0.5936904549598694,
                                    0.5859643220901489,
                                    0.5782170295715332,
                                    0.5704504251480103,
                                    0.5626664161682129,
                                    0.5548669099807739,
                                    0.5470539331436157,
                                    0.5392293334007263,
                                    0.5313950181007385,
                                    0.5235530138015747,
                                    0.5157051682472229,
                                    0.507853627204895,
                                    0.5,
                                    0.4921463429927826,
                                    0.484294593334198,
                                    0.4764467477798462,
                                    0.46860471367836,
                                    0.4607704281806946,
                                    0.4529458284378052,
                                    0.4451328217983246,
                                    0.437333345413208,
                                    0.42954933643341064,
                                    0.4217827320098877,
                                    0.4140354096889496,
                                    0.4063093066215515,
                                    0.3986063003540039,
                                    0.39092832803726196,
                                    0.3832772672176361,
                                    0.37565499544143677,
                                    0.36806342005729675,
                                    0.3605043888092041,
                                    0.35297977924346924,
                                    0.3454914391040802,
                                    0.338041216135025,
                                    0.33063095808029175,
                                    0.3232625126838684,
                                    0.3159376382827759,
                                    0.3086581826210022,
                                    0.3014259934425354,
                                    0.2942427396774292,
                                    0.28711026906967163,
                                    0.2800303101539612,
                                    0.2730046510696411,
                                    0.2660350203514099,
                                    0.2591230869293213,
                                    0.25227057933807373,
                                    0.24547919631004333,
                                    0.2387506067752838,
                                    0.23208650946617126,
                                    0.22548848390579224,
                                    0.21895819902420044,
                                    0.2124972641468048,
                                    0.2061072587966919,
                                    0.19978976249694824,
                                    0.1935463547706604,
                                    0.18737855553627014,
                                    0.18128788471221924,
                                    0.17527586221694946,
                                    0.1693439483642578,
                                    0.16349363327026367,
                                    0.15772631764411926,
                                    0.15204349160194397,
                                    0.14644649624824524,
                                    0.1409367322921753,
                                    0.13551557064056396,
                                    0.1301843225955963,
                                    0.12494435906410217,
                                    0.11979690194129944,
                                    0.11474326252937317,
                                    0.10978469252586365,
                                    0.10492238402366638,
                                    0.10015755891799927,
                                    0.09549137949943542,
                                    0.09092503786087036,
                                    0.08645960688591003,
                                    0.08209621906280518,
                                    0.07783591747283936,
                                    0.07367980480194092,
                                    0.06962886452674866,
                                    0.06568413972854614,
                                    0.06184655427932739,
                                    0.0581170916557312,
                                    0.0544966459274292,
                                    0.05098611116409302,
                                    0.04758638143539429,
                                    0.044298261404037476,
                                    0.04112258553504944,
                                    0.038060128688812256,
                                    0.03511166572570801,
                                    0.03227788209915161,
                                    0.02955952286720276,
                                    0.02695724368095398,
                                    0.024471670389175415,
                                    0.02210339903831482,
                                    0.01985308527946472,
                                    0.017721205949783325,
                                    0.015708357095718384,
                                    0.0138150155544281,
                                    0.012041598558425903,
                                    0.010388582944869995,
                                    0.008856356143951416,
                                    0.007445335388183594,
                                    0.006155818700790405,
                                    0.004988163709640503,
                                    0.003942638635635376,
                                    0.0030195116996765137,
                                    0.0022190213203430176,
                                    0.0015413165092468262,
                                    0.000986635684967041,
                                    0.0005550682544708252,
                                    0.0002467334270477295,
                                    0.0000616908073425293,
                                ],
                            },
                        }
                    },
                    {
                        "operation": {
                            "name": "log_mel_spectrogram",
                            "type": "LogMelSpectrum",
                            "attrs": {"chunk_size": 30, "hop_length": 160, "n_fft": 400, "n_mel": self.encoder.num_mel_bins},
                        }
                    },
                ]
            }
        }
        audio_processor_json = json.dumps(audio_processor_cfg, indent=4)

        with open(os.path.join(out_dir, "audio_processor_config.json"), "w") as f:
            f.write(audio_processor_json)

        genai_config = {
            "model": {
                "bos_token_id": self.bos_token_id,
                "context_length": self.decoder.max_target_positions,
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": [{self.decoder.ep: self.decoder.ep_attrs[self.decoder.ep]}] if self.decoder.ep != "cpu" else [],
                    },
                    "filename": self.decoder.filename,
                    "head_size": self.decoder.head_size,
                    "hidden_size": self.decoder.hidden_size,
                    "inputs": {
                        "input_ids": "input_ids",
                        "past_key_names": "past_key_self_%d",
                        "past_value_names": "past_value_self_%d",
                        "cross_past_key_names": "past_key_cross_%d",
                        "cross_past_value_names": "past_value_cross_%d",
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present_key_self_%d",
                        "present_value_names": "present_value_self_%d",
                    },
                    "num_attention_heads": self.decoder.num_attn_heads,
                    "num_hidden_layers": self.decoder.num_layers,
                    "num_key_value_heads": self.decoder.num_kv_heads,
                },
                "encoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "provider_options": [{self.encoder.ep: self.encoder.ep_attrs[self.encoder.ep]}] if self.encoder.ep != "cpu" else [],
                    },
                    "filename": self.encoder.filename,
                    "head_size": self.encoder.head_size,
                    "hidden_size": self.encoder.hidden_size,
                    "inputs": {"audio_features": "audio_features"},
                    "outputs": {
                        "encoder_hidden_states": "hidden_states",
                        "cross_present_key_names": "present_key_cross_%d",
                        "cross_present_value_names": "present_value_cross_%d",
                    },
                    "num_attention_heads": self.encoder.num_attn_heads,
                    "num_hidden_layers": self.encoder.num_layers,
                    "num_key_value_heads": self.encoder.num_kv_heads,
                },
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "type": "whisper",
                "vocab_size": self.vocab_size,
            },
            "search": {
                "diversity_penalty": 0.0,
                "do_sample": False,
                "early_stopping": True,
                "length_penalty": 1.0,
                "max_length": self.decoder.max_target_positions,
                "min_length": 0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": False,
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
            },
        }

        with open(os.path.join(out_dir, "genai_config.json"), "w") as f:
            json.dump(genai_config, f, indent=4)
