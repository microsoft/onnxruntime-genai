# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import math
import types

import onnx_ir as ir
import torch
import torch.nn.functional as functional

from .base import Model


class NemotronParseEncoderComponent(Model):
    """Build the fixed-resolution RADIO encoder and decoder cross-KV projections."""

    external_data_size_threshold_bytes = 1024

    def __init__(
        self,
        config,
        weights,
        io_dtype,
        onnx_dtype,
        ep,
        cache_dir,
        extra_options,
        *,
        image_height,
        image_width,
        encoder_sequence_length,
    ):
        self.source_config = config
        self.weights = weights
        self.encoder = weights.encoder
        self.decoder = weights.decoder
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.encoder_sequence_length = int(encoder_sequence_length)

        self.radio = self._radio_model()
        self.vit = self.radio.model
        self.patch_generator = self.vit.patch_generator
        self.patch_size = int(self.patch_generator.patch_size)
        self.patch_rows = self.image_height // self.patch_size
        self.patch_cols = self.image_width // self.patch_size
        self.patch_count = self.patch_rows * self.patch_cols
        self.prefix_count = int(self.patch_generator.num_skip)
        self.radio_sequence_length = self.prefix_count + self.patch_count
        self.radio_hidden_size = int(self.vit.embed_dim)
        self.radio_num_heads = int(self.vit.blocks[0].attn.num_heads)
        self.radio_head_size = self.radio_hidden_size // self.radio_num_heads

        component_config = copy.deepcopy(config.encoder)
        component_config._name_or_path = config._name_or_path
        component_config.architectures = ["RADIOModel"]
        component_config.hidden_size = self.radio_hidden_size
        component_config.intermediate_size = int(
            self.vit.blocks[0].mlp.fc1.out_features
        )
        component_config.hidden_act = "gelu"
        component_config.max_position_embeddings = self.radio_sequence_length
        component_config.num_attention_heads = self.radio_num_heads
        component_config.num_key_value_heads = self.radio_num_heads
        component_config.num_hidden_layers = len(self.vit.blocks)
        component_config.vocab_size = config.decoder.vocab_size

        component_options = copy.deepcopy(extra_options)
        component_options["filename"] = "encoder.onnx"
        component_options["prune_lm_head"] = False
        component_options["shared_embeddings"] = False
        super().__init__(
            component_config,
            io_dtype,
            onnx_dtype,
            ep,
            cache_dir,
            component_options,
        )
        self.graph.name = "nemotron_parse_radio_encoder"
        self._validate_architecture()

    def _radio_model(self):
        model_encoder = self.encoder.model_encoder
        return getattr(model_encoder, "radio_model", model_encoder)

    @staticmethod
    def _is_identity(module):
        return isinstance(module, torch.nn.Identity)

    def _validate_architecture(self):
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("Nemotron Parse image dimensions must be divisible by patch_size.")
        if not self.vit.blocks:
            raise ValueError("Nemotron Parse RADIO encoder has no transformer blocks.")
        if self.radio_hidden_size % self.radio_num_heads:
            raise ValueError("RADIO hidden size must be divisible by its attention head count.")
        if not self._is_identity(self.vit.norm):
            raise ValueError("Nemotron Parse builder currently requires RADIO's final norm to be Identity.")
        if not self._is_identity(self.patch_generator.patch_normalizer):
            raise ValueError("Nemotron Parse builder currently requires unnormalized RADIO patches.")
        if not self._is_identity(self.radio.feature_normalizer):
            raise ValueError("Nemotron Parse builder currently requires RADIO's feature normalizer to be Identity.")
        if getattr(self.radio, "adaptors", None):
            raise ValueError("Nemotron Parse builder does not support RADIO adaptors.")
        if self.prefix_count <= 0:
            raise ValueError("Nemotron Parse RADIO encoder requires prefix tokens.")
        if self.encoder.conv2.kernel_size != (1, 4) or self.encoder.conv2.stride != (1, 4):
            raise ValueError("Nemotron Parse builder requires the checkpoint's 1x4 compression neck.")
        compressed_width = (self.patch_cols - 4) // 4 + 1
        if self.patch_rows * compressed_width + 1 != self.encoder_sequence_length:
            raise ValueError("Encoder sequence length does not match the RADIO neck geometry.")
        for block in self.vit.blocks:
            if not all(
                self._is_identity(module)
                for module in (
                    block.ls1,
                    block.drop_path1,
                    block.ls2,
                    block.drop_path2,
                    block.attn.q_norm,
                    block.attn.k_norm,
                )
            ):
                raise ValueError("Nemotron Parse builder requires the checkpoint's plain ViT blocks.")
            if block.attn.num_heads != self.radio_num_heads:
                raise ValueError("All RADIO blocks must use the same attention head count.")

    def is_gqa_supported(self):
        return False

    def is_packed_attn_supported(self):
        return False

    def load_weights(self, input_path):
        raise RuntimeError("NemotronParseEncoderComponent receives weights from its parent builder")

    def build(self):
        self._make_inputs_and_outputs()
        hidden_states = self._make_patch_embedding()
        for layer_id, block in enumerate(self.vit.blocks):
            hidden_states = self._make_radio_block(layer_id, block, hidden_states)
        encoder_hidden_states = self._make_neck(hidden_states)
        self._make_cross_cache(encoder_hidden_states)

    def _make_inputs_and_outputs(self):
        self.graph.inputs.append(
            self.make_value(
                "pixel_values",
                self.io_dtype,
                [1, 3, self.image_height, self.image_width],
            )
        )
        self.graph.outputs.append(
            self.make_value(
                "encoder_hidden_states",
                self.io_dtype,
                [1, self.encoder_sequence_length, self.source_config.decoder.d_model],
            )
        )
        cross_shape = [
            1,
            self.source_config.decoder.decoder_attention_heads,
            self.encoder_sequence_length,
            self.source_config.decoder.d_model
            // self.source_config.decoder.decoder_attention_heads,
        ]
        for layer_id in range(self.source_config.decoder.decoder_layers):
            self.graph.outputs.append(
                self.make_value(f"cross_present.{layer_id}.key", self.io_dtype, cross_shape)
            )
            self.graph.outputs.append(
                self.make_value(f"cross_present.{layer_id}.value", self.io_dtype, cross_shape)
            )

    def _make_layer_norm(self, layer_norm, name, root_input, shape):
        weight = f"{name[1:].replace('/', '.')}.weight"
        bias = f"{name[1:].replace('/', '.')}.bias"
        self.make_initializer(layer_norm.weight, weight, to=self.io_dtype)
        self.make_initializer(layer_norm.bias, bias, to=self.io_dtype)
        output = f"{name}/output_0"
        self.make_node(
            "LayerNormalization",
            inputs=[root_input, weight, bias],
            outputs=[output],
            name=name,
            axis=-1,
            epsilon=float(layer_norm.eps),
            stash_type=1,
        )
        self.make_value(output, self.io_dtype, shape)
        return output

    def _make_linear(self, linear, name, root_input, sequence_length):
        matmul = self.make_matmul(linear, f"{name}/MatMul", root_input, seq_dim=sequence_length)
        output = f"{matmul}/output_0"
        if linear.bias is not None:
            self.make_add_bias(
                linear.bias,
                f"{name}/Add",
                root_input=output,
                seq_dim=sequence_length,
            )
            output = f"{name}/Add/output_0"
        return output

    def _make_patch_embedding(self):
        base = "/encoder/radio/patch_generator"
        projection = self.patch_generator.embedder
        weight_name = "encoder.radio.patch_generator.weight"
        conv_weight = projection.weight.reshape(
            self.radio_hidden_size, 3, self.patch_size, self.patch_size
        )
        self.make_initializer(conv_weight, weight_name, to=self.io_dtype)
        conv = f"{base}/Conv"
        self.make_conv(
            conv,
            ["pixel_values", weight_name],
            self.io_dtype,
            [1, self.radio_hidden_size, self.patch_rows, self.patch_cols],
            kernel_shape=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
        )
        transpose = f"{base}/Transpose"
        self.make_transpose(
            transpose,
            f"{conv}/output_0",
            self.io_dtype,
            [1, self.patch_rows, self.patch_cols, self.radio_hidden_size],
            [0, 2, 3, 1],
        )
        reshape_shape = f"{base}/patch_shape"
        self.make_initializer(
            torch.tensor([1, self.patch_count, self.radio_hidden_size], dtype=torch.int64),
            reshape_shape,
        )
        reshape = f"{base}/Reshape"
        self.make_reshape(
            reshape,
            [f"{transpose}/output_0", reshape_shape],
            self.io_dtype,
            [1, self.patch_count, self.radio_hidden_size],
        )

        position_name = "encoder.radio.patch_generator.position_embedding"
        self.make_initializer(self._specialized_position_embedding(), position_name, to=self.io_dtype)
        add = f"{base}/AddPosition"
        self.make_add(
            add,
            [f"{reshape}/output_0", position_name],
            self.io_dtype,
            [1, self.patch_count, self.radio_hidden_size],
        )

        prefix_name = "encoder.radio.patch_generator.prefix_tokens"
        self.make_initializer(
            self.patch_generator.cls_token.token.unsqueeze(0),
            prefix_name,
            to=self.io_dtype,
        )
        concat = f"{base}/ConcatPrefix"
        self.make_concat(
            concat,
            [prefix_name, f"{add}/output_0"],
            self.io_dtype,
            [1, self.radio_sequence_length, self.radio_hidden_size],
            axis=1,
        )
        return f"{concat}/output_0"

    def _specialized_position_embedding(self):
        position = self.patch_generator.pos_embed.detach().float().reshape(
            1,
            self.patch_generator.num_rows,
            self.patch_generator.num_cols,
            self.radio_hidden_size,
        )
        position = position.permute(0, 3, 1, 2)
        target = (self.patch_rows, self.patch_cols)
        if self.patch_generator.cpe_mode:
            max_dim = max(target)
            position = functional.interpolate(
                position,
                size=(max_dim, max_dim),
                mode="bilinear",
                align_corners=True,
            )
            position = position[..., : self.patch_rows, : self.patch_cols]
        else:
            position = position[..., : self.patch_rows, : self.patch_cols]
            if position.shape[-2:] != target:
                position = functional.interpolate(
                    position,
                    size=target,
                    mode="bilinear",
                    align_corners=True,
                )
        return position.permute(0, 2, 3, 1).reshape(
            1, self.patch_count, self.radio_hidden_size
        )

    def _make_radio_block(self, layer_id, block, hidden_states):
        base = f"/encoder/radio/layers.{layer_id}"
        token_shape = [1, self.radio_sequence_length, self.radio_hidden_size]
        normalized = self._make_layer_norm(block.norm1, f"{base}/norm1", hidden_states, token_shape)
        qkv = self._make_linear(
            block.attn.qkv,
            f"{base}/attn/qkv",
            normalized,
            self.radio_sequence_length,
        )

        qkv_shape_name = f"{base}/attn/qkv_shape"
        self.make_initializer(
            torch.tensor(
                [1, self.radio_sequence_length, 3, self.radio_num_heads, self.radio_head_size],
                dtype=torch.int64,
            ),
            qkv_shape_name,
        )
        qkv_reshape = f"{base}/attn/ReshapeQKV"
        self.make_reshape(
            qkv_reshape,
            [qkv, qkv_shape_name],
            self.io_dtype,
            [1, self.radio_sequence_length, 3, self.radio_num_heads, self.radio_head_size],
        )
        split_name = f"{base}/attn/SplitQKV"
        split_sizes = f"{base}/attn/split_sizes"
        self.make_initializer(torch.tensor([1, 1, 1], dtype=torch.int64), split_sizes)
        split_outputs = [f"{split_name}/q", f"{split_name}/k", f"{split_name}/v"]
        split_shape = [1, self.radio_sequence_length, 1, self.radio_num_heads, self.radio_head_size]
        self.make_split(
            split_name,
            [f"{qkv_reshape}/output_0", split_sizes],
            split_outputs,
            [self.io_dtype] * 3,
            [split_shape] * 3,
            axis=2,
        )
        squeeze_axes = f"{base}/attn/squeeze_axes"
        self.make_initializer(torch.tensor([2], dtype=torch.int64), squeeze_axes)
        qkv_bhsd = []
        for label, value in zip(("q", "k", "v"), split_outputs):
            squeeze = f"{base}/attn/{label}/Squeeze"
            self.make_squeeze(
                squeeze,
                [value, squeeze_axes],
                self.io_dtype,
                [1, self.radio_sequence_length, self.radio_num_heads, self.radio_head_size],
            )
            transpose = f"{base}/attn/{label}/Transpose"
            self.make_transpose(
                transpose,
                f"{squeeze}/output_0",
                self.io_dtype,
                [1, self.radio_num_heads, self.radio_sequence_length, self.radio_head_size],
                [0, 2, 1, 3],
            )
            qkv_bhsd.append(f"{transpose}/output_0")

        scale_name = f"{base}/attn/scale"
        self.make_initializer(
            torch.tensor(math.sqrt(float(block.attn.scale)), dtype=torch.float32),
            scale_name,
            to=self.io_dtype,
        )
        q_scale = f"{base}/attn/q/MulScale"
        self.make_mul(
            q_scale,
            [qkv_bhsd[0], scale_name],
            self.io_dtype,
            [1, self.radio_num_heads, self.radio_sequence_length, self.radio_head_size],
        )
        k_transpose = f"{base}/attn/k/TransposeScores"
        self.make_transpose(
            k_transpose,
            qkv_bhsd[1],
            self.io_dtype,
            [1, self.radio_num_heads, self.radio_head_size, self.radio_sequence_length],
            [0, 1, 3, 2],
        )
        k_scale = f"{base}/attn/k/MulScale"
        self.make_mul(
            k_scale,
            [f"{k_transpose}/output_0", scale_name],
            self.io_dtype,
            [1, self.radio_num_heads, self.radio_head_size, self.radio_sequence_length],
        )
        scores = f"{base}/attn/MatMulScores"
        score_shape = [1, self.radio_num_heads, self.radio_sequence_length, self.radio_sequence_length]
        self.make_node(
            "MatMul",
            inputs=[f"{q_scale}/output_0", f"{k_scale}/output_0"],
            outputs=[f"{scores}/output_0"],
            name=scores,
        )
        self.make_value(f"{scores}/output_0", self.io_dtype, score_shape)
        softmax = f"{base}/attn/Softmax"
        self.make_softmax(softmax, f"{scores}/output_0", self.io_dtype, score_shape, axis=-1)
        context = f"{base}/attn/MatMulContext"
        context_shape = [1, self.radio_num_heads, self.radio_sequence_length, self.radio_head_size]
        self.make_node(
            "MatMul",
            inputs=[f"{softmax}/output_0", qkv_bhsd[2]],
            outputs=[f"{context}/output_0"],
            name=context,
        )
        self.make_value(f"{context}/output_0", self.io_dtype, context_shape)
        context_transpose = f"{base}/attn/TransposeContext"
        self.make_transpose(
            context_transpose,
            f"{context}/output_0",
            self.io_dtype,
            [1, self.radio_sequence_length, self.radio_num_heads, self.radio_head_size],
            [0, 2, 1, 3],
        )
        merge_shape_name = f"{base}/attn/merge_shape"
        self.make_initializer(
            torch.tensor([1, self.radio_sequence_length, self.radio_hidden_size], dtype=torch.int64),
            merge_shape_name,
        )
        merge = f"{base}/attn/ReshapeContext"
        self.make_reshape(
            merge,
            [f"{context_transpose}/output_0", merge_shape_name],
            self.io_dtype,
            token_shape,
        )
        projection = self._make_linear(
            block.attn.proj,
            f"{base}/attn/proj",
            f"{merge}/output_0",
            self.radio_sequence_length,
        )
        attention_residual = f"{base}/AddAttention"
        self.make_add(
            attention_residual,
            [hidden_states, projection],
            self.io_dtype,
            token_shape,
        )

        normalized = self._make_layer_norm(
            block.norm2,
            f"{base}/norm2",
            f"{attention_residual}/output_0",
            token_shape,
        )
        fc1 = self._make_linear(
            block.mlp.fc1,
            f"{base}/mlp/fc1",
            normalized,
            self.radio_sequence_length,
        )
        activation = f"{base}/mlp/Gelu"
        self.make_node(
            "Gelu",
            inputs=[fc1],
            outputs=[f"{activation}/output_0"],
            name=activation,
            approximate="none",
        )
        self.make_value(
            f"{activation}/output_0",
            self.io_dtype,
            [1, self.radio_sequence_length, block.mlp.fc1.out_features],
        )
        fc2 = self._make_linear(
            block.mlp.fc2,
            f"{base}/mlp/fc2",
            f"{activation}/output_0",
            self.radio_sequence_length,
        )
        output = f"{base}/AddMlp"
        self.make_add(
            output,
            [f"{attention_residual}/output_0", fc2],
            self.io_dtype,
            token_shape,
        )
        return f"{output}/output_0"

    def _make_neck(self, hidden_states):
        base = "/encoder/neck"
        feature_starts = f"{base}/feature_starts"
        feature_ends = f"{base}/feature_ends"
        feature_axes = f"{base}/feature_axes"
        self.make_initializer(torch.tensor([self.prefix_count], dtype=torch.int64), feature_starts)
        self.make_initializer(torch.tensor([self.radio_sequence_length], dtype=torch.int64), feature_ends)
        self.make_initializer(torch.tensor([1], dtype=torch.int64), feature_axes)
        feature_slice = f"{base}/SliceFeatures"
        self.make_slice(
            feature_slice,
            [hidden_states, feature_starts, feature_ends, feature_axes],
            self.io_dtype,
            [1, self.patch_count, self.radio_hidden_size],
        )

        summary_indices = self.radio.summary_idxs.to(dtype=torch.int64)
        summary_indices_name = f"{base}/summary_indices"
        self.make_initializer(summary_indices, summary_indices_name)
        summary_gather = f"{base}/GatherSummary"
        self.make_gather(
            summary_gather,
            [hidden_states, summary_indices_name],
            self.io_dtype,
            [1, int(summary_indices.numel()), self.radio_hidden_size],
            axis=1,
        )
        summary_shape = f"{base}/summary_shape"
        summary_width = int(summary_indices.numel()) * self.radio_hidden_size
        self.make_initializer(torch.tensor([1, 1, summary_width], dtype=torch.int64), summary_shape)
        summary_reshape = f"{base}/ReshapeSummary"
        self.make_reshape(
            summary_reshape,
            [f"{summary_gather}/output_0", summary_shape],
            self.io_dtype,
            [1, 1, summary_width],
        )

        conv1_linear = types.SimpleNamespace(
            weight=self.encoder.conv1.weight.squeeze(-1),
            bias=self.encoder.conv1.bias,
        )
        projected = self._make_linear(
            conv1_linear,
            f"{base}/conv1",
            f"{feature_slice}/output_0",
            self.patch_count,
        )
        projected = self._make_layer_norm(
            self.encoder.layer_norm1,
            f"{base}/layer_norm1",
            projected,
            [1, self.patch_count, self.source_config.decoder.d_model],
        )
        grid_shape = f"{base}/grid_shape"
        self.make_initializer(
            torch.tensor(
                [1, self.patch_rows, self.patch_cols, self.source_config.decoder.d_model],
                dtype=torch.int64,
            ),
            grid_shape,
        )
        grid_reshape = f"{base}/ReshapeGrid"
        self.make_reshape(
            grid_reshape,
            [projected, grid_shape],
            self.io_dtype,
            [1, self.patch_rows, self.patch_cols, self.source_config.decoder.d_model],
        )
        grid_transpose = f"{base}/TransposeGrid"
        self.make_transpose(
            grid_transpose,
            f"{grid_reshape}/output_0",
            self.io_dtype,
            [1, self.source_config.decoder.d_model, self.patch_rows, self.patch_cols],
            [0, 3, 1, 2],
        )
        conv2_weight = f"{base[1:].replace('/', '.')}.conv2.weight"
        self.make_initializer(self.encoder.conv2.weight, conv2_weight, to=self.io_dtype)
        conv2 = f"{base}/conv2/Conv"
        compressed_cols = (self.patch_cols - 4) // 4 + 1
        self.make_conv(
            conv2,
            [f"{grid_transpose}/output_0", conv2_weight],
            self.io_dtype,
            [1, self.source_config.decoder.d_model, self.patch_rows, compressed_cols],
            kernel_shape=[1, 4],
            strides=[1, 4],
        )
        compressed_transpose = f"{base}/TransposeCompressed"
        self.make_transpose(
            compressed_transpose,
            f"{conv2}/output_0",
            self.io_dtype,
            [1, self.patch_rows, compressed_cols, self.source_config.decoder.d_model],
            [0, 2, 3, 1],
        )
        compressed_shape = f"{base}/compressed_shape"
        compressed_count = self.patch_rows * compressed_cols
        self.make_initializer(
            torch.tensor([1, compressed_count, self.source_config.decoder.d_model], dtype=torch.int64),
            compressed_shape,
        )
        compressed_reshape = f"{base}/ReshapeCompressed"
        self.make_reshape(
            compressed_reshape,
            [f"{compressed_transpose}/output_0", compressed_shape],
            self.io_dtype,
            [1, compressed_count, self.source_config.decoder.d_model],
        )
        compressed = self._make_layer_norm(
            self.encoder.layer_norm2,
            f"{base}/layer_norm2",
            f"{compressed_reshape}/output_0",
            [1, compressed_count, self.source_config.decoder.d_model],
        )

        summary = self._make_linear(
            self.encoder.sum_proj,
            f"{base}/sum_proj",
            f"{summary_reshape}/output_0",
            1,
        )
        summary = self._make_layer_norm(
            self.encoder.layer_norm3,
            f"{base}/layer_norm3",
            summary,
            [1, 1, self.source_config.decoder.d_model],
        )
        concat = f"{base}/ConcatOutput"
        self.make_node(
            "Concat",
            inputs=[compressed, summary],
            outputs=["encoder_hidden_states"],
            name=concat,
            axis=1,
        )
        self.make_value(
            "encoder_hidden_states",
            self.io_dtype,
            [1, self.encoder_sequence_length, self.source_config.decoder.d_model],
        )
        return "encoder_hidden_states"

    def _make_cross_cache(self, encoder_hidden_states):
        decoder_heads = self.source_config.decoder.decoder_attention_heads
        decoder_head_size = self.source_config.decoder.d_model // decoder_heads
        reshape_shape_name = "/encoder/cross_cache/reshape_shape"
        self.make_initializer(
            torch.tensor(
                [1, self.encoder_sequence_length, decoder_heads, decoder_head_size],
                dtype=torch.int64,
            ),
            reshape_shape_name,
        )
        for layer_id, layer in enumerate(self.decoder.layers):
            for kind, projection in (
                ("key", layer.encoder_attn.k_proj),
                ("value", layer.encoder_attn.v_proj),
            ):
                base = f"/encoder/cross_cache/layers.{layer_id}/{kind}"
                projected = self._make_linear(
                    projection,
                    f"{base}/proj",
                    encoder_hidden_states,
                    self.encoder_sequence_length,
                )
                reshape = f"{base}/Reshape"
                self.make_reshape(
                    reshape,
                    [projected, reshape_shape_name],
                    self.io_dtype,
                    [1, self.encoder_sequence_length, decoder_heads, decoder_head_size],
                )
                output = f"cross_present.{layer_id}.{kind}"
                self.make_node(
                    "Transpose",
                    inputs=[f"{reshape}/output_0"],
                    outputs=[output],
                    name=f"{base}/Transpose",
                    perm=[0, 2, 1, 3],
                )
                self.make_value(
                    output,
                    self.io_dtype,
                    [1, decoder_heads, self.encoder_sequence_length, decoder_head_size],
                )
