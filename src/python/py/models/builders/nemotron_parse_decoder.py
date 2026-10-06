# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import math

import onnx_ir as ir
import torch

from .base import Model


def make_decoder_config(config, cache_sequence_length):
    decoder_config = copy.deepcopy(config.decoder)
    decoder_config._name_or_path = config._name_or_path
    decoder_config.architectures = config.architectures
    decoder_config.num_hidden_layers = decoder_config.decoder_layers
    decoder_config.num_key_value_heads = decoder_config.decoder_attention_heads
    decoder_config.num_attention_heads = decoder_config.decoder_attention_heads
    decoder_config.hidden_size = decoder_config.d_model
    decoder_config.intermediate_size = decoder_config.decoder_ffn_dim
    decoder_config.hidden_act = decoder_config.activation_function
    decoder_config.max_position_embeddings = cache_sequence_length
    decoder_config.rms_norm_eps = getattr(decoder_config, "layer_norm_eps", None) or 1e-5
    if hasattr(decoder_config, "quantization_config") and decoder_config.quantization_config is None:
        delattr(decoder_config, "quantization_config")
    return decoder_config


class NemotronParseDecoderComponent(Model):
    """Build the unified Nemotron Parse mBART prefill/decode graph."""

    def __init__(
        self,
        config,
        io_dtype,
        onnx_dtype,
        ep,
        cache_dir,
        extra_options,
        *,
        encoder_sequence_length,
        cache_sequence_length,
    ):
        self.encoder_sequence_length = int(encoder_sequence_length)
        self.cache_sequence_length = int(cache_sequence_length)
        self.sequence_length = "sequence_length"

        component_options = copy.deepcopy(extra_options)
        component_options["filename"] = "decoder.onnx"
        component_options["prune_lm_head"] = True
        component_options["shared_embeddings"] = False
        super().__init__(
            make_decoder_config(config, self.cache_sequence_length),
            io_dtype,
            onnx_dtype,
            ep,
            cache_dir,
            component_options,
        )

        self.graph.name = "nemotron_parse_decoder"

        self.output_shapes["logits"] = [
            1,
            1,
            self.vocab_size,
        ]

    def make_value(self, name, dtype=None, shape=None):
        # Shared helpers use symbolic dimensions even though this component fixes them.
        if shape is not None:
            fixed_dims = {"batch_size": 1, "encoder_sequence_length": self.encoder_sequence_length}
            shape = [fixed_dims.get(dim, dim) for dim in shape]
        return super().make_value(name, dtype, shape)

    def is_gqa_supported(self):
        # This component emits the checkpoint's primitive mBART attention graph.
        # Provider-side fusion can then recognize the same graph in both phases.
        return False

    def is_packed_attn_supported(self):
        return False

    def load_weights(self, input_path):
        raise RuntimeError(
            "NemotronParseDecoderComponent receives weights from its parent builder"
        )

    def build(self, weights):
        self.weights = weights
        self.make_inputs_and_outputs()
        self.make_constants()

        hidden_states = self.make_decoder_embedding(weights.decoder)
        attention_mask = self.make_attention_mask()
        for layer_id, layer in enumerate(weights.decoder.layers):
            hidden_states = self.make_decoder_layer(
                layer_id, layer, hidden_states, attention_mask
            )

        hidden_states = self.make_layer_norm(
            weights.decoder.layer_norm,
            "/model/layer_norm",
            hidden_states,
            self.sequence_length,
        )
        self.layernorm_attrs["output_0"] = hidden_states
        self.make_lm_head(weights.lm_head)

    def make_inputs_and_outputs(self):
        batch = 1
        encoder_sequence = self.encoder_sequence_length
        head_shape = [
            batch,
            self.num_attn_heads,
            self.cache_sequence_length,
            self.head_size,
        ]
        cross_shape = [
            batch,
            self.num_attn_heads,
            encoder_sequence,
            self.head_size,
        ]

        self.graph.inputs.append(
            self.make_value(
                "decoder_input_ids",
                ir.DataType.INT64,
                [batch, self.sequence_length],
            )
        )
        self.graph.inputs.append(
            self.make_value(
                "decoder_attention_mask",
                ir.DataType.INT64,
                [batch, self.cache_sequence_length],
            )
        )
        for layer_id in range(self.num_layers):
            for name, shape in (
                (f"past_key_values.{layer_id}.key", head_shape),
                (f"past_key_values.{layer_id}.value", head_shape),
                (f"cross_past_key_values.{layer_id}.key", cross_shape),
                (f"cross_past_key_values.{layer_id}.value", cross_shape),
            ):
                self.graph.inputs.append(
                    self.make_value(name, self.io_dtype, shape)
                )
        self.graph.inputs.append(
            self.make_value(
                "cache_write_indices",
                ir.DataType.INT64,
                [batch],
            )
        )

        self.graph.outputs.append(
            self.make_value(
                "logits", self.output_types["logits"], self.output_shapes["logits"]
            )
        )
        for layer_id in range(self.num_layers):
            for name, shape in (
                (f"present.{layer_id}.key", head_shape),
                (f"present.{layer_id}.value", head_shape),
            ):
                self.graph.outputs.append(
                    self.make_value(name, self.io_dtype, shape)
                )

    def make_constants(self):
        torch_dtype = {
            ir.DataType.FLOAT16: torch.float16,
            ir.DataType.BFLOAT16: torch.bfloat16,
            ir.DataType.FLOAT: torch.float32,
        }.get(self.io_dtype)
        if torch_dtype is None:
            raise ValueError(
                "Nemotron Parse decoder inputs must use float, float16, or bfloat16"
            )
        mask_value = float(torch.finfo(torch_dtype).min)
        prefix = "/model/constants"
        dtype = self.to_str_dtype(self.io_dtype)
        self.constant_names = {
            "reshape_heads": f"{prefix}/INT64/[0, 0, {self.num_attn_heads}, {self.head_size}]",
            "merge_heads": f"{prefix}/INT64/[0, 0, {self.hidden_size}]",
            "mask_axes": f"{prefix}/INT64/[1, 2]",
            "mask_zero": f"{prefix}/INT64/0",
            "mask_value": f"{prefix}/{dtype}/{mask_value}",
            "float_zero": f"{prefix}/{dtype}/0.0",
            "attention_scale": f"{prefix}/{dtype}/{1.0 / math.sqrt(self.head_size)}",
            "shape_index": f"{prefix}/INT64/1",
            "range_start": f"{prefix}/INT64/0",
            "range_step": f"{prefix}/INT64/1",
            "write_index_axes": f"{prefix}/INT64/[1]",
            "query_position_axes": f"{prefix}/INT64/[1, 3]",
            "key_positions": "/model/attention_mask/key_positions",
        }
        self.make_initializer(
            torch.arange(
                self.cache_sequence_length, dtype=torch.int64
            ).reshape(1, 1, 1, self.cache_sequence_length),
            self.constant_names["key_positions"],
        )

    def make_attention_mask(self):
        key_length = self.cache_sequence_length
        base = "/model/attention_mask"
        equal = f"{base}/padding/Equal"
        self.make_equal(
            equal,
            ["decoder_attention_mask", self.constant_names["mask_zero"]],
            ["batch_size", key_length],
        )
        unsqueeze = f"{base}/Unsqueeze"
        self.make_unsqueeze(
            unsqueeze,
            [f"{equal}/output_0", self.constant_names["mask_axes"]],
            ir.DataType.BOOL,
            ["batch_size", 1, 1, key_length],
        )

        input_shape = f"{base}/input/Shape"
        self.make_shape(input_shape, "decoder_input_ids", [2])
        sequence_length = f"{base}/sequence_length/Gather"
        self.make_gather(
            sequence_length,
            [f"{input_shape}/output_0", self.constant_names["shape_index"]],
            ir.DataType.INT64,
            [],
            axis=0,
        )
        query_offsets = f"{base}/query_offsets/Range"
        self.make_range(
            query_offsets,
            [
                self.constant_names["range_start"],
                f"{sequence_length}/output_0",
                self.constant_names["range_step"],
            ],
            ir.DataType.INT64,
            [self.sequence_length],
        )
        write_indices = f"{base}/write_indices/Unsqueeze"
        self.make_unsqueeze(
            write_indices,
            [
                "cache_write_indices",
                self.constant_names["write_index_axes"],
            ],
            ir.DataType.INT64,
            ["batch_size", 1],
        )
        query_positions = f"{base}/query_positions/Add"
        self.make_add(
            query_positions,
            [f"{write_indices}/output_0", f"{query_offsets}/output_0"],
            ir.DataType.INT64,
            ["batch_size", self.sequence_length],
        )
        query_positions_4d = f"{base}/query_positions/Unsqueeze"
        self.make_unsqueeze(
            query_positions_4d,
            [
                f"{query_positions}/output_0",
                self.constant_names["query_position_axes"],
            ],
            ir.DataType.INT64,
            ["batch_size", 1, self.sequence_length, 1],
        )
        causal = f"{base}/causal/Greater"
        self.make_greater(
            causal,
            [
                self.constant_names["key_positions"],
                f"{query_positions_4d}/output_0",
            ],
            ["batch_size", 1, self.sequence_length, key_length],
        )
        invalid = f"{base}/padding_and_causal/Or"
        self.make_node(
            "Or",
            inputs=[f"{unsqueeze}/output_0", f"{causal}/output_0"],
            outputs=[f"{invalid}/output_0"],
            name=invalid,
        )
        self.make_value(
            f"{invalid}/output_0",
            ir.DataType.BOOL,
            ["batch_size", 1, self.sequence_length, key_length],
        )
        where = f"{base}/Where"
        self.make_where(
            where,
            [
                f"{invalid}/output_0",
                self.constant_names["mask_value"],
                self.constant_names["float_zero"],
            ],
            self.io_dtype,
            ["batch_size", 1, self.sequence_length, key_length],
        )
        return f"{where}/output_0"

    def make_decoder_embedding(self, decoder):
        base = "/model/embed_tokens"
        weight = "model.embed_tokens.weight"
        self.make_initializer(decoder.embed_tokens.weight, weight, to=self.io_dtype)
        gather = f"{base}/Gather"
        self.make_gather(
            gather,
            [weight, "decoder_input_ids"],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
            axis=0,
        )
        hidden_states = f"{gather}/output_0"

        embed_scale = float(getattr(decoder.embed_tokens, "embed_scale", 1.0))
        if embed_scale != 1.0:
            scale = f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{embed_scale}"
            mul = f"{base}/Mul"
            self.make_mul(
                mul,
                [hidden_states, scale],
                self.io_dtype,
                ["batch_size", self.sequence_length, self.hidden_size],
            )
            hidden_states = f"{mul}/output_0"

        return self.make_layer_norm(
            decoder.layernorm_embedding,
            "/model/layernorm_embedding",
            hidden_states,
            self.sequence_length,
        )

    def make_layer_norm(self, layer_norm, name, root_input, sequence_length):
        weight = f"{name[1:].replace('/', '.')}.weight"
        bias = f"{name[1:].replace('/', '.')}.bias"
        self.make_initializer(layer_norm.weight, weight, to=self.io_dtype)
        self.make_initializer(layer_norm.bias, bias, to=self.io_dtype)
        name = f"{name}/LayerNormalization"
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
        self.make_value(
            output,
            self.io_dtype,
            ["batch_size", sequence_length, self.hidden_size],
        )
        return output

    def make_linear(self, linear, name, root_input, sequence_length):
        matmul = self.make_matmul(
            linear,
            f"{name}/MatMul",
            root_input,
            seq_dim=sequence_length,
        )
        output = f"{matmul}/output_0"
        if linear.bias is not None:
            add = f"{name}/Add"
            self.make_add_bias(
                linear.bias,
                add,
                root_input=output,
                seq_dim=sequence_length,
            )
            output = f"{add}/output_0"
        return output

    def split_heads(
        self,
        root_input,
        name,
        sequence_length,
        *,
        output=None,
    ):
        reshape = f"{name}/Reshape"
        self.make_reshape(
            reshape,
            [root_input, self.constant_names["reshape_heads"]],
            self.io_dtype,
            [
                "batch_size",
                sequence_length,
                self.num_attn_heads,
                self.head_size,
            ],
        )
        transpose = f"{name}/Transpose"
        transpose_output = output or f"{transpose}/output_0"
        self.make_node(
            "Transpose",
            inputs=[f"{reshape}/output_0"],
            outputs=[transpose_output],
            name=transpose,
            perm=[0, 2, 1, 3],
        )
        self.make_value(
            transpose_output,
            self.io_dtype,
            [
                "batch_size",
                self.num_attn_heads,
                sequence_length,
                self.head_size,
            ],
        )
        return transpose_output

    def merge_heads(self, root_input, name, sequence_length):
        transpose = f"{name}/Transpose"
        self.make_transpose(
            transpose,
            root_input,
            self.io_dtype,
            [
                "batch_size",
                sequence_length,
                self.num_attn_heads,
                self.head_size,
            ],
            [0, 2, 1, 3],
        )
        reshape = f"{name}/Reshape"
        self.make_reshape(
            reshape,
            [f"{transpose}/output_0", self.constant_names["merge_heads"]],
            self.io_dtype,
            ["batch_size", sequence_length, self.hidden_size],
        )
        return f"{reshape}/output_0"

    def make_scaled_dot_product_attention(
        self,
        name,
        query,
        key,
        value,
        query_length,
        key_length,
        attention_mask=None,
    ):
        key_transpose = f"{name}/key/Transpose"
        self.make_transpose(
            key_transpose,
            key,
            self.io_dtype,
            [
                "batch_size",
                self.num_attn_heads,
                self.head_size,
                key_length,
            ],
            [0, 1, 3, 2],
        )
        scores = f"{name}/scores/MatMul"
        self.make_node(
            "MatMul",
            inputs=[query, f"{key_transpose}/output_0"],
            outputs=[f"{scores}/output_0"],
            name=scores,
        )
        scores_shape = [
            "batch_size",
            self.num_attn_heads,
            query_length,
            key_length,
        ]
        self.make_value(f"{scores}/output_0", self.io_dtype, scores_shape)

        scale = f"{name}/scores/Mul"
        self.make_mul(
            scale,
            [f"{scores}/output_0", self.constant_names["attention_scale"]],
            self.io_dtype,
            scores_shape,
        )
        softmax_input = f"{scale}/output_0"
        if attention_mask is not None:
            add_mask = f"{name}/scores/mask/Add"
            self.make_add(
                add_mask,
                [softmax_input, attention_mask],
                self.io_dtype,
                scores_shape,
            )
            softmax_input = f"{add_mask}/output_0"

        softmax = f"{name}/Softmax"
        self.make_softmax(
            softmax,
            softmax_input,
            self.io_dtype,
            scores_shape,
            axis=-1,
        )
        context = f"{name}/context/MatMul"
        self.make_node(
            "MatMul",
            inputs=[f"{softmax}/output_0", value],
            outputs=[f"{context}/output_0"],
            name=context,
        )
        self.make_value(
            f"{context}/output_0",
            self.io_dtype,
            [
                "batch_size",
                self.num_attn_heads,
                query_length,
                self.head_size,
            ],
        )
        return f"{context}/output_0"

    def make_self_attention(
        self, layer_id, attention, root_input, attention_mask
    ):
        base = f"/model/layers.{layer_id}/attn"
        query = self.split_heads(
            self.make_linear(
                attention.q_proj,
                f"{base}/q_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/q",
            self.sequence_length,
        )
        key_update = self.split_heads(
            self.make_linear(
                attention.k_proj,
                f"{base}/k_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/k",
            self.sequence_length,
        )
        value_update = self.split_heads(
            self.make_linear(
                attention.v_proj,
                f"{base}/v_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/v",
            self.sequence_length,
        )
        cache_shape = [
            "batch_size",
            self.num_attn_heads,
            self.cache_sequence_length,
            self.head_size,
        ]
        key = self.make_tensor_scatter(
            f"{base}/key/TensorScatter",
            f"past_key_values.{layer_id}.key",
            key_update,
            "cache_write_indices",
            self.io_dtype,
            cache_shape,
            output=f"present.{layer_id}.key",
        )
        value = self.make_tensor_scatter(
            f"{base}/value/TensorScatter",
            f"past_key_values.{layer_id}.value",
            value_update,
            "cache_write_indices",
            self.io_dtype,
            cache_shape,
            output=f"present.{layer_id}.value",
        )

        context = self.make_scaled_dot_product_attention(
            base,
            query,
            key,
            value,
            self.sequence_length,
            self.cache_sequence_length,
            attention_mask,
        )
        merged = self.merge_heads(
            context, f"{base}/merge_heads", self.sequence_length
        )
        return self.make_linear(
            attention.out_proj,
            f"{base}/o_proj",
            merged,
            self.sequence_length,
        )

    def make_cross_attention(self, layer_id, attention, root_input):
        base = f"/model/layers.{layer_id}/cross_attn"
        query = self.split_heads(
            self.make_linear(
                attention.q_proj,
                f"{base}/q_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/q",
            self.sequence_length,
        )
        key = f"cross_past_key_values.{layer_id}.key"
        value = f"cross_past_key_values.{layer_id}.value"

        context = self.make_scaled_dot_product_attention(
            base,
            query,
            key,
            value,
            self.sequence_length,
            self.encoder_sequence_length,
        )
        merged = self.merge_heads(
            context, f"{base}/merge_heads", self.sequence_length
        )
        return self.make_linear(
            attention.out_proj,
            f"{base}/o_proj",
            merged,
            self.sequence_length,
        )

    def make_decoder_layer(self, layer_id, layer, hidden_states, attention_mask):
        base = f"/model/layers.{layer_id}"
        self_norm = self.make_layer_norm(
            layer.self_attn_layer_norm,
            f"{base}/self_attn_layer_norm",
            hidden_states,
            self.sequence_length,
        )
        self_attention = self.make_self_attention(
            layer_id, layer.self_attn, self_norm, attention_mask
        )
        self_residual = f"{base}/attn/residual/Add"
        self.make_add(
            self_residual,
            [hidden_states, self_attention],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )

        cross_norm = self.make_layer_norm(
            layer.encoder_attn_layer_norm,
            f"{base}/encoder_attn_layer_norm",
            f"{self_residual}/output_0",
            self.sequence_length,
        )
        cross_attention = self.make_cross_attention(
            layer_id, layer.encoder_attn, cross_norm
        )
        cross_residual = f"{base}/cross_attn/residual/Add"
        self.make_add(
            cross_residual,
            [f"{self_residual}/output_0", cross_attention],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )

        final_norm = self.make_layer_norm(
            layer.final_layer_norm,
            f"{base}/final_layer_norm",
            f"{cross_residual}/output_0",
            self.sequence_length,
        )
        fc1 = self.make_linear(
            layer.fc1, f"{base}/mlp/fc1", final_norm, self.sequence_length
        )
        if self.activation != "gelu":
            raise ValueError(
                "Nemotron Parse builder currently supports the checkpoint's gelu MLP"
            )
        activation = f"{base}/mlp/activation_fn/Gelu"
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
            ["batch_size", self.sequence_length, self.intermediate_size],
        )
        fc2 = self.make_linear(
            layer.fc2,
            f"{base}/mlp/fc2",
            f"{activation}/output_0",
            self.sequence_length,
        )
        output = f"{base}/mlp/residual/Add"
        self.make_add(
            output,
            [f"{cross_residual}/output_0", fc2],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )
        return f"{output}/output_0"
