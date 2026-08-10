# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import math

import onnx_ir as ir
import torch

from .base import Model


class NemotronParseDecoderComponent(Model):
    """Build one Nemotron Parse mBART decoder phase with the common ONNX IR infra."""

    # ORT shape inference needs static Reshape inputs in the model protobuf.
    # Keep small graph constants embedded while externalizing decoder weights.
    external_data_size_threshold_bytes = 1024

    def __init__(
        self,
        config,
        io_dtype,
        onnx_dtype,
        ep,
        cache_dir,
        extra_options,
        *,
        phase,
        encoder_sequence_length,
        prefill_sequence_length,
        cache_sequence_length,
    ):
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"Unsupported Nemotron Parse decoder phase: {phase}")

        self.phase = phase
        self.encoder_sequence_length = int(encoder_sequence_length)
        self.prefill_sequence_length = int(prefill_sequence_length)
        self.cache_sequence_length = int(cache_sequence_length)
        self.sequence_length = (
            self.prefill_sequence_length if phase == "prefill" else 1
        )

        decoder_config = copy.deepcopy(config.decoder)
        decoder_config._name_or_path = config._name_or_path
        decoder_config.architectures = config.architectures
        decoder_config.num_hidden_layers = decoder_config.decoder_layers
        decoder_config.num_key_value_heads = (
            decoder_config.decoder_attention_heads
        )
        decoder_config.num_attention_heads = (
            decoder_config.decoder_attention_heads
        )
        decoder_config.hidden_size = decoder_config.d_model
        decoder_config.intermediate_size = decoder_config.decoder_ffn_dim
        decoder_config.hidden_act = decoder_config.activation_function
        decoder_config.max_position_embeddings = self.cache_sequence_length
        decoder_config.rms_norm_eps = (
            getattr(decoder_config, "layer_norm_eps", None) or 1e-5
        )
        if (
            hasattr(decoder_config, "quantization_config")
            and decoder_config.quantization_config is None
        ):
            delattr(decoder_config, "quantization_config")

        component_options = copy.deepcopy(extra_options)
        component_options["filename"] = (
            "decoder_prefill.onnx" if phase == "prefill" else "decoder.onnx"
        )
        component_options["prune_lm_head"] = phase == "prefill"
        component_options["shared_embeddings"] = False
        super().__init__(
            decoder_config,
            io_dtype,
            onnx_dtype,
            ep,
            cache_dir,
            component_options,
        )

        self.graph.name = f"nemotron_parse_decoder_{phase}"
        if phase == "decode":
            self.graph.opset_imports[""] = 24

        self.output_shapes["logits"] = [
            "batch_size",
            1 if phase == "prefill" else self.sequence_length,
            self.vocab_size,
        ]

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
        self._make_inputs_and_outputs()
        self._make_constants()

        hidden_states = self._make_embedding(weights.decoder)
        attention_mask = self._make_attention_mask()
        for layer_id, layer in enumerate(weights.decoder.layers):
            hidden_states = self._make_layer(
                layer_id, layer, hidden_states, attention_mask
            )

        hidden_states = self._make_layer_norm(
            weights.decoder.layer_norm,
            "/decoder/layer_norm",
            hidden_states,
            self.sequence_length,
        )
        self.layernorm_attrs["output_0"] = hidden_states
        self.make_lm_head(weights.lm_head)

    def _make_inputs_and_outputs(self):
        batch = "batch_size"
        encoder_sequence = "encoder_sequence_length"
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
        mask_length = (
            self.prefill_sequence_length
            if self.phase == "prefill"
            else self.cache_sequence_length
        )
        self.graph.inputs.append(
            self.make_value(
                "decoder_attention_mask",
                ir.DataType.INT64,
                [batch, mask_length],
            )
        )
        if self.phase == "prefill":
            self.graph.inputs.append(
                self.make_value(
                    "encoder_hidden_states",
                    self.io_dtype,
                    [batch, encoder_sequence, self.hidden_size],
                )
            )
        else:
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
        self_cache_shape = (
            [
                batch,
                self.num_attn_heads,
                self.prefill_sequence_length,
                self.head_size,
            ]
            if self.phase == "prefill"
            else head_shape
        )
        for layer_id in range(self.num_layers):
            for name, shape in (
                (f"present.{layer_id}.key", self_cache_shape),
                (f"present.{layer_id}.value", self_cache_shape),
            ):
                self.graph.outputs.append(
                    self.make_value(name, self.io_dtype, shape)
                )
            if self.phase == "prefill":
                for name in (
                    f"cross_present.{layer_id}.key",
                    f"cross_present.{layer_id}.value",
                ):
                    self.graph.outputs.append(
                        self.make_value(name, self.io_dtype, cross_shape)
                    )

    def _make_constants(self):
        prefix = "/nemotron_parse/constants"
        self._constants = {
            "reshape_heads": f"{prefix}/reshape_heads",
            "merge_heads": f"{prefix}/merge_heads",
            "mask_axes": f"{prefix}/mask_axes",
            "mask_zero": f"{prefix}/mask_zero",
            "mask_value": f"{prefix}/mask_value",
            "float_zero": f"{prefix}/float_zero",
            "attention_scale": f"{prefix}/attention_scale",
        }
        self.make_initializer(
            torch.tensor(
                [0, 0, self.num_attn_heads, self.head_size],
                dtype=torch.int64,
            ),
            self._constants["reshape_heads"],
        )
        self.make_initializer(
            torch.tensor([0, 0, self.hidden_size], dtype=torch.int64),
            self._constants["merge_heads"],
        )
        self.make_initializer(
            torch.tensor([1, 2], dtype=torch.int64),
            self._constants["mask_axes"],
        )
        self.make_initializer(
            torch.tensor(0, dtype=torch.int64),
            self._constants["mask_zero"],
        )
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
        self.make_initializer(
            torch.tensor(mask_value, dtype=torch.float32),
            self._constants["mask_value"],
            to=self.io_dtype,
        )
        self.make_initializer(
            torch.tensor(0.0, dtype=torch.float32),
            self._constants["float_zero"],
            to=self.io_dtype,
        )
        self.make_initializer(
            torch.tensor(
                1.0 / math.sqrt(self.head_size), dtype=torch.float32
            ),
            self._constants["attention_scale"],
            to=self.io_dtype,
        )

        if self.phase == "prefill":
            causal = torch.zeros(
                (
                    1,
                    1,
                    self.prefill_sequence_length,
                    self.prefill_sequence_length,
                ),
                dtype=torch.float32,
            )
            upper = torch.triu(
                torch.ones_like(causal, dtype=torch.bool), diagonal=1
            )
            causal.masked_fill_(upper, mask_value)
            self._constants["causal_mask"] = f"{prefix}/causal_mask"
            self.make_initializer(
                causal, self._constants["causal_mask"], to=self.io_dtype
            )

    def _make_attention_mask(self):
        key_length = (
            self.prefill_sequence_length
            if self.phase == "prefill"
            else self.cache_sequence_length
        )
        base = "/decoder/attention_mask"
        equal = f"{base}/Equal"
        self.make_equal(
            equal,
            ["decoder_attention_mask", self._constants["mask_zero"]],
            ["batch_size", key_length],
        )
        unsqueeze = f"{base}/Unsqueeze"
        self.make_unsqueeze(
            unsqueeze,
            [f"{equal}/output_0", self._constants["mask_axes"]],
            ir.DataType.BOOL,
            ["batch_size", 1, 1, key_length],
        )
        where = f"{base}/Where"
        self.make_where(
            where,
            [
                f"{unsqueeze}/output_0",
                self._constants["mask_value"],
                self._constants["float_zero"],
            ],
            self.io_dtype,
            ["batch_size", 1, 1, key_length],
        )
        if self.phase == "decode":
            return f"{where}/output_0"

        add = f"{base}/AddCausal"
        self.make_add(
            add,
            [f"{where}/output_0", self._constants["causal_mask"]],
            self.io_dtype,
            [
                "batch_size",
                1,
                self.prefill_sequence_length,
                self.prefill_sequence_length,
            ],
        )
        return f"{add}/output_0"

    def _make_embedding(self, decoder):
        base = "/decoder/embed_tokens"
        weight = "decoder.embed_tokens.weight"
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
            scale = f"{base}/scale"
            self.make_initializer(
                torch.tensor(embed_scale, dtype=torch.float32),
                scale,
                to=self.io_dtype,
            )
            mul = f"{base}/Mul"
            self.make_mul(
                mul,
                [hidden_states, scale],
                self.io_dtype,
                ["batch_size", self.sequence_length, self.hidden_size],
            )
            hidden_states = f"{mul}/output_0"

        return self._make_layer_norm(
            decoder.layernorm_embedding,
            "/decoder/layernorm_embedding",
            hidden_states,
            self.sequence_length,
        )

    def _make_layer_norm(self, layer_norm, name, root_input, sequence_length):
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
        self.make_value(
            output,
            self.io_dtype,
            ["batch_size", sequence_length, self.hidden_size],
        )
        return output

    def _make_linear(self, linear, name, root_input, sequence_length):
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

    def _split_heads(
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
            [root_input, self._constants["reshape_heads"]],
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

    def _merge_heads(self, root_input, name, sequence_length):
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
            [f"{transpose}/output_0", self._constants["merge_heads"]],
            self.io_dtype,
            ["batch_size", sequence_length, self.hidden_size],
        )
        return f"{reshape}/output_0"

    def _make_scaled_dot_product_attention(
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
            [f"{scores}/output_0", self._constants["attention_scale"]],
            self.io_dtype,
            scores_shape,
        )
        softmax_input = f"{scale}/output_0"
        if attention_mask is not None:
            add_mask = f"{name}/scores/AddMask"
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

    def _make_self_attention(
        self, layer_id, attention, root_input, attention_mask
    ):
        base = f"/decoder/layers.{layer_id}/self_attn"
        query = self._split_heads(
            self._make_linear(
                attention.q_proj,
                f"{base}/q_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/q",
            self.sequence_length,
        )
        key_update = self._split_heads(
            self._make_linear(
                attention.k_proj,
                f"{base}/k_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/k",
            self.sequence_length,
            output=(
                f"present.{layer_id}.key"
                if self.phase == "prefill"
                else None
            ),
        )
        value_update = self._split_heads(
            self._make_linear(
                attention.v_proj,
                f"{base}/v_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/v",
            self.sequence_length,
            output=(
                f"present.{layer_id}.value"
                if self.phase == "prefill"
                else None
            ),
        )

        if self.phase == "decode":
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
            key_length = self.cache_sequence_length
        else:
            key = key_update
            value = value_update
            key_length = self.prefill_sequence_length

        context = self._make_scaled_dot_product_attention(
            base,
            query,
            key,
            value,
            self.sequence_length,
            key_length,
            attention_mask,
        )
        merged = self._merge_heads(
            context, f"{base}/merge_heads", self.sequence_length
        )
        return self._make_linear(
            attention.out_proj,
            f"{base}/out_proj",
            merged,
            self.sequence_length,
        )

    def _make_cross_attention(self, layer_id, attention, root_input):
        base = f"/decoder/layers.{layer_id}/encoder_attn"
        query = self._split_heads(
            self._make_linear(
                attention.q_proj,
                f"{base}/q_proj",
                root_input,
                self.sequence_length,
            ),
            f"{base}/q",
            self.sequence_length,
        )
        if self.phase == "prefill":
            key = self._split_heads(
                self._make_linear(
                    attention.k_proj,
                    f"{base}/k_proj",
                    "encoder_hidden_states",
                    "encoder_sequence_length",
                ),
                f"{base}/k",
                "encoder_sequence_length",
                output=f"cross_present.{layer_id}.key",
            )
            value = self._split_heads(
                self._make_linear(
                    attention.v_proj,
                    f"{base}/v_proj",
                    "encoder_hidden_states",
                    "encoder_sequence_length",
                ),
                f"{base}/v",
                "encoder_sequence_length",
                output=f"cross_present.{layer_id}.value",
            )
        else:
            key = f"cross_past_key_values.{layer_id}.key"
            value = f"cross_past_key_values.{layer_id}.value"

        context = self._make_scaled_dot_product_attention(
            base,
            query,
            key,
            value,
            self.sequence_length,
            "encoder_sequence_length",
        )
        merged = self._merge_heads(
            context, f"{base}/merge_heads", self.sequence_length
        )
        return self._make_linear(
            attention.out_proj,
            f"{base}/out_proj",
            merged,
            self.sequence_length,
        )

    def _make_layer(self, layer_id, layer, hidden_states, attention_mask):
        base = f"/decoder/layers.{layer_id}"
        self_norm = self._make_layer_norm(
            layer.self_attn_layer_norm,
            f"{base}/self_attn_layer_norm",
            hidden_states,
            self.sequence_length,
        )
        self_attention = self._make_self_attention(
            layer_id, layer.self_attn, self_norm, attention_mask
        )
        self_residual = f"{base}/self_attn/Add"
        self.make_add(
            self_residual,
            [hidden_states, self_attention],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )

        cross_norm = self._make_layer_norm(
            layer.encoder_attn_layer_norm,
            f"{base}/encoder_attn_layer_norm",
            f"{self_residual}/output_0",
            self.sequence_length,
        )
        cross_attention = self._make_cross_attention(
            layer_id, layer.encoder_attn, cross_norm
        )
        cross_residual = f"{base}/encoder_attn/Add"
        self.make_add(
            cross_residual,
            [f"{self_residual}/output_0", cross_attention],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )

        final_norm = self._make_layer_norm(
            layer.final_layer_norm,
            f"{base}/final_layer_norm",
            f"{cross_residual}/output_0",
            self.sequence_length,
        )
        fc1 = self._make_linear(
            layer.fc1, f"{base}/fc1", final_norm, self.sequence_length
        )
        if self.activation != "gelu":
            raise ValueError(
                "Nemotron Parse builder currently supports the checkpoint's gelu MLP"
            )
        activation = f"{base}/activation_fn/Gelu"
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
        fc2 = self._make_linear(
            layer.fc2,
            f"{base}/fc2",
            f"{activation}/output_0",
            self.sequence_length,
        )
        output = f"{base}/fc2/AddResidual"
        self.make_add(
            output,
            [f"{cross_residual}/output_0", fc2],
            self.io_dtype,
            ["batch_size", self.sequence_length, self.hidden_size],
        )
        return f"{output}/output_0"
