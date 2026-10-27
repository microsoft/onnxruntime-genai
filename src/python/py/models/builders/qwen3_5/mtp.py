# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy
import os

from .text import Qwen35MoETextModel, Qwen35TextModel


class Qwen35MTPModel(Qwen35MoETextModel):
    """Qwen3.6 multi-token-prediction self-speculative head builder."""

    is_moe_mtp = True

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.is_mtp_head = True

        config = copy.deepcopy(config)
        text_config = getattr(config, "text_config", config)
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["full_attention"]
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]

        self.mtp_layer_config = copy.deepcopy(text_config)
        self.mtp_layer_config.layer_types = ["full_attention"]
        self.mtp_layer_config.num_hidden_layers = 1

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        quant_config = extra_options.get("_quant_config")
        self.preserve_mtp_quantization = quant_config is None or quant_config.checkpoint_policy == "preserve"
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape()

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True

        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_layernorm(1, self.mtp_weights.norm, skip=True, simple=True, location="final_norm")
        mtp_norm_output = self.layernorm_attrs["output_0"]
        self.make_lm_head(self.mtp_weights.lm_head)

        hidden_states_output = "hidden_states_out"
        self.rename_value(mtp_norm_output, hidden_states_output)
        hidden_states_value = self.make_value(
            hidden_states_output,
            self.io_dtype,
            shape=self.make_hidden_state_shape(),
        )
        self.model.graph.outputs.append(hidden_states_value)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        try:
            from loaders.qwen import QwenMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.qwen import QwenMTPModel  # noqa: PLC0415

        self.mtp_weights = QwenMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.mtp_layer_config,
            preserve_quantization=self.preserve_mtp_quantization,
            load_quantized_model=self.load_weights,
            is_moe=self.is_moe_mtp,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )

    def make_mtp_input_projection(self):
        basename = "/model/mtp"

        embed_output = self.make_mtp_embedding(basename)
        self.make_value(embed_output, self.io_dtype, shape=self.make_hidden_state_shape())

        embedding_norm = f"{basename}/pre_fc_norm_embedding/output_0"
        hidden_states_norm = f"{basename}/pre_fc_norm_hidden/output_0"
        for location, root_input, norm, output in (
            ("pre_fc_norm_embedding", embed_output, self.mtp_weights.pre_fc_norm_embedding, embedding_norm),
            ("pre_fc_norm_hidden", self.input_names["hidden_states"], self.mtp_weights.pre_fc_norm_hidden, hidden_states_norm),
        ):
            norm_name = f"{basename}/{location}"
            weight_name = f"{norm_name[1:].replace('/', '.')}.weight"
            self.make_initializer(norm.weight + self.layernorm_attrs["add_offset"], weight_name, to=self.io_dtype)
            self.make_layernorm_subgraph(
                norm_name, op_type="SimplifiedLayerNormalization", inputs=[root_input, weight_name], outputs=[output],
                skip=False, new_io_dtype=self.io_dtype, epsilon=self.layernorm_attrs["epsilon"], axis=-1, stash_type=1,
            )
            self.make_value(output, self.io_dtype, self.make_hidden_state_shape())

        concat_name = f"{basename}/fc/Concat"
        self.make_concat(
            concat_name,
            [embedding_norm, hidden_states_norm],
            self.io_dtype,
            self.make_hidden_state_shape(last_dim=2 * self.hidden_size),
            axis=-1,
        )

        fc_name = self.make_matmul(self.mtp_weights.fc, f"{basename}/fc/MatMul", f"{concat_name}/output_0")
        return f"{fc_name}/output_0"
    def make_mtp_embedding(self, basename):
        return self.make_embedding_lookup(
            self.mtp_weights.embedding.weight,
            f"{basename}/embed_tokens",
            self.mtp_weights.lm_head,
        )



class Qwen35DenseMTPModel(Qwen35MTPModel):
    """Dense Qwen3.5/Qwen3.8 MTP head with one full-attention decoder layer."""

    is_moe_mtp = False

    def make_layer(self, layer_id, layer):
        return Qwen35TextModel.make_layer(self, layer_id, layer)
