# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy
import os

from .expansions.mtp import Qwen38MTP
from .qwen3_8_text import Qwen4ExpTextModel


class Qwen4ExpMTPTextModel(Qwen38MTP, Qwen4ExpTextModel):
    """Qwen4-Exp one-layer self-speculative MTP head builder."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options, multimodal_rotary=False):
        config = copy.deepcopy(config)
        config.text_config.num_hidden_layers = 1
        config.text_config.layer_types = ["indexed_attention"]
        config.text_config.ple_layer_ids = []
        config.num_hidden_layers = 1
        config.layer_types = ["indexed_attention"]

        extra_options = copy.deepcopy(extra_options)
        extra_options["num_hidden_layers"] = 1
        extra_options["text_only"] = True
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_sparse_rotary_init(multimodal_rotary)

        self.hyper_connection_attrs["tile_first_hidden_state"] = False
        self.hyper_connection_attrs["emit_pre_final_hidden_states"] = True
        self.include_hidden_states = True
        self.output_names["hidden_states"] = "hidden_states_out"
        self.output_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"])
        self.input_names["hidden_states"] = "hidden_states"
        self.input_types["hidden_states"] = self.io_dtype
        self.input_shapes["hidden_states"] = self.make_hidden_state_shape(last_dim=self.hyper_connection_attrs["hidden_size"])

    def get_final_hyper_connection_mixer(self):
        return self.mtp_weights.hyper_connection_mixer

    def make_model(self, input_path):
        self.make_inputs_and_outputs()
        self.load_mtp_weights(input_path)
        self.make_preprocessing_nodes()

        projected = self.make_mtp_input_projection()
        self.layernorm_attrs["root_input"] = projected
        self.layernorm_attrs["skip_input"] = projected
        self.layernorm_attrs["first_layernorm"] = True
        self.make_layer(0, self.mtp_weights.layers[0])
        self.make_lm_head(self.mtp_weights.lm_head)

        self.make_postprocessing_nodes()
        del self.mtp_weights

    def load_mtp_weights(self, input_path):
        model_dir = input_path if input_path and os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            model_dir = snapshot_download(
                repo_id=model_dir,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        try:
            from loaders.mtp import Qwen4ExpMTPModel  # noqa: PLC0415
        except ImportError:
            from onnxruntime_genai.models.loaders.mtp import Qwen4ExpMTPModel  # noqa: PLC0415

        self.mtp_weights = Qwen4ExpMTPModel.from_pretrained(
            self.quant_type,
            input_path,
            model_dir,
            self.hf_load_config.text_config,
            preserve_quantization=self.quant_config.checkpoint_policy == "preserve",
            load_quantized_model=self.load_weights,
        )
