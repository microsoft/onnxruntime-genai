# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import copy

import onnx_ir as ir
from transformers.feature_extraction_utils import FeatureExtractionMixin

from .lfm2 import LFM2Model, LFM3Model
from .lfm2_vl import LFM2VLModel


class LFM3AudioEncoder(LFM2Model):
    # The LFM3-Audio speech encoder and projector, for one clip per run:
    # input_features [1, mels, frames] --> audio_features [ceil(frames / 8), text hidden size]

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        encoder_config = config.audio_encoder_config
        lfm_config = copy.deepcopy(encoder_config.text_config)
        lfm_config.architectures = config.architectures
        lfm_config._name_or_path = config._name_or_path
        # The checkpoint windows every attention layer; the window counts the frame itself
        lfm_config.layer_types = [
            "sliding_attention" if layer_type == "full_attention" else layer_type
            for layer_type in lfm_config.layer_types
        ]
        lfm_config.sliding_window = encoder_config.sliding_window
        self.num_mel_bins = encoder_config.num_mel_bins
        self.text_hidden_size = config.text_config.hidden_size
        # No KV cache to window: GroupQueryAttention gets no sliding_window_cache attribute
        extra_options = {**extra_options, "filename": "audio_encoder.onnx", "windowed_kv_cache": False}
        super().__init__(lfm_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self, config):
        super().make_attention_init(config)
        # The packed Attention and MultiHeadAttention fallbacks have no local window
        if self.attention_attrs["op_type"] != "GroupQueryAttention":
            raise NotImplementedError(
                f"The speech encoder needs GroupQueryAttention (local window), but {self.ep}/{self.io_dtype} selects "
                f"{self.attention_attrs['op_type']}; export for CPU or WebGPU with fp32 or for CUDA with fp16."
            )

    def resolve_model_loader(self):
        return self.resolve_transformers_class(LFM3Model.MULTIMODAL[self.model_type][0])

    def make_inputs_and_outputs(self):
        self.input_names = {"input_features": "input_features"}
        self.input_types = {"input_features": self.io_dtype}
        self.input_shapes = {"input_features": [1, self.num_mel_bins, "num_frames"]}
        self.output_names = {"audio_features": "audio_features"}
        self.output_types = {"audio_features": self.io_dtype}
        self.output_shapes = {"audio_features": ["num_audio_tokens", self.text_hidden_size]}
        super().make_inputs_and_outputs()
        # Each run is a whole clip: the conv layers start from zeros and keep no state
        self.input_names["past.conv"] = self.output_names["present.conv"] = dict.fromkeys(range(self.num_layers), "")

    def make_causal_conv_state_at_last_real_token(self, name, **kwargs):
        pass

    def make_key_value_cache_names(self, layer_id):
        # Unread, but CUDA's GroupQueryAttention requires the presents and on CPU they select the tiled kernel
        present = f"/model/layers.{layer_id}/attn/present"
        return "", "", f"{present}_key", f"{present}_value"

    def make_model(self, input_path):
        # Model.make_model's module scan would take the language model's layers first, so walk the encoder by name
        self.make_inputs_and_outputs()
        model = self.load_weights(input_path).model
        encoder = model.audio_encoder
        hidden_states = self.make_downsampler(encoder.downsample)
        self.make_attention_lengths(hidden_states)
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"] = hidden_states
        for layer_id, layer in enumerate(encoder.lfm.layers):
            self.make_layer(layer_id, layer)
        self.make_layernorm(self.num_layers, encoder.lfm.embedding_norm, skip=True, simple=True, location="final_norm")
        self.make_projector(model.audio_projector)

    def make_downsampler(self, downsample):
        # Make the following subgraph, three stride-2 convolutions over the (mel, frame) plane:
        #
        #   input_features [1,M,T] --> Unsqueeze [1,1,M,T] --> conv_in --> SiLU --> conv_mid --> lin_mid --> SiLU
        #     --> conv_end --> lin_end --> SiLU --> Reshape [1,C*M/8,N] --> lin_out --> Transpose [1,N,H]
        basename = "/model/downsample"
        unsqueeze_name = f"{basename}/Unsqueeze"
        unsqueeze_inputs = [self.input_names["input_features"], "/model/constants/INT64/[1]"]
        self.make_unsqueeze(unsqueeze_name, unsqueeze_inputs, self.io_dtype, [1, 1, self.num_mel_bins, "num_frames"])
        root_input = f"{unsqueeze_name}/output_0"
        stages = (("conv_in", True), ("conv_mid", False), ("lin_mid", True), ("conv_end", False), ("lin_end", True))
        for name, silu in stages:
            root_input = self.make_downsample_conv(f"{basename}/{name}", getattr(downsample, name), root_input, silu)

        channels = downsample.lin_out.in_channels
        reshape_name = f"{basename}/Reshape"
        reshape_inputs = [root_input, f"/model/constants/INT64/[0, {channels}, -1]"]
        self.make_reshape(reshape_name, reshape_inputs, self.io_dtype, [1, channels, "num_audio_tokens"])
        root_input = self.make_downsample_conv(f"{basename}/lin_out", downsample.lin_out, f"{reshape_name}/output_0")
        transpose_name = f"{basename}/Transpose"
        hidden_shape = [1, "num_audio_tokens", self.hidden_size]
        self.make_transpose(transpose_name, root_input, self.io_dtype, hidden_shape, [0, 2, 1])
        return f"{transpose_name}/output_0"

    def make_downsample_conv(self, basename, conv, root_input, silu=False):
        weight, bias = (f"{basename[1:].replace('/', '.')}.{name}" for name in ("weight", "bias"))
        self.make_initializer(conv.weight, weight, to=self.io_dtype)
        self.make_initializer(conv.bias, bias, to=self.io_dtype)
        shape = [1, conv.out_channels, *[None] * (conv.weight.dim() - 2)]
        conv_name = f"{basename}/Conv"
        self.make_conv(
            conv_name,
            [root_input, weight, bias],
            self.io_dtype,
            shape,
            kernel_shape=list(conv.kernel_size),
            strides=list(conv.stride),
            pads=list(conv.padding) * 2,
            group=conv.groups,
        )
        output = f"{conv_name}/output_0"
        if not silu:
            return output
        sigmoid_name = f"{basename}/act/Sigmoid"
        self.make_sigmoid(sigmoid_name, output, self.io_dtype, shape)
        mul_name = f"{basename}/act/Mul"
        self.make_mul(mul_name, [output, f"{sigmoid_name}/output_0"], self.io_dtype, shape)
        return f"{mul_name}/output_0"

    def make_attention_lengths(self, hidden_states):
        # Make the following subgraph, GroupQueryAttention's lengths for one clip with no past:
        #
        #   hidden_states [1,N,H] --> Shape --> Gather(1) = N --> Cast = total_sequence_length
        #                                                    \--> Sub(1) --> Unsqueeze --> Cast = seqlens_k [N-1]
        basename = "/model/attn_lengths"
        i64, i32 = ir.DataType.INT64, ir.DataType.INT32
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, hidden_states, shape=[3])
        gather_name = f"{basename}/Gather"
        self.make_gather(gather_name, [f"{shape_name}/output_0", "/model/constants/INT64/1"], i64, shape=[], axis=0)
        total_name = f"{basename}/total_sequence_length/Cast"
        self.make_cast(total_name, f"{gather_name}/output_0", i32, shape=[])
        sub_name = f"{basename}/Sub"
        self.make_sub(sub_name, [f"{gather_name}/output_0", "/model/constants/INT64/1"], i64, shape=[])
        unsqueeze_name = f"{basename}/Unsqueeze"
        self.make_unsqueeze(unsqueeze_name, [f"{sub_name}/output_0", "/model/constants/INT64/[0]"], i64, shape=[1])
        seqlens_name = f"{basename}/seqlens_k/Cast"
        self.make_cast(seqlens_name, f"{unsqueeze_name}/output_0", i32, shape=[1])
        self.mask_attrs["seqlens_k"], self.mask_attrs["total_seq_len"] = seqlens_name, total_name

    def make_projector(self, projector):
        # Make the following subgraph, on the tokens of the clip:
        #
        #   final norm [1,N,D] --> LayerNorm --> linear_1 --> Gelu --> linear_2 --> Reshape [N,H] --> audio_features
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["output_0"]
        # nn.LayerNorm's own epsilon, not the encoder's norm_eps
        self.layernorm_attrs["epsilon"] = projector.layer_norm.eps
        self.make_layernorm(self.num_layers, projector.layer_norm, skip=False, simple=False, location="projector")

        basename = "/model/projector"
        normed = self.layernorm_attrs["output_0"]
        hidden = self.make_projector_linear(projector.linear_1, f"{basename}/linear_1", normed)
        gelu_name = f"{basename}/Gelu"
        self.make_node("Gelu", [hidden], [f"{gelu_name}/output_0"], name=gelu_name, approximate="none")
        gelu_shape = self.make_hidden_state_shape(last_dim=projector.linear_1.out_features)
        self.make_value(f"{gelu_name}/output_0", self.io_dtype, shape=gelu_shape)
        features = self.make_projector_linear(projector.linear_2, f"{basename}/linear_2", f"{gelu_name}/output_0")
        reshape_name = f"{basename}/Reshape"
        reshape_inputs = [features, f"/model/constants/INT64/[-1, {self.text_hidden_size}]"]
        self.make_reshape(reshape_name, reshape_inputs, self.io_dtype, self.output_shapes["audio_features"])
        self.rename_value(f"{reshape_name}/output_0", self.output_names["audio_features"])

    def make_projector_linear(self, linear, basename, root_input):
        matmul_name = self.make_matmul(linear, f"{basename}/MatMul", root_input)
        add_name = f"{basename}/Add"
        self.make_add_bias(linear.bias, add_name, root_input=f"{matmul_name}/output_0")
        return f"{add_name}/output_0"


class LFM3AudioModel(LFM2VLModel):
    # The lfm3_audio pipeline (exclude_embeds=true): the LFM3 decoder, plus the speech encoder and the embedding model
    # that turn a clip's mel frames and the token ids into its inputs_embeds.

    vision_class = None
    decoder_class = LFM3Model
    pipeline_type = "lfm3_audio"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        if extra_options.get("use_paged_attention", False):
            raise NotImplementedError(
                f"use_paged_attention=true is not supported for the {self.pipeline_type} pipeline yet."
            )
        # TODO: allow other EPs once a real LFM3-Audio model has run on them.
        if ep not in {"cpu", "cuda", "webgpu"}:
            raise NotImplementedError(
                f"The {self.pipeline_type} pipeline runs on cpu, cuda and webgpu for now, not {ep}."
            )
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Read the mel settings before the build, so an unsupported checkpoint fails early
        self.mel_config = self.make_mel_config(self.read_audio_extractor(config, cache_dir))
        self.audio_token_id = config.audio_token_id

        # Never quantized. CUDA has no fp32 GroupQueryAttention, and on WebGPU an fp16 encoder drifts from fp32.
        speech_dtype = ir.DataType.FLOAT16 if ep == "cuda" else ir.DataType.FLOAT
        options = {key: extra_options[key] for key in ("hf_token", "hf_remote") if key in extra_options}
        self.speech = LFM3AudioEncoder(config, speech_dtype, speech_dtype, ep, cache_dir, options)
        if self.decoder.filename == self.speech.filename:
            raise ValueError(f"The decoder cannot be named {self.decoder.filename}: the speech encoder has that name")
        self.sub_models.append(self.speech)
        self.embedding.embed_attrs["features"].append(("audio_features", self.audio_token_id, self.speech.io_dtype))

    def read_audio_extractor(self, config, cache_dir):
        extractor, _ = FeatureExtractionMixin.get_feature_extractor_dict(
            self.decoder.model_name_or_path, cache_dir=cache_dir, token=self.decoder.hf_token
        )
        # The runtime's mel front end computes these settings as constants
        expected = {
            "feature_extractor_type": "Lfm3AudioFeatureExtractor",
            "log_floor": -12.0,
            "normalization_offset": 4.0,
            "normalization_divisor": 4.0,
            "padding_side": "right",
            "padding_value": 0.0,
            "win_length": extractor["n_fft"],
            "feature_size": config.audio_encoder_config.num_mel_bins,
            "encoder_stride": 8,
        }
        unsupported = {key: extractor.get(key) for key, value in expected.items() if extractor.get(key) != value}
        if unsupported:
            raise ValueError(f"Unsupported audio feature extractor settings {unsupported}: expected {expected}")
        return extractor

    def make_mel_config(self, extractor):
        return {
            "sample_rate": extractor["sampling_rate"],
            "num_mels": extractor["feature_size"],
            "fft_size": extractor["n_fft"],
            "win_length": extractor["win_length"],
            "hop_length": extractor["hop_length"],
            "subsampling_factor": extractor["encoder_stride"],
        }

    def update_genai_config(self, genai_config):
        super().update_genai_config(genai_config)
        model = genai_config["model"]
        model["audio_token_id"] = self.audio_token_id
        model.update(self.mel_config)
        session_options = model["embedding"]["session_options"]
        if self.decoder.ep == "cuda":
            # cuDNN's default search runs again whenever the clip length changes; HEURISTIC gives the same output
            session_options = {
                **session_options,
                "provider_options": [{"cuda": {"cudnn_conv_algo_search": "HEURISTIC"}}],
            }
        model["speech"] = {
            "filename": self.speech.filename,
            # audio_sizes names the tokens per clip of the processor, not an encoder input
            "inputs": {"audio_embeds": "input_features", "audio_sizes": "audio_sizes"},
            "outputs": {"audio_features": "audio_features"},
            "session_options": session_options,
        }
