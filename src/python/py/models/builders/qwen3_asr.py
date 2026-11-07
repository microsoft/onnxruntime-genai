# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import copy
import json
import os

import onnx_ir as ir
import torch
from onnx_ir.tensor_adapters import to_torch_dtype

from .base import Model
from .lfm2_vl import LFM2VLEmbedding
from .qwen import Qwen3Model


class Qwen3ASRAudioEncoder(Model):
    # The Qwen3-ASR audio tower and projector, run once per clip:
    #   mel_spectrogram [1, num_frames, num_mel_bins], mel_lengths [1] --> audio_embeddings [1, num_audio_tokens, H]
    #
    # num_frames is the clip's mel padded up to a multiple of chunk_frames (2 * n_window), the way transformers'
    # Qwen3ASRFeatureExtractor pads it; mel_lengths holds the real (unpadded) frame count. The tower runs its 2D
    # convolutions on each chunk of chunk_frames frames and attends inside windows of n_window_infer frames (8 s for the
    # published checkpoints). Each window is one row of the batch here, so no window sees another, every attention is
    # window x window, and only the last window, when it is partly filled, needs an attention bias. This is exactly the
    # windowing transformers applies through cu_seqlens.

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        audio_config = config.audio_config
        if audio_config.activation_function != "gelu":
            raise ValueError(f"Unsupported audio activation_function {audio_config.activation_function}: expected gelu")
        if audio_config.n_window_infer % (2 * audio_config.n_window) != 0:
            raise ValueError("n_window_infer must be a multiple of 2 * n_window")
        if getattr(audio_config, "scale_embedding", False):
            raise ValueError("scale_embedding=True is not supported")

        self.num_mel_bins = audio_config.num_mel_bins
        self.chunk_frames = 2 * audio_config.n_window  # mel frames per convolution chunk
        self.chunk_tokens = self.post_cnn_length(self.chunk_frames)  # tokens one full chunk yields (13)
        self.window_chunks = audio_config.n_window_infer // self.chunk_frames  # chunks per attention window (8)
        self.window_tokens = self.chunk_tokens * self.window_chunks  # tokens per attention window (104)
        self.conv_channels = audio_config.downsample_hidden_size
        self.conv_freqs = (
            (((audio_config.num_mel_bins + 1) // 2 + 1) // 2) + 1
        ) // 2  # mel bins left after 3 convs (16)
        self.output_size = audio_config.output_dim

        # Standardize config attributes for the encoder layers to simplify layer building logic
        config.hidden_size = audio_config.d_model
        config.intermediate_size = audio_config.encoder_ffn_dim
        config.num_attention_heads = audio_config.encoder_attention_heads
        config.num_key_value_heads = audio_config.encoder_attention_heads
        config.head_dim = audio_config.d_model // audio_config.encoder_attention_heads
        config.num_hidden_layers = audio_config.encoder_layers
        config.layer_types = ["full_attention"] * audio_config.encoder_layers
        config.hidden_act = audio_config.activation_function
        config.hidden_activation = None

        extra_options["filename"] = "audio_encoder.onnx"
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.layernorm_attrs["simple"] = False
        self.layernorm_attrs["epsilon"] = 1e-5  # nn.LayerNorm's default, which every audio LayerNorm uses
        self.attention_attrs["rope"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def post_cnn_length(self, num_frames):
        # Length after the three (kernel 3, stride 2, padding 1) convolutions
        return (((num_frames + 1) // 2 + 1) // 2 + 1) // 2

    def is_gqa_supported(self):
        # Tokens attend to every real token of their window, through an attention bias
        return False

    def is_packed_attn_supported(self):
        # MultiHeadAttention is the attention op with an attention bias that runs on CPU, CUDA and WebGPU
        return False

    def make_inputs_and_outputs(self):
        # Set input dicts
        self.input_names = {"mel_spectrogram": "mel_spectrogram", "mel_lengths": "mel_lengths"}
        self.input_types = {"mel_spectrogram": self.io_dtype, "mel_lengths": ir.DataType.INT64}
        self.input_shapes = {"mel_spectrogram": [1, "num_frames", self.num_mel_bins], "mel_lengths": [1]}

        # Set output dicts
        self.output_names = {"audio_embeddings": "audio_embeddings"}
        self.output_types = {"audio_embeddings": self.io_dtype}
        self.output_shapes = {"audio_embeddings": [1, "num_audio_tokens", self.output_size]}

        # Now set inputs and outputs
        super().make_inputs_and_outputs()

    def make_hidden_state_shape(self, seq_dim="sequence_length", last_dim=None):
        # The layers run on [num_windows, window_tokens, hidden]; the projector on the 2D real tokens
        if seq_dim == "num_audio_tokens":
            return [seq_dim, self.hidden_size if last_dim is None else last_dim]
        return super().make_hidden_state_shape(seq_dim, last_dim)

    def audio_tower(self, model):
        return model.model.audio_tower

    def make_preprocessing_nodes(self):
        # Make the following subgraph:
        #
        #   mel_spectrogram [1,T,M] --> Transpose --> Reshape [C,1,M,F] (C chunks of F frames)
        #     --> Conv --> Gelu --> Conv --> Gelu --> Conv --> Gelu [C,D',M',t] --> Transpose/Reshape [C,t,D'*M']
        #     --> MatMul (conv_out) --> Add (positions 0..t-1 of every chunk) [C,t,D]
        #     --> Pad to whole windows --> Reshape [W, window_tokens, D] --> layer 0
        #
        #   mel_lengths --> number of real tokens --> attention bias [W,1,L,L] for every layer
        basename = "/model/audio"
        tower = self.audio_tower(self.weights)
        dtype = self.io_dtype
        chunks, freqs, t = "num_chunks", self.conv_freqs, self.chunk_tokens

        transpose_name = f"{basename}/mel/Transpose"
        self.make_transpose(
            transpose_name, self.input_names["mel_spectrogram"], dtype, [1, self.num_mel_bins, "num_frames"], [0, 2, 1]
        )
        split_name = f"{basename}/mel/Reshape"
        split_shape = f"/model/constants/INT64/[1, {self.num_mel_bins}, -1, {self.chunk_frames}]"
        self.make_reshape(
            split_name,
            [f"{transpose_name}/output_0", split_shape],
            dtype,
            [1, self.num_mel_bins, chunks, self.chunk_frames],
        )
        chunks_name = f"{basename}/chunks/Transpose"
        self.make_transpose(
            chunks_name,
            f"{split_name}/output_0",
            dtype,
            [1, chunks, self.num_mel_bins, self.chunk_frames],
            [0, 2, 1, 3],
        )
        image_name = f"{basename}/chunks/Reshape"
        image_shape = f"/model/constants/INT64/[-1, 1, {self.num_mel_bins}, {self.chunk_frames}]"
        self.make_reshape(
            image_name,
            [f"{chunks_name}/output_0", image_shape],
            dtype,
            [chunks, 1, self.num_mel_bins, self.chunk_frames],
        )

        root_input = f"{image_name}/output_0"
        height, width = self.num_mel_bins, self.chunk_frames
        for i, conv in enumerate((tower.conv2d1, tower.conv2d2, tower.conv2d3), start=1):
            height, width = (height + 1) // 2, (width + 1) // 2
            weight, bias = f"model.audio.conv2d{i}.weight", f"model.audio.conv2d{i}.bias"
            self.make_initializer(conv.weight, weight, to=dtype)
            self.make_initializer(conv.bias, bias, to=dtype)
            conv_name = f"{basename}/conv2d{i}/Conv"
            self.make_conv(
                conv_name,
                [root_input, weight, bias],
                dtype,
                [chunks, self.conv_channels, height, width],
                dilations=[1, 1],
                group=1,
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
            gelu_name = f"{basename}/conv2d{i}/Gelu"
            gelu_output = f"{gelu_name}/output_0"
            self.make_node(
                "Gelu", inputs=[f"{conv_name}/output_0"], outputs=[gelu_output], name=gelu_name, approximate="none"
            )
            self.make_value(gelu_output, dtype, shape=[chunks, self.conv_channels, height, width])
            root_input = gelu_output

        # [C, D', M', t] --> [C, t, D' * M'], the order transformers flattens the channels and frequencies in
        flat_transpose_name = f"{basename}/conv_out/Transpose"
        self.make_transpose(
            flat_transpose_name, root_input, dtype, [chunks, t, self.conv_channels, freqs], [0, 3, 1, 2]
        )
        flat_name = f"{basename}/conv_out/Reshape"
        flat_shape = f"/model/constants/INT64/[0, 0, {self.conv_channels * freqs}]"
        self.make_reshape(
            flat_name, [f"{flat_transpose_name}/output_0", flat_shape], dtype, [chunks, t, self.conv_channels * freqs]
        )
        conv_out_name = self.make_matmul(tower.conv_out, f"{basename}/conv_out/MatMul", f"{flat_name}/output_0")
        self.make_value(f"{conv_out_name}/output_0", dtype, shape=[chunks, t, self.hidden_size])

        # Every chunk restarts at position 0
        positions = "model.audio.positional_embedding"
        self.make_initializer(tower.positional_embedding.positional_embedding[:t], positions, to=dtype)
        position_name = f"{basename}/positions/Add"
        self.make_add(position_name, [f"{conv_out_name}/output_0", positions], dtype, [chunks, t, self.hidden_size])

        # Group the chunks into attention windows: pad the chunk count to a whole number of windows, then one window per row
        chunk_count = self.make_chunk_count(f"{basename}/num_chunks")
        windows = self.make_ceil_div(f"{basename}/num_windows", chunk_count, self.window_chunks)
        missing_name = f"{basename}/missing_chunks/Sub"
        total_name = f"{basename}/padded_chunks/Mul"
        self.make_mul(total_name, [windows, f"/model/constants/INT64/[{self.window_chunks}]"], ir.DataType.INT64, [1])
        self.make_sub(missing_name, [f"{total_name}/output_0", chunk_count], ir.DataType.INT64, [1])
        pads_name = f"{basename}/pads/Concat"
        pads_inputs = ["/model/constants/INT64/[0, 0, 0]", f"{missing_name}/output_0", "/model/constants/INT64/[0, 0]"]
        self.make_concat(pads_name, pads_inputs, ir.DataType.INT64, [6])
        pad_name = f"{basename}/windows/Pad"
        pad_output = f"{pad_name}/output_0"
        self.make_node(
            "Pad",
            inputs=[f"{position_name}/output_0", f"{pads_name}/output_0"],
            outputs=[pad_output],
            name=pad_name,
            mode="constant",
        )
        self.make_value(pad_output, dtype, shape=["num_padded_chunks", t, self.hidden_size])
        windows_name = f"{basename}/windows/Reshape"
        windows_shape = f"/model/constants/INT64/[-1, {self.window_tokens}, {self.hidden_size}]"
        self.make_reshape(windows_name, [pad_output, windows_shape], dtype, self.make_hidden_state_shape())

        self.layernorm_attrs["root_input"] = f"{windows_name}/output_0"
        self.layernorm_attrs["skip_input"] = f"{windows_name}/output_0"

        self.make_num_audio_tokens()
        self.make_attention_bias(windows)

    def make_chunk_count(self, basename):
        # Chunks in the padded mel: num_frames / chunk_frames, as a 1D tensor
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, self.input_names["mel_spectrogram"], shape=[1], start=1, end=2)
        div_name = f"{basename}/Div"
        self.make_div(
            div_name,
            [f"{shape_name}/output_0", f"/model/constants/INT64/[{self.chunk_frames}]"],
            ir.DataType.INT64,
            [1],
        )
        return f"{div_name}/output_0"

    def make_ceil_div(self, basename, value, divisor):
        # (value + divisor - 1) / divisor for a non-negative 1D INT64 value
        add_name = f"{basename}/Add"
        self.make_add(add_name, [value, f"/model/constants/INT64/[{divisor - 1}]"], ir.DataType.INT64, [1])
        div_name = f"{basename}/Div"
        self.make_div(div_name, [f"{add_name}/output_0", f"/model/constants/INT64/[{divisor}]"], ir.DataType.INT64, [1])
        return f"{div_name}/output_0"

    def make_num_audio_tokens(self):
        # Real tokens of the clip, transformers' _get_feat_extract_output_lengths:
        #   chunk_tokens * (mel_lengths / chunk_frames) + post_cnn_length(mel_lengths % chunk_frames)
        # where post_cnn_length(r) = ceil(ceil(ceil(r / 2) / 2) / 2), which is 0 for r = 0
        basename = "/model/audio/num_audio_tokens"
        int64 = ir.DataType.INT64
        full_name = f"{basename}/full_chunks/Div"
        self.make_div(
            full_name, [self.input_names["mel_lengths"], f"/model/constants/INT64/[{self.chunk_frames}]"], int64, [1]
        )
        rest_name = f"{basename}/rest/Mod"
        self.make_mod(
            rest_name, [self.input_names["mel_lengths"], f"/model/constants/INT64/[{self.chunk_frames}]"], int64, [1]
        )
        rest = f"{rest_name}/output_0"
        for i in range(3):
            rest = self.make_ceil_div(f"{basename}/rest_conv{i + 1}", rest, 2)
        full_tokens_name = f"{basename}/full_chunks/Mul"
        self.make_mul(
            full_tokens_name, [f"{full_name}/output_0", f"/model/constants/INT64/[{self.chunk_tokens}]"], int64, [1]
        )
        add_name = f"{basename}/Add"
        self.make_add(add_name, [f"{full_tokens_name}/output_0", rest], int64, [1])
        self.num_audio_tokens = f"{add_name}/output_0"

    def make_attention_bias(self, windows):
        # Make the following subgraph, the additive mask that hides the padding tokens of the last window:
        #
        #   Range(0, W * L) --> Reshape [W,L] --> Less(num_audio_tokens) --> Where(0, lowest) --> Unsqueeze [W,1,1,L]
        #     --> Expand [W,1,L,L]
        basename = "/model/attn_mask"
        int64, dtype = ir.DataType.INT64, self.to_str_dtype(self.io_dtype)
        rows = ["batch_size", "sequence_length"]

        total_name = f"{basename}/total/Mul"
        self.make_mul(total_name, [windows, f"/model/constants/INT64/[{self.window_tokens}]"], int64, [1])
        scalar_name = f"{basename}/total/Squeeze"
        self.make_squeeze(scalar_name, [f"{total_name}/output_0", "/model/constants/INT64/[0]"], int64, [])
        range_name = f"{basename}/Range"
        self.make_range(
            range_name,
            ["/model/constants/INT64/0", f"{scalar_name}/output_0", "/model/constants/INT64/1"],
            int64,
            ["total_tokens"],
        )
        positions_name = f"{basename}/positions/Reshape"
        self.make_reshape(
            positions_name,
            [f"{range_name}/output_0", f"/model/constants/INT64/[-1, {self.window_tokens}]"],
            int64,
            rows,
        )
        less_name = f"{basename}/Less"
        self.make_less(less_name, [f"{positions_name}/output_0", self.num_audio_tokens])
        self.make_value(f"{less_name}/output_0", ir.DataType.BOOL, shape=rows)
        where_name = f"{basename}/Where"
        lowest = torch.finfo(to_torch_dtype(self.io_dtype)).min
        where_inputs = [f"{less_name}/output_0", f"/model/constants/{dtype}/0", f"/model/constants/{dtype}/{lowest}"]
        self.make_where(where_name, where_inputs, self.io_dtype, rows)
        unsqueeze_name = f"{basename}/Unsqueeze"
        self.make_unsqueeze(
            unsqueeze_name,
            [f"{where_name}/output_0", "/model/constants/INT64/[1, 2]"],
            self.io_dtype,
            ["batch_size", 1, 1, "sequence_length"],
        )

        # MultiHeadAttention needs the full [B,1,L,L] bias, not one row to broadcast
        expand_name = f"{basename}/Expand"
        expand_shape = f"/model/constants/INT64/[1, 1, {self.window_tokens}, 1]"
        self.make_expand(
            expand_name,
            [f"{unsqueeze_name}/output_0", expand_shape],
            self.io_dtype,
            ["batch_size", 1, "sequence_length", "sequence_length"],
        )
        self.mask_attrs["mask_name"] = expand_name

    def make_embedding(self, embedding):
        # The audio encoder has no token embedding
        pass

    def is_layer(self, module):
        return module.__class__.__name__ == "Qwen3ASRAudioEncoderLayer"

    def make_layer(self, layer_id, layer):
        layer.input_layernorm = layer.self_attn_layer_norm
        layer.post_attention_layernorm = layer.final_layer_norm

        class Qwen3ASRAudioMLP(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.fc1 = layer.fc1
                self.fc2 = layer.fc2

        layer.mlp = Qwen3ASRAudioMLP(layer)
        super().make_layer(layer_id, layer)

    def make_key_value_cache_names(self, layer_id):
        # The audio encoder does not use any KV caches
        return "", "", "", ""

    def has_final_norm(self, module, model):
        return module is self.audio_tower(model).ln_post

    def has_lm_head(self, module):
        # The audio encoder ends with the projector
        return False

    def make_postprocessing_nodes(self):
        # Make the following subgraph, which keeps the real tokens and projects them to the decoder's hidden size:
        #
        #   ln_post [W,L,D] --> Reshape [W*L,D] --> Slice [:num_audio_tokens] --> linear_1 --> Gelu --> linear_2
        #     --> Unsqueeze [1,N,H] --> audio_embeddings
        basename = "/model/projector"
        projector = self.weights.model.multi_modal_projector
        tokens = "num_audio_tokens"

        flat_name = f"{basename}/tokens/Reshape"
        flat_inputs = [self.layernorm_attrs["output_0"], f"/model/constants/INT64/[-1, {self.hidden_size}]"]
        self.make_reshape(flat_name, flat_inputs, self.io_dtype, ["total_tokens", self.hidden_size])
        slice_name = f"{basename}/tokens/Slice"
        slice_inputs = [
            f"{flat_name}/output_0",
            "/model/constants/INT64/[0]",
            self.num_audio_tokens,
            "/model/constants/INT64/[0]",
        ]
        self.make_slice(slice_name, slice_inputs, self.io_dtype, [tokens, self.hidden_size])

        root_input = self.make_projector_linear(projector.linear_1, f"{basename}/linear_1", f"{slice_name}/output_0")
        gelu_name = f"{basename}/act_fn/Gelu"
        gelu_output = f"{gelu_name}/output_0"
        self.make_node("Gelu", [root_input], [gelu_output], name=gelu_name, approximate="none")
        self.make_value(gelu_output, self.io_dtype, shape=[tokens, projector.linear_1.out_features])
        root_input = self.make_projector_linear(projector.linear_2, f"{basename}/linear_2", gelu_output)

        unsqueeze_name = f"{basename}/output/Unsqueeze"
        self.make_node(
            "Unsqueeze",
            [root_input, "/model/constants/INT64/[0]"],
            [self.output_names["audio_embeddings"]],
            name=unsqueeze_name,
        )

    def make_projector_linear(self, linear, basename, root_input):
        matmul_name = self.make_matmul(linear, f"{basename}/MatMul", root_input, seq_dim="num_audio_tokens")
        if linear.bias is None:
            return f"{matmul_name}/output_0"
        add_name = f"{basename}/Add"
        self.make_add_bias(linear.bias, add_name, root_input=f"{matmul_name}/output_0", seq_dim="num_audio_tokens")
        return f"{add_name}/output_0"


class Qwen3ASREmbedding(LFM2VLEmbedding):
    # input_ids, audio_features --> inputs_embeds, with the audio features in place of the audio tokens
    # (transformers' masked_scatter over audio_token_id). The LFM2-VL embedding graph, fed with audio features.

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        config.image_token_id = config.audio_token_id  # The token whose embeddings the features replace
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_inputs_and_outputs(self):
        # The runtime binds the features as embedding.inputs.audio_features
        self.input_names = {"input_ids": "input_ids", "image_features": "audio_features"}
        self.input_types = {"input_ids": ir.DataType.INT64, "image_features": self.embed_attrs["image_features_dtype"]}
        self.input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],
            "image_features": ["num_audio_tokens", self.hidden_size],
        }
        self.output_names = {"inputs_embeds": "inputs_embeds"}
        self.output_types = {"inputs_embeds": self.io_dtype}
        self.output_shapes = {"inputs_embeds": self.make_hidden_state_shape()}
        Model.make_inputs_and_outputs(self)


class Qwen3ASRModel:
    # The qwen3_asr pipeline: the Qwen3 decoder (exclude_embeds), plus the audio encoder and embedding models that turn
    # a clip's log-mel frames and the prompt's token ids into its inputs_embeds

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # The audio encoder and embedding models are never quantized. A bf16 tower drifts much further from fp32 than
        # an fp16 one, so it runs in fp16 and the embedding model casts its output.
        audio_dtype = ir.DataType.FLOAT16 if io_dtype == ir.DataType.BFLOAT16 else io_dtype
        # The remaining options configure the decoder
        options = {key: extra_options[key] for key in ("hf_token", "hf_remote") if key in extra_options}
        unquantized_config = copy.deepcopy(config)
        if hasattr(unquantized_config, "quantization_config"):
            delattr(unquantized_config, "quantization_config")
        self.audio = Qwen3ASRAudioEncoder(
            copy.deepcopy(unquantized_config), audio_dtype, audio_dtype, ep, cache_dir, dict(options)
        )
        self.embedding = Qwen3ASREmbedding(unquantized_config, io_dtype, io_dtype, ep, cache_dir, dict(options))
        self.embedding.embed_attrs["image_features_dtype"] = self.audio.io_dtype
        extra_options["exclude_embeds"] = True
        self.decoder = Qwen3Model(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.decoder.model_type = "qwen3_asr"
        if self.decoder.filename in (self.audio.filename, self.embedding.filename):
            raise ValueError(
                f"The decoder cannot be named {self.decoder.filename}: the audio encoder or embedding model has that name"
            )
        self.audio_token_id = config.audio_token_id
        self.audio_config = config.audio_config

    def make_model(self, input_path):
        # All three models read the same checkpoint, so load it once
        hf_model = self.audio.load_weights(input_path)
        self.audio.load_weights = self.embedding.load_weights = lambda input_path: hf_model
        if (
            self.decoder.quant_type is None
            and not {"adapter_path", "num_hidden_layers"} & self.decoder.extra_options.keys()
        ):
            self.decoder.load_weights = self.audio.load_weights
        self.audio.make_model(input_path)
        self.embedding.make_model(input_path)
        self.decoder.make_model(input_path)

    def save_model(self, out_dir):
        self.audio.save_model(out_dir)
        self.embedding.save_model(out_dir)
        self.decoder.save_model(out_dir)

    def make_genai_config(self, config, extra_kwargs, out_dir):
        self.decoder.make_genai_config(config, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        # The audio encoder and embedding models run on the decoder's EP. Graph capture replays one recorded run, so it
        # stays on the decoder: the other models get new shapes for every clip.
        model = genai_config["model"]
        graph_capture = {"enable_cuda_graph", "enable_graph_capture", "enableGraphCapture"}
        provider_options = [
            {name: {key: value for key, value in options.items() if key not in graph_capture}}
            for provider in model["decoder"]["session_options"]["provider_options"]
            for name, options in provider.items()
        ]
        session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
        model["audio_token_id"] = self.audio_token_id
        model["embedding"] = {
            "filename": self.embedding.filename,
            "inputs": {"input_ids": "input_ids", "audio_features": "audio_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
            "session_options": session_options,
        }
        model["speech"] = {
            "filename": self.audio.filename,
            "inputs": {"audio_embeds": "mel_spectrogram", "audio_lengths": "mel_lengths", "audio_sizes": "audio_sizes"},
            "outputs": {"audio_features": "audio_embeddings"},
            "session_options": session_options,
        }
        # The log-mel front end of Qwen3ASRFeatureExtractor; the runtime defaults to these values
        model["num_mels"] = self.audio_config.num_mel_bins
        model["sample_rate"] = 16000
        model["fft_size"] = 400
        model["hop_length"] = 160

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.decoder.save_processing(model_name_or_path, extra_kwargs, out_dir)
