#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate a tiny LFM3-Audio pipeline for testing.

Writes the three models the lfm3_audio pipeline runs, with the input/output contract of the model builder's exports,
the genai_config.json the builder writes, and a tokenizer and chat template made here:

  dummy_speech.onnx     input_features [1, num_mels, frames] -> audio_features [ceil(frames / 8), hidden]
                        One clip per run. The frames are zero-padded to a multiple of 8, each 8 are averaged and
                        projected over the mel axis, so a transposed or padded input changes the result or fails.
  dummy_embedding.onnx  input_ids, audio_features -> inputs_embeds, the LFM2-Audio fixture's scatter over <|audio|>.
  dummy_text.onnx       The LFM2 decoder of the LFM2-VL fixture, whose attention layers the config calls full and
                        sliding: each logit reads the current input embedding and the two before it.
  tokenizer.json        Byte-level BPE with no merges, plus the chat and audio markers as special tokens. It holds
                        no vocabulary from any released model.
  chat_template.jinja   Renders {"type": "audio"} as <|audio|>.

Usage:
    python create_dummy_lfm3_audio_models.py --output test/models/lfm3-audio
"""

import argparse
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import create_dummy_lfm2_audio_models as lfm2_audio
from create_dummy_lfm2_vl_models import HIDDEN_SIZE, _const, _save, create_dummy_decoder_model
from create_dummy_lfm3_vl_models import create_tokenizer as create_byte_level_tokenizer

NUM_MELS = 16
SUBSAMPLING_FACTOR = 8
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|audio|>",
    "<|audio_out|>",
    "<|audio_start|>",
    "<|text_end|>",
    "<|mixed_start|>",
]
# Special tokens follow the 256 byte tokens
PAD_ID, BOS_ID, IM_END_ID, AUDIO_TOKEN_ID = 256, 257, 259, 260
AUDIO_START_ID, TEXT_END_ID, MIXED_START_ID = 262, 263, 264
CHAT_TEMPLATE = """\
{{- '<|startoftext|>' -}}
{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' -}}
{%- if message['content'] is string -%}
{{- message['content'] -}}
{%- else -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'audio' -%}{{- '<|audio|>' -}}{%- else -%}{{- item['text'] -}}{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}
"""


def create_dummy_speech_model(output_path: str, rng: np.random.Generator, num_mels: int = NUM_MELS):
    """input_features [1, num_mels, frames] -> audio_features [ceil(frames / 8), hidden]"""
    inputs = [helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, num_mels, "num_frames"])]
    outputs = [helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, ["num_audio_tokens", HIDDEN_SIZE])]
    projection = numpy_helper.from_array(rng.standard_normal((num_mels, HIDDEN_SIZE)).astype(np.float32), "projection")
    nodes = [
        _const("frame_axis", [2]),
        _const("stride", [SUBSAMPLING_FACTOR]),
        _const("pads_before_frame_end", [0, 0, 0, 0, 0]),
        _const("token_shape", [-1, SUBSAMPLING_FACTOR, num_mels]),
        helper.make_node("Shape", ["input_features"], ["mel_shape"]),
        helper.make_node("Gather", ["mel_shape", "frame_axis"], ["frames"]),
        helper.make_node("Mod", ["frames", "stride"], ["past_multiple"]),
        helper.make_node("Sub", ["stride", "past_multiple"], ["to_multiple"]),
        helper.make_node("Mod", ["to_multiple", "stride"], ["frame_pad"]),
        helper.make_node("Concat", ["pads_before_frame_end", "frame_pad"], ["pads"], axis=0),
        helper.make_node("Pad", ["input_features", "pads"], ["padded"]),
        helper.make_node("Transpose", ["padded"], ["frame_major"], perm=[0, 2, 1]),
        helper.make_node("Reshape", ["frame_major", "token_shape"], ["token_frames"]),
        helper.make_node("ReduceMean", ["token_frames"], ["token_means"], axes=[1], keepdims=0),
        helper.make_node("MatMul", ["token_means", "projection"], ["audio_features"]),
    ]
    _save(helper.make_graph(nodes, "speech", inputs, outputs, initializer=[projection]), output_path)


def create_tokenizer(output_dir: str):
    create_byte_level_tokenizer(output_dir)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer = json.load(f)
    tokenizer["added_tokens"] = [
        {"id": 256 + i, "content": token, "single_word": False, "lstrip": False, "rstrip": False}
        | {"normalized": False, "special": True}
        for i, token in enumerate(SPECIAL_TOKENS)
    ]
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write(CHAT_TEMPLATE)


def create_genai_config(output_path: str):
    """The sections the builder writes for LFM3-Audio-2B, with this fixture's sizes and file names."""
    lfm2_audio.create_genai_config(output_path)
    with open(output_path, encoding="utf-8") as f:
        config = json.load(f)
    model = config["model"]
    model["decoder"]["layer_types"] = ["conv", "full_attention", "conv", "sliding_attention"]
    model.update(
        type="lfm3_audio",
        audio_token_id=AUDIO_TOKEN_ID,
        bos_token_id=BOS_ID,
        pad_token_id=PAD_ID,
        eos_token_id=[IM_END_ID, TEXT_END_ID, AUDIO_START_ID, MIXED_START_ID],
        sample_rate=16000,
        num_mels=NUM_MELS,
        fft_size=400,
        win_length=400,
        hop_length=160,
        subsampling_factor=SUBSAMPLING_FACTOR,
    )
    model["speech"]["inputs"] = {"audio_embeds": "input_features", "audio_sizes": "audio_sizes"}
    model["speech"]["outputs"] = {"audio_features": "audio_features"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny LFM3-Audio pipeline for testing")
    parser.add_argument("--output", type=str, default="test/models/lfm3-audio", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(42)
    create_dummy_speech_model(os.path.join(args.output, "dummy_speech.onnx"), rng)
    # The LFM2-Audio helper reads its module's token id, and this fixture's id sits past the byte tokens.
    lfm2_audio.AUDIO_TOKEN_ID = AUDIO_TOKEN_ID
    lfm2_audio.create_dummy_embedding_model(os.path.join(args.output, "dummy_embedding.onnx"), rng)
    create_dummy_decoder_model(os.path.join(args.output, "dummy_text.onnx"), rng)
    create_genai_config(os.path.join(args.output, "genai_config.json"))
    create_tokenizer(args.output)
    print(f"Wrote dummy LFM3-Audio models to {args.output}")


if __name__ == "__main__":
    main()
