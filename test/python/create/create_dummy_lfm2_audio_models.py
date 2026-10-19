#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Generate tiny ONNX models for LFM2-Audio pipeline testing.

Writes the three models the lfm2_audio pipeline runs, with the input/output contract of the published
LFM2-Audio / LFM2.5-Audio encoder export (LiquidAI/LFM2.5-Audio-1.5B-ONNX), plus the genai_config.json
that wires them together:

  dummy_speech.onnx     mel_spectrogram [N, T, 128], mel_lengths [N] (int64)
                        -> audio_embeddings [N, ceil(T / 8), hidden], audio_lengths [N] (int64)
                        Every 8 mel frames are averaged and projected, so the features depend on the
                        audio and the frame count matches the placeholders the processor writes.
  dummy_embedding.onnx  input_ids, audio_features -> inputs_embeds
                        Token lookup with audio_features scattered over the positions holding
                        AUDIO_TOKEN_ID, the graph the tutorial describes for the real model.
  dummy_text.onnx       The LFM2 decoder of the LFM2-VL fixture (see create_dummy_lfm2_vl_models.py):
                        each logit depends on the current token and the two before it through a causal
                        convolution over past.%d.conv, so generation only matches between runs when the
                        conv state is carried correctly.

  dummy_depthformer.onnx, dummy_audio_embedding.onnx
                        Speech output, with the contract of LiquidAI's vocoder_depthformer.onnx and
                        audio_embedding.onnx. The depthformer gives one codebook's logits per run; they
                        depend on the hidden state, the codebook, the code before it and the cache of
                        the runs before, so a frame only comes out right when all four are carried.
                        The decoder has the hidden_states output they read. genai_config.json leaves
                        them out: the tests that use them add model.audio_output themselves.

The tokenizer is checked in separately, and zipped: tokenizer.zip holds tokenizer.json and
tokenizer_config.json from LiquidAI/LFM2.5-Audio-1.5B (LFM2-Audio-1.5B ships the same files), whose
4.8 MB compresses to under 1 MB. test_lfm2_audio_models.py unpacks it into a temporary directory
next to these models.

Usage:
    python create_dummy_lfm2_audio_models.py --output test/models/lfm2-audio
"""

import argparse
import json
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_dummy_lfm2_vl_models import (
    CONV_CACHE_SIZE,
    HEAD_SIZE,
    HIDDEN_SIZE,
    LAYER_TYPES,
    LOGIT_VOCAB,
    NUM_KV_HEADS,
    VOCAB_SIZE,
    _const,
    _save,
    create_dummy_decoder_model,
)

NUM_MELS = 128
SUBSAMPLING_FACTOR = 8
# <|reserved_123|>: the placeholder the processor writes once per encoder frame. LFM2-Audio has no
# audio input token of its own, so any single token that never occurs in text will do.
AUDIO_TOKEN_ID = 133


def create_dummy_speech_model(output_path: str, rng: np.random.Generator):
    """mel_spectrogram, mel_lengths -> audio_embeddings [N, ceil(T / 8), hidden], audio_lengths"""
    inputs = [
        helper.make_tensor_value_info("mel_spectrogram", TensorProto.FLOAT, ["batch_size", "time_steps", NUM_MELS]),
        helper.make_tensor_value_info("mel_lengths", TensorProto.INT64, ["batch_size"]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "audio_embeddings", TensorProto.FLOAT, ["batch_size", "reduced_time", HIDDEN_SIZE]
        ),
        helper.make_tensor_value_info("audio_lengths", TensorProto.INT64, ["batch_size"]),
    ]
    projection = numpy_helper.from_array(
        (rng.standard_normal((NUM_MELS, HIDDEN_SIZE)) / np.sqrt(NUM_MELS)).astype(np.float32), name="projection"
    )
    nodes = [
        _const("subsampling_minus_one", [SUBSAMPLING_FACTOR - 1]),
        _const("subsampling", [SUBSAMPLING_FACTOR]),
        # [N, T, mels] -> [N, mels, T] so the pool runs over time; ceil_mode keeps the partial last window
        helper.make_node("Transpose", ["mel_spectrogram"], ["mel_nct"], perm=[0, 2, 1]),
        helper.make_node(
            "AveragePool",
            ["mel_nct"],
            ["pooled_nct"],
            kernel_shape=[SUBSAMPLING_FACTOR],
            strides=[SUBSAMPLING_FACTOR],
            ceil_mode=1,
        ),
        helper.make_node("Transpose", ["pooled_nct"], ["pooled"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["pooled", "projection"], ["audio_embeddings"]),
        # audio_lengths = ceil(mel_lengths / subsampling_factor)
        helper.make_node("Add", ["mel_lengths", "subsampling_minus_one"], ["lengths_rounded"]),
        helper.make_node("Div", ["lengths_rounded", "subsampling"], ["audio_lengths"]),
    ]
    _save(helper.make_graph(nodes, "speech", inputs, outputs, initializer=[projection]), output_path)


def create_dummy_embedding_model(output_path: str, rng: np.random.Generator):
    """input_ids, audio_features -> inputs_embeds, with the audio features scattered over the placeholders"""
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, ["num_audio_tokens", HIDDEN_SIZE]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", HIDDEN_SIZE]
        )
    ]
    table = numpy_helper.from_array(
        rng.standard_normal((LOGIT_VOCAB, HIDDEN_SIZE)).astype(np.float32), name="embed_table"
    )
    nodes = [
        _const("table_rows", [LOGIT_VOCAB]),
        _const("audio_token_id", [AUDIO_TOKEN_ID]),
        _const("flat_rows", [-1, HIDDEN_SIZE]),
        _const("flat", [-1]),
        helper.make_node("Mod", ["input_ids", "table_rows"], ["row_ids"]),
        helper.make_node("Gather", ["embed_table", "row_ids"], ["text_embeds"], axis=0),
        # Positions of the placeholders in the flattened [batch * sequence] token stream, in order,
        # receive the audio features row by row.
        helper.make_node("Shape", ["text_embeds"], ["embeds_shape"]),
        helper.make_node("Reshape", ["text_embeds", "flat_rows"], ["flat_embeds"]),
        helper.make_node("Reshape", ["input_ids", "flat"], ["flat_ids"]),
        helper.make_node("Equal", ["flat_ids", "audio_token_id"], ["is_audio"]),
        helper.make_node("NonZero", ["is_audio"], ["audio_positions_t"]),
        helper.make_node("Transpose", ["audio_positions_t"], ["audio_positions"], perm=[1, 0]),
        helper.make_node("ScatterND", ["flat_embeds", "audio_positions", "audio_features"], ["merged"]),
        helper.make_node("Reshape", ["merged", "embeds_shape"], ["inputs_embeds"]),
    ]
    _save(helper.make_graph(nodes, "embedding", inputs, outputs, initializer=[table]), output_path)


NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2049  # 2048 codes and the end-of-audio code
DEPTH_SIZE = 16
TABLE_ROWS = 251  # the embedding tables are folded to keep the files small; prime, so codebooks do not collide


def create_dummy_depthformer_model(output_path: str, rng: np.random.Generator):
    """One codebook's logits per run, with the inputs and outputs of LiquidAI's vocoder_depthformer.onnx"""
    cache_shape = [1, "batch", 1, "past_len", DEPTH_SIZE]
    inputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", HIDDEN_SIZE]),
        helper.make_tensor_value_info("depth_slices_in", TensorProto.FLOAT, ["batch", NUM_CODEBOOKS, DEPTH_SIZE]),
        helper.make_tensor_value_info("step_idx", TensorProto.INT64, []),
        helper.make_tensor_value_info("prev_token", TensorProto.INT64, ["batch"]),
        helper.make_tensor_value_info("past_keys", TensorProto.FLOAT, cache_shape),
        helper.make_tensor_value_info("past_values", TensorProto.FLOAT, cache_shape),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["batch"]),
        helper.make_tensor_value_info("total_seq_len", TensorProto.INT32, []),
    ]
    new_cache_shape = [1, "batch", 1, "new_len", DEPTH_SIZE]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", CODEBOOK_SIZE]),
        helper.make_tensor_value_info("depth_slices", TensorProto.FLOAT, ["batch", NUM_CODEBOOKS, DEPTH_SIZE]),
        helper.make_tensor_value_info("new_keys", TensorProto.FLOAT, new_cache_shape),
        helper.make_tensor_value_info("new_values", TensorProto.FLOAT, new_cache_shape),
    ]
    initializers = [
        numpy_helper.from_array(
            rng.standard_normal((HIDDEN_SIZE, NUM_CODEBOOKS * DEPTH_SIZE)).astype(np.float32), name="depth_linear"
        ),
        numpy_helper.from_array(rng.standard_normal((TABLE_ROWS, DEPTH_SIZE)).astype(np.float32), name="code_table"),
        numpy_helper.from_array(rng.standard_normal((DEPTH_SIZE, CODEBOOK_SIZE)).astype(np.float32), name="to_logits"),
    ]
    nodes = [
        _const("slices_shape", [-1, NUM_CODEBOOKS, DEPTH_SIZE]),
        _const("cache_entry_shape", [1, -1, 1, 1, DEPTH_SIZE]),
        _const("zero", [0], dims=[]),
        _const("past_axis", [3]),
        _const("table_rows", [TABLE_ROWS]),
        # The projection of the hidden state is only computed on the first codebook's run; later runs
        # are handed it back, as the real graph does to skip the projection.
        helper.make_node("MatMul", ["hidden_states", "depth_linear"], ["projected"]),
        helper.make_node("Reshape", ["projected", "slices_shape"], ["computed_slices"]),
        helper.make_node("Equal", ["step_idx", "zero"], ["is_first"]),
        helper.make_node("Where", ["is_first", "computed_slices", "depth_slices_in"], ["depth_slices"]),
        helper.make_node("Gather", ["depth_slices", "step_idx"], ["slice"], axis=1),
        helper.make_node("Mod", ["prev_token", "table_rows"], ["previous_row"]),
        helper.make_node("Gather", ["code_table", "previous_row"], ["previous"], axis=0),
        helper.make_node("Add", ["slice", "previous"], ["position"]),
        # What the earlier codebooks of this frame left in the cache.
        helper.make_node("ReduceSum", ["past_keys", "past_axis"], ["history_5d"], keepdims=0),
        helper.make_node("Reshape", ["history_5d", _shape_name("history")], ["history"]),
        helper.make_node("Add", ["position", "history"], ["mixed"]),
        helper.make_node("MatMul", ["mixed", "to_logits"], ["logits"]),
        helper.make_node("Reshape", ["position", "cache_entry_shape"], ["cache_entry"]),
        helper.make_node("Concat", ["past_keys", "cache_entry"], ["new_keys"], axis=3),
        helper.make_node("Concat", ["past_values", "cache_entry"], ["new_values"], axis=3),
    ]
    nodes.insert(0, _const(_shape_name("history"), [-1, DEPTH_SIZE]))
    _save(helper.make_graph(nodes, "depthformer", inputs, outputs, initializer=initializers), output_path)


def _shape_name(name: str) -> str:
    return f"{name}_shape"


def create_dummy_audio_embedding_model(output_path: str, rng: np.random.Generator):
    """audio_codes [batch, length], each offset into its codebook's rows -> audio_embeds [batch, length, hidden]"""
    inputs = [helper.make_tensor_value_info("audio_codes", TensorProto.INT64, ["batch_size", "audio_length"])]
    outputs = [
        helper.make_tensor_value_info("audio_embeds", TensorProto.FLOAT, ["batch_size", "audio_length", HIDDEN_SIZE])
    ]
    table = numpy_helper.from_array(
        rng.standard_normal((TABLE_ROWS, HIDDEN_SIZE)).astype(np.float32), name="audio_table"
    )
    nodes = [
        _const("table_rows", [TABLE_ROWS]),
        helper.make_node("Mod", ["audio_codes", "table_rows"], ["rows"]),
        helper.make_node("Gather", ["audio_table", "rows"], ["audio_embeds"], axis=0),
    ]
    _save(helper.make_graph(nodes, "audio_embedding", inputs, outputs, initializer=[table]), output_path)


def create_genai_config(output_path: str):
    session_options = {"log_id": "onnxruntime-genai", "provider_options": []}
    config = {
        "model": {
            "audio_token_id": AUDIO_TOKEN_ID,
            "bos_token_id": 1,
            "context_length": 4096,
            "decoder": {
                "session_options": session_options,
                "filename": "dummy_text.onnx",
                "head_size": HEAD_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                    "past_conv_names": "past.%d.conv",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                    "present_conv_names": "present.%d.conv",
                },
                "num_attention_heads": NUM_KV_HEADS,
                "num_hidden_layers": len(LAYER_TYPES),
                "num_key_value_heads": NUM_KV_HEADS,
                "layer_types": LAYER_TYPES,
                "conv_cache_size": CONV_CACHE_SIZE,
            },
            "eos_token_id": [7, 128, 130],  # <|im_end|>, <|audio_start|>, <|text_end|>: what the builder writes
            "pad_token_id": 0,
            "type": "lfm2_audio",
            "vocab_size": VOCAB_SIZE,
            "embedding": {
                "session_options": session_options,
                "filename": "dummy_embedding.onnx",
                "inputs": {"input_ids": "input_ids", "audio_features": "audio_features"},
                "outputs": {"inputs_embeds": "inputs_embeds"},
            },
            "speech": {
                "session_options": session_options,
                "filename": "dummy_speech.onnx",
                "inputs": {
                    "audio_embeds": "mel_spectrogram",
                    "audio_lengths": "mel_lengths",
                    "audio_sizes": "audio_sizes",
                },
                "outputs": {"audio_features": "audio_embeddings"},
            },
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 4096,
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
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate tiny ONNX models for LFM2-Audio pipeline testing")
    parser.add_argument("--output", type=str, default="test/models/lfm2-audio", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(42)
    create_dummy_speech_model(os.path.join(args.output, "dummy_speech.onnx"), rng)
    create_dummy_embedding_model(os.path.join(args.output, "dummy_embedding.onnx"), rng)
    create_dummy_decoder_model(os.path.join(args.output, "dummy_text.onnx"), rng, hidden_states_output=True)
    create_dummy_depthformer_model(os.path.join(args.output, "dummy_depthformer.onnx"), rng)
    create_dummy_audio_embedding_model(os.path.join(args.output, "dummy_audio_embedding.onnx"), rng)
    create_genai_config(os.path.join(args.output, "genai_config.json"))
    print(f"Wrote dummy LFM2-Audio models to {args.output}")


if __name__ == "__main__":
    main()
