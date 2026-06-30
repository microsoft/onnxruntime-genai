#!/usr/bin/env python3
"""
Creates minimal dummy multi-modal test models for testing DecoderState input_ids injection.

Two model variants are generated:
  - multimodal-decoder-no-input-ids/  (Mistral3-like: decoder has NO input_ids input)
  - multimodal-decoder-with-input-ids/ (Gemma4-like:  decoder HAS input_ids input)

These are used by test/python/test_decoder_state_input_ids.py.
"""

import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN_SIZE = 64
VOCAB_SIZE = 10
NUM_KV_HEADS = 4
HEAD_SIZE = 16
NUM_LAYERS = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# ONNX initializer helpers
# ---------------------------------------------------------------------------

def _logits_weight() -> onnx.TensorProto:
    """Zero weight matrix for logit projection: [HIDDEN_SIZE, VOCAB_SIZE]."""
    return numpy_helper.from_array(
        np.zeros((HIDDEN_SIZE, VOCAB_SIZE), dtype=np.float32), name="logits_weight"
    )


def _kv_pad_constant() -> onnx.TensorProto:
    """Pad descriptor that appends exactly 1 zero element on axis-2 (sequence axis).

    For a 4-D KV tensor [B, H, P, D] the pad vector layout is:
      [begin_B, begin_H, begin_P, begin_D, end_B, end_H, end_P, end_D]
    = [0, 0, 0, 0, 0, 0, 1, 0]

    This makes present.key.shape[2] == past.key.shape[2] + 1 each step,
    which satisfies ORT-GenAI's KV-cache shape validation.
    """
    return numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int64), name="kv_pad"
    )


# ---------------------------------------------------------------------------
# Shared decoder building block
# ---------------------------------------------------------------------------

def _kv_inputs() -> list:
    """KV cache inputs shared by both decoder variants."""
    return [
        helper.make_tensor_value_info(
            f"past_key_values.{i}.{k}",
            TensorProto.FLOAT,
            ["batch", NUM_KV_HEADS, "past_seq", HEAD_SIZE],
        )
        for i in range(NUM_LAYERS)
        for k in ("key", "value")
    ]


def _build_decoder_graph(extra_graph_inputs: list) -> tuple:
    """Return (nodes, initializers, outputs) for a decoder model.

    The decoder:
      - Projects inputs_embeds to logits via a zero-weight MatMul.
      - Grows KV cache by padding one zero slice on the sequence axis.

    extra_graph_inputs  additional graph inputs beyond inputs_embeds + KV cache.
    """
    nodes = [
        helper.make_node("MatMul", ["inputs_embeds", "logits_weight"], ["logits"]),
    ]
    initializers = [_logits_weight(), _kv_pad_constant()]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "seq", VOCAB_SIZE]),
    ]

    for i in range(NUM_LAYERS):
        for k in ("key", "value"):
            past = f"past_key_values.{i}.{k}"
            present = f"present.{i}.{k}"
            nodes.append(helper.make_node("Pad", [past, "kv_pad"], [present]))
            outputs.append(
                helper.make_tensor_value_info(
                    present,
                    TensorProto.FLOAT,
                    ["batch", NUM_KV_HEADS, "total_seq", HEAD_SIZE],
                )
            )

    return nodes, initializers, outputs


# ---------------------------------------------------------------------------
# Decoder model factories
# ---------------------------------------------------------------------------

def make_decoder_no_input_ids() -> onnx.ModelProto:
    """Decoder that does NOT declare input_ids (Mistral3-like).

    Inputs:  inputs_embeds [batch, seq, HIDDEN_SIZE]
             past_key_values.{i}.key/value
    Outputs: logits [batch, seq, VOCAB_SIZE]
             present.{i}.key/value  (KV cache grown +1 via Pad)
    """
    inputs_embeds = helper.make_tensor_value_info(
        "inputs_embeds", TensorProto.FLOAT, ["batch", "seq", HIDDEN_SIZE]
    )
    nodes, initializers, outputs = _build_decoder_graph([])
    graph = helper.make_graph(
        nodes, "decoder", [inputs_embeds] + _kv_inputs(), outputs, initializers
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_decoder_with_input_ids() -> onnx.ModelProto:
    """Decoder that DOES declare input_ids (Gemma4-like).

    Inputs:  input_ids     [batch, seq]
             inputs_embeds [batch, seq, HIDDEN_SIZE]
             past_key_values.{i}.key/value
    Outputs: logits [batch, seq, VOCAB_SIZE]
             present.{i}.key/value  (KV cache grown +1 via Pad)
    """
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, ["batch", "seq"]
    )
    inputs_embeds = helper.make_tensor_value_info(
        "inputs_embeds", TensorProto.FLOAT, ["batch", "seq", HIDDEN_SIZE]
    )
    nodes, initializers, outputs = _build_decoder_graph(["input_ids"])
    graph = helper.make_graph(
        nodes, "decoder", [input_ids, inputs_embeds] + _kv_inputs(), outputs, initializers
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# ---------------------------------------------------------------------------
# Embedding and vision model factories (shared by both variants)
# ---------------------------------------------------------------------------

def make_embedding_model() -> onnx.ModelProto:
    """Embedding model: converts input_ids → inputs_embeds.

    Inputs:  input_ids      [batch, seq]
             image_features [num_tokens, HIDDEN_SIZE]  (declared but unused in
                            computation; required so that
                            session_info_.GetInputDataType("image_features") succeeds
                            when MultiModalFeatures allocates an empty features tensor
                            for text-only generation)
    Outputs: inputs_embeds  [batch, seq, HIDDEN_SIZE]  (fixed zero initializer)
    """
    embeds_init = numpy_helper.from_array(
        np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32), name="inputs_embeds"
    )
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch", "seq"])
    image_features = helper.make_tensor_value_info(
        "image_features", TensorProto.FLOAT, ["num_tokens", HIDDEN_SIZE]
    )
    embeds_out = helper.make_tensor_value_info(
        "inputs_embeds", TensorProto.FLOAT, ["batch", "seq", HIDDEN_SIZE]
    )
    graph = helper.make_graph([], "embedding", [input_ids, image_features], [embeds_out], [embeds_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_vision_model() -> onnx.ModelProto:
    """Minimal vision model (not exercised by the text-only test).

    Inputs:  pixel_values [num_images, max_crops, 3, height, width]
             image_sizes  [num_images, 2]
    Outputs: image_features [num_tokens, HIDDEN_SIZE]  (fixed zero initializer)
    """
    feat_init = numpy_helper.from_array(
        np.zeros((1, HIDDEN_SIZE), dtype=np.float32), name="image_features"
    )
    pixel_values = helper.make_tensor_value_info(
        "pixel_values", TensorProto.FLOAT, ["num_images", "max_crops", 3, "height", "width"]
    )
    image_sizes = helper.make_tensor_value_info("image_sizes", TensorProto.INT64, ["num_images", 2])
    feat_out = helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_tokens", HIDDEN_SIZE])
    graph = helper.make_graph([], "vision", [pixel_values, image_sizes], [feat_out], [feat_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# ---------------------------------------------------------------------------
# Config and tokenizer helpers
# ---------------------------------------------------------------------------

def make_genai_config(decoder_filename: str) -> dict:
    """genai_config.json for a phi3v-type multimodal model with tiny dimensions."""
    return {
        "model": {
            "type": "phi3v",
            "bos_token_id": 1,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "vocab_size": VOCAB_SIZE,
            "context_length": 64,
            "decoder": {
                "filename": decoder_filename,
                "hidden_size": HIDDEN_SIZE,
                "head_size": HEAD_SIZE,
                "num_attention_heads": NUM_KV_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
                "num_hidden_layers": NUM_LAYERS,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "session_options": {
                    "provider_options": [],
                },
            },
            "embedding": {
                "filename": "dummy_embedding.onnx",
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
            "vision": {
                "filename": "dummy_vision.onnx",
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_sizes": "image_sizes",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
        },
        "search": {
            "do_sample": False,
            "max_length": 10,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "past_present_share_buffer": False,
        },
    }


def make_tokenizer_json() -> dict:
    """Minimal HuggingFace tokenizer JSON with a tiny vocabulary."""
    vocab = {str(i): i for i in range(VOCAB_SIZE)}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def make_tokenizer_config() -> dict:
    return {
        "bos_token": "<s>",
        "eos_token": "<s>",
        "model_max_length": 64,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
    }


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def create_model_dir(output_dir: str, decoder_model: onnx.ModelProto, decoder_filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for model_obj, filename in [
        (decoder_model, decoder_filename),
        (make_embedding_model(), "dummy_embedding.onnx"),
        (make_vision_model(), "dummy_vision.onnx"),
    ]:
        onnx.checker.check_model(model_obj)
        onnx.save(model_obj, os.path.join(output_dir, filename))

    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(make_genai_config(decoder_filename), f, indent=4)

    with open(os.path.join(output_dir, "tokenizer.json"), "w") as f:
        json.dump(make_tokenizer_json(), f, indent=4)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(make_tokenizer_config(), f, indent=4)

    print(f"Created: {output_dir}")


def main() -> None:
    # Mistral3-like: decoder does NOT declare input_ids
    create_model_dir(
        os.path.join(SCRIPT_DIR, "multimodal-decoder-no-input-ids"),
        make_decoder_no_input_ids(),
        "dummy_text.onnx",
    )

    # Gemma4-like: decoder DOES declare input_ids
    create_model_dir(
        os.path.join(SCRIPT_DIR, "multimodal-decoder-with-input-ids"),
        make_decoder_with_input_ids(),
        "dummy_text.onnx",
    )


if __name__ == "__main__":
    main()
