# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Generate the dummy ``gemma4_unified`` test model directory.

The gemma-4-12B "unified" (encoder-free) model shares the gemma4 decoder /
embedding contract but consumes raw 48px merged pixel patches
(``pixel_values`` last dim = 48*48*3 = 6912) and raw 640-sample waveform frames
(``audio_embeds`` last dim = 640) directly, instead of the SigLIP 16px /
128-dim log-mel contract.

This derives a temporary unified model from the existing
``test/models/gemma4`` fixtures: the embedding / text decoders are copied
verbatim, the vision / speech graphs consume the unified inputs, and the
genai / processor configs are rewritten for the ``gemma4_unified`` type.

Usage (from the repo root):
    python test/python/create/create_dummy_gemma4_unified_models.py --output /tmp/gemma4-unified-validation
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx

_UNIFIED_PIXEL_DIM = 48 * 48 * 3  # 6912
_UNIFIED_AUDIO_DIM = 640

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "test" / "models" / "gemma4"
_DST_DIR = _REPO_ROOT / "test" / "models" / "gemma4_unified"

_TOKENIZER_FILES = [
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _create_unified_vision_model(out_path: Path) -> None:
    """Select real patches by XY positions and expose their pixel means."""
    h, t = onnx.helper, onnx.TensorProto
    inputs = [
        h.make_tensor_value_info(
            "pixel_values", t.FLOAT, ["batch_size", 280, _UNIFIED_PIXEL_DIM]
        ),
        h.make_tensor_value_info("pixel_position_ids", t.INT64, ["batch_size", 280, 2]),
    ]
    outputs = [h.make_tensor_value_info("image_features", t.FLOAT, ["tokens", 2048])]
    nodes = [
        h.make_node("Gather", ["pixel_position_ids", "x_axis"], ["x"], axis=2),
        h.make_node("Greater", ["x", "negative_one"], ["valid"]),
        h.make_node("NonZero", ["valid"], ["indices_t"]),
        h.make_node("Transpose", ["indices_t"], ["indices"], perm=[1, 0]),
        h.make_node("GatherND", ["pixel_values", "indices"], ["patches"]),
        h.make_node("Cast", ["patches"], ["patches_double"], to=t.DOUBLE),
        h.make_node(
            "ReduceMean", ["patches_double"], ["means_double"], axes=[1], keepdims=1
        ),
        h.make_node("Cast", ["means_double"], ["means"], to=t.FLOAT),
        h.make_node("Shape", ["means"], ["mean_shape"]),
        h.make_node("Gather", ["mean_shape", "first_axis"], ["token_count"], axis=0),
        h.make_node(
            "Concat", ["token_count", "hidden_size"], ["feature_shape"], axis=0
        ),
        h.make_node("Expand", ["means", "feature_shape"], ["image_features"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.array(0, np.int64), "x_axis"),
        onnx.numpy_helper.from_array(np.array(-1, np.int64), "negative_one"),
        onnx.numpy_helper.from_array(np.array([0], np.int64), "first_axis"),
        onnx.numpy_helper.from_array(np.array([2048], np.int64), "hidden_size"),
    ]
    graph = h.make_graph(
        nodes, "unified_input_sensitive_vision", inputs, outputs, initializers
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)], ir_version=7)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def _create_unified_speech_model(out_path: Path) -> None:
    """Consume PCM, the actual frame mask, and the requested audio token count."""
    h, t = onnx.helper, onnx.TensorProto
    inputs = [
        h.make_tensor_value_info(
            "input_features", t.FLOAT, ["batch_size", "num_frames", _UNIFIED_AUDIO_DIM]
        ),
        h.make_tensor_value_info(
            "input_features_mask", t.BOOL, ["batch_size", "num_frames"]
        ),
        h.make_tensor_value_info("audio_sizes", t.INT64, ["batch_size"]),
    ]
    # SpeechState binds rank-3 output, then reshapes it to rank 2 for embedding.
    outputs = [h.make_tensor_value_info("audio_features", t.FLOAT, [1, "tokens", 2048])]
    nodes = [
        h.make_node(
            "ReduceMean", ["input_features"], ["frame_means"], axes=[2], keepdims=0
        ),
        h.make_node("Cast", ["input_features_mask"], ["mask_float"], to=t.FLOAT),
        h.make_node("Mul", ["frame_means", "mask_float"], ["masked_means"]),
        h.make_node("ReduceSum", ["masked_means"], ["audio_sum"], keepdims=0),
        h.make_node("Cast", ["audio_sizes"], ["sizes_float"], to=t.FLOAT),
        h.make_node("ReduceSum", ["sizes_float"], ["size_sum"], keepdims=0),
        h.make_node("Add", ["audio_sum", "size_sum"], ["signal"]),
        h.make_node(
            "Concat",
            ["batch_size", "audio_sizes", "hidden_size"],
            ["feature_shape"],
            axis=0,
        ),
        h.make_node("Expand", ["signal", "feature_shape"], ["audio_features"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(np.array([1], np.int64), "batch_size"),
        onnx.numpy_helper.from_array(np.array([2048], np.int64), "hidden_size"),
    ]
    graph = h.make_graph(
        nodes, "unified_input_sensitive_speech", inputs, outputs, initializers
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)], ir_version=7)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def create_model(source_dir: Path, output_dir: Path) -> None:
    """Derive an isolated fixture; never rewrite the tracked model directory."""
    if not source_dir.exists():
        raise SystemExit(
            f"Source gemma4 fixtures not found at {source_dir}. Generate/download the "
            "gemma4 test model directory first; gemma4_unified is derived from it."
        )
    if output_dir.resolve() in {source_dir.resolve(), _DST_DIR.resolve()}:
        raise ValueError(
            "Use a temporary output directory, not a tracked fixture directory"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Decoder + embedding are identical to gemma4.
    for name in ("dummy_text.onnx", "dummy_embedding.onnx"):
        shutil.copyfile(source_dir / name, output_dir / name)

    _create_unified_vision_model(output_dir / "dummy_vision.onnx")
    _create_unified_speech_model(output_dir / "dummy_speech.onnx")

    for name in _TOKENIZER_FILES:
        shutil.copyfile(source_dir / name, output_dir / name)

    # genai_config.json: switch the model type and vision processor config file.
    with open(source_dir / "genai_config.json") as f:
        genai_config = json.load(f)
    genai_config["model"]["type"] = "gemma4_unified"
    genai_config["model"]["vision"]["config_filename"] = "image_processor.json"
    genai_config["model"]["speech"]["inputs"] = {
        "audio_embeds": "input_features",
        "attention_mask": "input_features_mask",
        "audio_sizes": "audio_sizes",
    }
    with open(output_dir / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=4)

    # image_processor.json: reuse Gemma4ImageTransform with the merged geometry
    # (patch_size=48, pooling_kernel_size=1) that yields 6912-dim patches.
    image_processor = {
        "processor": {
            "name": "gemma_4_unified_image_processing",
            "transforms": [
                {
                    "operation": {
                        "name": "decode_image",
                        "type": "DecodeImage",
                        "attrs": {"color_space": "RGB"},
                    }
                },
                {
                    "operation": {
                        "name": "gemma4_image_transform",
                        "type": "Gemma4ImageTransform",
                        "attrs": {
                            "patch_size": 48,
                            "max_soft_tokens": 280,
                            "pooling_kernel_size": 1,
                        },
                    }
                },
            ],
        }
    }
    with open(output_dir / "image_processor.json", "w") as f:
        json.dump(image_processor, f, indent=4)

    # audio_feature_extraction.json: raw 640-sample waveform framing.
    audio_config = {
        "feature_extraction": {
            "sequence": [
                {
                    "operation": {
                        "name": "audio_decoder",
                        "type": "AudioDecoder",
                        "attrs": {"max_samples": 0},
                    }
                },
                {
                    "operation": {
                        "name": "gemma4_audio",
                        "type": "Gemma4Audio",
                        "attrs": {
                            "type": "raw_frames",
                            "audio_samples_per_token": 640,
                            "sampling_rate": 16000,
                            "padding_value": 0.0,
                        },
                    }
                },
            ]
        }
    }
    with open(output_dir / "audio_feature_extraction.json", "w") as f:
        json.dump(audio_config, f, indent=4)

    print(f"Wrote gemma4_unified dummy model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_SRC_DIR)
    parser.add_argument(
        "--output", type=Path, required=True, help="New temporary model directory"
    )
    args = parser.parse_args()
    create_model(args.source, args.output)


if __name__ == "__main__":
    main()
