# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Generate the dummy ``gemma4_unified`` test model directory.

The gemma-4-12B "unified" (encoder-free) model shares the gemma4 decoder /
embedding contract but consumes raw 48px merged pixel patches
(``pixel_values`` last dim = 48*48*3 = 6912) and raw 640-sample waveform frames
(``audio_embeds`` last dim = 640) directly, instead of the SigLIP 16px /
128-dim log-mel contract.

This derives ``test/models/gemma4_unified`` from the existing
``test/models/gemma4`` fixtures: the embedding / text decoders are copied
verbatim, the vision / speech dummies get the unified input dims, and the
genai / processor configs are rewritten for the ``gemma4_unified`` type.

Usage (from the repo root):
    python test/python/create/create_dummy_gemma4_unified_models.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


def _set_shape(value_info, shape) -> None:
    dims = value_info.type.tensor_type.shape.dim
    dims.clear()
    for value in shape:
        dim = dims.add()
        if isinstance(value, str):
            dim.dim_param = value
        else:
            dim.dim_value = value


def _rewrite_model_inputs(model_path: Path, out_path: Path, replacements, extra_inputs=()) -> None:
    """Rewrite names and shapes for inputs on a dummy constant-output graph."""
    model = onnx.load(str(model_path))
    pending = dict(replacements)
    for graph_input in model.graph.input:
        replacement = pending.pop(graph_input.name, None)
        if replacement is None:
            continue
        new_name, shape = replacement
        old_name = graph_input.name
        graph_input.name = new_name
        _set_shape(graph_input, shape)
        for node in model.graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == old_name:
                    node.input[index] = new_name
    if pending:
        raise ValueError(f"inputs {sorted(pending)} not found in {model_path}")
    model.graph.input.extend(extra_inputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def create_model(src_dir: Path, dst_dir: Path) -> None:
    """Create a unified fixture from the standard Gemma4 fixture."""
    if not src_dir.exists():
        raise SystemExit(
            f"Source gemma4 fixtures not found at {src_dir}. Generate/download the "
            "gemma4 test model directory first; gemma4_unified is derived from it."
        )
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Decoder + embedding are identical to gemma4.
    for name in ("dummy_text.onnx", "dummy_embedding.onnx"):
        shutil.copyfile(src_dir / name, dst_dir / name)

    # Vision / speech dummies: same trivial constant-output graphs, but declare
    # the unified input dims so the fixtures document the real contract.
    _rewrite_model_inputs(
        src_dir / "dummy_vision.onnx",
        dst_dir / "dummy_vision.onnx",
        {
            "pixel_values": ("pixel_values", ["batch_size", 280, _UNIFIED_PIXEL_DIM]),
            "pixel_position_ids": ("pixel_position_ids", ["batch_size", 280, 2]),
        },
    )
    _rewrite_model_inputs(
        src_dir / "dummy_speech.onnx",
        dst_dir / "dummy_speech.onnx",
        {
            "audio_embeds": ("input_features", ["batch_size", "num_frames", _UNIFIED_AUDIO_DIM]),
            "audio_sizes": ("audio_sizes", ["batch_size"]),
        },
        [
            onnx.helper.make_tensor_value_info(
                "input_features_mask", onnx.TensorProto.BOOL, ["batch_size", "num_frames"]
            )
        ],
    )

    for name in _TOKENIZER_FILES:
        shutil.copyfile(src_dir / name, dst_dir / name)

    # genai_config.json: switch the model type and vision processor config file.
    with open(src_dir / "genai_config.json") as f:
        genai_config = json.load(f)
    genai_config["model"]["type"] = "gemma4_unified"
    genai_config["model"]["vision"]["config_filename"] = "image_processor.json"
    genai_config["model"]["speech"]["inputs"]["audio_embeds"] = "input_features"
    with open(dst_dir / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=4)

    # image_processor.json: reuse Gemma4ImageTransform with the merged geometry
    # (patch_size=48, pooling_kernel_size=1) that yields 6912-dim patches.
    image_processor = {
        "processor": {
            "name": "gemma_4_unified_image_processing",
            "transforms": [
                {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
                {
                    "operation": {
                        "name": "gemma4_image_transform",
                        "type": "Gemma4ImageTransform",
                        "attrs": {"patch_size": 48, "max_soft_tokens": 280, "pooling_kernel_size": 1},
                    }
                },
            ],
        }
    }
    with open(dst_dir / "image_processor.json", "w") as f:
        json.dump(image_processor, f, indent=4)

    # audio_feature_extraction.json: raw 640-sample waveform framing.
    audio_config = {
        "feature_extraction": {
            "sequence": [
                {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
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
    with open(dst_dir / "audio_feature_extraction.json", "w") as f:
        json.dump(audio_config, f, indent=4)


def main() -> None:
    create_model(_SRC_DIR, _DST_DIR)
    print(f"Wrote gemma4_unified dummy model to {_DST_DIR}")


if __name__ == "__main__":
    main()
