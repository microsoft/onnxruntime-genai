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


def _set_input_last_dim(model_path: Path, out_path: Path, input_name: str, last_dim: int) -> None:
    """Rewrite the last dimension of a named graph input to a fixed value."""
    model = onnx.load(str(model_path))
    for inp in model.graph.input:
        if inp.name == input_name:
            dims = inp.type.tensor_type.shape.dim
            dims[-1].ClearField("dim_param")
            dims[-1].dim_value = last_dim
            break
    else:
        raise ValueError(f"input {input_name!r} not found in {model_path}")
    onnx.save(model, str(out_path))


def main() -> None:
    if not _SRC_DIR.exists():
        raise SystemExit(
            f"Source gemma4 fixtures not found at {_SRC_DIR}. Generate/download the "
            "gemma4 test model directory first; gemma4_unified is derived from it."
        )
    _DST_DIR.mkdir(parents=True, exist_ok=True)

    # Decoder + embedding are identical to gemma4.
    for name in ("dummy_text.onnx", "dummy_embedding.onnx"):
        shutil.copyfile(_SRC_DIR / name, _DST_DIR / name)

    # Vision / speech dummies: same trivial constant-output graphs, but declare
    # the unified input dims so the fixtures document the real contract.
    _set_input_last_dim(
        _SRC_DIR / "dummy_vision.onnx", _DST_DIR / "dummy_vision.onnx", "pixel_values", _UNIFIED_PIXEL_DIM
    )
    _set_input_last_dim(
        _SRC_DIR / "dummy_speech.onnx", _DST_DIR / "dummy_speech.onnx", "audio_embeds", _UNIFIED_AUDIO_DIM
    )

    for name in _TOKENIZER_FILES:
        shutil.copyfile(_SRC_DIR / name, _DST_DIR / name)

    # genai_config.json: switch the model type and vision processor config file.
    with open(_SRC_DIR / "genai_config.json") as f:
        genai_config = json.load(f)
    genai_config["model"]["type"] = "gemma4_unified"
    genai_config["model"]["vision"]["config_filename"] = "image_processor.json"
    with open(_DST_DIR / "genai_config.json", "w") as f:
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
    with open(_DST_DIR / "image_processor.json", "w") as f:
        json.dump(image_processor, f, indent=4)

    # audio_feature_extraction.json: raw 640-sample waveform framing.
    audio_config = {
        "feature_extraction": {
            "sequence": [
                {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
                {
                    "operation": {
                        "name": "gemma4_unified_audio_frames",
                        "type": "Gemma4UnifiedAudioFrames",
                        "attrs": {"audio_samples_per_token": 640, "sampling_rate": 16000, "padding_value": 0.0},
                    }
                },
            ]
        }
    }
    with open(_DST_DIR / "audio_feature_extraction.json", "w") as f:
        json.dump(audio_config, f, indent=4)

    print(f"Wrote gemma4_unified dummy model to {_DST_DIR}")


if __name__ == "__main__":
    main()
