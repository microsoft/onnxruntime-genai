# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
"""Make an ONNX Runtime GenAI package of VibeVoice-Realtime-0.5B.

Source: the ONNX FP16 export at https://huggingface.co/nenad1002/microsoft-vibevoice-0.5B-onnx-fp16 of
https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B (MIT). Download it first, for example:

    huggingface-cli download nenad1002/microsoft-vibevoice-0.5B-onnx-fp16 --revision 1227c4cb --local-dir vibevoice-hf

Then:

    python vibevoice-package.py --source vibevoice-hf --output vibevoice-realtime-0.5b

The package holds the five graphs, the tokenizer, the English voice presets in voices/ and a
genai_config.json for model type "vibevoice". Files are hard-linked when possible, otherwise copied.
Each .npz voice preset becomes an ONNX graph without inputs (voices/<name>.onnx) whose float16 outputs
are the cached voice prompt, so ONNX Runtime GenAI loads it like any other graph. Needs numpy and onnx.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

GRAPHS = {
    "text_lm": "text_lm_kv.onnx",
    "tts_lm": "tts_lm_kv.onnx",
    "diffusion_head": "diffusion_head.onnx",
    "acoustic_connector": "acoustic_connector.onnx",
    "vocoder": "vocoder.onnx",
}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
# The model card says the model is intended for English. "in" presets are Indian English.
LANGUAGES = {"en": "en", "in": "en-in"}
GENDERS = {"woman": "female", "man": "male"}
DEFAULT_VOICE = "Emma"
END_OF_TEXT = "<|endoftext|>"


def place(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def voice_graph(source: Path, target: Path):
    """The arrays that the runtime reads from an .npz preset, as Constant outputs of an ONNX graph."""
    with np.load(source) as arrays:
        names = [n for n in arrays.files if n.startswith(("lm_key_", "lm_value_", "tts_lm_", "neg_tts_lm_"))]
        names = [n for n in names if n != "tts_lm_hidden"]
        values = {name: arrays[name].astype(np.float16) for name in names}
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value, f"{name}_value"))
        for name, value in values.items()
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(value.shape)) for name, value in values.items()
    ]
    graph = helper.make_graph(nodes, f"vibevoice_voice_{source.stem}", [], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="vibevoice-package")
    model.ir_version = 8
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)


def voices(source: Path):
    """en-Emma_woman.npz -> (Emma, en, female), for the English presets only."""
    found = []
    for file in sorted(source.glob("*.npz")):
        prefix, _, rest = file.stem.partition("-")
        name, _, gender = rest.partition("_")
        if prefix in LANGUAGES and name and gender in GENDERS:
            found.append(
                {
                    "name": name,
                    "file": f"voices/{file.stem}.onnx",
                    "language": LANGUAGES[prefix],
                    "gender": GENDERS[gender],
                }
            )
    if not found:
        raise SystemExit(f"No English voice presets (en-<Name>_<man|woman>.npz) in {source}")
    return found


def token_id(tokenizer_config: dict, token: str) -> int:
    for token_id_text, value in tokenizer_config.get("added_tokens_decoder", {}).items():
        if value.get("content") == token:
            return int(token_id_text)
    raise SystemExit(f"{token} is not in tokenizer_config.json")


def genai_config(source: Path, voice_list):
    model = json.loads((source / "config.json").read_text())
    schedule = json.loads((source / "schedule.json").read_text())
    end_of_text = token_id(json.loads((source / "tokenizer_config.json").read_text()), END_OF_TEXT)
    if model.get("sample_rate", 24000) != 24000:
        raise SystemExit("The ONNX Runtime GenAI VibeVoice model type expects 24 kHz audio")
    default_voice = DEFAULT_VOICE if any(v["name"] == DEFAULT_VOICE for v in voice_list) else voice_list[0]["name"]
    return {
        "model": {
            "type": "vibevoice",
            # Positions of the Qwen2.5 TTS LM: the voice prompt, the text and the speech frames.
            "context_length": 8192,
            "vocab_size": 151936,
            "bos_token_id": end_of_text,
            "eos_token_id": end_of_text,
            "pad_token_id": end_of_text,
            "sample_rate": 24000,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    # Synthesis alternates between five sessions; spinning threads in the idle pools
                    # multiply the CPU time without making synthesis faster.
                    "session.intra_op.allow_spinning": "0",
                    "provider_options": [],
                }
            },
            "default_voice": default_voice,
            "voices": voice_list,
            "vibevoice": {
                **{graph: {"filename": filename} for graph, filename in GRAPHS.items()},
                "speech_scaling_factor": model["speech_scaling_factor"],
                "speech_bias_factor": model["speech_bias_factor"],
                "text_window_size": model["text_window_size"],
                "speech_window_size": model["speech_window_size"],
                "cfg_scale": model["cfg_scale"],
                "diffusion": {"timesteps": schedule["timesteps"], "sigmas": schedule["sigmas"]},
            },
        },
        # A fixed seed makes the diffusion noise, and so the audio, repeatable.
        "search": {"random_seed": 24301},
    }


def main():
    parser = argparse.ArgumentParser(description="Make an ONNX Runtime GenAI package of VibeVoice-Realtime-0.5B")
    parser.add_argument("--source", type=Path, required=True, help="Folder of the Hugging Face download")
    parser.add_argument("--output", type=Path, required=True, help="Package folder to write")
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename in GRAPHS.values():
        place(source / filename, output / filename)
        if (source / f"{filename}.data").exists():
            place(source / f"{filename}.data", output / f"{filename}.data")
    for filename in TOKENIZER_FILES:
        place(source / filename, output / filename)
    voice_list = voices(source)
    for voice in voice_list:
        voice_graph(source / f"{Path(voice['file']).stem}.npz", output / voice["file"])
    if (source / "README.md").exists():
        place(source / "README.md", output / "MODEL_CARD.md")

    config = genai_config(source, voice_list)
    (output / "genai_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {output} with voices {', '.join(v['name'] for v in voice_list)} (default {config['model']['default_voice']})"
    )


if __name__ == "__main__":
    main()
