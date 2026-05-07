# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Parakeet TDT speech recognition — Whisper-style API.

Mirrors examples/python/whisper.py so the same public API
(`og.Audios`, `model.create_multimodal_processor()`, `og.Generator`) is used:

  python parakeet.py --model_path <model_dir> --audio_file <audio>

The model loads the audio in one shot and decodes it with the standard
Generator loop. Internally the encoder is fed in fixed-length chunks
(0.8 s with 9 s left context + 1.6 s right context, matching NeMo).
"""

import argparse
import os

import onnxruntime_genai as og


def run(args: argparse.Namespace) -> None:
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    print(f"Loading audio: {args.audio_file}")
    audios = og.Audios.open(args.audio_file)

    print("Processing audio...")
    inputs = processor("", audios=audios)

    params = og.GeneratorParams(model)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    while not generator.is_done():
        generator.generate_next_token()

    # The processor injects a single placeholder token at index 0; skip it.
    tokens = list(generator.get_sequence(0))
    if tokens:
        tokens = tokens[1:]
    transcription = processor.decode(tokens)

    print()
    print("Transcription:")
    print(f"    {transcription.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the Parakeet model directory")
    parser.add_argument("-a", "--audio_file", type=str, required=True, help="Path to the audio file (WAV/MP3/...)")
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "follow_config"],
        help="Execution provider. Defaults to follow_config.",
    )
    args = parser.parse_args()
    run(args)
