# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Parakeet TDT speech recognition

  python parakeet.py --model_path <model_dir> --audio_file <audio>

The model loads the audio in one shot and decodes it with the standard
Generator loop.
"""

import argparse
import os
import time
import wave

import onnxruntime_genai as og


def _audio_duration_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except wave.Error:
        pass
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


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
    audio_seconds = _audio_duration_seconds(args.audio_file)

    print("Processing audio...")
    t0 = time.perf_counter()
    inputs = processor("", audios=audios)

    params = og.GeneratorParams(model)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    while not generator.is_done():
        generator.generate_next_token()
    elapsed = time.perf_counter() - t0

    transcription = processor.decode(generator.get_sequence(0))

    print()
    print("Transcription:")
    print(f"    {transcription.strip()}")

    print()
    if audio_seconds > 0:
        rtfx = audio_seconds / elapsed if elapsed > 0 else float("inf")
        print(f"Audio duration: {audio_seconds:.2f}s | Inference: {elapsed:.2f}s | RTFx: {rtfx:.2f}x")
    else:
        print(f"Inference: {elapsed:.2f}s")


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
