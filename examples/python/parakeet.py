# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Parakeet-TDT batch transcription example (mirrors whisper.py).

Usage:
    python parakeet.py -m /path/to/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 -a /path/to/audio.wav
"""
import argparse
import os
import time
import wave

import onnxruntime_genai as og


def _audio_duration_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(path)
            return info.frames / float(info.samplerate)
        except Exception:
            return 0.0


def run(args: argparse.Namespace):
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    audio_paths = [p.strip() for p in args.audio.split(",") if p.strip()]
    if not audio_paths:
        raise ValueError("No audio paths provided.")
    for p in audio_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    for audio_path in audio_paths:
        print(f"\nTranscribing: {audio_path}")
        duration = _audio_duration_seconds(audio_path)
        audios = og.Audios.open(audio_path)
        inputs = processor(prompts=[""], audios=audios)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_length=args.max_length,
            batch_size=1,
        )

        generator = og.Generator(model, params)
        t_total0 = time.perf_counter()
        generator.set_inputs(inputs)

        t0 = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
        gen_elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - t_total0

        tokens = generator.get_sequence(0)
        text = processor.decode(tokens)
        rtfx = (duration / total_elapsed) if total_elapsed > 0 and duration > 0 else 0.0
        print(f"  audio={duration:.2f}s  set_inputs+gen={total_elapsed:.2f}s  gen_only={gen_elapsed:.2f}s  RTFx={rtfx:.1f}x")
        print(f"  {text.strip()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_path", required=True, help="Path to model dir containing genai_config.json")
    p.add_argument("-a", "--audio", required=True, help="Comma-separated audio file paths")
    p.add_argument("-e", "--execution_provider", default="follow_config",
                   choices=["cpu", "cuda", "follow_config"])
    p.add_argument("--max_length", type=int, default=4096)
    args = p.parse_args()
    run(args)
