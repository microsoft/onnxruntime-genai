#!/usr/bin/env python3
"""Moonshine Streaming ASR using onnxruntime-genai StreamingProcessor API."""

import argparse
import json
import os
import sys
import time
import numpy as np
import onnxruntime_genai as og


def load_audio(audio_path, sample_rate):
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import scipy.signal
        num_samples = int(len(audio) * sample_rate / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def decode_tokens(generator, tokenizer_stream):
    text = ""
    while not generator.is_done():
        generator.generate_next_token()
        tokens = generator.get_next_tokens()
        if len(tokens) > 0:
            token_text = tokenizer_stream.decode(tokens[0])
            if token_text:
                print(token_text, end="", flush=True)
                text += token_text
    return text


def main():
    parser = argparse.ArgumentParser(description="Moonshine Streaming ASR")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: {args.audio} not found")
        sys.exit(1)

    # Read config
    config_path = os.path.join(args.model, "genai_config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    sample_rate = cfg["model"]["sample_rate"]
    chunk_samples = cfg["model"]["chunk_samples"]

    # Load audio
    audio = load_audio(args.audio, sample_rate)
    duration = len(audio) / sample_rate
    print(f"Audio: {duration:.1f}s, chunk={chunk_samples/sample_rate*1000:.0f}ms")

    # Load model
    config = og.Config(args.model)
    model = og.Model(config)
    processor = og.StreamingProcessor(model)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)

    print("-" * 60)
    t0 = time.perf_counter()
    full_transcript = ""

    # Feed audio in chunks - processor accumulates internally
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            full_transcript += decode_tokens(generator, tokenizer_stream)

    # Flush remaining audio - this runs the encoder
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        full_transcript += decode_tokens(generator, tokenizer_stream)

    total = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"  {full_transcript.strip()}")
    print(f"{'=' * 60}")
    print(f"  Audio: {duration:.2f}s | Wall: {total:.2f}s | RTF: {duration/total:.2f}x")


if __name__ == "__main__":
    main()
