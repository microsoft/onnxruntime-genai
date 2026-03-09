# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import sys
import time
import numpy as np
import onnxruntime_genai as og

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8960
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE


def load_audio(audio_path):
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import scipy.signal
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def simulate_microphone(model_path, audio_path):
    """Stream audio through Generator + StreamingAudioProcessor API."""
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    num_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    print(f"Audio: {duration:.1f}s | {num_chunks} chunks × {CHUNK_DURATION*1000:.0f}ms")

    config = og.Config(model_path)
    model = og.Model(config)
    processor = og.StreamingAudioProcessor(model)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)

    print("-" * 60)
    stream_start = time.time()
    full_transcript = ""

    def decode_chunk():
        nonlocal full_transcript
        while not generator.is_done():
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) > 0:
                text = tokenizer_stream.decode(tokens[0])
                if text:
                    print(text, end="", flush=True)
                    full_transcript += text

    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES].astype(np.float32)
        mel = processor.process(chunk)
        if mel is not None:
            generator.set_model_input("audio_features", mel)
            decode_chunk()

    # Flush remaining audio
    mel = processor.flush()
    if mel is not None:
        generator.set_model_input("audio_features", mel)
        decode_chunk()

    # Feed silence chunks for right context
    for _ in range(4):
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        mel = processor.process(silence)
        if mel is not None:
            generator.set_model_input("audio_features", mel)
            decode_chunk()

    total_wall = time.time() - stream_start

    print(f"\n{'=' * 60}")
    print(f"  {full_transcript.strip()}")
    print(f"{'=' * 60}")
    print(f"  Audio: {duration:.2f}s | Wall: {total_wall:.2f}s | RTF: {duration/total_wall:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)
    simulate_microphone(args.model_path, args.audio_file)


if __name__ == "__main__":
    main()