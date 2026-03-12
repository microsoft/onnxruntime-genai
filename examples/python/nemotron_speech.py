# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import sys
import time
import numpy as np
import onnxruntime_genai as og


def load_config(model_path):
    """Read sample_rate and chunk_samples from genai_config.json."""
    config_path = os.path.join(model_path, "genai_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    sample_rate = config["model"]["sample_rate"]
    chunk_samples = config["model"]["chunk_samples"]
    return sample_rate, chunk_samples


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


def simulate_microphone(model_path, audio_path):
    """Stream audio through Generator + StreamingProcessor API."""
    sample_rate, chunk_samples = load_config(model_path)
    audio = load_audio(audio_path, sample_rate)
    duration = len(audio) / sample_rate
    chunk_duration = chunk_samples / sample_rate
    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    config = og.Config(model_path)
    model = og.Model(config)
    processor = og.StreamingProcessor(model)
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

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            decode_chunk()

    # Flush remaining audio
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
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