# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import sys
import time
import numpy as np
import psutil
import onnxruntime_genai as og
from common import get_config


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


def decode_tokens(generator, tokenizer_stream):
    """Decode all available tokens from the generator, returning the text."""
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


def simulate_microphone(model_path, audio_path, execution_provider, enable_vad=False, vad_threshold=0.5):
    """Stream audio through Generator + StreamingProcessor API."""
    sample_rate, chunk_samples = load_config(model_path)
    audio = load_audio(audio_path, sample_rate)
    duration = len(audio) / sample_rate

    config = get_config(model_path, execution_provider)
    model = og.Model(config)
    processor = og.StreamingProcessor(model)

    process = psutil.Process()

    # VAD is controlled via genai_config.json (disabled by default).
    # Override programmatically:
    if enable_vad and processor.get_option("vad_enabled") != "true":
        processor.set_option("vad_enabled", "true")
    if vad_threshold != 0.5 and processor.get_option("vad_enabled") == "true":
        processor.set_option("vad_threshold", str(vad_threshold))
    if processor.get_option("vad_enabled") == "true":
        actual_threshold = processor.get_option("vad_threshold")
        print(f"  VAD: enabled (threshold={actual_threshold})")
    else:
        print("  VAD: disabled")

    # Measure memory after first chunk (VAD session is created lazily)
    mem_before_vad = process.memory_info().rss

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)

    print("-" * 60)
    stream_start = time.time()
    full_transcript = ""
    vad_enabled = processor.get_option("vad_enabled") == "true"
    chunks_total = 0
    chunks_processed = 0
    chunks_skipped = 0
    vad_time_total = 0.0

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        chunks_total += 1
        chunk_start = time.time()
        inputs = processor.process(chunk)
        chunk_elapsed = time.time() - chunk_start
        # Measure memory after first chunk (VAD session created lazily on first process call)
        if chunks_total == 1 and vad_enabled:
            mem_after_vad = process.memory_info().rss
        if inputs is not None:
            chunks_processed += 1
            generator.set_inputs(inputs)
            full_transcript += decode_tokens(generator, tokenizer_stream)
        else:
            chunks_skipped += 1
        if vad_enabled:
            vad_time_total += chunk_elapsed

    # Flush remaining audio
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        full_transcript += decode_tokens(generator, tokenizer_stream)

    total_wall = time.time() - stream_start

    print(f"\n{'=' * 60}")
    print(f"  {full_transcript.strip()}")
    print(f"{'=' * 60}")
    print(f"  Audio: {duration:.2f}s | Wall: {total_wall:.2f}s | RTF: {duration/total_wall:.2f}x")
    if vad_enabled:
        avg_vad_ms = (vad_time_total / max(chunks_total, 1)) * 1000
        chunk_duration_ms = (chunk_samples / sample_rate) * 1000
        vad_pct = (avg_vad_ms / chunk_duration_ms) * 100
        print(f"  VAD Metrics: {chunks_total} total chunks, {chunks_processed} processed, "
              f"{chunks_skipped} skipped ({chunks_skipped / max(chunks_total, 1) * 100:.1f}% compute saved)")
        print(f"  VAD Overhead: {vad_pct:.1f}% avg per chunk ({avg_vad_ms:.1f}ms)")
        vad_mem_mb = (mem_after_vad - mem_before_vad) / (1024 * 1024)
        print(f"  VAD Memory: {vad_mem_mb:.1f}MB (Total Additional RSS)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--enable_vad", action="store_true",
                        help="Enable Voice Activity Detection (disabled by default; or set in genai_config.json).")
    parser.add_argument("--vad_threshold", type=float, default=0.5,
                        help="VAD speech probability threshold (default: 0.5).")
    parser.add_argument("-e", "--execution_provider", type=str, required=False, default="follow_config",
                        choices=["cpu", "cuda", "dml", "follow_config"],
                        help="Execution provider to run with. Defaults to follow_config.")
    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)
    simulate_microphone(args.model_path, args.audio_file, args.execution_provider,
                        enable_vad=args.enable_vad, vad_threshold=args.vad_threshold)


if __name__ == "__main__":
    main()