"""
Parakeet TDT Speech Streaming ASR — Simulated real-time microphone demo.

Reads an audio file and feeds it chunk-by-chunk in real-time,
simulating live microphone input with actual wall-clock delays.

Usage:
  python parakeet_simulate_mic.py --model_path ./parakeet-tdt-0.6b-v3-onnx-fp32/fp32 --audio_file recording.wav
"""

import argparse
import os
import sys
import time
import re
import numpy as np

import onnxruntime_genai as og

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 32000  # 2 seconds at 16kHz (larger chunks since encoder has no cache)
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE


def load_audio(audio_path):
    """Load a WAV file as float32 mono at 16kHz."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("pip install soundfile")

    audio, sr = sf.read(audio_path, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        try:
            import scipy.signal
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            print(f"  Resampled {sr}Hz -> {SAMPLE_RATE}Hz")
        except ImportError:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz. pip install scipy")

    return audio


def simulate_microphone(model_path, audio_path):
    """Simulate real-time microphone streaming with wall-clock delays."""

    print("=" * 60)
    print("  PARAKEET TDT — SIMULATED REAL-TIME MICROPHONE")
    print("=" * 60)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    num_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    print(f"  Duration: {duration:.1f}s  |  Chunks: {num_chunks} x {CHUNK_DURATION*1000:.0f}ms")

    # Load model
    print(f"\nLoading model: {model_path}")
    config = og.Config(model_path)
    model = og.Model(config)
    asr = og.StreamingASR(model)
    print("  Model ready.\n")

    # Simulate
    print("-" * 60)
    print("LIVE TRANSCRIPTION (simulated real-time):")
    print("-" * 60)
    print()

    stream_start = time.time()

    for i in range(0, len(audio), CHUNK_SAMPLES):
        # Wait for real-time moment
        audio_time = i / SAMPLE_RATE
        target_time = stream_start + audio_time
        now = time.time()
        if now < target_time:
            time.sleep(target_time - now)

        # Feed chunk
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunk = chunk.astype(np.float32)

        raw_text = asr.transcribe_chunk(chunk)

        if raw_text:
            print(raw_text, end="", flush=True)

    # Flush remaining audio
    flush_text = asr.flush()
    if flush_text:
        print(flush_text, end="", flush=True)

    elapsed = time.time() - stream_start
    print(f"\n\n{'='*60}")
    print(f"Audio duration: {duration:.1f}s  |  Processing: {elapsed:.1f}s  |  RTF: {elapsed/duration:.2f}x")
    print(f"\nFull transcript:")
    print(asr.get_transcript())
    print(f"{'='*60}")


def batch_transcribe(model_path, audio_path):
    """Feed entire audio at once (no chunking) for quality comparison."""

    print("=" * 60)
    print("  PARAKEET TDT — BATCH TRANSCRIPTION")
    print("=" * 60)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    # Load model
    print(f"\nLoading model: {model_path}")
    config = og.Config(model_path)
    model = og.Model(config)
    asr = og.StreamingASR(model)

    start = time.time()
    text = asr.transcribe_chunk(audio)
    text += asr.flush()
    elapsed = time.time() - start

    print(f"\nTranscript ({elapsed:.2f}s, RTF={elapsed/duration:.2f}x):")
    print(asr.get_transcript())
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parakeet TDT Speech Streaming ASR")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--audio_file", required=True, help="Path to audio file (WAV)")
    parser.add_argument("--mode", choices=["stream", "batch"], default="stream",
                        help="stream=simulate realtime, batch=process all at once")
    args = parser.parse_args()

    if args.mode == "stream":
        simulate_microphone(args.model_path, args.audio_file)
    else:
        batch_transcribe(args.model_path, args.audio_file)
