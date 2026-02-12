"""
Nemotron Speech Streaming ASR — Simulated real-time microphone demo.

Reads an audio file and feeds it chunk-by-chunk in real-time,
simulating live microphone input with actual wall-clock delays.

Usage:
  python nemotron_simulate_mic.py --model_path ./nemotron-speech-streaming-en-0.6b --audio_file recording.wav
"""

import argparse
import os
import sys
import time
import re
import numpy as np

import onnxruntime_genai as og

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8960  # 560ms at 16kHz
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
            print(f"  Resampled {sr}Hz → {SAMPLE_RATE}Hz")
        except ImportError:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz. pip install scipy")

    return audio


def load_tokenizer(model_path):
    """Load SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Warning: pip install sentencepiece for text decoding")
        return None
    path = os.path.join(model_path, "tokenizer.model")
    if not os.path.exists(path):
        return None
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def parse_token_ids(raw_text):
    """Parse token IDs from '<id>' format."""
    return [int(m.group(1)) for m in re.finditer(r'<(\d+)>', raw_text)]


def simulate_microphone(model_path, audio_path):
    """Simulate real-time microphone streaming with wall-clock delays."""

    print("=" * 60)
    print("  🎤 NEMOTRON SPEECH — SIMULATED REAL-TIME MICROPHONE")
    print("=" * 60)

    # Load audio
    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    num_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    print(f"  Duration: {duration:.1f}s  |  Chunks: {num_chunks} × {CHUNK_DURATION*1000:.0f}ms")

    # Load model
    print(f"\nLoading model: {model_path}")
    config = og.Config(model_path)
    model = og.Model(config)
    sp = load_tokenizer(model_path)
    asr = og.StreamingASR(model)
    print("  Model ready.\n")

    # Simulate
    print("-" * 60)
    print("LIVE TRANSCRIPTION (simulated real-time):")
    print("-" * 60)
    print()

    all_token_ids = []
    stream_start = time.time()

    for i in range(0, len(audio), CHUNK_SAMPLES):
        # ── Wait for real-time moment ──
        audio_time = i / SAMPLE_RATE
        target_time = stream_start + audio_time
        now = time.time()
        if now < target_time:
            time.sleep(target_time - now)

        # ── Feed chunk ──
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunk = chunk.astype(np.float32)

        raw_text = asr.transcribe_chunk(chunk)

        if raw_text:
            token_ids = parse_token_ids(raw_text)
            if token_ids:
                all_token_ids.extend(token_ids)
                if sp:
                    print(sp.Decode(token_ids), end="", flush=True)
                else:
                    print(raw_text, end="", flush=True)

    # ── Flush remaining ──
    for _ in range(3):
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        raw_text = asr.transcribe_chunk(silence)
        if raw_text:
            token_ids = parse_token_ids(raw_text)
            if token_ids:
                all_token_ids.extend(token_ids)
                if sp:
                    print(sp.Decode(token_ids), end="", flush=True)

    total_wall = time.time() - stream_start

    if sp and all_token_ids:
        final_text = sp.Decode(all_token_ids)
    else:
        final_text = asr.get_transcript()

    print(f"\n\n{'=' * 60}")
    print(f"FINAL TRANSCRIPT:")
    print(f"  {final_text.strip()}")
    print(f"{'=' * 60}")
    print(f"  Audio duration : {duration:.2f}s")
    print(f"  Wall clock     : {total_wall:.2f}s")
    print(f"  RTF            : {duration / total_wall:.2f}x realtime")
    print(f"  Tokens emitted : {len(all_token_ids)}")
    print(f"  Execution      : CPU")


def main():
    parser = argparse.ArgumentParser(description="Nemotron Speech — Simulated Microphone")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: {args.audio_file} not found")
        sys.exit(1)

    simulate_microphone(args.model_path, args.audio_file)


if __name__ == "__main__":
    main()