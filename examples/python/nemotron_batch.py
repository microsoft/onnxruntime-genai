"""
Non-streaming (batch) Nemotron ASR example.
Feeds entire audio at once through StreamingASR (no chunking).

Usage:
  python nemotron_batch.py --model_path ./nemotron-speech-streaming-en-0.6b --audio_file test.wav
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import onnxruntime_genai as og
except ImportError:
    raise ImportError("Please install onnxruntime-genai: pip install onnxruntime-genai")


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8960  # 560ms at 16kHz — model's native chunk size


def load_audio(audio_path):
    """Load a WAV file as float32 mono at 16kHz."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    audio, sr = sf.read(audio_path, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        try:
            import scipy.signal
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            print(f"Resampled from {sr}Hz to {SAMPLE_RATE}Hz")
        except ImportError:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. Install scipy for resampling.")

    return audio


def main():
    parser = argparse.ArgumentParser(description="Batch (non-streaming) Nemotron ASR")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Nemotron ONNX model directory")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to 16kHz mono WAV file")
    parser.add_argument("--execution_provider", type=str, default="cpu",
                        help="Execution provider (cpu, cuda, dml, etc.)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    config = og.Config(args.model_path)
    if args.execution_provider != "cpu":
        config.clear_providers()
        config.append_provider(args.execution_provider)
    model = og.Model(config)

    asr = og.StreamingASR(model)

    print(f"Loading audio from {args.audio_file}...")
    audio = load_audio(args.audio_file)
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"Audio duration: {duration_sec:.2f}s ({len(audio)} samples)")

    print("Running batch inference (feeding all chunks at once)...")
    t0 = time.perf_counter()

    # Feed all audio in chunk-sized pieces (no waiting between chunks)
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        asr.transcribe_chunk(chunk.astype(np.float32))

    # Flush with silence to get remaining tokens
    for _ in range(4):
        asr.transcribe_chunk(np.zeros(CHUNK_SAMPLES, dtype=np.float32))

    elapsed = time.perf_counter() - t0

    transcript = asr.get_transcript()

    print(f"\n--- Transcript ---")
    print(transcript.strip())
    print(f"\n--- Stats ---")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"Audio duration: {duration_sec:.2f}s")
    print(f"RTF (real-time factor): {elapsed / duration_sec:.3f}x")


if __name__ == "__main__":
    main()