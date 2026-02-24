"""
Parakeet TDT — Real microphone demo.

Captures audio from the system microphone and transcribes in real-time
using onnxruntime-genai StreamingASR.

Requirements:
  pip install pyaudio numpy onnxruntime-genai

Usage:
  python parakeet_mic.py --model_path ./parakeet-tdt-0.6b-v3-onnx-fp32/fp32
"""

import argparse
import sys
import threading
import time
import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.4  # seconds per chunk
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
FRAMES_PER_BUFFER = 1600  # 100ms at 16kHz


def main(model_path):
    import onnxruntime_genai as og

    try:
        import pyaudio
    except ImportError:
        print("ERROR: pip install pyaudio")
        sys.exit(1)

    print("=" * 60)
    print("  PARAKEET TDT — LIVE MICROPHONE")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_path}")
    config = og.Config(model_path)
    model = og.Model(config)
    asr = og.StreamingASR(model)
    print("  Model ready.\n")

    # Audio buffer (thread-safe via lock)
    audio_buffer = []
    buffer_lock = threading.Lock()
    running = True

    # PyAudio callback — runs in a separate thread
    def audio_callback(in_data, frame_count, time_info, status):
        if status:
            print(f"  [Audio status: {status}]", file=sys.stderr)
        samples = np.frombuffer(in_data, dtype=np.float32)
        with buffer_lock:
            audio_buffer.extend(samples.tolist())
        return (None, pyaudio.paContinue)

    # Start microphone
    pa = pyaudio.PyAudio()

    # Find default input device
    default_device = pa.get_default_input_device_info()
    print(f"  Microphone: {default_device['name']}")
    print(f"  Sample rate: {SAMPLE_RATE}Hz")
    print(f"  Chunk size: {CHUNK_DURATION}s ({CHUNK_SAMPLES} samples)\n")

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        stream_callback=audio_callback,
    )

    print("-" * 60)
    print("LIVE TRANSCRIPTION (press Ctrl+C to stop):")
    print("-" * 60)
    print()

    stream.start_stream()

    try:
        while True:
            # Wait until we have enough samples for a chunk
            time.sleep(0.1)

            with buffer_lock:
                if len(audio_buffer) < CHUNK_SAMPLES:
                    continue
                chunk = np.array(audio_buffer[:CHUNK_SAMPLES], dtype=np.float32)
                del audio_buffer[:CHUNK_SAMPLES]

            text = asr.transcribe_chunk(chunk)
            if text:
                print(text, end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n[Stopping...]")

    # Flush
    flush_text = asr.flush()
    if flush_text:
        print(flush_text, end="", flush=True)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print(f"\n\n{'=' * 60}")
    print("Full transcript:")
    print(asr.get_transcript())
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parakeet TDT — Live Microphone")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    args = parser.parse_args()
    main(args.model_path)
