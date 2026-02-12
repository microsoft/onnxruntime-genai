"""
Nemotron Speech Streaming ASR — Live Microphone Demo.

Captures audio from your microphone and transcribes in real-time.

Usage:
  python nemotron_mic.py --model_path ./nemotron-speech-streaming-en-0.6b

Requirements:
  pip install sounddevice sentencepiece soundfile onnxruntime-genai
"""

import argparse
import os
import sys
import re
import time
import threading
import numpy as np

import onnxruntime_genai as og

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8960  # 560ms at 16kHz


def load_tokenizer(model_path):
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
    return [int(m.group(1)) for m in re.finditer(r'<(\d+)>', raw_text)]


def main():
    parser = argparse.ArgumentParser(description="Nemotron Speech — Live Microphone ASR")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: pip install sounddevice")
        sys.exit(1)

    # List audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print()

    # Load model
    print(f"Loading model from {args.model_path}...")
    config = og.Config(args.model_path)
    model = og.Model(config)
    sp = load_tokenizer(args.model_path)
    asr = og.StreamingASR(model)
    print("Model loaded.\n")

    print("=" * 60)
    print("  LIVE MICROPHONE TRANSCRIPTION")
    print("  Speak into your microphone. Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    # Shared buffer
    audio_buffer = []
    buffer_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[Audio warning: {status}]", file=sys.stderr)
        with buffer_lock:
            audio_buffer.append(indata[:, 0].copy())

    all_token_ids = []
    start_time = time.time()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=args.device,
            callback=audio_callback,
        ):
            print("🎤 Listening...\n")

            while True:
                # Collect enough samples
                chunk = None
                with buffer_lock:
                    if audio_buffer:
                        combined = np.concatenate(audio_buffer)
                        if len(combined) >= CHUNK_SAMPLES:
                            chunk = combined[:CHUNK_SAMPLES]
                            leftover = combined[CHUNK_SAMPLES:]
                            audio_buffer.clear()
                            if len(leftover) > 0:
                                audio_buffer.append(leftover)

                if chunk is not None:
                    raw_text = asr.transcribe_chunk(chunk)
                    if raw_text:
                        token_ids = parse_token_ids(raw_text)
                        if token_ids:
                            all_token_ids.extend(token_ids)
                            if sp:
                                decoded = sp.Decode(token_ids)
                                print(decoded, end="", flush=True)
                            else:
                                print(raw_text, end="", flush=True)
                else:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time

        if sp and all_token_ids:
            final_text = sp.Decode(all_token_ids)
        else:
            final_text = asr.get_transcript()

        print(f"\n\n{'=' * 60}")
        print(f"FINAL TRANSCRIPT:")
        print(f"  {final_text.strip()}")
        print(f"{'=' * 60}")
        print(f"  Duration : {elapsed:.1f}s")
        print(f"  Tokens   : {len(all_token_ids)}")
        print(f"  Execution: CPU")


if __name__ == "__main__":
    main()