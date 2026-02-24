"""
Mic audio receiver — run this on the VM.

Listens for audio streamed from parakeet_mic_sender.py (on your Windows machine)
and transcribes it in real-time using onnxruntime-genai StreamingASR.

Requirements (VM):
  pip install numpy onnxruntime-genai

Usage:
  python parakeet_mic_receiver.py --model_path ~/parakeet-tdt-0.6b-v3-onnx-fp32/fp32 --port 5555
"""

import argparse
import socket
import struct
import sys
import threading
import time
import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.4
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)


def main(model_path, port):
    import onnxruntime_genai as og

    print("=" * 60)
    print("  PARAKEET TDT — REMOTE MICROPHONE RECEIVER")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_path}")
    config = og.Config(model_path)
    model = og.Model(config)
    asr = og.StreamingASR(model)
    print("  Model ready.\n")

    # Listen for connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"Listening on port {port}... (run parakeet_mic_sender.py on your Windows machine)")

    conn, addr = server.accept()
    print(f"Connected from {addr}\n")

    print("-" * 60)
    print("LIVE TRANSCRIPTION:")
    print("-" * 60)
    print()

    audio_buffer = []

    def recv_exact(sock, n):
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    try:
        while True:
            # Read length prefix
            header = recv_exact(conn, 4)
            if header is None:
                break
            length = struct.unpack("<I", header)[0]
            if length == 0:
                break  # End signal

            # Read audio data
            raw = recv_exact(conn, length)
            if raw is None:
                break

            samples = np.frombuffer(raw, dtype=np.float32)
            audio_buffer.extend(samples.tolist())

            # Process when we have enough
            while len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = np.array(audio_buffer[:CHUNK_SAMPLES], dtype=np.float32)
                del audio_buffer[:CHUNK_SAMPLES]

                text = asr.transcribe_chunk(chunk)
                if text:
                    print(text, end="", flush=True)

    except (ConnectionResetError, BrokenPipeError):
        print("\n[Connection closed]")
    except KeyboardInterrupt:
        print("\n[Stopped]")

    # Flush remaining
    if audio_buffer:
        chunk = np.array(audio_buffer, dtype=np.float32)
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        asr.transcribe_chunk(chunk)

    flush_text = asr.flush()
    if flush_text:
        print(flush_text, end="", flush=True)

    conn.close()
    server.close()

    print(f"\n\n{'=' * 60}")
    print("Full transcript:")
    print(asr.get_transcript())
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parakeet TDT — Remote Microphone Receiver")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    main(args.model_path, args.port)
