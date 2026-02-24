"""
Mic audio sender — run this on your Windows machine.

Captures microphone audio and streams it over TCP to the VM
where parakeet_mic_receiver.py runs the ASR.

Requirements (Windows only):
  pip install pyaudio numpy

Usage:
  python parakeet_mic_sender.py --host <VM_IP> --port 5555
"""

import argparse
import socket
import sys
import struct
import numpy as np

SAMPLE_RATE = 16000
FRAMES_PER_BUFFER = 1600  # 100ms


def main(host, port):
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pip install pyaudio")
        sys.exit(1)

    print(f"Connecting to {host}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print("Connected! Streaming mic audio... (Ctrl+C to stop)\n")

    pa = pyaudio.PyAudio()
    default_device = pa.get_default_input_device_info()
    print(f"  Microphone: {default_device['name']}")

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    try:
        while True:
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            # Send length-prefixed chunk
            sock.sendall(struct.pack("<I", len(data)) + data)
    except KeyboardInterrupt:
        print("\n[Stopped]")
    finally:
        # Send zero-length to signal end
        sock.sendall(struct.pack("<I", 0))
        stream.stop_stream()
        stream.close()
        pa.terminate()
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mic audio sender")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    main(args.host, args.port)
