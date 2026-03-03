# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Nemotron Batch ASR Example
==========================

Offline (non-streaming) speech recognition using the Nemotron ASR model.
Unlike the streaming API which processes audio chunk-by-chunk in real time,
the batch API takes complete audio files and returns the full transcript at once.

Usage:
    python nemotron_batch.py --model_path /path/to/nemotron_model --audio_file /path/to/audio.wav

    # Transcribe multiple files:
    python nemotron_batch.py --model_path /path/to/nemotron_model --audio_file file1.wav file2.wav file3.wav
"""

import argparse
import os
import sys
import time
import re
import numpy as np
import onnxruntime_genai as og

SAMPLE_RATE = 16000


def load_audio(audio_path):
    """Load an audio file and resample to 16kHz mono float32."""
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    if sr != SAMPLE_RATE:
        import scipy.signal

        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def load_tokenizer(model_path):
    """Load the SentencePiece tokenizer if available."""
    import sentencepiece as spm

    path = os.path.join(model_path, "tokenizer.model")
    if not os.path.exists(path):
        return None
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def parse_token_ids(raw_text):
    """Extract token IDs from raw text like '<123><456>'."""
    return [int(m.group(1)) for m in re.finditer(r"<(\d+)>", raw_text)]


def transcribe_batch(model_path, audio_files):
    """Transcribe one or more audio files using the batch ASR API."""
    # Load model once
    config = og.Config(model_path)
    model = og.Model(config)
    sp = load_tokenizer(model_path)

    # Create batch ASR instance
    asr = og.BatchASR(model)

    print(f"Model loaded from: {model_path}")
    print(f"Files to transcribe: {len(audio_files)}")
    print("=" * 60)

    total_audio_duration = 0.0
    total_wall_time = 0.0

    for audio_path in audio_files:
        # Load audio
        audio = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE

        print(f"\nFile: {os.path.basename(audio_path)}")
        print(f"  Duration: {duration:.2f}s ({len(audio)} samples)")

        # Transcribe
        start_time = time.time()
        raw_text = asr.transcribe(audio)
        wall_time = time.time() - start_time

        # Post-process with SentencePiece if available
        if sp:
            token_ids = parse_token_ids(raw_text)
            final_text = sp.Decode(token_ids) if token_ids else raw_text
        else:
            final_text = raw_text

        total_audio_duration += duration
        total_wall_time += wall_time

        print(f"  Wall time: {wall_time:.2f}s | RTF: {duration / wall_time:.2f}x")
        print(f"  Transcript: {final_text.strip()}")

    print(f"\n{'=' * 60}")
    print(f"Total audio: {total_audio_duration:.2f}s | Total wall: {total_wall_time:.2f}s | "
          f"Avg RTF: {total_audio_duration / total_wall_time:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Offline batch transcription using Nemotron ASR"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Nemotron ASR model directory",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to audio file(s) to transcribe (WAV, FLAC, etc.)",
    )
    args = parser.parse_args()

    # Validate files exist
    for f in args.audio_file:
        if not os.path.exists(f):
            print(f"Error: {f} not found")
            sys.exit(1)

    transcribe_batch(args.model_path, args.audio_file)


if __name__ == "__main__":
    main()
