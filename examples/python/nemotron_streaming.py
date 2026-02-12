"""
Nemotron Speech Streaming ASR example for onnxruntime-genai.

Real-time cache-aware streaming speech recognition using the
Nemotron Speech Streaming 0.6B model via onnxruntime-genai's StreamingASR API.

Usage:
  python nemotron_streaming.py --model_path ./nemotron-speech-streaming-en-0.6b --audio_file test.wav
  python nemotron_streaming.py --model_path ./nemotron-speech-streaming-en-0.6b --realtime

Setup:
  1. pip install onnxruntime-genai numpy soundfile sentencepiece
  2. pip install sounddevice  # Only for --realtime mic input
  3. Download model files from:
     https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-speech-streaming-en-0.6b
  4. Place genai_config.json (see nemotron_genai_config.json template) in the model directory

Model directory should contain:
  encoder.onnx          - Cache-aware FastConformer streaming encoder
  encoder.onnx.data     - Encoder weights
  decoder_joint.onnx    - RNNT decoder + joint network
  tokenizer.model       - SentencePiece tokenizer
  genai_config.json     - onnxruntime-genai configuration
"""

import argparse
import os
import sys
import time
import numpy as np

import onnxruntime_genai as og


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8960  # 560ms at 16kHz


def load_audio(audio_path):
    """Load a WAV file as float32 mono at 16kHz."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        # Resample if needed
        try:
            import scipy.signal
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            print(f"Resampled from {sr}Hz to {SAMPLE_RATE}Hz")
        except ImportError:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. Install scipy for resampling.")

    return audio


def load_sentencepiece_tokenizer(model_path):
    """Load SentencePiece tokenizer from model directory."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Warning: sentencepiece not installed. Token IDs will not be decoded.")
        print("  Install with: pip install sentencepiece")
        return None

    tokenizer_path = os.path.join(model_path, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print(f"Warning: tokenizer.model not found at {tokenizer_path}")
        return None

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp


def transcribe_file(model_path, audio_path, execution_provider="cpu"):
    """Transcribe an audio file using streaming chunks via onnxruntime-genai."""
    print("Loading model...")

    # Use onnxruntime-genai to load the model
    config = og.Config(model_path)
    if execution_provider != "cpu":
        config.clear_providers()
        config.append_provider(execution_provider)
    model = og.Model(config)

    # Load SentencePiece tokenizer for decoding
    sp_tokenizer = load_sentencepiece_tokenizer(model_path)

    # Create StreamingASR instance — handles encoder cache + RNNT decode internally
    asr = og.StreamingASR(model)

    # Load audio
    print(f"Loading audio from {audio_path}...")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    print(f"Audio duration: {duration:.1f}s")
    print(f"Chunk size: {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES / SAMPLE_RATE * 1000:.0f}ms)")
    print(f"Execution: CPU")
    print("Streaming: ", end="", flush=True)

    start_time = time.time()

    # Process audio in streaming chunks
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]

        # Pad last chunk if needed
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        chunk = chunk.astype(np.float32)
        raw_text = asr.transcribe_chunk(chunk)

        if raw_text:
            print(raw_text, end="", flush=True)

    # Flush with silence to get remaining tokens
    # (4 chunks: 1 extra to flush the one-chunk right-context buffer)
    for _ in range(4):
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        raw_text = asr.transcribe_chunk(silence)
        if raw_text:
            print(raw_text, end="", flush=True)

    elapsed = time.time() - start_time

    # Final full transcript from internal accumulation
    final_text = asr.get_transcript()

    print(f"\n\n{'='*60}")
    print(f"Final transcript: {final_text.strip()}")
    print(f"{'='*60}")
    print(f"Completed in {elapsed:.2f}s (audio: {duration:.2f}s, RTF: {duration / elapsed:.2f}x realtime)")

    return final_text.strip()


def parse_token_ids(raw_text):
    """Parse token IDs from <id> format like '<946><666><298>'."""
    import re
    ids = []
    for match in re.finditer(r'<(\d+)>', raw_text):
        ids.append(int(match.group(1)))
    if ids:
        return ids
    # If no <id> pattern, return empty (text was already decoded)
    return []


def stream_from_microphone(model_path, execution_provider="cpu"):
    """Real-time streaming transcription from the microphone via onnxruntime-genai."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("Please install sounddevice: pip install sounddevice")

    print("Loading model...")
    config = og.Config(model_path)
    if execution_provider != "cpu":
        config.clear_providers()
        config.append_provider(execution_provider)
    model = og.Model(config)

    sp_tokenizer = load_sentencepiece_tokenizer(model_path)
    asr = og.StreamingASR(model)

    print("Real-time streaming transcription (Ctrl+C to stop)")
    print(f"Chunk size: {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES / SAMPLE_RATE * 1000:.0f}ms)")
    print("Listening...\n")

    audio_buffer = np.zeros(0, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status:
            print(f"[Audio: {status}]", file=sys.stderr)
        audio_buffer = np.append(audio_buffer, indata[:, 0])

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            while True:
                while len(audio_buffer) >= CHUNK_SAMPLES:
                    chunk = audio_buffer[:CHUNK_SAMPLES].copy()
                    audio_buffer = audio_buffer[CHUNK_SAMPLES:]

                    raw_text = asr.transcribe_chunk(chunk)
                    if raw_text:
                        token_ids = parse_token_ids(raw_text)
                        if token_ids and sp_tokenizer:
                            decoded = sp_tokenizer.Decode(token_ids)
                            print(decoded, end="", flush=True)
                        elif raw_text:
                            print(raw_text, end="", flush=True)

                time.sleep(0.01)

    except KeyboardInterrupt:
        if sp_tokenizer:
            full_raw = asr.get_transcript()
            all_ids = parse_token_ids(full_raw)
            if all_ids:
                final = sp_tokenizer.Decode(all_ids)
            else:
                final = full_raw
        else:
            final = asr.get_transcript()
        print(f"\n\nFinal transcript: {final.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Nemotron Speech Streaming ASR (onnxruntime-genai)",
        epilog="Example: python nemotron_streaming.py --model_path ./nemotron-speech-streaming-en-0.6b --audio_file test.wav",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory (with genai_config.json)")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Path to WAV file (16kHz mono)")
    parser.add_argument("--realtime", action="store_true",
                        help="Stream from microphone in real-time")
    parser.add_argument("--execution_provider", type=str, default="cpu",
                        help="Execution provider (cpu, cuda, dml, etc.)")
    args = parser.parse_args()

    if args.realtime:
        stream_from_microphone(args.model_path, args.execution_provider)
    elif args.audio_file:
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file not found: {args.audio_file}")
            sys.exit(1)
        transcribe_file(args.model_path, args.audio_file, args.execution_provider)
    else:
        print("Please specify --audio_file <path> or --realtime")
        parser.print_help()


if __name__ == "__main__":
    main()