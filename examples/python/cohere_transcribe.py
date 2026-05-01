"""
Cohere Transcribe ONNX inference via onnxruntime-genai.
Long audio is automatically split into chunks using energy-based boundaries
(matching the PyTorch reference implementation).

Usage:
  OMP_NUM_THREADS=4 python cohere_transcribe.py -m /path/to/model -a audio.wav
"""

import argparse
import io
import math
import os
import struct
import time
import wave

import numpy as np
import onnxruntime_genai as og


PROMPT_TOKENS = [
    "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
    "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
]

# Chunking defaults (from Cohere config)
MAX_AUDIO_CLIP_S = 35.0
OVERLAP_CHUNK_S = 5.0
MIN_ENERGY_WINDOW_SAMPLES = 1600


def read_wav_mono_f32(path):
    """Read a WAV file and return (samples_float32, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(n)

    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)

    return samples, sr


def samples_to_wav_bytes(samples_f32, sample_rate):
    """Convert float32 samples to in-memory WAV bytes."""
    int16 = (samples_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def _find_split_point_energy(waveform, start_idx, end_idx, min_window):
    """Find the quietest point in waveform[start_idx:end_idx]."""
    segment = waveform[start_idx:end_idx]
    if segment.shape[0] <= min_window:
        return (start_idx + end_idx) // 2
    min_energy = float("inf")
    quietest = start_idx
    upper = segment.shape[0] - min_window
    for i in range(0, upper, min_window):
        w = segment[i : i + min_window]
        energy = float(np.sqrt(np.mean(w * w)))
        if energy < min_energy:
            min_energy = energy
            quietest = start_idx + i
    return quietest


def split_audio_chunks(waveform, sample_rate,
                       max_clip_s=MAX_AUDIO_CLIP_S,
                       overlap_s=OVERLAP_CHUNK_S,
                       min_energy_window=MIN_ENERGY_WINDOW_SAMPLES):
    """Split waveform into chunks at quiet boundaries. Returns list of 1-D arrays."""
    chunk_size = max(1, int(round(max_clip_s * sample_rate)))
    boundary_ctx = max(1, int(round(overlap_s * sample_rate)))
    total = waveform.shape[0]
    if total <= chunk_size:
        return [waveform]

    chunks = []
    idx = 0
    while idx < total:
        if idx + chunk_size >= total:
            chunks.append(waveform[idx:total])
            break
        search_start = max(idx, idx + chunk_size - boundary_ctx)
        search_end = min(idx + chunk_size, total)
        if search_end <= search_start:
            split = idx + chunk_size
        else:
            split = _find_split_point_energy(waveform, search_start, search_end, min_energy_window)
        split = max(idx + 1, min(split, total))
        chunks.append(waveform[idx:split])
        idx = split

    return chunks


def transcribe_chunk(model, processor, tokenizer, chunk_wav_bytes, prompt):
    """Transcribe a single audio chunk. Returns decoded text."""
    audios = og.Audios.open_bytes(chunk_wav_bytes)
    inputs = processor([prompt], audios=audios)

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=1024,
        num_beams=1,
        batch_size=1,
    )

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    tokens = []
    while not generator.is_done():
        generator.generate_next_token()
        tokens.append(generator.get_sequence(0)[-1])

    # Decode all tokens
    text = "".join(tokenizer.create_stream().decode(t) for t in tokens)
    return text.strip()


def run(args):
    print(f"Loading model from {args.model_path} ({args.execution_provider}) ...")
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider == "cuda":
        config.append_provider("cuda")
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    audio_paths = [p.strip() for p in args.audio.split(",")]
    for p in audio_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p}")

    prompt = "".join(PROMPT_TOKENS)

    t0 = time.time()
    all_texts = []

    for audio_path in audio_paths:
        samples, sr = read_wav_mono_f32(audio_path)
        duration = len(samples) / sr
        chunks = split_audio_chunks(samples, sr)
        print(f"Audio: {audio_path} ({duration:.1f}s, {len(chunks)} chunk(s))")

        chunk_texts = []
        for i, chunk in enumerate(chunks):
            chunk_dur = len(chunk) / sr
            print(f"  Chunk {i+1}/{len(chunks)} ({chunk_dur:.1f}s) ... ", end="", flush=True)
            wav_bytes = samples_to_wav_bytes(chunk, sr)
            text = transcribe_chunk(model, processor, tokenizer, wav_bytes, prompt)
            chunk_texts.append(text)
            print(text[:80] + ("..." if len(text) > 80 else ""))

        # Join chunks: strip each, remove leading punct from non-first chunks
        # (matches PyTorch join_chunk_texts behavior)
        parts = [t.strip() for t in chunk_texts if t.strip()]
        for i in range(1, len(parts)):
            parts[i] = parts[i].lstrip(".,;:!? ")
        full_text = " ".join(parts)
        all_texts.append(full_text)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    for text in all_texts:
        print(text)

    total_dur = sum(len(read_wav_mono_f32(p)[0]) / read_wav_mono_f32(p)[1] for p in audio_paths)
    print(f"\nElapsed: {elapsed:.2f}s | Audio: {total_dur:.1f}s | RTFx: {total_dur / max(elapsed, 1e-9):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohere Transcribe ONNX inference")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file(s), comma separated")
    parser.add_argument("-e", "--execution_provider", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution provider")
    args = parser.parse_args()
    run(args)
