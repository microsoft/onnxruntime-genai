"""
Cohere Transcribe ONNX inference via onnxruntime-genai.

Usage:
  OMP_NUM_THREADS=4 python cohere_transcribe.py -m /datadisks/disk3/nebanfic/cohere-transcribe-onnx-int2 -a audio.wav
"""

import argparse
import os
import time
import wave

import onnxruntime_genai as og


PROMPT_TOKENS = [
    "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
    "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
]


def run(args):
    print(f"Loading model from {args.model_path} ...")
    config = og.Config(args.model_path)
    config.clear_providers()  # CPU only
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()

    audio_paths = [p.strip() for p in args.audio.split(",")]
    for p in audio_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p}")

    print(f"Loading {len(audio_paths)} audio file(s) ...")
    audios = og.Audios.open(*audio_paths)

    prompt = "".join(PROMPT_TOKENS)
    batch_size = len(audio_paths)
    prompts = [prompt] * batch_size

    print("Processing ...")
    inputs = processor(prompts, audios=audios)

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=1024,
        num_beams=1,
        batch_size=batch_size,
    )

    t0 = time.time()
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    # Stream tokens as they are generated
    print("\nTranscription: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_sequence(0)[-1]
        text = stream.decode(token)
        print(text, end="", flush=True)
    print()

    elapsed = time.time() - t0

    total_audio_dur = 0.0
    for i in range(batch_size):
        with wave.open(audio_paths[i], 'rb') as wf:
            audio_dur = wf.getnframes() / wf.getframerate()
        total_audio_dur += audio_dur

    print(f"\nElapsed: {elapsed:.2f}s | Audio: {total_audio_dur:.2f}s | RTFx: {total_audio_dur / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohere Transcribe ONNX inference")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file(s), comma separated")
    args = parser.parse_args()
    run(args)
