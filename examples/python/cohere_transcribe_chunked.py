"""
Cohere Transcribe — C++ integrated chunking test.
Audio chunking is handled internally by CohereProcessor + Generator.
The user just provides the full audio file, no manual splitting needed.

Usage:
  python cohere_transcribe_chunked.py -m /path/to/model -a audio.wav [-e cuda]

To compare with the Python-side chunking reference:
  python cohere_transcribe.py -m /path/to/model -a audio.wav [-e cuda]
"""

import argparse
import os
import time

import onnxruntime_genai as og


PROMPT_TOKENS = [
    "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
    "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
]


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
        print(f"Audio: {audio_path}")

        # Load full audio — chunking happens inside C++ CohereProcessor
        print("  Loading audio...", flush=True)
        audios = og.Audios.open(audio_path)
        print("  Audio loaded, running processor...", flush=True)
        inputs = processor([prompt], audios=audios)
        print("  Processor done, creating params...", flush=True)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=1024,
            num_beams=1,
            batch_size=1,
        )
        print("  Creating generator...", flush=True)

        generator = og.Generator(model, params)
        print("  Setting inputs...", flush=True)
        generator.set_inputs(inputs)
        print("  Starting generation...", flush=True)

        # Generate — chunk transitions are handled internally by Generator.IsDone()
        tokens = []
        step = 0
        while True:
            print(f"    step {step}: checking is_done...", end="", flush=True)
            try:
                done = generator.is_done()
            except Exception as e:
                print(f" EXCEPTION in is_done: {e}")
                break
            print(f" done={done}", flush=True)
            if done:
                break
            print(f"    step {step}: generating next token...", end="", flush=True)
            try:
                generator.generate_next_token()
            except Exception as e:
                print(f" EXCEPTION in generate_next_token: {e}")
                break
            new_token = generator.get_sequence(0)[-1]
            tokens.append(new_token)
            step += 1
            print(f" token={new_token}", flush=True)

        # Decode all tokens (includes tokens from all chunks)
        stream = tokenizer.create_stream()
        text = "".join(stream.decode(t) for t in tokens)
        text = text.strip()
        all_texts.append(text)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("TRANSCRIPTION (C++ chunking):")
    print("=" * 60)
    for text in all_texts:
        print(text)

    print(f"\nElapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohere Transcribe — C++ integrated chunking")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file(s), comma separated")
    parser.add_argument("-e", "--execution_provider", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution provider")
    args = parser.parse_args()
    run(args)
