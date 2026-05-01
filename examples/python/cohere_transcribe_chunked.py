"""
Cohere Transcribe — C++ integrated chunking test.
Audio chunking is handled internally by CohereProcessor + Generator.
The user just provides the full audio file, no manual splitting needed.

Usage:
  python cohere_transcribe_chunked.py -m /path/to/model -a audio.wav [-e cuda]
"""

import argparse
import os
import time

import onnxruntime_genai as og


# <|noitn|> disables Inverse Text Normalization: numbers, dates, currency and
# similar entities stay in spoken form (e.g. "twenty twenty four", "five
# dollars", "three p m"). Swap to "<|itn|>" to get written/numeric form
# instead ("2024", "$5", "3 PM"), which is usually what end users want.
# Additionally, swap <|en|> for a different language (e.g. <|es|> for Spanish) 
# if the audio is not English.
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

        audios = og.Audios.open(audio_path)
        inputs = processor([prompt], audios=audios)

        params = og.GeneratorParams(model)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        stream = tokenizer.create_stream()
        while not generator.is_done():
            generator.generate_next_token()
            print(stream.decode(generator.get_sequence(0)[-1]), end="", flush=True)

        seq = generator.get_sequence(0)
        text = tokenizer.decode(list(seq))
        all_texts.append(text.strip())
        print()  # newline after streaming

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("TRANSCRIPTION (C++ chunking):")
    print("=" * 60)
    for text in all_texts:
        print(text)

    print(f"\nElapsed: {elapsed:.2f}s")

    # Print RTFx if we can determine audio duration
    try:
        import wave
        total_dur = 0.0
        for p in audio_paths:
            with wave.open(p, "rb") as wf:
                total_dur += wf.getnframes() / wf.getframerate()
        if total_dur > 0:
            print(f"Audio: {total_dur:.1f}s | RTFx: {total_dur / elapsed:.1f}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohere Transcribe — C++ integrated chunking")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file(s), comma separated")
    parser.add_argument("-e", "--execution_provider", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution provider")
    args = parser.parse_args()
    run(args)
