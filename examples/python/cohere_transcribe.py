"""
Audio chunking is handled internally by CohereProcessor + Generator.
The user just provides the full audio file, no manual splitting needed.

Usage:
  python cohere_transcribe.py -m /path/to/model -a audio.wav [-e cuda]
"""

import argparse
import os
import time

import onnxruntime_genai as og


# <|itn|> keeps numbers, dates, currency and similar entities in spoken form
# (e.g. "twenty twenty four", "five dollars", "three p m"). Swap to "<|noitn|>"
# to get written/numeric form instead ("2024", "$5", "3 PM"), which is usually
# what end users want.
# <|pnc|> enables Punctuation and Capitalization (sentence-case + commas,
# periods, etc). Swap to "<|nopnc|>" for lowercase, punctuation-free output.
# Additionally, swap <|en|> for a different language (e.g. <|es|> for Spanish)
# if the audio is not English. The model requires, by design, two same language tags.
# Other tags should largely remain as it is.
PROMPT_TOKENS = [
    "<|startofcontext|>", "<|startoftranscript|>", "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>",
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
        print()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
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
        # RTFx is informational only; ignore failures (non-WAV input, missing wave module, etc.).
        pass

    # Optional WER check used by the E2E harness.
    if args.expected_transcription is not None:
        wer = _compute_wer(args.expected_transcription, all_texts[0])
        print(f"WER: {wer:.4f} (max allowed: {args.max_wer})")
        if wer > args.max_wer:
            raise SystemExit(
                f"WER {wer:.4f} exceeds threshold {args.max_wer}\n"
                f"  expected: {args.expected_transcription}\n"
                f"  got:      {all_texts[0]}"
            )


def _normalize(text):
    import re
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def _compute_wer(reference, hypothesis):
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Standard Levenshtein word-distance / len(ref).
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cohere Transcribe — C++ integrated chunking")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to audio file(s), comma separated")
    parser.add_argument("-e", "--execution_provider", type=str, default="cpu", choices=["cpu", "cuda"], help="Execution provider")
    parser.add_argument("--expected_transcription", type=str, default=None, help="Reference transcript for WER validation")
    parser.add_argument("--max_wer", type=float, default=0.10, help="Maximum acceptable WER when --expected_transcription is provided")
    args = parser.parse_args()
    run(args)
