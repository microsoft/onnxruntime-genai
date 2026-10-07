# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import os
import readline
import re

import onnxruntime_genai as og
from common import register_ep

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)


def _complete(text, state):
    return ([*glob.glob(text + "*"), None])[state]


class Format:
    end = "\033[0m"
    underline = "\033[4m"


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = re.findall(r"\w+", reference.lower())
    hypothesis_words = re.findall(r"\w+", hypothesis.lower())
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / max(1, len(reference_words))


def run(args: argparse.Namespace):
    print("Loading model...")
    register_ep(args.execution_provider, "", False)
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)

        if args.non_interactive:
            audio_paths = [audio_path.strip() for audio_path in args.audio.split(",")]
        else:
            audio_paths = [audio_path.strip() for audio_path in input("Audio Paths (comma separated): ").split(",")]
        if len(audio_paths) == 0:
            raise ValueError("No audio provided.")

        print("Loading audio...")
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audios = og.Audios.open(*audio_paths)

        print("Processing audio...")
        batch_size = len(audio_paths)
        decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
        if not args.timestamps:
            decoder_prompt_tokens.append("<|notimestamps|>")
        prompts = ["".join(decoder_prompt_tokens)] * batch_size
        inputs = processor(prompts, audios=audios)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            max_length=448,
            batch_size=batch_size,
            whisper_timestamps=args.timestamps,
        )

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            generator.generate_next_token()

        print()
        transcriptions = []
        tokenizer = og.Tokenizer(model)
        for i in range(batch_size * args.num_beams):
            tokens = generator.get_sequence(i)
            if args.timestamps:
                timestamp_tokens = [int(token) for token in tokens if tokenizer.is_timestamp_token(int(token))]
                if len(timestamp_tokens) < 2:
                    raise RuntimeError("Timestamp-enabled Whisper output did not contain timestamp boundaries.")
                if timestamp_tokens != sorted(timestamp_tokens):
                    raise RuntimeError("Whisper timestamp tokens are not monotonic.")
                if timestamp_tokens[0] > tokenizer.timestamp_begin_token_id + 50:
                    raise RuntimeError("The first Whisper timestamp exceeds the configured initial boundary.")
                timestamp_seconds = [tokenizer.timestamp_to_seconds(token) for token in timestamp_tokens]
                print(f"Timestamp token IDs: {timestamp_tokens}")
                print(f"Timestamp seconds: {timestamp_seconds}")

            transcription = processor.decode(tokens)

            print("Transcription:")
            print(
                f"    {Format.underline}batch {i // args.num_beams}, beam {i % args.num_beams}{Format.end}: {transcription}"
            )
            transcriptions.append(transcription.strip())

        for _ in range(3):
            print()

        if args.non_interactive:
            args.output = args.output.strip()
            if args.max_word_error_rate is None:
                if args.output in transcriptions:
                    print("One of the model's transcriptions matches the expected transcription.")
                    return
                raise Exception("None of the model's transcriptions match the expected transcription.")

            best_word_error_rate = min(_word_error_rate(args.output, text) for text in transcriptions)
            print(f"Best word error rate: {best_word_error_rate}")
            if best_word_error_rate <= args.max_word_error_rate:
                return
            raise Exception(f"No transcription met the maximum word error rate of {args.max_word_error_rate}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.",
    )
    parser.add_argument("-b", "--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("-a", "--audio", type=str, default="", help="Path to audio file for CI testing purposes")
    parser.add_argument(
        "-o", "--output", type=str, default="", help="Expected transcribed output for CI testing purposes"
    )
    parser.add_argument(
        "-ni",
        "--non_interactive",
        default=False,
        action="store_true",
        help="Non-interactive mode for CI testing purposes",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Enable Whisper timestamp-token generation and validate the generated timestamp sequence.",
    )
    parser.add_argument(
        "--max_word_error_rate",
        type=float,
        default=None,
        help="Accept a non-interactive transcription when its word error rate is at most this value.",
    )
    args = parser.parse_args()
    run(args)
