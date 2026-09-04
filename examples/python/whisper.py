# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import json
import os
import readline

import numpy as np
import onnxruntime_genai as og
from common import register_ep
from whisper_timestamps import word_timestamps

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)


def _complete(text, state):
    return ([*glob.glob(text + "*"), None])[state]


class Format:
    end = "\033[0m"
    underline = "\033[4m"


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
        decoder_prompt_tokens = ["<|startoftranscript|>", f"<|{args.language}|>", f"<|{args.task}|>"]
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
        )

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        if args.word_timestamps:
            if args.execution_provider != "cuda":
                raise ValueError("Word timestamps require CUDA because Whisper alignment is finalized on the GPU.")
            if batch_size != 1 or args.num_beams != 1:
                raise ValueError("Word timestamps require one audio and one beam.")
            generator.set_model_input("alignment_heads", np.asarray(args.alignment_heads, dtype=np.int32))

        while not generator.is_done():
            generator.generate_next_token()

        print()
        transcriptions = []
        for i in range(batch_size * args.num_beams):
            tokens = generator.get_sequence(i)
            transcription = processor.decode(tokens)

            print("Transcription:")
            print(
                f"    {Format.underline}batch {i // args.num_beams}, beam {i % args.num_beams}{Format.end}: {transcription}"
            )
            transcriptions.append(transcription.strip())
            if args.word_timestamps:
                generated_tokens = [token for token in tokens[len(decoder_prompt_tokens) :] if token < args.eot_token_id]
                token_text = [processor.decode([token]) for token in generated_tokens]
                cross_qk = generator.get_output("cross_qk")[0, 0]
                print("Word timestamps:")
                for word in word_timestamps(token_text, cross_qk):
                    print(f"    [{word.start:.2f} --> {word.end:.2f}] {word.word}")

        for _ in range(3):
            print()

        if args.non_interactive:
            args.output = args.output.strip()
            matching = False
            for transcription in transcriptions:
                if transcription == args.output:
                    matching = True
                    break

            if matching:
                print("One of the model's transcriptions matches the expected transcription.")
                return
            raise Exception("None of the model's transcriptions match the expected transcription.")


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
    parser.add_argument("--language", default="en", help="Whisper language token (for example, en or fr)")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--timestamps", action="store_true", help="Enable Whisper segment timestamp tokens")
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Print word timestamps using CUDA cross-attention alignment; requires --alignment-heads",
    )
    parser.add_argument(
        "--alignment-heads",
        type=json.loads,
        default=[],
        help='JSON alignment-head pairs exported for this model, e.g. "[[3, 0], [3, 1]]"',
    )
    parser.add_argument("--eot-token-id", type=int, default=50257, help="Whisper end-of-text token ID")
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
    args = parser.parse_args()
    if args.word_timestamps and not args.alignment_heads:
        parser.error("--word-timestamps requires --alignment-heads.")
    run(args)
