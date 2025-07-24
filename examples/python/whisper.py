# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import os
import readline

import onnxruntime_genai as og
# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)

def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


class Format:
    end = "\033[0m"
    underline = "\033[4m"


def run(args: argparse.Namespace):
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)

        if args.non_interactive:
            audio_paths = [args.audio]
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
        decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        prompts = ["".join(decoder_prompt_tokens)] * batch_size
        inputs = processor(prompts, audios=audios)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            max_length=448,
        )

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            generator.generate_next_token()

        print()
        transcriptions = []
        for i in range(batch_size * args.num_beams):
            tokens = generator.get_sequence(i)
            transcription = processor.decode(tokens)

            print(f"Transcription:")
            print(
                f"    {Format.underline}batch {i // args.num_beams}, beam {i % args.num_beams}{Format.end}: {transcription}"
            )
            transcriptions.append(transcription.strip())

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
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        '-e', '--execution_provider', type=str, required=False, default='follow_config', choices=["cpu", "cuda", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead."
    )
    parser.add_argument(
        "-b", "--num_beams", type=int, default=4, help="Number of beams"
    )
    parser.add_argument(
        "-a", "--audio", type=str, default="", help="Path to audio file for CI testing purposes"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="", help="Expected transcribed output for CI testing purposes"
    )
    parser.add_argument(
        "-ni", "--non_interactive", default=False, action="store_true", help="Non-interactive mode for CI testing purposes"
    )
    args = parser.parse_args()
    run(args)
