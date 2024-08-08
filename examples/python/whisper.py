# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import glob
import os
import readline

import onnxruntime_genai as og


def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


class Format:
    end = "\033[0m"
    underline = "\033[4m"


def run(args: argparse.Namespace):
    print("Loading model...")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        audio_path = input("Audio Path: ")
        if len(audio_path) == 0:
            raise ValueError("No audio provided.")

        print("Loading audio...")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = og.Audios.open(audio_path)

        print("Processing audio...")
        inputs = processor(audios=audio, lang="en", task="transcribe")

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            max_length=256,
        )

        batch_size = 1
        params.set_inputs(inputs)

        generator = og.Generator(model, params)

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

        print()
        for i in range(batch_size * args.num_beams):
            tokens = generator.get_sequence(i)
            transcription = processor.decode(tokens)

            print(f"Transcription:")
            print(
                f"    {Format.underline}batch {i // args.num_beams}, beam {i % args.num_beams}{Format.end}: {transcription}"
            )

        for _ in range(3):
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "-b", "--num_beams", type=int, default=4, help="Number of beams"
    )
    args = parser.parse_args()
    run(args)
