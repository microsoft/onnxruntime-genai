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
    tokenizer = og.Tokenizer(model)

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        audio_paths = [audio_path.strip() for audio_path in input("Audio Paths (comma separated): ").split(",")]
        if len(audio_paths) == 0:
            raise ValueError("No audio provided.")

        print("Loading audio...")
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audios = og.Audios.open(*audio_paths)

        print("Processing audio...")
        mel = processor(audios=audios)
        decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            max_length=256,
        )

        batch_size = len(audio_paths)
        params.set_inputs(mel)
        params.input_ids = [[tokenizer.to_token_id(token) for token in decoder_prompt_tokens]] * batch_size

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
