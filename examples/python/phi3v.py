# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
import readline
import glob
from pathlib import Path

import onnxruntime_genai as og

REPO_ROOT = Path(__file__).parents[2]

def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def run(args: argparse.Namespace):
    print("Loading model...")
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.provider != "cpu":
        print(f"Setting model to {args.provider}...")
        config.append_provider(args.provider)
    model = og.Model(config)
    print("Model loaded")
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    interactive = args.interactive

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        if interactive:
            image_paths = [
                image_path.strip()
                for image_path in input(
                    "Image Path (comma separated; leave empty if no image): "
                ).split(",")
            ]
        else:
            if args.image_paths:
                image_paths = args.image_paths
            else:
                image_paths = [str(REPO_ROOT / "test" / "test_models" / "images" / "australia.jpg")]

        image_paths = [image_path for image_path in image_paths]
        print(image_paths)

        images = None
        prompt = "<|user|>\n"
        if len(image_paths) == 0:
            print("No image provided")
        else:
            print(f"Loading images: {image_paths}")
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                prompt += f"<|image_{i+1}|>\n"

            images = og.Images.open(*image_paths)

        if interactive:
            text = input("Prompt: ")
        else:
            if args.prompt:
                text = args.prompt
            else:
                text = "What is shown in this image?"
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        print("Processing images and prompt...")
        inputs = processor(prompt, images=images)

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=7680)

        generator = og.Generator(model, params)

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator

        if not interactive:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the folder containing the model"
    )
    parser.add_argument(
        "-p", "--provider", type=str, required=True, help="Provider to run model"
    )
    parser.add_argument(
        "--image_paths", nargs='*', type=str, required=False, help="Path to the images"
    )
    parser.add_argument(
        '-pr', '--prompt', required=False, help='Input prompts to generate tokens from.'
    )
    parser.add_argument(
        '--interactive', default=False, required=False, help='Interactive mode'
    )
    args = parser.parse_args()
    run(args)
