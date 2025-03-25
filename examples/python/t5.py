# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
import glob
import time
from pathlib import Path

import onnxruntime_genai as og

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True)

def _find_dir_contains_sub_dir(current_dir: Path, target_dir_name):
    curr_path = Path(current_dir).absolute()
    target_dir = glob.glob(target_dir_name, root_dir=curr_path)
    if target_dir:
        return Path(curr_path / target_dir[0]).absolute()
    else:
        if curr_path.parent == curr_path:
            # Root dir
            return None
        return _find_dir_contains_sub_dir(curr_path / '..', target_dir_name)


def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def get_paths(modality, user_provided_paths, default_paths, interactive):
    paths = None

    if interactive:
        try:
            import readline
            readline.set_completer_delims(" \t\n;")
            readline.parse_and_bind("tab: complete")
            readline.set_completer(_complete)
        except ImportError:
            # Not available on some platforms. Ignore it.
            pass
        paths = [
            path.strip()
            for path in input(
                f"{modality.capitalize()} Path (comma separated; leave empty if no {modality}): "
            ).split(",")
        ]
    else:
        paths = user_provided_paths if user_provided_paths else default_paths

    paths = [path for path in paths if path]
    return paths


def run(args: argparse.Namespace):
    print("Loading model...")
    model = og.Model(args.model_path)
    print("Model loaded")

    # processor = model.create_multimodal_processor()
    # tokenizer_stream = processor.create_stream()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    interactive = not args.non_interactive

    while True:
        prompt = "<|user|>\n"

        if interactive:
            text = input("Prompt: ")
        else:
            if args.prompt:
                text = args.prompt
            else:
                text = "Does the audio summarize what is shown in the image? If not, what is different?"
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        
        print("Processing inputs...")
        inputs = tokenizer.encode(prompt)
        print("Processor complete.")

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=7680)

        generator = og.Generator(model, params)
        start_time = time.time()

        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        print()
        total_run_time = time.time() - start_time
        print(f"Total Time : {total_run_time:.2f}")

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
        "-e", "--execution_provider", type=str, required=True, choices=["cpu", "cuda", "dml"], help="Execution provider to run model"
    )
    parser.add_argument(
        "--image_paths", nargs='*', type=str, required=False, help="Path to the images, mainly for CI usage"
    )
    parser.add_argument(
        "--audio_paths", nargs='*', type=str, required=False, help="Path to the audios, mainly for CI usage"
    )
    parser.add_argument(
        '-pr', '--prompt', required=False, help='Input prompts to generate tokens from, mainly for CI usage'
    )
    parser.add_argument(
        '--non-interactive', action=argparse.BooleanOptionalAction, required=False, help='Non-interactive mode, mainly for CI usage'
    )
    args = parser.parse_args()
    run(args)