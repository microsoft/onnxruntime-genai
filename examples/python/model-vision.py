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


def run(args: argparse.Namespace):
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting model to {args.execution_provider}...")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    print("Model loaded")

    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    interactive = not args.non_interactive

    while True:
        if interactive:
            try:
                import readline
                readline.set_completer_delims(" \t\n;")
                readline.parse_and_bind("tab: complete")
                readline.set_completer(_complete)
            except ImportError:
                # Not available on some platforms. Ignore it.
                pass
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
                image_paths = [str(_find_dir_contains_sub_dir(Path(__file__).parent, "test") / "test_models" / "images" / "australia.jpg")]

        image_paths = [image_path for image_path in image_paths if image_path]

        user_tag = image_tag = assistant_tag = end_tag = ""
        if model.type == "phi3v":
            user_tag = "<|user|>\n"
            image_tag = "<|image_{image_id}|>\n"
            assistant_tag = "<|assistant|>\n"
            end_tag = "<|end|>\n"
        elif model.type == "gemma3":
            user_tag = "<start_of_turn>user\n"
            image_tag = "<start_of_image>"
            assistant_tag = "<start_of_turn>model\n"
            end_tag = "<end_of_turn>\n"

        images = None
        prompt = f"{user_tag}"
        if len(image_paths) == 0:
            print("No image provided")
        else:
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                print(f"Using image: {image_path}")
                prompt += f"{image_tag.replace('{image_id}', str(i+1))}"

            images = og.Images.open(*image_paths)

        if interactive:
            text = input("Prompt: ")
        else:
            if args.prompt:
                text = args.prompt
            else:
                text = "What is shown in this image?"
        prompt += f"{text}{end_tag}{assistant_tag}"
        print("Processing images and prompt...")
        inputs = processor(prompt, images=images)

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=7680)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
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
        "-e", "--execution_provider", type=str, required=False, default='follow_config', choices=["cpu", "cuda", "dml", "follow_config"], help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead."
    )
    parser.add_argument(
        "--image_paths", nargs='*', type=str, required=False, help="Path to the images, mainly for CI usage"
    )
    parser.add_argument(
        '-pr', '--prompt', required=False, help='Input prompts to generate tokens from, mainly for CI usage'
    )
    parser.add_argument(
        '--non-interactive', action=argparse.BooleanOptionalAction, required=False, help='Non-interactive mode, mainly for CI usage'
    )
    args = parser.parse_args()
    run(args)
