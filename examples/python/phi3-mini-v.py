# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os

import onnxruntime_genai as og

def run(args: argparse.Namespace):
    print("Loading model...")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    print("Loading image...")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    image = og.Images.open(args.image_path)

    print("Processing image and prompt...")
    prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"
    inputs = processor(prompt, image)

    print("Generating response...")
    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=3072)

    generator = og.Generator(model, params)

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    args = parser.parse_args()
    run(args)
