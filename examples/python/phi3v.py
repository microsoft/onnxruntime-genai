# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
import readline
import glob

import onnxruntime_genai as og

def _complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

def run(args: argparse.Namespace):
    print("Loading model...")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    while True:
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        image_path = input("Image Path (leave empty if no image): ")

        image = None
        prompt = "<|user|>\n"
        if (len(image_path) == 0):
            print("No image provided")
        else:
            print("Loading image...")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = og.Images.open(image_path)
            prompt += "<|image_1|>\n"

        text = input("Prompt: ")
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        print("Processing image and prompt...")
        inputs = processor(prompt, images=image)

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
        
        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model")
    args = parser.parse_args()
    run(args)
