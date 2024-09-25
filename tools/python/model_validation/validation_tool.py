import onnxruntime_genai as og
import argparse
import time
from onnxruntime_genai.models.builder import create_model
import json
import os

def validate_model(args, model_directory, inputs):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{model_directory}')
    if args.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()
    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    
    # Set the max length to something sensible by default, unless it is specified by the user,
    # since otherwise it will be set to the entire context length

    search_options['max_length'] = 512

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    
    for input in inputs:

        complete_text = ''
        
        if args.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=input)}'


        # Tokenizer has interesting behavior, it creates multiple inputs
        input_tokens = tokenizer.encode(prompt)


        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if args.verbose: print("Generator created")

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                print(tokenizer_stream.decode(new_token), end='', flush=True)

                
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

                        

        with open('output.txt', 'a') as file:
            file.write(complete_text)


        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")


def folder_exists(folder_path):
    return os.path.isdir(folder_path)

def create_folder(folder_path):
    os.mkdir(folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")

    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')

    args = parser.parse_args()

    with open(args.json, 'r') as file:
        data = json.load(file)
        models = data['models']
        inputs = data['inputs']

    # Before model creation, do a check and see if the folder existsS
    model_output_dir = "../../../models_outputs"
    model_cache_dir = "../../../cache_models"

    if not folder_exists(model_output_dir):
        create_folder(model_output_dir)

    if not folder_exists(model_cache_dir):
        create_folder(model_cache_dir)

    for model in models:
        # Need to give the entire length
        onnx_model = create_model(model, '', model_output_dir, 'int4', 'cpu', model_cache_dir)
        # Add checks after model creation
        # validate_model(args, './models_output', inputs)
        #Table values
        #columns, model name, validation complete (y/n), third - exception / failure msgs
    
    # Print the table out once loop is completed 