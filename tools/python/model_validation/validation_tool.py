import onnxruntime_genai as og
import argparse
import time
from onnxruntime_genai.models.builder import create_model
import json
import os
import pandas as pd

def create_table(output):
    df = pd.DataFrame(output, columns=['Model Name', 'Validation Completed', 'Exceptions / Failures'])
    return df

def validate_model(config, model_directory):
    if config["verbose"]: print("Loading model...")
    if config["timings"]:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{model_directory}')

    if config["verbose"]: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if config["verbose"]: print("Tokenizer created")
    if config["verbose"]: print()

    search_option_keys = [
        'do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 
        'temperature', 'repetition_penalty'
    ]   

    search_options = {key: config[key] for key in search_option_keys} 

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    
    for input in config["inputs"]:

        complete_text = ''
        
        if config["timings"]: started_timestamp = time.time()

        prompt = f'{chat_template.format(input=input)}'

        input_tokens = tokenizer.encode(prompt)


        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if config["verbose"]: print("Generator created")

        if config["verbose"]: print("Running generation loop ...")
        if config["timings"]:
            first = True
            new_tokens = []

        print()
        # print("Output: ", end='', flush=True)

        generation_successful = True

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if config["timings"]:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                # print(tokenizer_stream.decode(new_token), end='', flush=True)
                
                if config["timings"]: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
            generation_successful = False
        except Exception as e:
            print(f"An error occurred: {e}")
            generation_successful = False

        print()
        print()


        with open('output.txt', 'a') as file:
            file.write(complete_text)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if config["timings"]:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"Prompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}, Time to first: {(prompt_time):.2f}s, Prompt tokens per second: {len(input_tokens)/prompt_time:.2f} tps, New tokens per second: {len(new_tokens)/run_time:.2f} tps")

    return generation_successful

def folder_exists(folder_path):
    return os.path.isdir(folder_path)

def create_folder(folder_path):
    os.mkdir(folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")

    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')

    args = parser.parse_args()

    with open(args.json, 'r') as file:
        config = json.load(file)

    # Check and see if the folder exists, if not create the folder
    model_output_dir = "../../../models_outputs/"
    model_cache_dir = "../../../cache_models"

    if not folder_exists(model_output_dir):
        create_folder(model_output_dir)

    if not folder_exists(model_cache_dir):
        create_folder(model_cache_dir)

    output = []

    for model in config["models"]:
        try:
            create_model(model, '', model_output_dir+f'/{model}', 'int4', 'cpu', model_cache_dir+f'/{model}')
            generation_successful = validate_model(config, model_output_dir)
            exception_message = None
        except Exception as e:
            exception_message = str(e)
        
        output.append([model, generation_successful, exception_message])
    
    df = create_table(output)
    print(df)