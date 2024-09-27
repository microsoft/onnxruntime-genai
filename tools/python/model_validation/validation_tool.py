import onnxruntime_genai as og
import argparse
import time
from onnxruntime_genai.models.builder import create_model
import json
import os


class validationConfigObject:
    def __init__(self, models, inputs, max_length, min_length, do_sample, top_p, top_k, temperature, reptition_penalty, verbose, timings):
        self.models = models
        self.inputs = inputs
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.reptition_penalty = reptition_penalty
        self.verbose = verbose
        self.timings = timings

# Return true or false
def validate_model(args, model_directory, validationConfigObject):
    if validationConfigObject.verbose: print("Loading model...")
    if validationConfigObject.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model = og.Model(f'{model_directory}')

    if validationConfigObject.verbose: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if validationConfigObject.verbose: print("Tokenizer created")
    if validationConfigObject.verbose: print()
    search_options = {name: getattr(validationConfigObject, name, None) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature']}

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    
    for input in validationConfigObject.inputs:

        complete_text = ''
        
        if validationConfigObject.timings: started_timestamp = time.time()

        # If there is a chat template, use it
        prompt = f'{chat_template.format(input=input)}'


        # Tokenizer has interesting behavior, it creates multiple inputs
        input_tokens = tokenizer.encode(prompt)


        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if validationConfigObject.verbose: print("Generator created")

        if validationConfigObject.verbose: print("Running generation loop ...")
        if validationConfigObject.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                if validationConfigObject.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                print(tokenizer_stream.decode(new_token), end='', flush=True)
                
                if validationConfigObject.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()


        with open('output.txt', 'a') as file:
            file.write(complete_text)


        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        if validationConfigObject.timings:
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

    validationConfigObject = validationConfigObject(**data)


        # Create a json object that holds all this information and 
        # then pass that in

    # Check and see if the folder exists, if not create the folder
    model_output_dir = "../../../models_outputs/"
    model_cache_dir = "../../../cache_models"

    if not folder_exists(model_output_dir):
        create_folder(model_output_dir)

    if not folder_exists(model_cache_dir):
        create_folder(model_cache_dir)

    for model in validationConfigObject.models:
        # Wrap in a try catch 
        create_model(model, '', model_output_dir+f'/{model}', 'int4', 'cpu', model_cache_dir+f'/{model}')
        validate_model(args, model_output_dir, validationConfigObject)
        #Table values
        #columns, model name, validation complete (y/n), third - exception / failure msgs
    
    # Print the table out once loop is completed 