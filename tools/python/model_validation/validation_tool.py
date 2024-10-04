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

    model = og.Model(f'{model_directory}')

    if config["verbose"]: print("Model loaded")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream() 
    if config["verbose"]: print("Tokenizer created")
    if config["verbose"]: print()  

    chat_template = get_chat_template(model_directory)
    
    for input in config["inputs"]:

        complete_text = ''
        
        prompt = f'{chat_template.format(input=input)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        if config["verbose"]: print("Generator created")

        if config["verbose"]: print("Running generation loop ...")

        print()
        print("Output: ", end='', flush=True)

        generation_successful = True

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                
                value_to_save = tokenizer_stream.decode(new_token)

                complete_text += value_to_save

                print(tokenizer_stream.decode(new_token), end='', flush=True)
                
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
            generation_successful = False
        except Exception as e:
            print(f"An error occurred: {e}")
            generation_successful = False

        with open(f'{model_directory}/output.txt', 'a') as file:
            file.write(complete_text)

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

    return generation_successful

def get_chat_template(output_directory):
    tokenizer_json = output_directory + '/tokenizer_config.json'
    with open(tokenizer_json, 'r') as file:
        config = json.load(file)
    return config["chat_template"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")

    parser.add_argument('-j', '--json', type=str, required=True, help='Path to the JSON file containing the arguments')

    args = parser.parse_args()

    with open(args.json, 'r') as file:
        config = json.load(file)

    os.makedirs(config["output_directory"], exist_ok=True)
    os.makedirs(config["cache_directory"], exist_ok=True)

    output = []

    validation_complete = False

    for model in config["models"]:

        print(f"We are validating {model}")
        adjusted_model = model.replace("/", "_")
        output_path = config["output_directory"] + f'/{adjusted_model}'
        # From the output directory, there exist a file named tokenizer_config.json which contains the chat 
        cache_path = config["cache_directory"] + f'/{adjusted_model}'
        
        try:
            create_model(model, '', output_path, config["precision"], config["executive_provider"], cache_path)
        except Exception as e:
            print(f'Failure after create model {e}')
            output.append([model, validation_complete, e])
            continue
        try:          
            validation_complete = validate_model(config, output_path)
        except Exception as e:
            print(f'Failure after validation model {e}')
            output.append([model, validation_complete, e])
        
            
    df = create_table(output)

    df.to_csv("models.csv")

    print(df)

    # From the folder name, get the chat template 