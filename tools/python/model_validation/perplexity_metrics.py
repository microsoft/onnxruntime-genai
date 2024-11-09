import json
from datasets import load_dataset
import numpy as np
import onnxruntime_genai as og
import torch

def get_wikitext2():
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # Concatenate the text with "\n\n" separator,
    result = "\n\n".join(text for text in test["text"])
    return result

def perplexity_eval(model_dir):
    # Load the model and tokenizer
    model = og.Model(f'{model_dir}')
    tokenizer = og.Tokenizer(model)

    total_log_probs = 0
    total_token_count = 0

    # Concatenated text 
    dataset = get_wikitext2()

    # Encode the entire dataset as one batch
    input_ids = tokenizer.encode_batch([dataset])
    input_ids = torch.tensor(input_ids)
    print(f"input_ids shape: {input_ids.shape}")

    # Need to retreive the Model's maximum via the ORT GenAI configuration
    ## Explore the biggest max length vs the context length in genai config and calculate the lower of the two
    with open(model_dir+'/genai_config.json', 'r') as file:
        config = json.load(file)

    max_length = config["model"]["context_length"]-1 # This is the default for qwen
    stride = 8192 
    # Just get the perplexity for one position
    seq_len = input_ids.size(1)

    prev_end_loc = 0

    # Hugging face looping logic
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        print(f"input_ids_chunk shape: {input_ids_chunk.shape}")
        print(f"target_ids shape: {target_ids.shape}")

        params = og.GeneratorParams(model)
        params.input_ids = input_ids_chunk.numpy()

        print(f"params input ids shape: {params.input_ids.shape}")

        generator = og.Generator(model, params)

        # Get Logits 
        with torch.no_grad():
            generator.compute_logits()
            logits = generator.get_output("logits")
        print(f"logits shape: {logits.shape}")

        # Calculate LogSoftMax
        log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=2).numpy()
        print(f"log_probs shape: {log_probs.shape}")

        target_ids_flat = target_ids.flatten()

        target_log_probs = log_probs[0, np.arange(3), target_ids_flat]
     
        target_log_probs_sliced = target_log_probs[:, -trg_len:]
        
        print(f"target_log_probs shape: {target_log_probs_sliced.shape}")

        total_log_probs += np.sum(target_log_probs_sliced)
        total_token_count += target_ids.numel()

        print(f"This is the total_log_probs {total_log_probs}")
        print(f"This is the total_token_count {total_token_count}")

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    avg_log_prob = total_log_probs / total_token_count
    perplexity = np.exp(-avg_log_prob)

    print(f"The perplexity of {model_dir} is {perplexity}")
    return perplexity