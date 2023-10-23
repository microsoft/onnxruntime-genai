import ort_generators as og
import numpy as np
from transformers import LlamaTokenizer

text = "The best hotel in bay area is"

# Generate input tokens from the text prompt
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
input_tokens = tokenizer.encode(text, return_tensors='np')

model=og.Llama("/dev_data/kvaishnavi/llama2/hf_models/llama2-7b-fp32/Llama-2-7b-hf_decoder_merged_model_fp32_opt.onnx")
print("Model loaded")

params=og.SearchParams()
params.max_length = 36
params.batch_size = input_tokens.shape[0]
params.sequence_length = input_tokens.shape[1]
params.input_ids = input_tokens
params.vocab_size = model.GetVocabSize()
params.eos_token_id = tokenizer.eos_token_id
params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

search=og.GreedySearch(params)
model.CreateInputs(search.GetSequenceLengths(), params)

print("Inputs:")
print(input_tokens)
print("Input prompt:", text)

print("Running greedy search loop...")
while not search.IsDone():
    print(f"Iteration, seq len = {search.GetSequenceLength()}")
    model.Run(search.GetNextTokens(), search.GetSequenceLength())
    search.SetLogits(model.GetLogits())

    # Scoring
    # Generators::Processors::MinLength(search, 1)
    # Generators::Processors::RepetitionPenalty(search, 1.0f)

    search.SelectTop1();

print("Outputs:")
output_tokens=search.GetSequence(0)
print(output_tokens)
decoded_output=tokenizer.decode(output_tokens)
print(decoded_output)
