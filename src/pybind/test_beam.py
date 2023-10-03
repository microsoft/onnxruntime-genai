import ort_generators as og
import numpy as np
from transformers import GPT2Tokenizer

text = "best hotel in bay area"

# Generate input tokens from the text prompt
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input_tokens = tokenizer.encode(text, return_tensors='np')

gpt=og.Gpt("../../python/onnx_models/gpt2.onnx")

params=og.GreedySearchParams()
params.num_beams = 4
params.max_length = 64
params.batch_size = input_tokens.shape[0]
params.sequence_length = input_tokens.shape[1]
params.input_ids = input_tokens
params.vocab_size = gpt.GetVocabSize()
params.eos_token_id = tokenizer.eos_token_id
params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

search=og.BeamSearch(params)
gpt.CreateInputs(search.GetSequenceLengths(), params)

print("Inputs:")
print(input_tokens)
print("Input prompt:", text)

print("Running loop...")
while not search.IsDone():
    gpt.Run(search.GetNextTokens(), [], search.GetSequenceLength())
    search.SetLogits(gpt.GetLogits())

    # Scoring
    # Generators::Processors::MinLength(search, 1)
    # Generators::Processors::RepetitionPenalty(search, 1.0f)

    search.NextTokensFromLogits()
    search.AppendNextTokensToSequences()

print("Outputs:")
output_tokens=search.GetSequence(0)
print(output_tokens)
decoded_output=tokenizer.decode(output_tokens)
print(decoded_output)
