import ort_generators as og
import numpy as np

input_tokens=np.array([[0, 0, 0, 52], [0, 0, 195, 731]], dtype=np.float32)

mparams=og.GptModelParams()
mparams.vocab_size=1000
mparams.head_count=4
mparams.hidden_size=8
mparams.layer_count=5
gpt=og.Gpt(mparams, "../models/files/gpt2_fp32.onnx")

params=og.GreedySearchParams()
params.max_length = 10
params.batch_size = input_tokens.shape[0]
params.sequence_length = input_tokens.shape[1]
params.input_ids = input_tokens
params.vocab_size = gpt.GetVocabSize()

input("Press enter to start")
search=og.GreedySearch(params)
gpt.CreateInputs(search.GetSequenceLengths(), params)

print("Inputs:")
print(mparams)
print(params)
print(input_tokens)

print("Running loop...")
while not search.IsDone():
    gpt.Run(search.GetNextTokens(), [], search.GetSequenceLength())
    search.SetLogits(gpt.GetLogits())

    # Scoring
    # Generators::Processors::MinLength(search, 1)
    # Generators::Processors::RepetitionPenalty(search, 1.0f)

    # Sampling goes here

    # Should NextTokensFromLogits() return an array? Then 'CheckForEOS()' is obvious along with AppendNextTokensToSequences()
    search.NextTokensFromLogits()
    search.CheckForEOS()
    search.AppendNextTokensToSequences()
    print("Getting next token...")

print("Outputs:")
print(search.GetSequence(0))
print(search.GetSequence(1))
