import onnxruntime_genai as og
from transformers import GPT2Tokenizer

device_type = og.DeviceType.CPU
# device_type = og.DeviceType.CUDA
text = "The best hotel in bay area is"

# Generate input tokens from the text prompt
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
input_tokens = tokenizer.encode(text, return_tensors='np')

model=og.Gpt_Model("../../python/onnx_models/gpt2.onnx", device_type)

params=og.SearchParams()
params.num_beams = 4
params.max_length = 64
params.batch_size = input_tokens.shape[0]
params.sequence_length = input_tokens.shape[1]
params.input_ids = input_tokens
params.vocab_size = model.GetVocabSize()
params.eos_token_id = tokenizer.eos_token_id
params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

search=og.BeamSearch(params, model.DeviceType)
state=og.Gpt_State(model, search.GetSequenceLengths(), params)

print("Inputs:")
print(input_tokens)
print("Input prompt:", text)

print("Running beam search loop...")
while not search.IsDone():
    search.SetLogits(state.Run(search.GetSequenceLength(), search.GetNextTokens(), search.GetNextIndices()))

    # Scoring
    # Generators::Processors::MinLength(search, 1)
    # Generators::Processors::RepetitionPenalty(search, 1.0f)

    search.SelectTop();

print("Outputs:")
output_tokens=search.GetSequence(0).GetArray()
print(output_tokens)
decoded_output=tokenizer.decode(output_tokens)
print(decoded_output)
