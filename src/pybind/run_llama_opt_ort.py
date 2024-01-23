import datetime
from transformers import LlamaTokenizer
import onnxruntime_genai as og


#device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

print("Loading model...")

#model=og.Llama_Model("models/PY007/TinyLlama-1.1B-Chat-V0.3/rank_0_TinyLlama-1.1B-Chat-V0.3_decoder_merged_model_int4.onnx", device_type)
#model=og.Llama_Model("models/meta-llama/Llama-2-7b-chat-hf/rank_0_Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx", device_type)
model=og.Llama_Model("../../test_models/natke-llama/rank_0_Llama-2-7b-chat-hf_decoder_merged_model_fp16.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-chat-int4-gpu/rank_0_Llama-2-7b-chat-hf_decoder_merged_model_int4.onnx", device_type)

print("Model loaded")


# Keep asking for input prompts in an loop
prompt = "I like walking my cute dog"
input_ids = tokenizer.encode(prompt, return_tensors='np')

params=og.SearchParams()
params.max_length = 128
params.batch_size = input_ids.shape[0]
params.sequence_length = input_ids.shape[1]
params.input_ids = input_ids
params.vocab_size = model.GetVocabSize()
params.eos_token_id = tokenizer.eos_token_id
params.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else params.eos_token_id

search=og.GreedySearch(params, model.DeviceType)

start_time = datetime.datetime.now()  

state=og.Llama_State(model, search.GetSequenceLengths(), params)

generate_ids = []
while not search.IsDone():
    search.SetLogits(state.Run(search.GetSequenceLength(), search.GetNextTokens()))

    # search.Apply_MinLength(1)
    # search.Apply_RepetitionPenalty(1.0)

    search.SampleTopP(0.9, 0.6)

    # Get the next generated token
    generate_ids.append(search.GetNextTokens().GetArray()[0])

end_time = datetime.datetime.now()

print(generate_ids)

output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

print(prompt, end=' ', flush=True)
print(output, end='', flush=True)
print("\n")

new_tokens = len(generate_ids)
num_tokens = input_ids.shape[1] + new_tokens
seconds = (end_time - start_time).total_seconds()
print(f"GenAI + ONNX Runtime, {num_tokens}, {new_tokens}, {round(seconds, 2)}, {round(num_tokens / seconds, 1)}, {round(new_tokens / seconds, 1)}")