import onnxruntime_genai as og
import numpy as np
import time
from transformers import LlamaTokenizer

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

print("Loading model...")
#model=og.Llama_Model("../../test_models/llama2-7b-fp32-cpu/Llama-2-7b-hf_decoder_merged_model_fp32_opt.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-fp16-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_fp16.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-int4-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_int4.onnx", device_type)
model=og.Model("../../test_models/llama2-7b-chat-int4-gpu", device_type)
print("Model loaded")

# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer.encode(text, return_tensors='np')

    params=og.SearchParams(model)
    params.max_length = 64
    params.input_ids = input_tokens

    start_time=time.time()
    output_tokens=model.Generate(params)
    run_time=time.time()-start_time;
    print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

    print("Output:")
    print(tokenizer.decode(output_tokens))

    print()
    print()
