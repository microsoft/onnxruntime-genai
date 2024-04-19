import onnxruntime_genai as og
import os
input(os.getpid())

prompt = "Human: What is the lightest element?\nAI:"

prompt = "<|user|>What is the difference between nuclear fission and nuclear fusion?<|end|>\n<|assistant|>"

model=og.Model(r"phi-3")

tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)
# tokens = [20490, 25, 1867, 318, 262, 1657, 395, 5002, 30, 198, 20185, 25] # What is the lightest element?
# tokens = [20490, 25, 1867, 318, 262, 3580, 1022, 4523, 277, 1480, 290, 4523, 21748, 198, 20185, 25] # What is the difference between nuclear fission and nuclear fusion

params=og.GeneratorParams(model)
params.set_search_options({"do_sample": True, "max_length":50})
params.try_use_cuda_graph_with_max_batch_size(1)
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)