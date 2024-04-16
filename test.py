import onnxruntime_genai as og
import os
input(os.getpid())

prompt = "What is the lightest element?"

model=og.Model(r"mistral")

tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":200})
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)