import onnxruntime_genai as og

print("Loading model...")

# The first argument is the name of the folder containing the model files
model=og.Model("example-models/llama2-7b-chat-int4-cpu", og.DeviceType.CPU)
print("Model loaded")
tokenizer=model.create_tokenizer()
print("Tokenizer created")

prompts = ["I like walking my cute dog",
           "What is the best restaurant in town?",
           "Hello, how are you today?"]

input_tokens = tokenizer.encode_batch(prompts)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":256})
params.set_input_sequences(input_tokens)


generator=og.Generator(model, params)

print("Generator created")

print("Running generation loop ...")

while not generator.is_done():
    generator.compute_logits()

    # Customize generation parameters
    # TODO: these do not work yet
    #generator.apply_min_length(1)
    #generator.apply_repetition_penalty(1.0)

    generator.generate_next_token_top_p(0.9, 1.0)

print("Outputs:")

for i in range(len(prompts)):
   print(tokenizer.decode(generator.get_sequence(i).get_array()))
   print()
