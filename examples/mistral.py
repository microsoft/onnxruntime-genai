import onnxruntime_genai as og
from transformers import LlamaTokenizer

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = LlamaTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

print("Loading model...")
model=og.Model("../test_models/mistral", device_type)
print("Model loaded")

# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer.encode(text, return_tensors='np')

    params=og.SearchParams(model)
    params.max_length = 128
    params.input_ids = input_tokens

    search=params.CreateSearch()
    state=model.CreateState(search.GetSequenceLengths(), params)

    print("Output:")

    print(text, end='', flush=True)
    while not search.IsDone():
        search.SetLogits(state.Run(search.GetSequenceLength(), search.GetNextTokens()))
        search.SampleTopP(0.7, 0.6)

        # Print each token as we compute it, we have to do some work to get newlines & spaces to appear properly:
        word=tokenizer.convert_ids_to_tokens([search.GetNextTokens().GetArray()[0]])[0]
        if word=='<0x0A>':
          word = '\n'
        if word.startswith('▁'):
          word = ' ' + word[1:]
        print(word, end='', flush=True)

    print()
    print()
