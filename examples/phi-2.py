import onnxruntime_genai as og
from transformers import AutoTokenizer

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')

print("Loading model...")
model=og.Model("../../test_models/phi-2", device_type)
print("Model loaded")

tokenizer2=model.CreateTokenizer()

# Keep asking for input prompts in an loop
while True:
    print("Enter your text (type 'END' to stop):")
    lines = []
    while True:
        line = input()
        if line == 'END':
            break
        lines.append(line)
    text = '\n'.join(lines)

    # text = input("Input:")
    input_tokens = tokenizer.encode(text, return_tensors='np')

    params=og.SearchParams(model)
    params.max_length = 256
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
        if word=='Ċ':
          word = '\n'
        if word.startswith('Ġ'):
          word = ' ' + word[1:]
        print(word, end='', flush=True)

    print()
    print()
