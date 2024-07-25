# Generate() API C# example 

## Obtain a model

You can download a published model from HuggingFace. For example, this is Phi-3 mini optimized for CPU and mobile. You can find other models here: 

```script
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir models
move models\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4 models\phi-3
```

Alternatively you can build a model yourself using the model builder tool. See [Generate models using Model Builder](https://onnxruntime.ai/docs/genai/howto/build-model.html) for more details.


## Run the model

Open [HelloPhi.sln](HelloPhi.sln) and run the console application.

Note that this application does not add a template to the prompt that you enter. If your model needs a template (e.g. `<|user|>\n{input} <|end|>\n<|assistant|>` for Phi-3) then please add this to your prompt.
