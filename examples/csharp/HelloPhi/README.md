# ONNX Runtime GenAI C# example 

## Obtain a model

You can download a published model from Hugging Face. For example, this is Phi-4 mini optimized for CPU and mobile. You can find other models here: 

```script
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir models
move models\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4 models\phi
```

Alternatively you can build a model yourself using the model builder. See [here](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md) for more details.


## Run the model

Open [HelloPhi.sln](HelloPhi.sln) and run the console application.

Notes:

1. The `executionProvider` must be one of the following: `cpu`, `cuda`, or `dml`.
