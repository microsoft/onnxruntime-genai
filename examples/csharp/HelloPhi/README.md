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

## Use constrained decoding

Constrained Decoding is useful when using function/tool calling as it helps in ensuring the output is in the correct format.

We have integrated [LLGuidance](https://github.com/guidance-ai/llguidance) for constrained decoding. There are three types of constrained decoding enabled right now:
1. Lark Grammar (Recommended): This option allows you to have an option for a regular output as well as function/tool output in JSON format.
2. JSON Schema: Output will be JSON schema and it will be one of the function/tools provided.
3. Regex: If a particular regular expression is desired.

To ensure that the function/tool call works correctly with constrained decoding, you need to modify your tokenizer.json file. For each model that has its own tool calling token, the tool calling token's `special` attribute needs to be set to true. For example, Phi-4 mini uses the <|tool_call|> and <|/tool_call|> tokens so you should set the `special` attribute for them as `true` inside `tokenizer.json`.

To run the C# examples with function/tool calling:
```
# Using JSON Schema with only tool call output
.\HelloPhi.exe -m {path to model folder} -e {execution provider} --response_format json_schema --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with only tool call output
.\HelloPhi.exe -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with text or tool call output
.\HelloPhi.exe -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --text_output --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"
```
