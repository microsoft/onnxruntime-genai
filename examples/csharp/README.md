# ONNX Runtime GenAI C# Example 

> 📝 **Note:** The examples from the main branch of this repository are compatible with the binaries built from the same commit. Therefore, if using the example from `main`, ONNX Runtime GenAI needs to be built from source. If this is your scenario, just build the library and the examples will be auto built along with the library. If this is not your scenario, please use prebuilt binaries from the release you're interested in and use the examples from the same version tag and follow the steps below.

## Install ONNX Runtime GenAI

Install the C# package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install) or [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html).

## Download a Model

There are many places to obtain a model. Please read through [our download options](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/DownloadModels.md).

## Run an Example

Open `ModelChat.sln` or `ModelMM.sln` and run the console application.

## Tool Calling

Please read through [our constrained decoding](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/ConstrainedDecoding.md) options to learn more.

To run the C# examples with function/tool calling:
```
# Using JSON Schema with only tool call output
.\ModelChat.exe -m {path to model folder} -e {execution provider} --response_format json_schema --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with only tool call output
.\ModelMM.exe -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with text or tool call output
.\ModelChat.exe -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --text_output --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"
```
