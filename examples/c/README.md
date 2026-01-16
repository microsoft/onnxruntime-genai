# ONNX Runtime GenAI C Example

> 📝 **Note:** The examples from the main branch of this repository are compatible with the binaries built from the same commit. Therefore, if using the example from `main`, ONNX Runtime GenAI needs to be built from source. If this is your scenario, just build the library and the examples will be auto built along with the library. If this is not your scenario, please use prebuilt binaries from the release you're interested in and use the examples from the same version tag and follow the steps below.

## Download the model

1. Download and install [foundry-local](https://github.com/microsoft/Foundry-Local/releases)
2. List available models: `foundry model list`
3. Download a model you would like to run. For example: `foundry model download Phi-4-generic-cpu`
4. Find out where the model is saved on disk: `foundry cache location`
5. Identify the path to the model on disk. For example: `C:\Users\<user>\.foundry\Microsoft\Phi-4-generic-cpu\cpu-int4-rtn-block-32-acc-level-4`

> 📝 **Note:** Foundry Local CLI is not available on Linux at the moment. Please download the model from a Windows or a macOS machine and copy it over to your Linux machine if you would like to run on Linux.

For other options to download models, read through [our download options](https://github.com/microsoft/onnxruntime-genai/blob/main/documents/DownloadModels.md).

## Build the C++ Example

1. Clone the repo: `git clone https://github.com/microsoft/onnxruntime-genai.git`
   - Use the relevant release tag that aligns with the version of the libraries you're planning to use.
2. `cd onnxruntime-genai`
3. Download the ONNX Runtime libraries.
   - Depending on the execution provider of interest, download one of the following:
      - [CPU execution provider](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime)
      - [CUDA execution provider on Windows](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu.Windows)
      - [CUDA execution provider on Linux](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu.Linux)
      - [QNN execution provider](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.QNN)
      - [DirectML execution provider](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)
   - Rename extension from `*.nupkg` to `*.zip`.
   - Extract the zip and copy over all the files from `<package>\runtimes\<platform>\native` to `examples\c\lib`
4. Download the [ONNX Runtime GenAI libraries](https://github.com/microsoft/onnxruntime-genai/releases).
   - Depending on the execution provider of interest, download one of the following:
      - CPU / QNN execution provider: `onnxruntime-genai-<version>-<platform>.zip/tar.gz`
      - CUDA execution provider: `onnxruntime-genai-<version>-<platform>-cuda.zip/tar.gz`
      - DirectML execution provider: `onnxruntime-genai-<version>-<platform>-dml.zip/tar.gz`
      - TRT-RTX execution provider: `onnxruntime-genai-trt-rtx-win-x64-<version>.zip` 
        - **Note**: TRT-RTX packages are distributed separately from [NVIDIA's onnxruntime-genai repository](https://github.com/NVIDIA/onnxruntime-genai/releases)
        - Download [onnxruntime-genai-trt-rtx-win-x64-0.10.0.zip](https://github.com/NVIDIA/onnxruntime-genai/releases/download/v0.10.0/onnxruntime-genai-trt-rtx-win-x64-0.10.0.zip) for v0.10.0 
   - Extract the zip and copy over all the files from `<package>\lib` to `examples\c\lib`
   - Copy over the header files from `<package>\include` to `examples\c\include`
5. Build using cmake
   - Windows x64:
      ```sh
      cd examples\c
      cmake -G "Visual Studio 17 2022" -S . -B build -DMODEL_QA=ON -DMODEL_CHAT=ON
      cmake --build build --parallel --config Debug
      ```
   - Windows arm64:
      ```sh
      cd examples\c
      cmake -G "Visual Studio 17 2022" -S . -B build -DMODEL_QA=ON -DMODEL_CHAT=ON -A ARM64
      cmake --build build --parallel --config Debug
      ```   
   - Linux x64 and macOS:
      ```sh
      cd examples/c
      cmake -G "Unix Makefiles" -S . -B build -DMODEL_QA=ON -DMODEL_CHAT=ON
      cmake --build build --parallel --config Debug
      ```

## Run the sample

1. On Windows:
   - cd build\Debug
   - .\model_qa.exe <path/to/model/from/above> <execution_provider>
2. On Linux and macOS:
   - cd build
   - ./model_qa <path/to/model/from/above> <execution_provider>

## Use constrained decoding

Constrained Decoding is useful when using function/tool calling as it helps in ensuring the output is in the correct format.

We have integrated [LLGuidance](https://github.com/guidance-ai/llguidance) for constrained decoding. There are three types of constrained decoding enabled right now:
1. Lark Grammar (Recommended): This option allows you to have an option for a regular output as well as function/tool output in JSON format.
2. JSON Schema: Output will be JSON schema and it will be one of the function/tools provided.
3. Regex: If a particular regular expression is desired.

To ensure that the function/tool call works correctly with constrained decoding, you need to modify your tokenizer.json file. For each model that has its own tool calling token, the tool calling token's `special` attribute needs to be set to true. For example, Phi-4 mini uses the <|tool_call|> and <|/tool_call|> tokens so you should set the `special` attribute for them as `true` inside `tokenizer.json`.

To run the C examples with function/tool calling:
```
# Using JSON Schema with only tool call output
.\model_chat -m {path to model folder} -e {execution provider} --response_format json_schema --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with only tool call output
.\model_chat -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with text or tool call output
.\model_chat -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --text_output --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"
```