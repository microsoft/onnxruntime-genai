# ONNX Runtime GenAI C/C++ Examples

> 📝 **Note:** The examples from the main branch of this repository are compatible with the binaries built from the same commit. Therefore, if using the example from `main`, ONNX Runtime GenAI needs to be built from source. If this is your scenario, just build the library and the examples will be auto built along with the library. If this is not your scenario, please use prebuilt binaries from the release you're interested in and use the examples from the same version tag and follow the steps below.

## Install ONNX Runtime GenAI

Install the C headers according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install) or [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html).

## Download a Model

There are many places to obtain a model. Please read through [our download options](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/DownloadModels.md).

## Build a C/C++ Example

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

## Run an Example

1. On Windows:

```powershell
# Prerequisite: navigate to the compiled binaries.
cd build\Debug
```

```powershell
# The `model-chat` script allows for multi-turn conversations.
.\model_chat.exe -m {path to model folder} -e {execution provider}
```

```powershell
# The `model-qa` script streams the output text token by token.
.\model_qa.exe -m {path to model folder} -e {execution provider}
```

```powershell
# The `model-mm` script works for multi-modal models and streams the output text token by token.
.\model_mm.exe -m {path to model folder} -e {execution provider}
```

2. On Linux and macOS:

```powershell
# Prerequisite: navigate to the compiled binaries.
cd build
```

```bash
# The `model-chat` script allows for multi-turn conversations.
./model_chat -m {path to model folder} -e {execution provider}
```

```bash
# The `model-qa` script streams the output text token by token.
./model_qa -m {path to model folder} -e {execution provider}
```

```bash
# The `model-mm` script works for multi-modal models and streams the output text token by token.
./model_mm -m {path to model folder} -e {execution provider}
```

## Tool Calling

Please read through [our constrained decoding](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/ConstrainedDecoding.md) options to learn more.

Here are some examples of how you can run the C/C++ examples with function/tool calling.

```bash
# Using JSON Schema with only tool call output
./model_qa -m {path to model folder} -e {execution provider} --response_format json_schema --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with only tool call output
./model_mm -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with text or tool call output
./model_chat -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --text_output --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"
```
