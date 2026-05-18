# ONNX Runtime GenAI Python Examples

> 📝 **Note:** The examples from the main branch of this repository are compatible with the binaries built from the same commit. Therefore, if using the example from `main`, ONNX Runtime GenAI needs to be built from source. If this is your scenario, just build the library and the examples will be auto built along with the library. If this is not your scenario, please use prebuilt binaries from the release you're interested in and use the examples from the same version tag and follow the steps below.

## Install ONNX Runtime GenAI

Install the Python package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install) or [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html).

## Download a Model

There are many places to obtain a model. Please read through [our download options](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/DownloadModels.md).

## Run an Example

```bash
# The `model-chat` script allows for multi-turn conversations.
python model-chat.py -m {path to model folder} -e {execution provider}
```

```bash
# The `model-generate` script generates the entire output sequence in one function call.
python model-generate.py -m {path to model folder} -e {execution provider}
```

```bash
# The `model-qa` script streams the output text token by token.
python model-qa.py -m {path to model folder} -e {execution provider}
```

```bash
# The `model-mm` script works for multi-modal models and streams the output text token by token.
python model-mm.py -m {path to model folder} -e {execution provider}
```

```bash
# Pass one or more images via --image_paths (space-separated). Supported by Qwen2.5-VL, Qwen3-VL, Phi-3-vision, etc.
# In non-interactive mode the default prompt is "What color is the sky?" (override with --user_prompt).
python model-mm.py -m {path to model folder} -e {execution provider} --image_paths image1.jpg image2.jpg --non_interactive
```

## Execution Providers

The ONNX Runtime GenAI Python package supports the following execution providers (EPs):

- `CPUExecutionProvider`
- `CUDAExecutionProvider`
- `NvTensorRTRTXExecutionProvider`
- `OpenVINOExecutionProvider`
- `QNNExecutionProvider`
- `VitisAIExecutionProvider`
- `WebGpuExecutionProvider`

To use an EP with the example scripts, make sure it is available to ONNX Runtime using one of the three approaches below. Some scenarios require explicit registration arguments, while provider-bridge EPs do not. Pick the one that matches your scenario:

### 1. Register a custom / locally-built EP (`--ep_path`)

Use this when you are developing an EP locally and want to test it with ONNX Runtime GenAI.

- Pass the EP name with `-e` and the path to the EP shared library with `--ep_path`.
- Example:
  ```bash
  python model-qa.py -m {path to model folder} -e {execution provider} --ep_path {path to onnxruntime_providers_ep.dll}
  ```

### 2. Use a provider-bridge EP

Use this when the EP you want is already built into the underlying `onnxruntime` Python package as a provider-bridge EP. No registration arguments are required.

- By default, the EP listed in the model's `genai_config.json` is used.
- Optionally pass `-e` to override the default EP at runtime.
- Example:
  ```bash
  python model-qa.py -m {path to model folder}
  ```

### 3. Register EPs via Windows ML (`--use_winml`) — Windows only

Use this when you want Windows ML to acquire, install, and register the EP for you (useful for testing model changes against existing EP libraries).

- Requires the [`windowsml`](https://pypi.org/project/windowsml/) Python module to be installed.
- `--use_winml` fetches the EP from Windows Update, installs it, and registers it with ONNX Runtime GenAI.
- Example:
  ```bash
  python model-qa.py -m {path to model folder} --use_winml
  ```


## Tool Calling

Please read through [our constrained decoding](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/ConstrainedDecoding.md) options to learn more.

Here are some examples of how you can run the Python examples with function/tool calling.

```bash
# Using JSON Schema with only tool call output
python model-qa.py -m {path to model folder} -e {execution provider} --response_format json_schema --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with only tool call output
python model-mm.py -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"

# Using Lark Grammar with text or tool call output
python model-chat.py -m {path to model folder} -e {execution provider} --response_format lark_grammar --tools_file {path to json file} --text_output --tool_output --tool_call_start "{starting tool call token}" --tool_call_end "{ending tool call token}"