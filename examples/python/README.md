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

After the initial image prompt, the same `Generator` can continue with text by calling
`append_tokens`. The decoder KV cache retains the image context, so the image does not need to be
processed again. For supported vision models, later turns may also contain new images:

```python
processor = model.create_multimodal_processor()
generator = og.Generator(model, params)
generator.set_inputs(processor(first_turn_prompt, images=og.Images.open("image_a.jpg")))
while not generator.is_done():
    generator.generate_next_token()

# Include the model's turn separators/chat formatting, but only the NEW turn's text and images.
generator.set_inputs(processor(second_turn_prompt, images=og.Images.open("image_b.jpg")))
while not generator.is_done():
    generator.generate_next_token()

# Text-only turns can be interleaved without preprocessing any previous images.
generator.append_tokens(tokenizer.encode(next_text_turn))
```

Later image turns are enabled for Phi-3V, Mistral3/Pixtral, Qwen2.5-VL, Qwen3-VL, and Fara,
using a single sequence and ordinary decoding on a supported continuous-decoding KV-cache
device. The decoder and its cache persist; only the new images run through vision and embedding
prefill. Image numbering is local to each processor call (for Phi, restart at `<|image_1|>`).
Reserve enough `max_length` for the entire conversation, including image placeholders and
responses; reaching EOS can be resumed, but appending beyond that total limit is rejected.

Later images are not supported for Gemma3/4, Phi-4MM, Qwen3.5/hybrid recurrent models,
VideoChat, audio/modality-adapter changes, beam/speculative/constrained decoding, graph capture,
multi-profile execution,
split decoder pipelines (`decoder.pipeline`), or sliding/windowed/model-managed caches.
Existing image-prefill followed by text behavior is
unchanged for those families where text continuation is supported. This is a classic
`Generator` feature, not a continuous-batching `Engine` feature.

Rewind is restricted to the text suffix strictly after the **latest** multimodal prompt;
it cannot reach or cross that boundary. Invalid later-turn inputs are rejected before the
sequence is appended. A failure during model execution is not transactional: discard the
failed Generator rather than retrying on partially updated decoder state.

```bash
# The `qwen-3.6-mtp` script runs Qwen3.6 with its multi-token-prediction (MTP) head for
# self-speculative decoding. See qwen-3.6-mtp.md for export instructions and design details.
python qwen-3.6-mtp.py -m {path to main model folder} -d {path to MTP head folder}
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
```

## Engine Tool Calling

Run a complete Engine tool-calling round trip with host execution and incremental continuation. Add `--guidance` to constrain the tool call in a guidance-enabled build:

```bash
python build.py --use_cuda --use_guidance --skip_tests --skip_examples
python examples/python/engine/tool-calling.py -m {path to model folder} -e cuda --tools_file test/tool-definitions/weather.json --guidance
```

`USE_GUIDANCE` is off by default. Guidance is request-local and optional; omit `--guidance` for models that natively produce tool-call output.