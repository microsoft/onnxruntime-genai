# QNN Execution Provider

ONNX Runtime GenAI supports running models on Qualcomm Snapdragon NPUs via the QNN Execution
Provider (EP). This enables hardware-accelerated LLM inference on Snapdragon-based devices on
Windows ARM64 and Linux ARM64.

ONNX models containing an EPContext node with `ep_context_type: "dlc"` are routed directly
through the Genie API — Qualcomm's optimized LLM inference runtime, part of the QAIRT SDK,
which provides accelerated token generation on the Snapdragon NPU. This routing is transparent
to OGA users; no additional configuration is required. Models in this format are produced by
newer QAIRT-targeting [olive-recipes](https://github.com/microsoft/olive-recipes) pipelines.
Older QNN-targeting recipes produce models without the DLC EPContext type and do not use the
Genie pathway.

## Prerequisites

**Packages**

- `onnxruntime-genai` — the OGA runtime
- `onnxruntime-qnn` — the QNN EP plugin ([PyPI](https://pypi.org/project/onnxruntime-qnn/) /
  [NuGet](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.QNN))

**Model**

QNN inference requires a model prepared for the QNN EP, packaged as an ONNX file containing
compiled QNN graph artifacts. The `genai_config.json` and model files are produced by the
[olive-recipes](https://github.com/microsoft/olive-recipes) pipeline targeting the QNN backend.

## Usage

The QNN EP is a plugin and must be registered before loading the model. Pass the path to the
`onnxruntime_providers_qnn` shared library from the `onnxruntime-qnn` package
(`onnxruntime_providers_qnn.dll` on Windows, `libonnxruntime_providers_qnn.so` on Linux):

```python
import onnxruntime_genai as og

# Register the QNN EP plugin before loading the model.
# The library is included in the onnxruntime-qnn package.
# Note: registration uses the full EP name "QNNExecutionProvider"; provider options
# (Config.set_provider_option / genai_config.json) use the short name "QNN".
og.register_execution_provider_library(
    "QNNExecutionProvider",
    "/path/to/onnxruntime_providers_qnn.dll"  # or libonnxruntime_providers_qnn.so on Linux
)

# Load model. The genai_config.json from the olive-recipes pipeline already
# specifies the QNN provider, so no further config is needed in the common case.
# To override options, use og.Config directly:
#   config = og.Config("/path/to/model")
#   config.set_provider_option("QNN", "htp_performance_mode", "burst")
#   model = og.Model(config)
model = og.Model("/path/to/model")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
generator = og.Generator(model, params)
generator.append_tokens(tokenizer.encode("What color is the sky?"))

while not generator.is_done():
    generator.generate_next_token()
    print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
print()
```

## Provider Options

Options are set via `config.set_provider_option("QNN", key, value)`.

| Option | Description | Values |
|---|---|---|
| `htp_performance_mode` | NPU power/performance profile | `burst`, `balanced`, `high_performance`, `power_saver` |
| `vtcm_mb` | VTCM allocation size in MB | `"8"`, `"16"`, etc. |
| `enable_htp_shared_memory_allocator` | Enable shared memory allocator for direct OGA/QNN tensor handoff | `"1"` to enable |
