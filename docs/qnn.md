# QNN Execution Provider

ONNX Runtime GenAI supports running models on Qualcomm Snapdragon NPUs via the QNN Execution
Provider (EP). This enables hardware-accelerated LLM inference on Snapdragon-based devices on
Windows ARM64 and Linux ARM64.

For best performance, use the newer QAIRT-targeting pipelines from
[olive-recipes](https://github.com/microsoft/olive-recipes), which optimize models for
accelerated NPU inference via the Genie runtime. Older QNN-targeting recipes produce
standard QNN models that also work but do not use this acceleration pathway.

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

The QNN EP is a plugin and must be registered before loading the model. The `onnxruntime-qnn`
package provides helpers to locate the library and EP name:

```python
import onnxruntime_genai as og
import onnxruntime_qnn

# Register the QNN EP plugin before loading the model.
og.register_execution_provider_library(
    onnxruntime_qnn.get_ep_name(),      # "QNNExecutionProvider"
    onnxruntime_qnn.get_library_path()  # platform-aware path to the QNN EP shared library
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
