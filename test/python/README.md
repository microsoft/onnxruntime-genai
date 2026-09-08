To run a test:

python -m pytest -sv test_onnxruntime_genai_api.py -k "<your_test_name>" --test_models ..\models

For example:

python -m pytest -sv test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models ..\models

## Consecutive multimodal turns

Run from the repository root with the current source-built wheel installed:

```powershell
$env:ORTGENAI_MULTIMODAL_TEST_EPS = "cpu"
python -m pytest -q test\python\models\test_multimodal_turns.py test\python\models\test_multimodal_ep_selection.py
```

Use `cpu,cuda` or `cpu,webgpu` on the appropriate accelerator runner. An explicitly
requested missing provider is an error, not a CPU fallback or successful skip.
Without the environment variable, the suite discovers installed providers and
always retains CPU coverage. Compiled `is_*_available()` flags alone do not prove
that a provider or physical device is present.

| Coverage | Fixture and assertions |
|---|---|
| Repeated images, mixed text/image turns, EOS, suffix rewind, text regression | One EP-parametrized scenario suite with a pixel/KV/position-sensitive numerical oracle and full-prefix references |
| Unequal image grids within later turns | Qwen per-image execution and device-safe feature assembly |
| Input lifetime and asynchronous execution | Repeated turns with caller inputs released, allocation pressure, and deferred readback; explicit unsynchronized runs require CUDA/TensorRT-RTX |
| Deterministic rejection | Invalid inputs, capacity, unsupported family/configuration, and unchanged sequence/logits after rejection |
| Execution-failure poisoning | CPU out-of-range Gather injector; this is not portable GPU fault injection and is explicitly skipped there |

The arithmetic fixture uses dynamic Concat KV tensors; it is **not** evidence of
shared-buffer attention correctness. Shared-cache and capture/device assertions
use the separate real-attention/native fixtures. Public NumPy output copies do
not establish device residency.

```powershell
python -m pytest -q test\python\models\test_multimodal_device.py
$env:PYTHONPATH = "$PWD\test\python"
python -m create.create_multimodal_gqa_model --suite --output_dir build\Windows\RelWithDebInfo\test\multimodal_gqa --devices cpu
.\build\Windows\RelWithDebInfo\RelWithDebInfo\multimodal_device_tests.exe --devices cpu
```

For native accelerators, generate the matching `--devices cpu,cuda` or
`--devices cpu,webgpu` fixtures, then pass the same selection and `--ep_dir`
plugin directory to the executable. Missing requested fixtures/providers fail.
The GQA dtype matrix is CPU FP32, CUDA FP16, and WebGPU FP32/FP16.
`ENABLE_MULTIMODAL_DEVICE_TESTS=ON` optionally registers this executable with
CTest after fixtures are provisioned. It is off by default so native-only builds
do not acquire a Python ONNX fixture-generation dependency.

Accelerator arithmetic sessions forbid CPU EP fallback and select the requested
provider for vision, embedding, and decoder. WebGPU's intentional host-side
input/embedding staging and CPU Search are not CPU model-kernel fallback.
Graph-capture guard tests must be distinguished from tests that actually execute
and replay a captured decoder; later image turns remain unsupported with capture.

The normal `test_onnxruntime_genai.py` launcher collects these model tests.
The Windows/Linux CUDA and Windows WebGPU workflows explicitly require their
respective EP selections. Real-model integration uses its own model-root and
execution-provider options; see `integration` for artifact provisioning and suite
selection. Configuring a CI lane is not evidence that it has executed successfully.
