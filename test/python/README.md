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

Use `cpu,cuda` or `cpu,webgpu` on accelerator runners. Requested missing providers
fail, never fall back to CPU or skip. Without the variable, the suite discovers
installed providers; CPU coverage is always retained.

| Coverage | Fixture and assertions |
|---|---|
| Repeated images, mixed turns, EOS, suffix rewind, text regression | Pixel/KV/position-sensitive oracle and full-prefix references |
| Unequal image grids within later turns | Qwen per-image execution and device-safe feature assembly |
| Input lifetime | Released caller inputs, allocation pressure, deferred readback; unsynchronized arithmetic runs require CUDA/TensorRT-RTX |
| Atomic rejection | Invalid inputs, capacity, unsupported family/configuration; unchanged sequence/logits |
| Execution-failure poisoning | CPU-only out-of-range Gather injector |

The arithmetic fixture uses dynamic Concat KV tensors, not real attention.
Separate GQA/native tests cover shared caches, device residency, and capture;
NumPy output copies alone cannot establish residency.

```powershell
python -m pytest -q test\python\models\test_multimodal_device.py
$env:PYTHONPATH = "$PWD\test\python"
python -m create.create_multimodal_gqa_model --suite --output_dir build\Windows\RelWithDebInfo\test\multimodal_gqa --devices cpu
.\build\Windows\RelWithDebInfo\RelWithDebInfo\multimodal_device_tests.exe --devices cpu
```

For native accelerators, generate and run with the same `--devices cpu,cuda` or
`--devices cpu,webgpu`; pass the plugin directory with `--ep_dir`.
The GQA matrix is CPU FP32, CUDA FP16, and WebGPU FP32/FP16.
After provisioning fixtures, `ENABLE_MULTIMODAL_DEVICE_TESTS=ON` registers the
executable with CTest. It defaults off to avoid requiring Python ONNX in native builds.

Arithmetic sessions forbid CPU EP fallback; GQA allows only its exact named
attention-length metadata nodes on CPU. WebGPU host-side staging and CPU Search
are intentional, not numerical fallback. GPU capture tests execute/replay the
decoder, unlike CPU guard tests; later image turns remain unsupported with capture.

`test_onnxruntime_genai.py` collects these tests. CUDA and WebGPU workflows require
their respective EPs. See `integration` for real-model provisioning and options.
