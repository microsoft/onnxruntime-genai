# ONNX Runtime GenAI Test Folder Structure

This directory contains assets, scripts, and definitions used by the test suite.

## Layout

- `test/audios/*` contains all audio files used for testing.
- `test/images/*` contains all image files used for testing.
- `test/models/model_name/*` contains uploaded assets for `model_name` such as `model.onnx`, `genai_config.json`, `tokenizer.json`, and related files. These are already-created models stored in the repo.
- `test/python/*` contains all Python scripts used for testing.
    - `test/python/builder/*` contains Python scripts used to validate the model builder.
    - `test/python/create/*` contains Python scripts used to create ONNX models. Scripts follow the `create_*.py` naming format.
    - `test/python/integration/*` contains Python scripts used to run integrating testing on Foundry Local models.
    - `test/python/models/*` contains Python scripts used to test ONNX models. Scripts follow the `test_*.py` naming format.
- `test/tool-definitions/*` contains all tool definitions used for testing.

## Legacy hybrid-cache dispatch regression

`test/models/cache-dispatch` contains tiny, executed arithmetic state machines
for the `kv_cache_unit_tests` target. The recurrent fixture preserves per-layer
`mamba`/`moe`/`attention` metadata, sparse global KV indices, and paired convolution
and recurrent states. The other fixtures exercise the legacy
`conv`/`full_attention` descriptors under both `decoder` and `lfm2` model types.
Nonempty layer metadata alone must not select the legacy convolution-only cache.

The tests check complete cache tensors, logits and selected tokens after prefill
and cached decode, then recurrent reset or fresh-generator initialization.
They validate runtime cache plumbing, not neural-network or checkpoint parity.
CPU tests require no checkpoint download or accelerator. CUDA builds also run
the controls on CUDA and exercise BF16 KV/convolution with FP32 recurrent state.
The `multimodal` fixture adds a prepared-image/text-only three-model pipeline;
`multimodal-retention` exercises scalar-BOOL vision dispatch, dynamic retained
rows, host prompt compaction and the prepared-feature entry without rerunning
vision. Invalid shapes, masks, counts, mixed raw inputs and beams are rejected.
See [its runtime contract](../docs/nemotron_h_omni.md) for supported inputs and
the remaining processor/video-pruning limitations.

```bash
# Regenerate checked-in fixtures (requires numpy, onnx and onnx-ir).
python test/python/create/create_cache_dispatch_models.py
python test/python/create/create_multimodal_cache_model.py

# Use an existing configured build with ENABLE_TESTS=ON.
cmake --build <build-dir> --target kv_cache_unit_tests
<build-dir>/kv_cache_unit_tests --gtest_filter='LegacyGenerator/*'
```

### Opt-in native packed-operator interoperability

`native_quantized_cache_probe.cpp` additionally exercises raw `MatMulGGMLQ8_0`
and dedicated `MatMulNVFP4A16` through the prepared three-model entry. It requires
the corresponding ORT CUDA operators (see
[ORT prerequisite #33400](https://github.com/microsoft/onnxruntime/pull/33400));
it is deliberately separate from the default tests that use released ORT.

Both fixtures pass genuine C++ BF16 feature buffers and CPU INT32 tokens, then
check BF16 native outputs, complete FP32 logits, all sparse KV/conv/SSM states,
and greedy selection through prefill and three cached decode steps. Tokens are
internally converted to the embedding graph's declared INT64 type. Scalar
reference formulas discriminate FP32 Q8 weight reconstruction from native BF16
weights, and globally scaled NVFP4 weights from the dedicated operator's FP32
global epilogue. Q8 uses its optional native bias; A16 uses a separate native
BF16 `Add` at opset 21. Capture and shared caches are disabled, batch/beam are
one. No Python BF16 conversion is used.

```bash
# Requires numpy, ml_dtypes, onnx and onnx-ir; no checkpoint downloads.
python test/python/create/create_native_quantized_cache_models.py
c++ -std=c++20 -O2 -Isrc test/cpp/native_quantized_cache_probe.cpp \
  -L<cuda-build-dir> -lonnxruntime-genai \
  -Wl,-rpath,<absolute-cuda-build-dir> -o <probe-output>
CUDA_VISIBLE_DEVICES=2 \
  LD_PRELOAD=<absolute-qualified-ort-dir>/libonnxruntime.so.1.32.0 \
  LD_LIBRARY_PATH=<qualified-ort-dir>:<cuda-lib-dir>:<cudnn-lib-dir>:<cuda-build-dir> \
  <probe-output> test/models/cache-dispatch
```

Use a fresh profiling/config directory when collecting evidence against another
ORT build. The probe prints loaded runtime paths on Linux. These are small
arithmetic interoperability fixtures, not source-exact image/video processing,
full-package generation, checkpoint parity, footprint or performance evidence.
