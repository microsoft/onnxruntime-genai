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
No checkpoint download or accelerator is required.

```bash
# Regenerate checked-in fixtures (requires numpy, onnx and onnx-ir).
python test/python/create/create_cache_dispatch_models.py

# Use an existing configured build with ENABLE_TESTS=ON.
cmake --build <build-dir> --target kv_cache_unit_tests
<build-dir>/kv_cache_unit_tests --gtest_filter='LegacyGenerator/*'
```
