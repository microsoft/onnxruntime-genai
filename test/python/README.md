To run a test:

python -m pytest -sv test_onnxruntime_genai_api.py -k "<your_test_name>" --test_models ..\models

For example:

python -m pytest -sv test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models ..\models

The runtime-library selection tests run without a model or installed native
bindings, from the repository root:

```bash
python -m unittest discover -s test/python -p test_runtime_library_selection.py
```

On Linux and macOS, set `ORT_LIB_PATH` before importing `onnxruntime_genai`
to select a compatible custom-built ONNX Runtime shared library. This explicit
path takes precedence over the installed `onnxruntime` package. An empty or
unloadable path raises an error instead of falling back to another runtime.
When the variable is unset, installed-package selection is unchanged.
