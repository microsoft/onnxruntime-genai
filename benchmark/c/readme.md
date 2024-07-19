# model_benchmark

`model_benchmark` is an end-to-end benchmark program for ONNX Runtime GenAI models.
It is written in C++ and built as part of the ONNX Runtime GenAI build (e.g., via [build.py](../../build.py)).

It is an alternative to the [Python benchmark script](../python/benchmark_e2e.py) that can be run in environments where Python is not available.

Example usage:
```
model_benchmark -i <path to model directory>
```

Run with `--help` to see information about additional options.

Note: On some platforms, such as Android, you may need to set the environment variable `LD_LIBRARY_PATH` to the directory containing the onnxruntime shared library for `model_benchmark` to be able to run.
