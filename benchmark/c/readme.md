# model_benchmark

`model_benchmark` is an end-to-end benchmark program for ONNX Runtime GenAI models.
It is written in C++ and built as part of the ONNX Runtime GenAI build (e.g., via [build.py](../../build.py)).

It is an alternative to the [Python benchmark script](../python/benchmark_e2e.py) that can be run in environments where Python is not available.

Example usage:
```
model_benchmark -i <path to model directory>
```

For LoRA models whose ONNX graph expects adapter weights as inputs, place
`adapter.safetensors` in the model directory or pass
`--adapter <path>`. The benchmark binds all adapter tensors via
`SetModelInput` before the first `AppendTokens` call.

Folded Gemm LoRA exports use fp16 weights keyed by `*.weight_fp16`, or int8
weights keyed by `*.weight_quantized` (file and graph input are both int8).
MatMulNBits LoRA exports use packed uint8 weights keyed by `*.weight_quantized`.

Run with `--help` to see information about additional options.

Note: On some platforms, such as Android, you may need to set the environment variable `LD_LIBRARY_PATH` to the directory containing the onnxruntime shared library for `model_benchmark` to be able to run.
