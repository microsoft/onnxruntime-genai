# model_benchmark

`model_benchmark` is an end-to-end benchmark program for ONNX Runtime GenAI models.
It is written in C++ and built as part of the ONNX Runtime GenAI build (e.g., via [build.py](../../build.py)).

It is an alternative to the [Python benchmark script](../python/benchmark_e2e.py) that can be run in environments where Python is not available.

Example usage:
```
model_benchmark -i <path to model directory>
```

Run with `--help` to see information about additional options.

To benchmark a locally built plugin EP, pass its shared library path:

```powershell
.\model_benchmark.exe -i C:\models\my-model -e webgpu --ep_library_path C:\ort\onnxruntime_providers_webgpu.dll -l 128 -g 128 -w 2 -r 5 --use_random_tokens --reuse_generator
```

The library is registered on GenAI's environment before model creation and remains loaded until
shutdown. Use compatible ONNX Runtime and GenAI libraries, and make the plugin's dependencies
available to the OS loader.

The default `-e follow_config` preserves the providers selected in `genai_config.json`. An explicit
`-e` selects the decoder provider while preserving its configured options, including graph capture.
With `--ep_library_path`, custom provider names are accepted; use the provider name expected by GenAI's
configuration (for example, `webgpu`, `cuda`, or a custom EP's advertised name).
The program prints the registered library path and provider selection before benchmarking.

Note: On some platforms, such as Android, you may need to set the environment variable `LD_LIBRARY_PATH` to the directory containing the onnxruntime shared library for `model_benchmark` to be able to run.
