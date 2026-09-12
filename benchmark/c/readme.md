# model_benchmark

`model_benchmark` is an end-to-end benchmark program for ONNX Runtime GenAI models.
It is written in C++ and built as part of the ONNX Runtime GenAI build (e.g., via [build.py](../../build.py)).

It is an alternative to the [Python benchmark script](../python/benchmark_e2e.py) that can be run in environments where Python is not available.

Example usage:
```
model_benchmark -i <path to model directory>
```

Run with `--help` to see information about additional options.

## Multimodal models

`model_benchmark` can also benchmark multimodal (vision-language / audio-language) models end to end,
as a C++ alternative to the [Python multimodal benchmark script](../python/benchmark_multimodal.py).

Pass `-im`/`--image_path` and/or `-au`/`--audio_path` (comma-separated for multiple files) to enable
multimodal mode. Inputs are built with the model's multimodal processor and fed to the generator via
`SetInputs()`, and the benchmark additionally reports the input (image/audio/text) preprocessing time.

```
model_benchmark -i <path to model directory> -im image.jpg
model_benchmark -i <path to model directory> -im image.jpg --prompt "<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n"
```

If no `--prompt`/`--prompt_file` is given, a default prompt containing the appropriate media tags for
the model type is used. `--prompt_length` and `--use_random_tokens` are not supported in multimodal
mode, and the batch size must be 1.

Note: On some platforms, such as Android, you may need to set the environment variable `LD_LIBRARY_PATH` to the directory containing the onnxruntime shared library for `model_benchmark` to be able to run.
