# Build Qwen 3.8 Flash from Raw Checkpoints

This specialized builder creates the INT2 target and FP8 DFlash2 package from
raw Qwen 3.8 Flash Next target and drafter checkpoints. It complements the
generic Qwen4-Exp graph export and constructs the `qwen38_flash_graph` target
with checkpoint-specific NVFP4 conversion and fresh KV-cache calibration.

The generated `qwen38_flash_graph` package also needs the runtime support from
the stacked [GenAI runtime PR #2714](https://github.com/microsoft/onnxruntime-genai/pull/2714)
and its matching ONNX Runtime changes. PR #2713 adds the builder-side export
workflow; it does not by itself make the package runnable.

## Requirements

- Linux with a CUDA GPU, CUDA-enabled ONNX Runtime, and the matching GenAI build
  that supports `qwen38_flash_graph`.
- A local raw target checkpoint from
  `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
- A local raw drafter checkpoint from
  `incoai/Qwen3.8-Flash-Next-DFlash2`.
- A CUDA provider library built with the same ONNX Runtime core as
  `--qwen38_ort_home`.
  The SDK directory must contain `include/onnxruntime_cxx_api.h` and
  `lib/libonnxruntime.so`.
- Python dependencies used by the GenAI model builder, including PyTorch,
  NumPy, ONNX, `onnx_ir`, `safetensors`, and `tokenizers`; and a C++17 compiler
  for the calibration runner.
- At least 100 GiB of free disk space for the generated package, in addition to
  the raw checkpoints and build artifacts.

Build ORT and GenAI as described in [Development Guide](../../../../../../docs/DEVELOPMENT.md),
using the runtime changes required by `qwen38_flash_graph`. Set
`PYTHONPATH` to the matching GenAI build's Python package and add its native
libraries, ORT, CUDA, and cuDNN libraries to `LD_LIBRARY_PATH`.

## Calibration input

Create a UTF-8 JSONL file with one sample per line. Each sample can contain text:

```json
{"text":"A representative prompt for calibration."}
```

or pre-tokenized IDs:

```json
{"input_ids":[1,2,3,4]}
```

Samples contain 4 to 4096 tokens by default. Use representative prompts; the
small smoke-calibration set used during development is not an accuracy
qualification. Inputs outside the configured calibration capacity fail rather
than being silently truncated. `--qwen38_calibration_capacity` changes this
calibration-only limit, not the model's public context length.

## Build command

```bash
python src/python/py/models/builder.py \
  --input /models/nvidia-qwen38-nvfp4 \
  --output /models/qwen38-int2-dflash2 \
  --precision bf16 \
  --execution_provider cuda \
  --qwen38_flash_graph \
  --qwen38_draft_checkpoint /models/qwen38-dflash2-checkpoint \
  --qwen38_calibration_prompts /models/calibration.jsonl \
  --qwen38_ort_home /build/ort-sdk \
  --qwen38_cuda_provider /build/ort/Release/libonnxruntime_providers_cuda.so
```

The checkpoint paths must refer to local directories. The output directory must
be new or empty. This uses the existing GenAI model-builder entry point; the
`--qwen38_flash_graph` option dispatches to `Qwen38FlashBuilder`. The command
procedurally creates the target graph, streams
NVFP4 expert conversion to INT2, writes target weights and PLE data, performs
fresh CUDA KV-cache calibration, finalizes the target, exports the FP8 DFlash2
drafter with GenAI's `DFlash2Builder`, assembles the model package, and runs
native generation and speculative-decoding checks.

The generated package has a 262,144-token public prompt-plus-output limit, an
internal KV capacity of 262,148, and 8,192-token prefill chunks. The current
checkpoint and runtime contract exports three DFlash2 draft tokens. This is an
experimental export path, not a byte-for-byte reproduction of the optimized
prepared graph. A successful smoke verification demonstrates execution only;
it does not qualify task accuracy, long-context accuracy, or general
performance.

For an interrupted build, `--qwen38_resume` may reuse a completed raw-target export
from the same checkpoint and matching inputs. It will revalidate calibration
provenance and rerun native verification. Incomplete target weight files are
not resumable; inspect and preserve them rather than assuming the pipeline can
continue safely.

## Focused tests

From the repository root:

```bash
python -m pytest -q \
  test/python/builder/test_qwen38_flash_builder.py \
  test/python/builder/test_qwen38_flash_quantization.py \
  test/python/builder/test_qwen38_flash_graph.py \
  test/python/builder/test_qwen38_flash_calibration.py
```
