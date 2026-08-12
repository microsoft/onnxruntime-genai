# ORT GenAI integration tests

Real-model integration tests for ONNX Runtime GenAI. The same test code
runs in the ADO `integration-tests` pipeline and on a contributor
machine; only the model source differs.

## What it tests

For each (model, execution provider) pair the suite loads the model,
generates a short continuation of the prompt `"The capital of France is"`
with greedy decoding, and asserts non-empty bounded output. A soft check
warns (without failing) when the expected substring (`paris`) is absent.

## Layout

```
test/python/integration/
  models.py                  # MODELS catalog + suite lists + PINNED_VERSIONS
  resolver.py                # get_path_for(model, device) -> Path
  suite_paths.py             # blob prefix for one (model, device) pair
  conftest.py                # --model, --execution-provider, --model-root, --run-engine-tests
  test_integration_text.py   # text-generation test (text pipeline)
  test_integration_engine.py # paged-attention Engine test (Engine stage)
```

## Running locally

Point `--model-root` at a directory laid out like
`<root>/<model>/onnx/<device_dir>/v<N>/genai_config.json` (this is the
exact shape of the `foundrylocalmodels/models` blob container, minus the
`foundrylocal/models/` prefix). Then run:

```bash
pip install -r test/python/requirements.txt
pip install pytest

# all models that support cpu
python -m pytest test/python/integration -sv \
    --model-root /path/to/models \
    --execution-provider cpu

# a single model on cuda
python -m pytest test/python/integration -sv \
    --model-root /path/to/models \
    --execution-provider cuda \
    --model qwen3-0.6b
```

`ORTGENAI_MODEL_ROOT` works the same as `--model-root` and is what CI
sets.

For WebGPU, additionally `pip install onnxruntime-ep-webgpu`. The test
registers the plug-in EP automatically and skips cleanly if the package
isn't installed.

## Running in CI

The ADO `integration-tests` pipeline:

1. Builds ORT GenAI from source for the target OS/EP (one wheel per OS).
2. Fans out test jobs - **one ADO job per (model, ep)** - so each agent
   only needs disk for one model.
3. Each test job `azcopy`s its model from `foundrylocalmodels/models`
   using the agent's managed identity, then runs `pytest --model <id>`.

The Foundry Local SDK is intentionally not installed in CI: it bundles a
native ORT GenAI runtime that would shadow the source-built wheel under
test.

## Paged-attention Engine test

`test_integration_engine.py` drives the continuous-batching `og.Engine`
end to end against `qwen2.5-0.5b-instruct-paged`. It asserts bounded
generation, simultaneous and staggered request isolation, stop conditions,
request removal, and engine reuse. It does **not** measure performance,
memory, capacity, scaling, or longevity.

See [Engine test selection](../../engine/README.md) for guidance on choosing
between C++ unit tests, synthetic-model tests, and real-model integration
tests.

It is **CUDA only**: the PagedAttention operator has no CPU kernel, and this
build's paged KV cache sizes its block pool from free GPU memory.

### How it is selected

The module is marked `engine` and is **skipped unless `--run-engine-tests`
is passed** (registered in `conftest.py`). The text pipeline collects the
whole directory but never passes that flag, so the Engine test stays inert
there and does not require the paged artifact - existing tests are
unaffected.

It runs as the separate `integration_engine_test_linux_x64` stage in the
existing `.pipelines/integration-tests.yml` pipeline. The stage depends on
`build_linux_x64`, downloads the same published Linux CUDA wheel used by the
existing integration jobs, and reuses the shared pytest setup.

Run it locally with the flag (CUDA required):

```bash
python -m pytest test/python/integration/test_integration_engine.py -sv \
    --run-engine-tests \
    --model-root /path/to/models \
    --execution-provider cuda
```

### The paged model artifact

The model already exists in the `staging` container:

```text
https://foundrylocalmodels.blob.core.windows.net/staging/
    paged-attention/qwen2.5-0.5b-instruct/
```

The Engine stage copies that flat directory into its local resolver layout
as `qwen2.5-0.5b-instruct-paged/onnx/cuda/v1`. The logical `-paged` suffix
keeps it separate from the standard Qwen integration model.

The export contains 24 `com.microsoft::PagedAttention` nodes, paged KV inputs,
the external weights file, tokenizer files, and an `engine.dynamic_batching`
configuration:

```jsonc
"engine": {
  "dynamic_batching": {
    "block_size": 256,
    "gpu_utilization_factor": 0.6,   // block pool sized from free GPU memory
    "max_batch_size": 100
  }
}
```

`PINNED_IDENTITY` records SHA-256 hashes for `genai_config.json`,
`model.onnx`, and `model.onnx.data`. The test streams and verifies all three
files so changes to the mutable staging artifact fail explicitly.

### Regenerating the artifact

The staged artifact was produced by the model builder:

```bash
python src/python/py/models/builder.py -m Qwen/Qwen2.5-0.5B-Instruct \
    -o <out> -p int4 -e cuda -c <cache> \
    --extra_options use_paged_attention=true shared_embeddings=false
```

If the staged files are intentionally regenerated, refresh the three hashes in
`PINNED_IDENTITY` in the same change.

## Adding a new model

The blob container is populated by the Foundry team. The integration
suites are kept intentionally small - the goal is **coverage of model
architectures**, not coverage of every checkpoint Foundry ships.

### When to add a model to the suites

| Situation | Add to `MODELS`? | Add to `pr`? | Add to `all`? |
|---|---|---|---|
| New architecture family arrives (e.g. first time `mamba`/`falcon`/`glm`) | yes | **yes** - pick the smallest size with a real release | yes |
| New size/version of a family we already cover (e.g. `qwen3-32b` when we already test `qwen3-0.6b`) | yes | no | yes |
| Finetune/specialized variant of a model we already cover (e.g. `qwen3-0.6b-pp-finetuned`) | yes | no | no - run manually if you need it |
| Model isn't text-to-text (VLM, MMM, ASR, embeddings) | no - out of scope today | no | no |

The `pr` suite gates every PR, so its size directly affects developer
wait time. Add to `pr` only when a genuinely new architecture lands; one
representative model per family is enough.

### Mechanical steps

1. Confirm the model exists in the container by listing parent
   directories of every `genai_config.json` (command in `models.py`).
2. Add an entry to `MODELS` in `models.py` with the device tags it
   supports.
3. Append the logical id to `pr` and/or `all` in `models.py` per the
   table above.
4. Mirror the change in the `pr_models` / `all_models` default lists in
   `.pipelines/integration-tests.yml`. The two must agree; CI fans one
   ADO job out per entry in those lists.

> A paged **Engine** model is different: add it to `MODELS` (CUDA only) and
> the `engine` suite (not `pr`/`all`), pin it in `PINNED_VERSIONS` and
> `PINNED_IDENTITY`, then update the Engine stage in
> `.pipelines/stages/integration-stage.yml`. See "Paged-attention Engine test"
> above.

### Scope

Today: text-to-text only. Vision-language, multimodal, and ASR models
are deliberately out of scope and will be enabled in a separate effort
with a different scenario adapter.
