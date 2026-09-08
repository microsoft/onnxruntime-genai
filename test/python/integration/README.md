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
  fetch_public_models.py    # revision/hash-pinned public artifacts, normalized locally
  conftest.py                # shared CLI and independent Engine/multimodal opt-ins
  test_integration_text.py   # text-generation test (text pipeline)
  test_integration_engine.py # paged-attention Engine test (Engine stage)
  test_integration_multimodal.py # retained-cache Phi Vision turns (opt-in)
  test_integration_infrastructure.py # small offline catalog/resolver/partition checks
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

## Real Phi Vision continuation suite

The `multimodal` suite is independent of the text `pr`/`all` and Engine suites.
It is skipped unless `--run-multimodal-tests` is passed. Text tests do not
parametrize over the VLM, even when collecting this whole directory.

### Immutable public artifacts

Both variants come from
[`microsoft/Phi-3.5-vision-instruct-onnx`](https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx/tree/672d73375fa86f3d7787e40ac593e33a4f04a055),
revision `672d73375fa86f3d7787e40ac593e33a4f04a055`:

| EP | Upstream subdirectory | Approximate artifact size |
|---|---|---|
| CPU | `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` | 3.2 GB |
| CUDA | `gpu/gpu-int4-rtn-block-32` | 2.6 GB |

`models.PUBLIC_IDENTITY` pins SHA-256 for **all eleven files** in each variant:
three ONNX graphs, all three external weight files, configuration, processor,
and tokenizer data. The fetcher uses the existing Hugging Face dependency,
requests the immutable revision without credentials, verifies each file, and
copies it to `<root>/Phi-3.5-vision-instruct/onnx/<device_dir>/v1`.
The model fixture independently verifies the entire artifact before loading.
These are public artifacts, **not** invented Foundry blob prefixes.

The Hugging Face download cache lives under `<root>/.huggingface`; allow roughly
twice the artifact size on disk. Ordinary copies are intentional: ORT rejects
external-data files with multiple hard links.

```powershell
# Run from the repository root with a source-built wheel installed.
python test\python\integration\fetch_public_models.py `
  --model Phi-3.5-vision-instruct --device cpu `
  --model-root build\models\multimodal-integration
python -m pytest test\python\integration\test_integration_multimodal.py -sv `
  --run-multimodal-tests --model Phi-3.5-vision-instruct `
  --execution-provider cpu --model-root build\models\multimodal-integration
```

Use `--device cuda` when fetching and `--execution-provider cuda` when testing
on a CUDA host. Both provider availability and missing/unsupported required
artifacts are errors, never successful skips. Weights and generated evidence
stay in ignored `build` directories.

### Scenarios and numerical contract

The subject retains one Generator throughout image → response → image,
image → multi-token text → image, initial text → image → text, EOS resume,
latest-image suffix rewind, and deferred-readback payload-lifetime scenarios.
The image pair uses distinct RGB patterns and opposite aspect ratios. The
caller releases image and processed input objects and creates allocation
pressure after each image turn. The stress case teacher-forces response tokens
without intermediate `get_logits`/`get_sequence` inspection.

A new reference Generator replays the same exact full-prefix token history
**only in tests**. The real processor constructs the complete image payload;
the reference correctly numbers later images `image_2`/`-2` while each new
subject turn uses `image_1`/`-1`. Reference response tokens are teacher-forced
from the subject rather than allowing divergent argmax tie handling.
Comparisons begin at a committed prefix before sampling EOS. Terminal
`get_logits()` returns retained scores, which may include caller overrides or
sampling processors, and is not an
equivalent readback of that prefix.

All vocabulary logits must satisfy fixed NumPy `assert_allclose` bounds:
CPU FP32 uses `atol=0.002, rtol=0.0002`; CUDA FP16 uses
`atol=0.06, rtol=0.002`. The INT4 variants are never compared across EPs.
There is no adaptive tolerance or argmax-based fallback.

CPU explicitly sets the documented decoder session option
`session.disable_prepacking=1`. The export's accuracy-level-4 packed kernel
quantizes activations to INT8 and is chunk-dependent even on text-only
continuations. The unpacked path retains the pinned INT4 weights and selected
CPU EP but accumulates in FP32, making the fixed replay oracle meaningful.
This is **not** validation of the default packed INT8 numerical path.
CUDA retains its normal FP16 execution path; no CPU numerical substitute is
permitted there.

`test_default_kernel_retained_turns` separately exercises **unmodified default
numerical kernels**, with both dynamic and shared GQA caches. It retains one
Generator through image → generated response → multi-token text → generated
response → image → generated response, releases caller payloads, checks exact
sequence retention, and compares all vocabulary logits against teacher-forced
replay with **identical chunk widths** using the same fixed tolerances. It also
requires the continued second-image logits to differ from a fresh Generator
given only that second image. These are deterministic retained-runtime and
context-influence checks, not a claim that packed INT8 full-prefix evaluation
is numerically equivalent. CPU default-kernel sessions never disable prepacking
and are cached/profiled separately from the unpacked full-prefix oracle.

The pinned decoder has 32 real GroupQueryAttention nodes. The image-pair
scenario runs with both dynamic and shared cache settings; the other cases
use the export's shared cache mode. Public tests establish numerical cache
continuity, **not native pointer alias/residency**; those assertions belong
in the white-box device tests.

Tests remain below **4096 tokens**. The export changes LongRoPE tables above
4096; full-prefix replay beyond that point is not equivalent to retaining
keys rotated before the transition. The current ORT Extensions processor
uses up to sixteen crops even for small source images, so the chosen aspect
ratios bound the two-image history within the short-context regime. No graph
capture is enabled, and no later-image capture support is claimed.

### Actual provider evidence and CI

Provider selection explicitly replaces vision, embedding, and decoder session
options and filters accelerator selection to GPU hardware. Every session writes an ORT profile to
`build/multimodal-integration-results` (override with
`--multimodal-output-dir`). After successful scenarios the fixture requires
real numerical nodes on the selected EP in **all three sessions**.
For CUDA, CPU execution is permitted only for Shape/Size, the explicit
integer/bool-only index/shape operator list in `_METADATA_OPS`, and exact
If/Loop dispatch nodes found in the hash-pinned graphs. Their body kernels
are individually audited. `SequenceConstruct` and `SplitToSequence` also
require exclusively integer/bool inputs: their output element types are
preserved, although ORT omits sequence outputs from profile shape metadata.
Memory-copy events are not numerical work.
Floating-point CPU embedding/vision/attention fallback fails the suite.
Neither `model.device_type` nor NumPy host copies establish GPU execution.

`.pipelines/integration-tests.yml` enables the suite with
`run_multimodal_tests: true` or its existing nightly schedule. The dedicated
`integration_multimodal_test_linux_x64` stage reuses the source-built Linux
wheel and `integration-test-job.yml`:

- `integration_cpu_Phi_3_5_vision_instruct`: existing Linux CPU pool.
- `integration_cuda_Phi_3_5_vision_instruct`: Linux A10 NV32 pool (24 GB GPU).

The existing `linux_x64_cpu`/`linux_x64_cuda` switches control these lanes.
The public fetch route is explicit; the default Foundry route and text/Engine
jobs are unchanged. JUnit and per-session profiles are published separately.
`check_models_in_sync.py --multimodal ...` checks the mirrored pipeline list.

Real **WebGPU and Qwen-VL coverage remain NOT RUN**: no compatible pinned
export has been verified. The GPU artifact directory name does not imply
WebGPU compatibility. No unverified WebGPU lane or internal storage path is
created; synthetic device tests cover their separately supported scenarios.

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
| Verified VLM export with a supported scenario adapter | yes, in a separate multimodal suite | no | no |
| MMM, ASR, embeddings without a supported adapter | no | no | no |

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

Text-to-text, separately opted-in paged Engine, and the pinned Phi Vision
continuation adapter. Other multimodal families and ASR require verified
artifacts and their own compatible scenario adapters.
