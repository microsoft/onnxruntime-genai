# Pipeline-as-Config examples

These examples demonstrate the **schema v2** (`"version": 2`) "Pipeline-as-Config" surface added in
[issue #2114](https://github.com/microsoft/onnxruntime-genai/issues/2114). A v2 `genai_config.json`
describes a model's execution as an explicit **pipeline** — the ONNX sessions it runs, the order and
lifecycle phase they run in (`flow`), how their tensors are wired together (`dataflow`), and the
state it carries (KV cache, cross-attention cache, position ids) — instead of relying on a hard-coded
`model.type` string.

Every example here is a real, parseable config. Examples 1, 2, 3 and 5 are covered by unit tests
(`ExamplePipelineConfigs.*` in `test/pipeline_config_tests.cpp`) that load each config, lower it, and
assert it routes to the expected model class. (They reference ONNX files by name but the tests only
exercise config parsing/lowering/dispatch, so no model weights are required.)

| # | Example | Folder | Demonstrates |
|---|---------|--------|--------------|
| 1 | Preset usage | [`01-preset-decoder/`](01-preset-decoder/) | The minimal v2 decoder: select a built-in preset by name |
| 2 | Explicit multi-stage dataflow | [`02-explicit-encoder-decoder/`](02-explicit-encoder-decoder/) | Hand-written `flow`/`dataflow`, `init`/`step` phases, cross-attention cache |
| 3 | VLM per-image flow | [`03-vlm-per-image/`](03-vlm-per-image/) | The Qwen-VL style vision pipeline (`loop: "per_image"`, 3D mRoPE) |
| 4 | Plugin escape hatch | [`04-plugin-escape-hatch/`](04-plugin-escape-hatch/) | The opaque-handle plugin shape (**syntax only — requires `USE_GENAI_PLUGINS=ON`**) |
| 5 | v1 → v2 side by side | [`05-v1-to-v2/`](05-v1-to-v2/) | A legacy v1 config and its hand-written v2 equivalent |

---

## The `version` field

`version` selects the schema:

* **`version: 1`** (the default when omitted) — the legacy layout, where everything lives under the
  top-level `model` object (`model.decoder.filename`, `model.vision.filename`, …). **Every existing
  v1 config keeps working unchanged.**
* **`version: 2`** — adds the top-level `pipeline` section (plus the v2 conveniences `tokens`,
  `generation` and `metadata`). The runtime still reads `model.*` internally; on load it *lowers* the
  v2 sections into the same fields, so the rest of the engine is unaffected.

You only need v2 if you want to express a pipeline explicitly (or use a preset / plugin). If v1 works
for your model, there is no reason to migrate.

## v1 → v2 migration (v1 is still fully supported)

Internally, the runtime always builds a normalized `Config::Pipeline` view:

* For a **v1** config, `TranslateV1ToPipeline()` *derives* the pipeline view from `model.type` and the
  populated `model.*` fields, **without changing `model.*` at all**. This is how a legacy gpt2 config
  becomes an "autoregressive-decoder with a combined KV cache" internally.
* For a **v2** config, the pipeline is parsed directly, `extends` presets are resolved, and the
  session files / tokens / generation settings are lowered back into `model.*` / `search.*`.

Because both paths converge on the same structural view, the model-dispatch decision
(`ClassifyStructuralRoute`) is made on config **structure** — which sessions exist, whether there is a
cross-attention cache, whether the KV cache is combined — not on a `model.type` string. Example 5
shows a v1 gpt2 config and a v2 config that produce the *same* dispatch decision.

To migrate a config by hand:

1. Set `"version": 2`.
2. Move each ONNX file into `pipeline.sessions` under a logical name (`decoder`, `encoder`, `vision`,
   `speech`, `embedding`).
3. Either set `pipeline.extends` to a preset (see below) or write `flow`/`dataflow` explicitly.
4. (Optional) Move token ids into `tokens`, generation settings into `generation`, and
   `model_type`/provenance into `metadata`.

## Presets

A **preset** is a fully-formed pipeline skeleton for a common topology. Select one with
`pipeline.extends`. The four built-in presets (see `src/pipeline_presets.cpp`) are:

| Preset name | Topology |
|-------------|----------|
| `autoregressive-decoder` | A single decoder session run every step (plain LLM). |
| `vision-language` | `vision` (init) → `embedding` (init) → `decoder` (step), with vision/embedding/decoder dataflow wired. |
| `encoder-decoder` | `encoder` (init) + `decoder` (step) with a frozen cross-attention cache. |
| `speech-language` | `speech` (init) → `embedding` (init) → `decoder` (step). |

**Override semantics** (`replace, don't merge` — documented canonically in `src/pipeline_presets.h`):

* The named preset is the **base**.
* Any top-level array you specify (`flow`, `dataflow`) **replaces** the preset's array wholesale — no
  per-element merge. Omit the array to inherit the preset's.
* For `state`, sub-objects you touch override the preset's; untouched sub-objects keep preset defaults.
* `sessions` always come from your config (presets never invent file names).

Example 1 is the smallest case: `extends: "autoregressive-decoder"` plus a single `decoder` session —
the preset supplies the one-step `flow`.

## Flow phases and guardrails

`flow` is an ordered list of steps; each step has:

* `run` — the session name to execute.
* `when` — the lifecycle phase: **`init`** (run once during prompt processing, e.g. a vision or
  encoder pass) or **`step`** (run every token-generation step, e.g. the decoder). A **`final`** phase
  also exists (a stage run exactly once *after* the generation loop completes), but no in-tree model
  declares a final stage today, so in practice you will only use `init` and `step`.
* `loop` — **`batched`** (default) or **`per_image`** (run the step once per input image; used by the
  Qwen-VL family — see example 3).
* `cross_attention_from` — for encoder-decoder, the session that produces the cross-attention KV
  (see example 2).

`dataflow` is a list of `{from, to}` wires, each `"session.tensor"`, that connect one session's output
to another's input.

The pipeline runtime (`src/models/decoder_only_pipeline.cpp`) enforces two guardrails on the resulting
stage graph:

* **Cycle detection** — the dataflow graph must be acyclic; a cycle is rejected at load.
* **A 10-stage cap** — a pipeline may not exceed 10 stages.

Keep hand-written pipelines acyclic and within the cap (all examples here use 1–3 stages).

## Plugin opt-in (the escape hatch)

For a fully custom pipeline that no preset/flow can express, a config may declare a `pipeline.plugin`
`{library, entry_point}` (example 4). The runtime loads the shared library and calls the C entry point
to build the model (see `src/models/plugin_api.h`).

**This path is build-gated and OFF by default.** It is only linked when the runtime is built with
`USE_GENAI_PLUGINS=ON`. In a default build, a config that declares `pipeline.plugin` parses fine but
**fails at model construction** with a clear *"plugin support is not enabled in this build"* error.
Example 4 is therefore a **syntax/documentation example only** and is intentionally not exercised by
the example unit tests.

## Not yet implemented (future schema revisions)

The v2 schema shipped here is intentionally scoped to the topologies above. The following ideas were
discussed but are **not implemented** and must not be expected to work in this build:

* TTS `single_pass` flows and `when: "final"` vocoder steps.
* Diffusion / denoising loops.
* RNNT/transducer loop strategies expressed as `flow` (transducers still use their dedicated classes).
* `repeat` / `counter` loop constructs.

If a future config uses any of these, treat them as forward-looking and unsupported until a later
schema revision lands.
