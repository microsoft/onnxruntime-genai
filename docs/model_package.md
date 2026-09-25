# Model packages

onnxruntime-genai can load models from an ONNX Runtime *model package* directory in
addition to the traditional flat model directory. A package bundles multiple build
*variants* of the same model — for example builds targeting different hardware or
compiled with different execution-provider flags — and lets ONNX Runtime pick the variant
that best matches the local hardware at load time. Files that are common across variants,
such as tokenizer or processor configuration, can live in a sibling directory and be
referenced from each variant's configuration with a path scheme.

## Package layout

A model package is a directory containing a top-level `manifest.json`. The package owns a
single component that holds the variants onnxruntime-genai loads. By convention the
component is named `model`. The component is declared inline in the manifest, so its
variant directories sit directly at the package root. Each variant has its own directory
containing a complete `genai_config.json` along with the ONNX graph files and any other
per-variant assets. Files shared across variants (such as tokenizer assets) live in a
content-addressed `shared_assets/sha256-<hex>/` directory and are referenced with the
`sha256:` scheme.

```
my-model.ortpackage/
├── manifest.json
├── openvino-1/
│   ├── genai_config.json
│   ├── model.onnx
│   └── model.onnx.data
├── openvino-2/
│   ├── genai_config.json
│   ├── model.onnx
│   └── model.onnx.data
└── shared_assets/
    └── sha256-<hex>/
        ├── tokenizer.json
        └── processor_config.json
```

The shape of `manifest.json` and the variant declarations follows the ONNX Runtime model
package specification. The minimum needed for onnxruntime-genai is a single inline
component named `model`:

```jsonc
// manifest.json
{
  "schema_version": "1.0",
  "components": {
    "model": {
      "variants": {
        "openvino-1": {
          "ep": "OpenVINOExecutionProvider",
          "device": "npu",
          "compatibility_string": "<compat-string-1>"
        },
        "openvino-2": {
          "ep": "OpenVINOExecutionProvider",
          "device": "npu",
          "compatibility_string": "<compat-string-2>"
        }
      }
    }
  }
}
```

Each variant's directory defaults to its name (`openvino-1`, `openvino-2`) relative to the
package root; declare `variant_directory` on a variant to override that.

In this example both variants target the same execution provider but compile for
different hardware or software versions. The OpenVINO EP scores each variant's `compatibility_string` at load time and ORT selects the highest-scoring match — callers do
not need to know which build is best for their machine.

Each variant directory must contain a `genai_config.json` describing the model to
onnxruntime-genai. Variant directories are otherwise self-contained: they may hold the
ONNX graph, external weights, custom op libraries, LoRA adapters, and any other files
specific to that build. Files that are identical across variants can instead live in a
content-addressed shared asset and be referenced with the `sha256:` scheme (see below),
which lets multiple variants share a single on-disk copy.

## Sharing tokenizer and processor files across variants

Tokenizer assets and processor configuration are typically identical across variants of a
model. Rather than duplicating them in every variant directory, store them once as a
content-addressed *shared asset* and reference them from each variant's `genai_config.json`
using the `sha256:` path scheme.

A shared asset is a directory under the package's `shared_assets/` area named
`sha256-<hex>`, where `<hex>` is the asset's content digest. ONNX Runtime owns the
shared-asset model: it discovers `shared_assets/sha256-<hex>/` directories at load time and
can remap a digest to a custom or external directory through a `shared_assets` override in
the manifest. Referencing an asset by digest therefore resolves to the right location even
when the manifest overrides it.

Set `model.tokenizer_dir` to the asset's `sha256:` URI. An optional `/`-separated tail
selects a subdirectory within the asset.

```jsonc
// openvino-1/genai_config.json
{
  "model": {
    "type": "phi",
    "tokenizer_dir": "sha256:<hex>",
    ...
  },
  "search": {}
}
```

If `tokenizer_dir` is left unset the tokenizer files are looked up alongside
`genai_config.json` exactly like a flat (non-package) model directory.

Processor-specific configuration (`model.vision.config_filename`,
`model.speech.config_filename`) is intentionally left per-variant: those files are small
and keeping them next to the ONNX graphs avoids needing a shared asset for them.

## Path scheme

`model.tokenizer_dir` accepts either a plain path or a scheme-prefixed value. Resolution is
performed by ONNX Runtime's model package resolver:

| Form | Resolution |
| --- | --- |
| *(empty)* | Treated as the variant directory (the directory containing `genai_config.json`). |
| `sha256:<hex>` or `sha256:<hex>/<tail>` | A content-addressed shared asset. Resolves to the asset's directory (honoring manifest `shared_assets` overrides), optionally joined with the confined tail. Only valid when loading from a model package. |
| Any other value | A relative path resolved against the variant directory, with portable-layout confinement (no absolute paths, no `..`). |

## Loading a package

### Python

```python
import onnxruntime_genai as og

# Auto-detects the execution provider when the package declares only one across its
# variants. ORT still picks among multiple variants of the same EP using their
# compatibility strings, so a single Model() call resolves to the right build for the
# local hardware.
model = og.Model("./my-model.ortpackage")

# Multiple execution providers across the variants: load an OgaConfig with an explicit
# ep, then create the Model from it. Passing ep with a flat (non-package) directory is
# rejected.
config = og.Config.from_package_ep("./my-model.ortpackage", "openvino")
model = og.Model(config)
```

The execution provider name accepts the short form used in `genai_config.json`
(`"cuda"`, `"dml"`, `"openvino"`, ...) or the full ONNX Runtime name
(`"CUDAExecutionProvider"`, `"DmlExecutionProvider"`, ...).

### C API

```c
OgaModel* model = NULL;

// Auto-detect.
OgaCheckResult(OgaCreateModel("./my-model.ortpackage", &model));

// Multi-EP package: create a config with an explicit ep, then the model from it.
OgaConfig* config = NULL;
OgaCheckResult(OgaCreateConfigFromPackageEp("./my-model.ortpackage", "openvino", &config));
OgaCheckResult(OgaCreateModelFromConfig(config, &model));
OgaDestroyConfig(config);
```

`OgaCreateModel` and `OgaCreateConfig` continue to work on flat directories unchanged.
When passed a package they auto-detect the execution provider; an ambiguous package
returns an error pointing at `OgaCreateConfigFromPackageEp`. The `FromPackageEp` entry
point rejects flat directories.

A directory is treated as a package when it has a top-level `manifest.json` and no
top-level `genai_config.json`. A flat model directory always keeps `genai_config.json` at
its root, so one that happens to carry an unrelated `manifest.json` is still loaded as a
flat directory.

Combining a model package with `OgaRuntimeSettings` is not supported and returns an
error. Runtime settings work as before with flat directories via
`OgaCreateModelWithRuntimeSettings`.

### C++ wrapper

```cpp
auto model = OgaModel::Create("./my-model.ortpackage");                                // auto-detect

// Multi-EP package: two-step load.
auto config = OgaConfig::CreateFromPackageEp("./my-model.ortpackage", "openvino");
auto model_ov = OgaModel::Create(*config);
```

`OgaConfig::CreateFromPackageEp` requires the path to be a model package.

## Runtime profiles

The selected variant's `genai_config.json` may contain typed runtime profiles for CUDA GPU memory
classes. Base `engine.dynamic_batching` values must remain safe for the minimum supported device;
profiles are optional tuning upgrades applied once during Model creation, after provider/device
resolution and before Engine allocation.

Memory profiles currently require visible CUDA device 0, both as the current device and in
the provider options. Explicit `hardware_device_id` filtering is not supported with profiles.
To select a different physical GPU, set `CUDA_VISIBLE_DEVICES` before starting the process;
that GPU becomes visible ordinal 0. This restriction prevents selecting a profile using one
GPU's memory while loading its graph on another GPU.

```json
{
  "engine": {
    "dynamic_batching": {
      "num_blocks": 512,
      "max_batch_size": 8,
      "max_scheduled_tokens": 2048
    }
  },
  "runtime_profiles": [
    {
      "id": "24gib-to-32gib",
      "eligibility": {
        "minimum_total_device_memory_bytes": 25769803776,
        "maximum_total_device_memory_bytes": 34359738368
      },
      "overlay": {
        "model": {
          "decoder": {
            "filename": "model_32gib.onnx"
          }
        },
        "engine": {
          "dynamic_batching": {
            "num_blocks": 768,
            "max_batch_size": 12,
            "max_scheduled_tokens": 3072
          }
        },
        "search": {
          "chunk_size": 512,
          "max_length": 65536
        },
        "speculative": {
          "max_draft_tokens": 6
        }
      }
    }
  ]
}
```

Profile IDs must be non-empty and unique. Every profile requires a minimum total-memory value;
the maximum is optional and inclusive. Ranges must be valid and non-overlapping. Zero matches uses
the base settings. `overlay` is typed and may contain any subset of these fields:

- `model.decoder.filename`
- `engine.dynamic_batching.num_blocks`
- `engine.dynamic_batching.max_batch_size`
- `engine.dynamic_batching.max_scheduled_tokens`
- `search.chunk_size`
- `search.max_length`
- `speculative.max_draft_tokens`

Omitted fields retain their base values. Every other config field is rejected from a runtime
profile overlay. Alternate decoder graphs may share one external-data file; each graph still owns
its graph-specific inputs, outputs, attributes, and small initializers.

An overlaid `search.max_length` remains the default Engine Request length. The Engine's
`max_request_length` capability reports the hard per-request limit derived from the resolved target
cache and model context. A caller may explicitly choose a `max_session_tokens` value above the
default but not above that capability when it is nonzero. A zero capability means the cache-backed
ceiling is unavailable and preserves the `search.max_length` default and ceiling.

Selection uses total device memory from the primary CUDA interface selected by the normal provider
append path. The query intentionally uses the existing device-ID-agnostic interface for the current
single-discrete-GPU scope. Distinguishing CUDA device ordinals on heterogeneous multi-GPU machines
is deferred to a future interface change. Actual model and Engine allocation remains the
authoritative fit check; profiles do not add a separate free-memory gate.

## Authoring notes

- Every variant declared in the component's `variants` map must have its own directory
  containing a `genai_config.json`. Variant directories are completely independent — they
  may differ in any genai_config field, including `context_length`, tokenizer-related
  defaults, or the set of ONNX graphs they load.
- Variants targeting the same execution provider must each declare a distinct
  `compatibility_string` (and typically a `device`) so the EP can score them against the
  local hardware. ONNX Runtime treats the string as opaque and forwards it to the EP's
  validator; the EP defines the syntax.
- Files shared across variants live in a content-addressed shared asset under
  `shared_assets/sha256-<hex>/`. Reference them from `genai_config.json` with the `sha256:`
  scheme so multiple variants share a single on-disk copy.
- The model package's execution provider names must match what ONNX Runtime expects
  (`CPUExecutionProvider`, `CUDAExecutionProvider`, `DmlExecutionProvider`,
  `OpenVINOExecutionProvider`, ...). The short forms used by genai_config (`cuda`, `dml`,
  `openvino`, ...) are not accepted by ORT in this position.
