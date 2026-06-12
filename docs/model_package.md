# Model packages

onnxruntime-genai can load models from an ONNX Runtime *model package* directory in
addition to the traditional flat model directory. A package bundles multiple build
*variants* of the same model — such as 
execution-provider compile flag — and lets ONNX Runtime pick the variant that best
matches the local hardware at load time. Files that are common across variants, such as
tokenizer or processor configuration, can live in a sibling directory and be referenced
from each variant's configuration with a path scheme.

## Package layout

A model package is a directory containing a top-level `manifest.json`. The package owns a
single component that holds the variants onnxruntime-genai loads. By convention the
component is named `model`. The component directory under `models/` carries a
`metadata.json` enumerating the variants. Each variant has its own directory containing a
complete `genai_config.json` along with the ONNX graph files and any other per-variant
assets.

```
my-model.ortpackage/
├── manifest.json
├── models/
│   └── model/
│       ├── metadata.json
│       ├── openvino-1/
│       │   ├── genai_config.json
│       │   ├── model.onnx
│       │   └── model.onnx.data
│       └── openvino-2/
│           ├── genai_config.json
│           ├── model.onnx
│           └── model.onnx.data
└── shared/
    ├── tokenizer.json
    └── processor_config.json
```

The shape of `manifest.json`, `metadata.json`, and the variant declarations follows the
ONNX Runtime model package specification. The minimum needed for onnxruntime-genai is:

```jsonc
// manifest.json
{
  "schema_version": 1,
  "components": ["model"]
}
```

```jsonc
// models/model/metadata.json
{
  "component_name": "model",
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
```

In this example both variants target the same execution provider but compile for
different hardware or software versions. The OpenVINO EP scores each variant's `compatibility_string` at load time and ORT selects the highest-scoring match — callers do
not need to know which build is best for their machine.

Each variant directory must contain a `genai_config.json` describing the model to
onnxruntime-genai. Variant directories are otherwise self-contained: they may hold the
ONNX graph, external weights, custom op libraries, LoRA adapters, and any other files
specific to that build.

> **Heads-up.** External weight files (`model.onnx.data`) currently sit inside each
> variant directory. An upcoming ONNX Runtime change adds content-addressed shared
> assets, which will let multiple variants of the same model reference a single copy of
> the weights from the package's `shared_assets/` area. Packages authored under the
> conventions in this document migrate without genai-side changes: only the weight
> references inside the variant's `genai_config.json` change form.

## Sharing tokenizer and processor files across variants

Tokenizer assets and processor configuration are typically identical across variants of a
model. They can live in any sibling directory under the package root and be referenced
from each variant's `genai_config.json` using the `package:` path scheme.

Set `model.tokenizer_dir` to the directory holding `tokenizer.json` and friends. The
value uses the path-scheme syntax described below.

```jsonc
// models/model/openvino-1/genai_config.json
{
  "model": {
    "type": "phi",
    "tokenizer_dir": "package:shared",
    "vision": { "config_filename": "package:shared/processor_config.json" },
    ...
  },
  "search": {}
}
```

If `tokenizer_dir` is left unset the tokenizer files are looked up alongside
`genai_config.json` exactly like a flat (non-package) model directory.

## Path scheme

The following fields in `genai_config.json` accept either a plain path or a
scheme-prefixed value:

- `model.tokenizer_dir`
- `model.vision.config_filename`
- `model.speech.config_filename`

| Form | Resolution |
| --- | --- |
| *(empty)* | Treated as the variant directory (the directory containing `genai_config.json`). |
| `package:<relative_path>` | Joined with the package root. Only valid when loading from a model package. |
| Any other value | Joined with the variant directory, matching how every other path in `genai_config.json` is resolved. |

## Loading a package

### Python

```python
import onnxruntime_genai as og

# Auto-detects the execution provider when the package declares only one across its
# variants. ORT still picks among multiple variants of the same EP using their
# compatibility strings, so a single Model() call resolves to the right build for the
# local hardware.
model = og.Model("./my-model.ortpackage")

# Pass an explicit execution provider when the package declares more than one ep.
# Passing ep with a flat (non-package) directory is rejected.
model = og.Model("./my-model.ortpackage", ep="openvino")

# The same options apply to og.Config.
config = og.Config("./my-model.ortpackage", ep="openvino")
```

The execution provider name accepts the short form used in `genai_config.json`
(`"cuda"`, `"dml"`, `"openvino"`, ...) or the full ONNX Runtime name
(`"CUDAExecutionProvider"`, `"DmlExecutionProvider"`, ...).

### C API

```c
OgaModel* model = NULL;

// Auto-detect the execution provider.
OgaCheckResult(OgaCreateModel("./my-model.ortpackage", &model));

// Or specify it explicitly. Only valid for model packages.
OgaCheckResult(OgaCreateModelFromPackage("./my-model.ortpackage", "openvino", &model));

// Same for OgaConfig.
OgaConfig* config = NULL;
OgaCheckResult(OgaCreateConfigFromPackage("./my-model.ortpackage", "openvino", &config));
```

`OgaCreateModel` and `OgaCreateConfig` continue to work on flat directories unchanged.
When passed a package they auto-detect the execution provider; an ambiguous package
returns an error pointing at `OgaCreateModelFromPackage` or `OgaCreateConfigFromPackage`.
The `FromPackage` entry points reject flat directories.

Combining a model package with `OgaRuntimeSettings` is not supported and returns an
error. Runtime settings work as before with flat directories via
`OgaCreateModelWithRuntimeSettings`.

### C++ wrapper

```cpp
auto model = OgaModel::Create("./my-model.ortpackage");                  // auto-detect
auto model_ov = OgaModel::Create("./my-model.ortpackage", "openvino");   // explicit ep
```

The two-argument `Create` overload routes through `OgaCreateModelFromPackage` and
therefore requires the path to be a model package.

## Authoring notes

- Every variant declared in `metadata.json` must have its own directory containing a
  `genai_config.json`. Variant directories are completely independent — they may differ
  in any genai_config field, including `context_length`, tokenizer-related defaults, or
  the set of ONNX graphs they load.
- Variants targeting the same execution provider must each declare a distinct
  `compatibility_string` (and typically a `device`) so the EP can score them against the
  local hardware. ONNX Runtime treats the string as opaque and forwards it to the EP's
  validator; the EP defines the syntax.
- Files outside variant directories (such as the `shared/` directory in the example
  above) are not interpreted by the runtime. Reference them from
  `genai_config.json` with the `package:` scheme to keep them shared across variants.
- The model package's execution provider names must match what ONNX Runtime expects
  (`CPUExecutionProvider`, `CUDAExecutionProvider`, `DmlExecutionProvider`,
  `OpenVINOExecutionProvider`, ...). The short forms used by genai_config (`cuda`, `dml`,
  `openvino`, ...) are not accepted by ORT in this position.
