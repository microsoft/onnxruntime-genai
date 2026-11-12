# GGUF Initializers in ONNX Runtime GenAI

ONNX Runtime GenAI supports loading ONNX model initializers directly from GGUF v2/v3 weight files (`.gguf`), eliminating the need to duplicate model weights into `.onnx.data` external data files.

## Configuration (`genai_config.json`)

The `model.decoder`, `model.draft`, `model.dflash2`, and `model.mtp` sections can specify `"gguf_initializers"` either as a relative path to an external JSON mapping file or as an inline JSON object.

Mappings apply to all decoder sessions, including separately configured speech decoders and Nemotron Parse prefill sessions. Pipelined decoders and `streaming_enc_dec_asr` (Moonshine) models are not supported and reject GGUF mappings explicitly.

### 1. External Mapping File (`gguf_initializers.json`)

In `genai_config.json`:

```json
{
  "model": {
    "type": "nemotron_bolt",
    "decoder": {
      "filename": "model.onnx",
      "gguf_initializers": "gguf_initializers.json"
    },
    "dflash2": {
      "filename": "dflash2.onnx",
      "gguf_initializers": "dflash2_gguf_initializers.json"
    }
  }
}
```

In `gguf_initializers.json`:

```json
{
  "files": [
    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
  ],
  "tensors": {
    "embed_tokens": {
      "gguf_name": "token_embd.weight",
      "share": true
    },
    "language_model.backbone.layers.0.norm.weight": "blk.0.attn_norm.weight",
    "l0.in_proj.fp4": "blk.0.ssm_in.weight",
    "l0.in_proj.fp4g": "blk.0.ssm_in.scale",
    "l1.fc1_w": {
      "gguf_name": "blk.1.ffn_up_exps.weight",
      "data_type": 2,
      "shape": [240, 1536, 264]
    }
  }
}
```

### 2. Inline Object in `genai_config.json`

```json
{
  "model": {
    "decoder": {
      "filename": "model.onnx",
      "gguf_initializers": {
        "files": ["model.gguf"],
        "share": false,
        "tensors": {
          "embed_tokens": {
            "gguf_name": "token_embd.weight",
            "share": true
          },
          "lm_head.weight": "output.weight"
        }
      }
    }
  }
}
```

## Mapping Fields

| Field | Type | Description |
| --- | --- | --- |
| `config_file` | `string` | Optional path (relative to `genai_config.json`) to an external JSON mapping file. Can also be passed directly as `"gguf_initializers": "path.json"`. An external mapping file cannot specify `config_file` itself; nested mapping files are rejected. |
| `files` | `string[]` | One or more `.gguf` files (relative to `genai_config.json`) searched when a tensor mapping does not specify its own `file`. Aliases of the same file are deduplicated; matches in distinct files are rejected as ambiguous. Per-tensor overrides do not add files to this default search list. |
| `share` | `bool` | Default cross-session device sharing flag (`false` by default) inherited by tensor entries that do not specify `share`. |
| `tensors` | `object` | Map from ONNX initializer name to either a GGUF tensor name (string) or a mapping object with the fields below. |
| `tensors.<name>.gguf_name` | `string` | Required in a mapping object: the GGUF tensor name. |
| `tensors.<name>.file` | `string` | Optional `.gguf` file override for a specific tensor when weights span multiple GGUF shards. |
| `tensors.<name>.data_type` | `int` | Optional `ONNXTensorElementDataType` enum override. Defaults to the GGUF type's natural ONNX element type (`FLOAT` for `F32`, `FLOAT16` for `F16`, `BFLOAT16` for `BF16`, `UINT8` for block-quantized types). |
| `tensors.<name>.shape` | `int64[]` | Optional ONNX tensor shape override. An explicit `[]` creates a rank-zero scalar. Must match the exact byte size of the GGUF tensor. When omitted, defaults to `[ggml_dims[n-1], ..., ggml_dims[1], (ggml_dims[0] / blck_size) * type_size]` for block-quantized tensors or `[ggml_dims[n-1], ..., ggml_dims[0]]` for unquantized tensors. |
| `tensors.<name>.share` | `bool` | When `true` on a non-CPU device, copies the GGUF tensor once into a cached device `OrtValue` and registers it via `SessionOptions::AddInitializer` so multiple sessions (e.g. target decoder and `dflash2` / `mtp` drafter) share a single GPU allocation. Ignored on CPU. Rejected on DML, whose allocator is closed while shared tensors may still be alive. |

Each field has exactly one spelling, and unknown fields are rejected.

Only the tensors that a mapping selects are validated, so a GGUF file can contain tensors of types this loader does not support (or with inconsistent geometry) as long as no mapping selects them. Selecting such a tensor fails with an error naming the tensor and file.

## Runtime Behavior & Zero-Copy Loading

1. **Cached Read-Only Mapping and Header Parsing**:
   `GetOrOpenGgufFile` maps each `.gguf` file read-only once (`mmap` with `PROT_READ, MAP_SHARED` on POSIX; `CreateFileMappingW` with `PAGE_READONLY` and `MapViewOfFile` on Windows, opened by UTF-16 path) and caches the parsed GGUF v2/v3 tensor directory by file identity (`st_dev:st_ino` on POSIX, canonical path on Windows). Pages are loaded on demand and remain reclaimable by the OS. Every model that uses a file holds a reference to its mapping, so the mapping stays valid until the last such model is destroyed.
2. **Direct Buffer Injection (`"share": false`)**:
   `Model::AddGgufInitializers` creates non-owning CPU `OrtValue` views pointing directly into the mapped GGUF data region, sets `session.use_external_initializer_file_buffers_directly = "1"`, and injects them via `SessionOptions::AddExternalInitializers`. Whether ONNX Runtime borrows a view instead of copying it depends on the runtime and the tensor:
   - The runtime must include [microsoft/onnxruntime#33877](https://github.com/microsoft/onnxruntime/pull/33877) (ONNX Runtime 1.32.0 or later). Older runtimes apply this option only to `AddExternalInitializersFromFilesInMemory`, so every tensor supplied through `AddExternalInitializers` is copied into memory owned by ONNX Runtime during session creation.
   - Even then, ONNX Runtime borrows only tensors larger than 127 bytes whose data address is a multiple of their element size, and only on little-endian hosts; it copies all other tensors. GGUF tensor data is aligned to `general.alignment` (32 bytes by default), so tensors are naturally aligned unless the file uses a smaller alignment than the element size of the mapped ONNX type.
   - Borrowing removes only the duplicate CPU copy. Execution providers that keep initializers in device memory still upload them during session creation.

   When tensors are copied, budget host memory for those copies in addition to the mapped file pages.
3. **Cross-Session Device Sharing (`"share": true`)**:
   Tensors marked with `"share": true` (such as `token_embd.weight` and `output.weight` shared between the target decoder and the LiLiCorr/DFlash2 drafter) are uploaded once to device memory directly from the mapped file, cached in `g_shared_initializers`, and bound into both sessions via `SessionOptions::AddInitializer`. The device buffers keep no host staging copy after the upload, so sharing costs only the device allocation besides the mapped pages. The cache distinguishes allocator names as well as device types and IDs, so different execution providers cannot share incompatible allocations. Sharing is not supported on DML.
