# GGUF Initializers in ONNX Runtime GenAI

ONNX Runtime GenAI supports loading ONNX model initializers directly from GGUF v2/v3 weight files (`.gguf`), eliminating the need to duplicate model weights into `.onnx.data` external data files.

## Configuration (`genai_config.json`)

The `model.decoder`, `model.draft`, `model.dflash2`, and `model.mtp` sections can specify `"gguf_initializers"` either as a relative path to an external JSON mapping file or as an inline JSON object.

Mappings apply to all decoder sessions, including separately configured speech decoders and Nemotron Parse prefill sessions. Pipelined decoders are not supported and reject GGUF mappings explicitly.

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
| `config_file` | `string` | Optional path (relative to `genai_config.json`) to an external JSON mapping file. Can also be passed directly as `"gguf_initializers": "path.json"`. |
| `files` / `file` | `string[]` / `string` | One or more `.gguf` files (relative to `genai_config.json`) searched when a tensor mapping does not specify its own `file`. Aliases of the same file are deduplicated; matches in distinct files are rejected as ambiguous. Per-tensor overrides do not add files to this default search list. |
| `share` | `bool` | Default cross-session device sharing flag (`false` by default) inherited by tensor entries that do not specify `share`. |
| `tensors` | `object` or `array` | Map of `onnx_initializer_name -> gguf_tensor_name` (or mapping object), or array of mapping objects with `onnx_name` (`name`) and `gguf_name` (`gguf_tensor` / `tensor`). |
| `tensors[].file` | `string` | Optional `.gguf` file override for a specific tensor when weights span multiple GGUF shards. |
| `tensors[].data_type` (`dtype`) | `int` | Optional `ONNXTensorElementDataType` enum override. Defaults to the GGUF type's natural ONNX element type (`FLOAT` for `F32`, `FLOAT16` for `F16`, `BFLOAT16` for `BF16`, `UINT8` for block-quantized types). |
| `tensors[].shape` | `int64[]` | Optional ONNX tensor shape override. An explicit `[]` creates a rank-zero scalar. Must match the exact byte size of the GGUF tensor. When omitted, defaults to `[ggml_dims[n-1], ..., ggml_dims[1], (ggml_dims[0] / blck_size) * type_size]` for block-quantized tensors or `[ggml_dims[n-1], ..., ggml_dims[0]]` for unquantized tensors. |
| `tensors[].share` | `bool` | When `true` on a non-CPU device, copies the GGUF tensor once into a cached device `OrtValue` and registers it via `SessionOptions::AddInitializer` so multiple sessions (e.g. target decoder and `dflash2` / `mtp` drafter) share a single GPU allocation. |

## Runtime Behavior & Zero-Copy Loading

1. **Cached `mmap` and Header Parsing**:
   `GetOrOpenGgufFile` memory-maps each `.gguf` file once (`PROT_READ, MAP_SHARED` on POSIX) and caches the parsed GGUF v2/v3 tensor directory by file identity (`st_dev:st_ino`). Windows uses a retained file buffer and canonical UTF-16 paths.
2. **Direct Buffer Injection (`"share": false`)**:
   `Model::AddGgufInitializers` creates non-owning CPU `OrtValue` views pointing directly into the `mmap`'d GGUF data region, sets `session.use_external_initializer_file_buffers_directly = "1"`, and injects them via `SessionOptions::AddExternalInitializers`. ONNX Runtime aliases the `mmap`'d pages without allocating or copying a duplicate CPU buffer.
3. **Cross-Session Device Sharing (`"share": true`)**:
   Tensors marked with `"share": true` (such as `token_embd.weight` and `output.weight` shared between the target decoder and the LiLiCorr/DFlash2 drafter) are uploaded once to device memory, cached in `g_shared_initializers`, and bound into both sessions via `SessionOptions::AddInitializer`. The cache distinguishes allocator names as well as device types and IDs, so different execution providers cannot share incompatible allocations.
