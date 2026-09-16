# Build your LFM2-VL / LFM2.5-VL ONNX models for ONNX Runtime GenAI

LFM2-VL pairs a SigLIP2 NaViT vision tower with an LFM2 decoder (the hybrid conv/attention model
described in [the LFM2 support PR](https://github.com/microsoft/onnxruntime-genai/pull/1979)). Like
every other vision-language model in ONNX Runtime GenAI it runs as three ONNX models:

| Model | Inputs | Outputs |
| --- | --- | --- |
| `vision.onnx` | `pixel_values`, `pixel_attention_mask`, `spatial_shapes` | `image_features` |
| `embeddings.onnx` | `input_ids`, `image_features` | `inputs_embeds` |
| `model.onnx` (decoder) | `inputs_embeds`, `attention_mask`, `position_ids`, KV + conv cache | `logits` |

Only the decoder is produced by the model builder in this repository; the vision tower and the
embedding model are exported separately, as with Gemma-3 and Phi-3 vision.

## Steps

1. [Build the decoder](#1-build-the-decoder)
2. [Get the vision and embedding models](#2-get-the-vision-and-embedding-models)
3. [Write `genai_config.json` and `processor_config.json`](#3-write-genai_configjson-and-processor_configjson)
4. [Run the model](#4-run-the-model)
5. [Known limitations](#5-known-limitations)

## 1. Build the decoder

```bash
# Download the PyTorch model
$ huggingface-cli download LiquidAI/LFM2.5-VL-1.6B --local-dir ./lfm2.5-vl-1.6b/pytorch

# Build the decoder as INT4 with FP32 inputs/outputs for CPU
$ python3 -m onnxruntime_genai.models.builder \
    -i ./lfm2.5-vl-1.6b/pytorch \
    -o ./lfm2.5-vl-1.6b/cpu \
    -p int4 \
    -e cpu \
    --extra_options exclude_embeds=true
```

`exclude_embeds=true` is what makes this a vision pipeline stage: the decoder then takes
`inputs_embeds` instead of `input_ids`, so the embedding model can splice image features into the
token embeddings before the decoder runs. The builder writes `"type": "lfm2_vl"` into
`genai_config.json` for this case, and `"type": "lfm2_vl_text"` when the flag is omitted — that
second form is a plain text-only LFM2 model that happens to come from a VLM checkpoint, and it runs
through the normal `AppendTokens` path with no vision or embedding model.

## 2. Get the vision and embedding models

LiquidAI publishes ONNX exports of the vision tower and projector for the LFM2.5-VL models. The
vision graph is `onnx/embed_images.onnx` in
[LiquidAI/LFM2.5-VL-1.6B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX) and
`onnx/vision_encoder.onnx` in the
[450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX) and
[3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-ONNX) repositories. Download the graph together
with every `*.onnx_data*` file next to it (the 3B export is split into four). All three have the
signature ONNX Runtime GenAI expects:

| Name | Shape | Type |
| --- | --- | --- |
| `pixel_values` (input) | `[num_images, num_patches, 768]` | float |
| `pixel_attention_mask` (input) | `[num_images, num_patches]` | int64 |
| `spatial_shapes` (input) | `[num_images, 2]` | int64 |
| `image_features` (output) | `[num_image_tokens, hidden_size]` | float |

`768` is `encoder_patch_size * encoder_patch_size * 3`, `spatial_shapes` holds each image's patch
grid as `(rows, cols)`, and the graph drops the masked padding positions itself, so
`image_features` is the concatenation of every image's projected tokens with nothing in between.
`hidden_size` is the decoder's hidden size (1024 for 450M, 2048 for 1.6B and 3B).

The embedding model is a small graph that looks up `input_ids` in the decoder's embedding table and
scatters `image_features` into the positions holding the image token (`image_token_id` in
`config.json`: 396 for the 450M and 1.6B models, 124907 for the 3B model). A Gather followed by a
ScatterND over the flattened sequence is enough; build it from
`model.language_model.embed_tokens.weight` in the checkpoint, or reuse `onnx/embed_tokens.onnx` from
the repository above and add the scatter.

## 3. Write `genai_config.json` and `processor_config.json`

Add `embedding` and `vision` sections to the `genai_config.json` the builder produced. Leave the
`decoder` and `search` sections it wrote alone:

```json
{
    "model": {
        "bos_token_id": 1,
        "context_length": 128000,
        "decoder": { "...": "written by the model builder" },
        "embedding": {
            "filename": "embeddings.onnx",
            "inputs": {
                "input_ids": "input_ids",
                "image_features": "image_features"
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds"
            },
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            }
        },
        "eos_token_id": 7,
        "pad_token_id": 0,
        "type": "lfm2_vl",
        "vision": {
            "filename": "vision.onnx",
            "config_filename": "processor_config.json",
            "patch_size": 16,
            "spatial_merge_size": 2,
            "max_num_patches": 1024,
            "inputs": {
                "pixel_values": "pixel_values",
                "attention_mask": "pixel_attention_mask",
                "image_sizes": "spatial_shapes"
            },
            "outputs": {
                "image_features": "image_features"
            },
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            }
        },
        "vocab_size": 65536
    },
    "search": { "...": "written by the model builder" }
}
```

The three vision fields drive the image processor and must match the model's `config.json`:

| `genai_config.json` | `config.json` | LFM2-VL / LFM2.5-VL value |
| --- | --- | --- |
| `vision.patch_size` | `encoder_patch_size` | 16 |
| `vision.spatial_merge_size` | `downsample_factor` | 2 |
| `vision.max_num_patches` | `max_image_tokens * downsample_factor²` | 1024 |

Every LFM2-VL and LFM2.5-VL model published so far shares these values, so the same
`processor_config.json` serves the whole line, with one exception: set the `Resize` step's
`interpolation` to match `resample` in the model's `processor_config.json`. `2` (bilinear) is
`LINEAR`, which the shipped file uses and which LFM2.5-VL-450M and LFM2.5-VL-1.6B need; `3` (bicubic)
is `CUBIC`, which LFM2.5-VL-3B and the LFM2-VL models need.

`max_num_patches` is the sequence length every image is padded to so that several images can share
one vision run; set it to `0` to pad each batch to its own longest image instead, which is cheaper
but only valid if the vision graph accepts a dynamic patch count.

Copy [`test/models/lfm2-vl/processor_config.json`](../../test/models/lfm2-vl/processor_config.json)
next to `genai_config.json`. It decodes each image, smart-resizes it so the patch count lands
between `min_image_tokens` and `max_image_tokens` after the projector's 2× pixel unshuffle, then
rescales and normalizes with mean/std `0.5`. If you change `min_image_tokens` or `max_image_tokens`,
update the `min_pixels` / `max_pixels` attributes of the `Resize` step to
`tokens * encoder_patch_size² * downsample_factor²`.

## 4. Run the model

[`model-mm.py`](model-mm.py) drives any multi-modal model in this repository:

```bash
$ python3 model-mm.py -m ./lfm2.5-vl-1.6b/cpu -e cpu
```

The prompt is built by the chat template, which emits one `<image>` per image. The C++ image
processor rewrites each `<image>` into `<|image_start|>`, one `<image>` per projected vision
feature, and `<|image_end|>`, so the number of placeholders always matches the number of features
the vision model produced. Images the prompt never referenced are prepended rather than dropped.

## 5. Known limitations

**LFM2.5-VL-3B needs a tokenizer regex substitution.** Its `tokenizer.json` pre-tokenizes with the
`'(?i:[sdmt]|ll|ve|re)|...` pattern, which the tokenizer in onnxruntime-extensions does not parse
("Invalid '(?...)' zero-width assertion"). Replace that pattern with the equivalent one the other
models use, `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
before loading the model. It selects the same contractions and produces the same tokens on chat
prompts.

**The LFM2-VL (non-2.5) models have no published vision export.** The runtime handles them the same
way, but you have to export the vision tower and projector yourself with the signature above.

**Image splitting (tiling) is not supported.** The Hugging Face processor cuts a large image into up
to `max_tiles` 512×512 tiles plus a thumbnail, which lets it spend thousands of tokens on a
high-resolution image. ONNX Runtime GenAI resizes each image once, to at most `max_image_tokens`
tokens — the same thing the reference implementation does when `do_image_splitting` is `false`.
Large images therefore lose detail compared to the PyTorch model. Adding tiling needs a dedicated
transform in onnxruntime-extensions, because the preprocessing pipeline can only resize an image
once.

**Small images may be resized one step differently.** For images below `min_image_tokens` worth of
pixels, the smart resize in onnxruntime-extensions truncates an intermediate product before rounding
up to a multiple of `encoder_patch_size * downsample_factor`, so the result can be one 32-pixel step
smaller on one axis than the Hugging Face processor's. The token accounting stays consistent because
the runtime counts tokens from the size the image actually arrived at.
