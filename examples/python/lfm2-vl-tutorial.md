# Build your LFM2-VL / LFM2.5-VL ONNX models for ONNX Runtime GenAI

LFM2-VL pairs a SigLIP2 NaViT vision tower with an LFM2 decoder (the hybrid conv/attention model
described in [the LFM2 support PR](https://github.com/microsoft/onnxruntime-genai/pull/1979)). Like
every other vision-language model in ONNX Runtime GenAI it runs as three ONNX models:

| Model | Inputs | Outputs |
| --- | --- | --- |
| `vision.onnx` | `pixel_values`, `pixel_attention_mask`, `spatial_shapes` | `image_features` |
| `embeddings.onnx` | `input_ids`, `image_features` | `inputs_embeds` |
| `model.onnx` (decoder) | `inputs_embeds`, `attention_mask`, `position_ids`, KV + conv cache | `logits` |

The model builder builds all three from the Hugging Face checkpoint.

## Steps

1. [Build the model](#1-build-the-model)
2. [Run the model](#2-run-the-model)
3. [Known limitations](#3-known-limitations)

## 1. Build the model

```bash
# Download the PyTorch model
$ hf download LiquidAI/LFM2.5-VL-1.6B --local-dir ./lfm2.5-vl-1.6b/pytorch

# Build the decoder as INT4 with FP32 inputs/outputs for CPU, with the vision and embedding models
$ python3 -m onnxruntime_genai.models.builder \
    -i ./lfm2.5-vl-1.6b/pytorch \
    -o ./lfm2.5-vl-1.6b/cpu \
    -p int4 \
    -e cpu \
    --extra_options exclude_embeds=true
```

The vision and embedding models are newer than onnxruntime-genai 0.18.0. Until the next release,
clone this repository and run `python3 onnxruntime-genai/src/python/py/models/builder.py` with the
same arguments instead.

`exclude_embeds=true` is what makes this a vision pipeline: the decoder then takes `inputs_embeds`
instead of `input_ids`, so the embedding model can splice image features into the token embeddings
before the decoder runs. The builder writes `"type": "lfm2_vl"` into `genai_config.json` for this
case, and `"type": "lfm2_vl_text"` when the flag is omitted — that second form is a plain text-only
LFM2 model that happens to come from a VLM checkpoint, and it runs through the normal `AppendTokens`
path with no vision or embedding model.

Next to the decoder, the builder writes:

- `vision.onnx`, the vision tower and the projector. It takes the patches of a batch of images and
  returns the projected tokens of all of them:

  | Name | Shape | Type |
  | --- | --- | --- |
  | `pixel_values` (input) | `[num_images, num_patches, 768]` | float |
  | `pixel_attention_mask` (input) | `[num_images, num_patches]` | int64 |
  | `spatial_shapes` (input) | `[num_images, 2]` | int64 |
  | `image_features` (output) | `[num_image_tokens, hidden_size]` | float |

  `768` is `encoder_patch_size * encoder_patch_size * 3`, `spatial_shapes` holds each image's patch
  grid as `(rows, cols)`, and the model drops the masked padding positions itself, so
  `image_features` is the concatenation of every image's projected tokens with nothing in between.
  `hidden_size` is the decoder's hidden size (1024 for 450M, 2048 for 1.6B and 3B).
- `embeddings.onnx`. It looks up `input_ids` in the checkpoint's embedding table and puts
  `image_features` at the positions holding the image token (`image_token_id` in `config.json`:
  124907 for LFM2.5-VL-3B, 396 for the others).
- the `embedding` and `vision` sections of `genai_config.json`. Both models get the decoder's
  execution provider options without graph capture, which only suits the decoder: a captured graph
  replays the shapes of its first run.
- `processor_config.json`, from the checkpoint's image processor settings.

The vision and embedding models are never quantized. They use the decoder's input/output precision
(float32 in the CPU example above), except that the vision model uses float16 when the decoder
uses bf16.

The three vision fields drive the image processor and come from the image processor settings:

| `genai_config.json` | image processor setting | LFM2-VL / LFM2.5-VL value |
| --- | --- | --- |
| `vision.patch_size` | `encoder_patch_size` | 16 |
| `vision.spatial_merge_size` | `downsample_factor` | 2 |
| `vision.max_num_patches` | `max_num_patches` (`max_image_tokens * downsample_factor²`) | 1024 |

`max_num_patches` is the sequence length every image is padded to so that several images can share
one vision run; set it to `0` to pad each batch to its own longest image instead, which is cheaper
but only valid if the vision graph accepts a dynamic patch count.

`processor_config.json` decodes each image, smart-resizes it so the patch count lands between
`min_image_tokens` and `max_image_tokens` after the projector's 2× pixel unshuffle, then rescales and
normalizes with mean/std `0.5`. The `Resize` step's `interpolation` follows the upstream model's
`resample`, which is not the same across the line:

| Model | Upstream `resample` | `interpolation` |
| --- | --- | --- |
| LFM2-VL-450M, LFM2-VL-1.6B, LFM2-VL-3B | `3` (bicubic), in `preprocessor_config.json` | `CUBIC` |
| LFM2.5-VL-3B | `3` (bicubic), in `processor_config.json` | `CUBIC` |
| LFM2.5-VL-450M, LFM2.5-VL-1.6B | `2` (bilinear), in `processor_config.json` | `LINEAR` |

Using `LINEAR` on a bicubic model still runs, but the pixels drift from the reference: on
LFM2-VL-450M the largest per-pixel difference from the Hugging Face processor grows from 1 to 33
levels out of 255. The builder refuses other `resample` values.

## 2. Run the model

[`model-mm.py`](model-mm.py) drives any multi-modal model in this repository:

```bash
$ python3 model-mm.py -m ./lfm2.5-vl-1.6b/cpu -e cpu
```

The prompt is built by the chat template, which emits one `<image>` per image. The C++ image
processor rewrites each `<image>` into `<|image_start|>`, one `<image>` per projected vision
feature, and `<|image_end|>`, so the number of placeholders always matches the number of features
the vision model produced. The prompt must hold exactly one `<image>` per image, in image order; the
processor rejects any other count, as the Hugging Face processor does, and it accepts one prompt at
a time (a single-entry list is fine, batching is not).

## 3. Known limitations

**LFM2.5-VL-3B needs a tokenizer regex substitution.** Its `tokenizer.json` pre-tokenizes with the
`'(?i:[sdmt]|ll|ve|re)|...` pattern, which the tokenizer in onnxruntime-extensions does not parse
("Invalid '(?...)' zero-width assertion"). Replace that pattern with the equivalent one the other
models use, `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
before loading the model. It selects the same contractions and produces the same tokens on chat
prompts.


**Image splitting (tiling) is not supported.** The Hugging Face processor cuts a large image into up
to `max_tiles` 512×512 tiles plus a thumbnail, which lets it spend thousands of tokens on a
high-resolution image. ONNX Runtime GenAI resizes each image once, to at most `max_image_tokens`
tokens — the same thing the reference implementation does when `do_image_splitting` is `false`.
Large images therefore lose detail compared to the PyTorch model. Adding tiling needs a dedicated
transform in onnxruntime-extensions, because the preprocessing pipeline can only resize an image
once.

**Aspect ratios above 200:1 are rejected.** The `Resize` step in onnxruntime-extensions refuses
an image whose longer side is more than 200 times the shorter one ("Absolute aspect ratio must be
smaller than 200"); the Hugging Face processor has no such limit.

**Small images may be resized one step differently.** For images below `min_image_tokens` worth of
pixels, the smart resize in onnxruntime-extensions truncates an intermediate product before rounding
up to a multiple of `encoder_patch_size * downsample_factor`, so the result can be one 32-pixel step
smaller on one axis than the Hugging Face processor's. The same one-step difference occurs when a side
divided by 32 lands exactly on .5 (for example a width of 2000): Python's `round` goes to the even
neighbour, `std::round` goes away from zero. The token accounting stays consistent in both cases
because the runtime counts tokens from the size the image actually arrived at.
