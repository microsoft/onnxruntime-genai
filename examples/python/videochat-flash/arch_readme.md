# Export Pipeline Architecture — `internVideo2_builder.py`

Detailed architecture reference for the ONNX export of VideoChat-Flash's vision and embedding components.

Source model: [`OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B`](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B)

---

## Table of Contents

1. [Export Modes](#export-modes)
2. [Script Execution Flow](#script-execution-flow)
3. [Vision Export — `export_vision()`](#vision-export--export_vision)
4. [Embedding Export — `export_embedding()`](#embedding-export--export_embedding)
5. [Meta-Device Parameter Fix](#meta-device-parameter-fix)
6. [ONNX Weight Consolidation](#onnx-weight-consolidation)
7. [Full Shape Reference](#full-shape-reference)

---

## Export Modes

The script supports three mutually exclusive modes via CLI flags:

```
python internVideo2_builder.py                  # vision only (image mode)
python internVideo2_builder.py --video          # vision only (video mode)
python internVideo2_builder.py --embed          # vision + embedding
python internVideo2_builder.py --video --embed  # vision (video) + embedding
python internVideo2_builder.py --embed-only     # embedding only (no vision)
```

| Flag | Exports | Output files |
|------|---------|-------------|
| *(none)* | Vision (image) | `vcf-vision.onnx` + `.data` |
| `--video` | Vision (video) | `vcf-vision-video.onnx` + `.data` |
| `--embed` | Vision + Embedding | `vcf-vision*.onnx` + `vcf-embed.onnx` |
| `--embed-only` | Embedding only | `vcf-embed.onnx` + `.data` |

---

## Script Execution Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLI Argument Parse                            │
│                    --video  --embed  --embed-only                        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         --embed-only    --embed       (default)
              │              │              │
              │         ┌────┴────┐         │
              │         ▼         ▼         ▼
              │   export_vision() │   export_vision()
              │         │         │
              │         ▼         │
              │   export_embedding()
              │                   │
              ▼                   │
        export_embedding()        │
                                  │
                             ◄────┘
                            Done
```

---

## Vision Export — `export_vision()`

Exports the InternVideo2-1B vision tower and mm_projector (ToMe + MLP connector) as a single ONNX model. The mode (image vs. video) changes the wrapper class and compression behavior.

### Step-by-step

```
[1/5] Load HF model (float16)
       │
       ├── Extract vision_tower     (InternVideo2-1B ViT)
       ├── Extract mm_projector     (ToMe token compression + MLP)
       └── Delete LLM backbone      (free ~7 GB: layers, embed_tokens, lm_head)
                │
[2/5] Fix meta-device parameters    (see "Meta-Device Parameter Fix" below)
                │
[3/5] Wrap in mode-specific nn.Module, cast to float32, test forward pass
                │
[4/5] torch.onnx.export             (opset 18, dynamic batch + frames)
                │
[5/5] Consolidate external weights   (single .onnx.data file)
```

### Image Mode — `VisionWithProjectorImage`

```
                      images
                  [B, 1, 3, 224, 224]
                        │
                        ▼
            ┌───────────────────────┐
            │   InternVideo2-1B     │
            │   (vision_tower)      │
            │                       │
            │   ViT patch embed:    │
            │   224 / 14 = 16       │
            │   16 × 16 = 256       │
            │   spatial patches     │
            │   per frame           │
            │                       │
            │   hidden_dim = 1408   │
            └───────────┬───────────┘
                        │
                  [B, 256, 1408]
                        │
                        ▼
            ┌───────────────────────┐
            │   mm_projector        │
            │                       │
            │   compress = False    │
            │   ToMe: 4× merge     │
            │   256 → 64 tokens    │
            │                       │
            │   MLP: 1408 → 3584   │
            └───────────┬───────────┘
                        │
                  visual_tokens
                  [B, 64, 3584]
```

**Output:** `vcf-vision.onnx` — 64 visual tokens per image, matching `HIDDEN_SIZE=3584` of the Qwen2.5-7B LLM.

### Video Mode — `VisionWithProjectorVideo`

```
                      images
                  [B, T=4, 3, 224, 224]
                        │
                        ▼
            ┌───────────────────────┐
            │   InternVideo2-1B     │
            │   (vision_tower)      │
            │                       │
            │   T frames × 256     │
            │   patches = 1024      │
            │   spatiotemporal      │
            │   tokens              │
            └───────────┬───────────┘
                        │
                  [B, T×256, 1408]
                  = [B, 1024, 1408]
                        │
                        ▼  reshape
                  [B×T, 256, 1408]
                  = [4, 256, 1408]
                        │
                        ▼
            ┌───────────────────────┐
            │   mm_projector        │
            │                       │
            │   compress = True     │
            │   local_num_frames=T  │
            │   ToMe: 16× merge    │
            │   256 → 16 tok/frame │
            │                       │
            │   MLP: 1408 → 3584   │
            └───────────┬───────────┘
                        │
                  visual_tokens
                  [B, 16×T, 3584]
                  = [B, 64, 3584]
```

**Output:** `vcf-vision-video.onnx` — 64 visual tokens per 4-frame segment (16 tokens/frame, temporally compressed).

**Important:** `T=4` (`mm_local_num_frames` from config) is baked as a constant in the reshape op at export time. At inference, each call must provide exactly T frames. Longer videos are processed as multiple segments.

### ToMe Compression Comparison

```
                   Image Mode                     Video Mode
                ┌──────────────┐              ┌──────────────┐
  Input patches │  256 / frame │              │  256 / frame │
                └──────┬───────┘              └──────┬───────┘
                       │                             │
                       ▼                             ▼
  ToMe merge    256 → 64 (4×)              256 → 16 (16×) per frame
  ratio         compress=False              compress=True
                       │                             │
                       ▼                             ▼
  Per call      64 tokens                    16 × T = 64 tokens
                (1 frame)                    (T=4 frames)
                       │                             │
                       ▼                             ▼
  Tokens/frame  64                           16
  Info density  High spatial detail          Temporal context, less spatial
```

### ONNX Dynamic Axes

```
input:  "images"        dim 0 = "batch"       (variable)
                        dim 1 = "num_frames"  (variable in graph, but T baked in reshape)

output: "visual_tokens" dim 0 = "batch"              (variable)
                        dim 1 = "num_visual_tokens"   (variable)
```

---

## Embedding Export — `export_embedding()`

Exports the `EmbeddingWithMerge` module: a Qwen2.5 embedding table that also injects visual tokens at `<|image_pad|>` positions.

### Step-by-step

```
[1/3] Load embed_tokens.weight from safetensors (fp32, ~2 GB)
       │   Only reads a single tensor — no full model load needed.
       │   Scans shards for key "model.embed_tokens.weight"
       │
[2/3] Build EmbeddingWithMerge module, run test forward pass
       │   Validates that <|image_pad|> positions are correctly replaced
       │
[3/3] torch.onnx.export → consolidate weights → vcf-embed.onnx
```

### `EmbeddingWithMerge` — Internal Data Flow

```
 input_ids [B, seq_len]             image_features [1, N, 3584]
     │                                      │
     ▼                                      │
 ┌────────────────────────┐                  │
 │ embed_tokens           │                  │
 │ nn.Embedding           │                  │
 │ (152064 × 3584)        │                  │
 └──────────┬─────────────┘                  │
            │                                │
     text_embeds                             │
     [B, seq_len, 3584]                      │
            │                                │
            ▼                                ▼
 ┌──────────────────────────────────────────────────────────┐
 │                      Merge Logic                         │
 │                                                          │
 │  1. mask = (input_ids == 151655)         bool [B, seq]   │
 │                                                          │
 │  2. indices = cumsum(mask) - 1           int  [B, seq]   │
 │     Maps each <|image_pad|> to its                       │
 │     0-based position in image_features                   │
 │                                                          │
 │  3. safe_features = cat(                                 │
 │       image_features.flatten,   [N, 3584]                │
 │       zeros(1, 3584)            ← safety row             │
 │     )                           [N+1, 3584]              │
 │                                                          │
 │  4. visual_at_pos = embedding(indices, safe_features)    │
 │                                 [B, seq, 3584]           │
 │                                                          │
 │  5. output = where(mask, visual_at_pos, text_embeds)     │
 │                                                          │
 └──────────────────────────┬───────────────────────────────┘
                            │
                      inputs_embeds
                      [B, seq_len, 3584]
```

**Why the safety row?** When `image_features` has 0 tokens (text-only prompt), indices would index into an empty tensor. The appended zero row makes the gather always safe — those values are never selected by the `where` because `mask` is all-False.

### ONNX Dynamic Axes

```
input:  "input_ids"      dim 0 = "batch"            int64
                         dim 1 = "seq_len"

input:  "image_features" dim 0 = "num_images"        float32
                         dim 1 = "num_image_tokens"

output: "inputs_embeds"  dim 0 = "batch"             float32
                         dim 1 = "seq_len"
```

All spatial dimensions are dynamic. `image_features` dim 1 can be 0 for text-only inference.

---

## Meta-Device Parameter Fix

The InternVideo2-1B checkpoint has a naming mismatch:

```
  Checkpoint key:    model.vision_tower.blocks.0.attn.proj.ls1.gamma
  Model parameter:   blocks.0.attn.proj.ls1.weight
```

`from_pretrained()` fails to match `ls1.gamma` → `ls1.weight`, leaving those `LayerScale` parameters on the `meta` device (empty placeholders).

### Fix procedure (step [2/5])

```
┌──────────────────────────────────────────────────────────────────┐
│  1. Identify meta-device params                                  │
│     {n: p for n, p in vision_tower.named_parameters()            │
│      if p.device.type == "meta"}                                 │
│                                                                  │
│  2. Download/locate safetensors shards                           │
│     snapshot_download(model_id)                                  │
│                                                                  │
│  3. Build lookup: strip "model.vision_tower." prefix             │
│     from checkpoint keys                                         │
│                                                                  │
│  4. For each meta param, try matching:                           │
│     param_name            → direct match                         │
│     param_name(.weight→.gamma) → gamma variant                   │
│                                                                  │
│  5. Load matched tensor from safetensors shard                   │
│     Set on parent module via setattr()                           │
│     Cast to float16 to match model dtype                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## ONNX Weight Consolidation

`torch.onnx.export` with large models creates many individual external data files. The consolidation step (step [5/5]) merges them:

```
Before consolidation:              After consolidation:
  vcf-vision.onnx                    vcf-vision.onnx        (graph only)
  vision_tower.block0.weight         vcf-vision.onnx.data   (all weights)
  vision_tower.block1.weight
  mm_projector.mlp.0.weight
  mm_projector.mlp.2.weight
  ... (hundreds of files)
```

### Process

```
1. onnx.load(path, load_external_data=True)    ← all weights into RAM
2. Delete individual weight files from disk
3. convert_model_to_external_data(
     all_tensors_to_one_file=True,
     location="*.onnx.data",
     size_threshold=1024                        ← only externalize tensors ≥1KB
   )
4. onnx.save_model()                           ← writes graph + single .data file
```

---

## Full Shape Reference

### Constants

```
IMAGE_SIZE         = 224          Input image resolution
PATCH_SIZE         = 14           ViT patch size (224/14 = 16 grid)
PATCHES_PER_FRAME  = 256          16 × 16 spatial patches
VIT_HIDDEN_SIZE    = 1408         InternVideo2-1B hidden dimension
LLM_HIDDEN_SIZE    = 3584         Qwen2.5-7B hidden dimension
LOCAL_NUM_FRAMES   = 4            T — baked into video mode at export
VOCAB_SIZE         = 152064       Qwen2.5 vocabulary size
IMAGE_PAD_TOKEN_ID = 151655       <|image_pad|> token id
```

### Image Mode — End-to-End Shapes

```
                         ┌────────────────────────────────────────┐
                         │            vcf-vision.onnx             │
images ─────────────────▶│                                        │──────▶ visual_tokens
[B, 1, 3, 224, 224]      │  ViT: [B, 256, 1408]                  │       [B, 64, 3584]
                         │  ToMe 4×: [B, 64, 1408]               │
                         │  MLP: [B, 64, 3584]                   │
                         └────────────────────────────────────────┘

                         ┌────────────────────────────────────────┐
input_ids ──────────────▶│            vcf-embed.onnx              │
[B, seq_len]             │                                        │──────▶ inputs_embeds
image_features ─────────▶│  embed: [B, seq_len, 3584]            │       [B, seq_len, 3584]
[1, 64, 3584]            │  merge at <|image_pad|> positions     │
                         └────────────────────────────────────────┘
```

### Video Mode — End-to-End Shapes

```
                         ┌────────────────────────────────────────┐
                         │         vcf-vision-video.onnx          │
images ─────────────────▶│                                        │──────▶ visual_tokens
[B, 4, 3, 224, 224]      │  ViT: [B, 1024, 1408]                 │       [B, 64, 3584]
                         │  reshape: [B×4, 256, 1408]             │
                         │  ToMe 16×: [B×4, 16, 1408]            │
                         │  MLP + rebatch: [B, 64, 3584]         │
                         └────────────────────────────────────────┘

                         ┌────────────────────────────────────────┐
input_ids ──────────────▶│            vcf-embed.onnx              │
[B, seq_len]             │                                        │──────▶ inputs_embeds
image_features ─────────▶│  embed: [B, seq_len, 3584]            │       [B, seq_len, 3584]
[1, S×64, 3584]          │  merge at <|image_pad|> positions     │
  S = num_segments       └────────────────────────────────────────┘
```

### Multi-Segment Video (at inference time)

```
  Video: 16 frames sampled
    │
    ▼ group into segments of T=4
  Segment 0: frames [0,1,2,3]  → [1, 4, 3, 224, 224] → vcf-vision-video → [1, 64, 3584]
  Segment 1: frames [4,5,6,7]  → [1, 4, 3, 224, 224] → vcf-vision-video → [1, 64, 3584]
  Segment 2: frames [8,9,10,11] → [1, 4, 3, 224, 224] → vcf-vision-video → [1, 64, 3584]
  Segment 3: frames [12,13,14,15] → [1, 4, 3, 224, 224] → vcf-vision-video → [1, 64, 3584]
    │
    ▼ concatenate
  visual_tokens [1, 256, 3584]   (4 segments × 64 tokens)
    │
    ▼
  Prompt needs 256 × <|image_pad|> tokens to match
```

---

## Output Files

| File | Size | Contents |
|------|------|----------|
| `vcf-vision.onnx` | ~100 KB | ONNX graph (image mode) |
| `vcf-vision.onnx.data` | ~1 GB | InternVideo2-1B + mm_projector weights |
| `vcf-vision-video.onnx` | ~100 KB | ONNX graph (video mode) |
| `vcf-vision-video.onnx.data` | ~1 GB | Same weights, different graph topology |
| `vcf-embed.onnx` | ~100 KB | ONNX graph (embed + merge) |
| `vcf-embed.onnx.data` | ~2 GB | embed_tokens weight (152064 × 3584 × fp32) |
