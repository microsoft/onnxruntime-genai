# Prepared-input Nemotron-H Omni models

`model.type = "nemotron_h_omni"` selects the generic three-model multimodal
pipeline: vision, embedding, then decoder. This registration establishes the
prepared-tensor runtime path; it does **not** provide a native C-RADIO image/video
processor or establish checkpoint numerical parity.

## Model and cache bindings

Provide `model.vision.filename`, `model.embedding.filename` and
`model.decoder.filename`. The standard role mappings are:

| Section | Inputs | Outputs |
| --- | --- | --- |
| vision | Prepared `pixel_values`, scalar `is_video` | `image_features`, `retention_mask`, `kept_indices` |
| embedding | `input_ids`, flat `image_features` | `inputs_embeds` |
| decoder | `inputs_embeds`, integer `attention_mask`, integer `position_ids`, caches | `logits`, caches |

The embedding and vision feature tensors should use the flat shape
`[total_image_tokens, decoder_hidden_size]`. The embedding model must support
empty features `[0, decoder_hidden_size]` during text-only input and cached
decoding. Attention masks and position IDs, when present, must both use int32
or both use int64.

Preserve architecture layer metadata and global cache indices. Attention layers
have paired key/value caches; Mamba layers have paired convolution/SSM states;
MoE layers need neither. Configure the actual graph names rather than adding
dummy cache inputs for stateless layers. For exports using `.ssm_state`, the
decoder binding fragment is:

```json
{
  "inputs": {
    "inputs_embeds": "inputs_embeds",
    "attention_mask": "attention_mask",
    "position_ids": "position_ids",
    "past_key_names": "past_key_values.%d.key",
    "past_value_names": "past_key_values.%d.value",
    "past_conv_names": "past_key_values.%d.conv_state",
    "past_recurrent_names": "past_key_values.%d.ssm_state"
  },
  "outputs": {
    "logits": "logits",
    "present_key_names": "present.%d.key",
    "present_value_names": "present.%d.value",
    "present_conv_names": "present.%d.conv_state",
    "present_recurrent_names": "present.%d.ssm_state"
  }
}
```

Supply the decoder dimensions and full architecture layer count separately.
State dimensions and the independent convolution/SSM element types come from
the ONNX session. Keep ordinary initial validation at batch size 1, beam count 1,
without speculation. Buffer sharing must match the exported cache operators,
not the architecture name.

For CUDA, configure the decoder's provider as usual. Omit auxiliary
`session_options` to reuse the decoder settings. If supplying separate vision
or embedding session options in `genai_config.json`, explicitly include
`"provider_options": [{"cuda": {}}]` there as well; those sessions otherwise
use their own settings, not the decoder's enabled providers. Check placement
for each role rather than inferring it from the decoder's device type.

## Prepared inputs, not a processor alias

Use `Generator.SetInputs` with named tensors containing:

- `input_ids`: int32 prompt tokens, already expanded with media placeholders;
- `num_image_tokens`: int64 counts totaling the vision output's feature rows;
- the exact prepared media inputs required by the vision graph.

`num_image_tokens` is runtime metadata, not necessarily a vision graph input.
For images, `pixel_values` also supplies the image count. Additional graph-named
media inputs can be routed as extra tensors, but that does not implement video
sampling, temporal packing or prompt expansion.

The source-specific vision metadata bindings are accepted:

```json
{
  "inputs": {"pixel_values": "pixel_values", "is_video": "is_video"},
  "outputs": {
    "image_features": "image_features",
    "retention_mask": "retention_mask",
    "kept_indices": "kept_indices"
  }
}
```

`is_video` is a scalar BOOL graph input, not an image count. The vision graph
receives externally normalized FLOAT32 `[N,3,H,W]`; its image path processes
an image stream, and its video path processes **one clip per call**. The
video export owns temporal-2 packing, odd-last-frame repetition and the separate
learned video patch projection. Reusing the image projection is not equivalent.

No `MultiModalProcessor` is registered for this family. Do not alias it to a
Qwen processor: Nemotron-H Omni does not have Qwen's image-grid/MRoPE contract.
Tokenizer, chat-template and generation metadata must come from the same pinned
checkpoint. Preserve its actual generation stop IDs, rather than choosing only
the outer or nested text config's EOS value.

## Host-prepared retained features

The normal raw-media pipeline runs vision once and preallocates its features
from token counts. It cannot discover content-dependent retention, run several
clips, and then shrink search's already-appended prompt. Use a host processor
before constructing the generator's prompt:

1. Run the image stream and each video clip separately through the same vision
   graph, setting the scalar dispatch flag for each call.
2. Read model-dtype features `[K,hidden]`, BOOL retention `[dense_tokens]` and
   INT64 ascending kept indices `[K]`. Validate that indices match the true mask
   positions and feature-row counts. Preserve image-first, then clip order.
3. Map the dense masks to the corresponding original image-placeholder ID 18
   positions. Remove discarded positions from both prompt IDs and attention
   mask, without removing surrounding text. Concatenate retained feature rows
   in that same order.
4. Call `Generator.SetInputs` with CPU int32 `input_ids[1,compacted_length]`
   and model-dtype `image_features[K,hidden]`. Optional `num_image_tokens` is
   an int64 vector whose sum must equal K; without it K comes from the features.
   Use the embedding graph's feature input name if it is not `image_features`.

This prepared-feature entry is specific to `nemotron_h_omni`. It copies features
to the embedding session's input device and skips vision execution, including
on decode. Features must exactly match the embedding graph's type and width.
Empty `[0,hidden]` features are supported. The configured vision session is
still constructed, so release a separate host encoder session before loading
GenAI if duplicated encoder weights would exceed the memory budget.

Initial support is **unpadded batch 1, beam 1**. An optional int32/int64
`attention_mask[1,compacted_length]` must contain only ones; GenAI creates the
corresponding managed mask and position IDs from the compacted prompt. Do not
supply dense position IDs, raw media/dispatch inputs, retention masks or kept
indices in the same prepared-feature call. Counts, placeholder numbers, feature
shapes/types, masks and unsupported beams are validated rather than silently
falling back to a different path.

The C++/C tensor API can supply explicit BF16 feature buffers. The current
Python `Tensor`/`NamedTensors` NumPy converter does not recognize BF16 and
exposes no DLPack constructor. A Python host using BF16 features therefore
needs a separately validated binding bridge or a C++/C consumer; silently
casting those features to FP16 changes the contract. The prepared-input tests
here establish the C++/C runtime path, not Python BF16 interoperability.

This does not make GenAI's raw-media processor implement EVS. Correct resizing,
normalization, video sampling, placeholder expansion and retention application
remain the host processor's responsibility. Full features plus an unapplied
mask are not a valid prepared prompt.

## Regression tests

`test/models/cache-dispatch/multimodal` contains executed arithmetic vision,
embedding and sparse-state decoder graphs. The tests cover text-only and
nonzero prepared-image prefill, `.ssm_state` mappings, complete logits and cache
values, and decode with no new media. They are cache/pipeline contracts, not
neural model parity tests.

`multimodal-retention` adds a real scalar-BOOL ONNX `If` with dynamic retained
features, masks and kept indices. Its arithmetic video branch retains rows
whose means exceed 4; that is a binding/compaction fixture, **not** EVS or
temporal-packing mathematics. Tests run an image call and two clip calls,
compact the prompt before search, then verify all logits and sparse states
through cached generation without a redundant vision call. Invalid prepared
inputs and empty-media inference are also covered.

```bash
python test/python/create/create_multimodal_cache_model.py
cmake --build <build-dir> --target kv_cache_unit_tests
<build-dir>/kv_cache_unit_tests
```

CUDA builds additionally execute the recurrent and legacy-convolution controls
with CUDA sessions, including BF16 KV/convolution state paired with FP32 SSM
state. They require a compatible CUDA-enabled ONNX Runtime and a usable GPU.
