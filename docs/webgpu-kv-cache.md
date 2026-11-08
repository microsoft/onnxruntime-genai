# WebGPU KV cache growth and graph capture

Native WebGPU models with `past_present_share_buffer: true` and symbolic KV
sequence dimensions can use shared
past/present buffers without reserving the full `search.max_length` at generator
creation. Supported FP16/FP32 caches allocate on the first prompt, reserve at
least 512 tokens (clamped to `max_length`), and grow geometrically as needed.
Longer prompts also reserve room for the first decode token. Rewinding retains
the allocated capacity for subsequent use.

Symbolic/fixed KV shapes describe the ONNX sequence dimension.
`past_present_share_buffer` controls whether past and present alias the same
tensor. Symbolic shapes can use either separate tensors or shared tensors with
fixed or growable capacity. Fixed positive KV shapes are detected automatically
and use shared tensors sized to the model's declared capacity.

The share flag remains explicit: `false` keeps the existing separate-buffer
dynamic path. Other backends keep their existing sharing and capacity policies.
For future WebGPU model packages, prefer `true` when the exported attention
graph supports aliased past/present buffers and uses a separate logical sequence
length, as the tested `com.microsoft::GroupQueryAttention` exports do. This PR
does not change model-builder defaults or automatically inspect graphs to enable
sharing. Verify each export before selecting the flag.

Graph capture is opt-in through the existing WebGPU provider option. Keep the
model package's other provider options, including any rotary-cache options, and
set these fields in its configuration:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "provider_options": [{ "webgpu": { "enableGraphCapture": "1" } }]
      }
    }
  },
  "search": {
    "past_present_share_buffer": true,
    "num_beams": 1
  }
}
```

This fragment shows the relevant fields; it is not a complete model
configuration. Shared exports must support aliased past/present buffers and
consume the current logical sequence length during replay. Symbolic ONNX
dimensions alone do not establish those semantics. In particular, deriving
logical length from `Shape(past)` reads capacity, and concatenating the entire
past tensor includes spare rows; setting the flag does not fix either graph.

Within one capacity, decode reuses stable buffer addresses. At a growth boundary,
GenAI releases all captured lengths and variants owned by that generator,
migrates KV tensors on device with the correct batch/head strides, and binds the
larger buffers. Later decode captures again. This requires ORT API 27 or newer
for captured-graph release and a runtime with the model-editor API for the
migration graph. Other generators' captures remain independently owned.

Graphs that retain unshared dynamic KV allocate new outputs each step and do not
support graph capture. Growable shared KV currently
excludes beam search, decoder pipelines, configured sliding windows and
quantized KV. Fixed-declared cache shapes retain their declared capacity.

Evaluate both memory and latency for the intended workload. With the share flag
already true, both the previous `max_length` allocation and the growable policy
reuse past/present storage. Growth reduces reserved KV capacity and adds occasional
allocation/migration work; it does not remove per-step present allocation from
that already-shared baseline. Unshared dynamic KV keeps separate tensors.
Capture can reduce decode
CPU overhead; crossing a
capacity boundary adds migration and recapture latency. Bound KV tensor bytes
exclude retained allocator buffers, migration overlap, model weights and capture
resources, so they do not measure total GPU memory or its peak. Check logits and
tokens against capture-disabled execution through growth boundaries before
enabling capture for a model package.

The provider may retain retired graph buffers for reuse by later captures.
Releasing a captured graph therefore need not immediately reduce process GPU
memory, and lower bound KV capacity does not guarantee a lower full-process peak.
