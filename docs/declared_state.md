# Dense declared-state producer and owner envelope (K1)

This opt-in follow-up to the dense per-cache colocation work produces ORT's version-1
KV declaration before decoder session creation. It enforces the owner's cache
envelope; it does not implement placement selection, budget admission, reservations,
M2 accounting, paged caches, or ORT bind-time validation.

## Configuration

Set `model.decoder.declared_state` in `genai_config.json`, or apply it through the
existing config overlay **before creating the model**. Generator parameters are
created after the ORT session and cannot establish or enlarge its declaration.

For example, this overlay limits a model with a context length of at least 32:

```json
{
  "model": {
    "decoder": {
      "declared_state": {
        "max_batch_size": 1,
        "max_sequence_length": 32,
        "max_live_instances": 1
      }
    }
  },
  "search": {
    "max_length": 16,
    "num_beams": 1,
    "past_present_share_buffer": false
  }
}
```

`max_sequence_length` is required and must be positive and no greater than the model's
context length. Batch and resident-instance bounds default to 1; other values are
currently rejected. The effective default `search.max_length` must fit the envelope:
reducing the envelope may require changing `search.max_length` in the same overlay.
Invalid overlays leave the original config unchanged; limits are never silently clamped.

The initial scope is one flat, unquantized dense-GQA decoder on CPU or CUDA, one batch,
one beam, and one resident generator. Engine, pipelines, multimodal/companion sessions,
hybrid/recurrent state, windowed cache eviction, speculative decoding, graph capture,
and multi-profile execution are rejected.

## Declaration and allocation

Each configured K/V input and output template must contain exactly one `%d`, with no
other percent formats or braces. The producer replaces it with `{layer}`, expands the
configured contiguous layer range, and emits separate K and V groups with:

- `capacity: max_sequence_length`, `lifetime: per_generator`, and `location: follow_consumer`;
- `alias: past_present` for shared buffers, or `alias: none` for disjoint buffers.

ORT derives graph dtype, BNSH geometry, head count, head size, and GQA wiring. Cache
allocators still come from the originating session's per-input/output memory locations,
as in the lower colocation layer; the declaration does not choose a CPU/GPU cut.

After runtime profiles and session options are applied, the model freezes the limits,
alias policy, and serialized declaration. The dedicated ORT setter copies the JSON into
session options before any decoder session is created, including memory-backed models.
Session-option clones retain it.

Each generator can choose a smaller positive `max_length`, but that value, batch/beam
geometry, and alias policy cannot change after construction. Cache construction checks
the actual capacity and alias policy, including automatic fixed-shape alias detection,
before allocating cache tensors. Shared and disjoint cache updates check total length
before changing bindings or allocating larger tensors.
All cache layers must be declared; head geometry must be concrete in session metadata,
and each pair must match the allocation dtype.

A generator holds the single-resident lease until destruction, even after generation
finishes. Rewind reuses that lease. Failed construction releases it automatically,
and destruction on another thread is supported. This is a cache-owner limit, not a
whole-session memory budget or a guarantee about application-retained tensor copies.

## ORT dependency and validation

Opt-in requires a matching SDK and runtime containing the APIs from
[microsoft/onnxruntime#34113](https://github.com/microsoft/onnxruntime/pull/34113):
`GetDeclaredStateSupport` and `SessionOptionsSetDeclaredStateContract`. The negotiated
API table must be at least version 33, and the support query must positively advertise
schema version 1. Errors, including `NOT_IMPLEMENTED`, propagate; there is no fallback
that drops the declaration.

Older SDKs without the slots still compile, but reject this option. Older negotiated
runtime API tables also reject it before slot access. Experimental same-version
nightlies predating the new slots must not be mixed with K1-enabled headers: API 33 is
still under development, so use the matching K1 SDK/runtime pair.

Without `declared_state`, existing configuration, session creation, cache allocation,
and generator behavior are unchanged.

The `KvCacheUnitTests` CTest target covers serialization, invalid/atomic overlays,
legacy loading and unsupported-capability rejection. With a matching K1-enabled ORT,
it additionally runs real CPU GQA cases for shared/disjoint buffers, resident leases,
construction rollback, rewind/reuse, frozen geometry, and growth rejection before
bindings change. Only explicitly unavailable capabilities skip these integration cases;
session creation or execution errors are failures. CUDA and mixed CPU/CUDA execution
still require separate hardware validation.

On Windows, run the focused target from an existing Release build:

```powershell
ctest --test-dir build\Windows\Release -C Release -R "^KvCacheUnitTests$" --output-on-failure
```

Regenerate the checked-in tiny GQA fixture with:

```powershell
python test\python\create_dense_k1_model.py
```
