# Declared-state producers and owner envelopes (K1)

This opt-in follow-up to per-cache colocation produces ORT's version-1 dense or
version-2 paged KV declaration before decoder session creation. It enforces the
owner's cache envelope; it does not implement placement selection, whole-device
budget admission/reservations, M2 accounting, or ORT bind-time validation.

## Dense configuration

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
context length. Dense batch and resident-instance bounds default to 1; other values
are rejected. The effective default `search.max_length` must fit the envelope:
reducing the envelope may require changing `search.max_length` in the same overlay.
Invalid overlays leave the original config unchanged; limits are never silently clamped.

The initial scope is one flat, unquantized dense-GQA decoder on CPU or CUDA, one batch,
one beam, and one resident generator. Engine, pipelines, multimodal/companion sessions,
hybrid/recurrent state, windowed cache eviction, speculative decoding, graph capture,
and multi-profile execution are rejected.

## Dense declaration and allocation

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

## Paged Engine configuration and ownership

With `engine.dynamic_batching`, the same opt-in produces schema version 2. For a
flat decoder with context length at least 32, configure both limits and fixed geometry:

```json
{
  "model": {
    "decoder": {
      "declared_state": {
        "max_batch_size": 2,
        "max_sequence_length": 32,
        "max_live_instances": 1
      }
    }
  },
  "search": {"max_length": 32, "num_beams": 1},
  "engine": {
    "dynamic_batching": {
      "block_size": 16,
      "num_blocks": 4,
      "max_batch_size": 2,
      "prefix_caching": false
    }
  }
}
```

The declared batch bound must equal the Engine batch limit. One session-wide pool
set is permitted (`max_live_instances=1`); `num_blocks` must cover
`max_batch_size * ceil(max_sequence_length / block_size)`. The declaration charges
each entire pool, including free blocks, once rather than multiplying that backing
by the number of Requests again. Geometry and flat physical slot indices must fit
int32. There is no prefix-sharing credit.

Each K/V group uses `alias: past_present`, `lifetime: per_session`, and a `paging`
object containing `block_size`, `num_blocks`, and `block_table`. The configured table
name can be literal or contain one `%d`, expanded to `{layer}` in the declaration.
Per-layer tables mirror the same logical block IDs on each consuming layer's device.
Table rows and columns are bounded by the declared batch and rounded sequence limits.
Pool dtype, head geometry, fixed dimensions, layer coverage, and session locations
are checked before allocating backing; past and present bind the same pool tensors.

A pool lease follows the actual backing, including a cache manager retained after
its Engine is destroyed. A separate Engine lease prevents sharing that pool between
two Engines. Failed construction releases leases automatically. Only the model's
production paged cache manager can be injected into a declared Engine.

Every unclosed Request counts toward the batch bound, including idle and pending
Requests. Excess Requests and explicit `max_session_tokens` above the declared
sequence bound fail before Request allocation. Existing append and turn checks
enforce each Request's immutable, possibly smaller limit. Reservations and table
growth check the envelope before changing ownership or bindings. Session geometry,
cache/table names, and workload limits cannot be mutated after model creation.
Paged declarations cannot execute through a dense Generator.

The initial scope is unquantized FP16/BF16 `SEPARATE` PagedAttention, with explicit
pool sizing and **explicit `prefix_caching: false`**. Automatic/device-budget sizing,
prefix retention/snapshots, windowed rings, graph capture, hybrid state, auxiliary
models, speculative decoding (including external drafts), pipelines, and quantized
caches are rejected. This is placement-aware ownership, not offload: CPU storage
does not make it usable by CUDA PagedAttention. There is currently no native CPU
PagedAttention kernel; the paged K1 integration cases require CUDA.

## ORT dependency and validation

Opt-in requires a matching SDK and runtime containing the APIs from
[microsoft/onnxruntime#34113](https://github.com/microsoft/onnxruntime/pull/34113):
`GetDeclaredStateSupport` and `SessionOptionsSetDeclaredStateContract`. The negotiated
API table must be at least version 33, and the support query must positively advertise
schema version 1 for dense or version 2 for paged. Errors, including
`NOT_IMPLEMENTED`, propagate; there is no fallback that drops the declaration.

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

The same target covers paged serialization, rounded pool coverage, int32 boundaries,
atomic overlays, and mandatory schema-v2 negotiation without requiring CUDA. Native
paged cases require both a matching K1-v2 SDK/runtime and a CUDA build; they cover
lease rollback/lifetime, Request and pool/table bounds, frozen configuration, and
prefill/decode parity across a block boundary. Stock SDKs skip these positive native
cases rather than exercising an undeclared fallback.

On Windows, run the focused target from an existing Release build:

```powershell
ctest --test-dir build\Windows\Release -C Release -R "^KvCacheUnitTests$" --output-on-failure
```

Regenerate the checked-in tiny GQA fixture with:

```powershell
python test\python\create_dense_k1_model.py
```

Regenerate the native FP16 PagedAttention fixture (no downloaded weights) with:

```powershell
python test\python\create_paged_k1_model.py
```
