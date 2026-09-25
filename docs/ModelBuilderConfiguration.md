# Shared Model Builder and Olive Configuration

Status: experimental implementation. The GenAI version-2 entry points normalize
legacy and structured policy, reject unsupported drafter combinations, validate
runtime overlays against the exported package, and apply ordered quantization
overrides. Olive pass integration, target checkpoint conversion policy, and INT8
`GatherBlockQuantized` export are still pending.

Date: 2026-09-18

This document proposes a shared configuration contract for the ONNX Runtime
GenAI model builder and Olive's `ModelBuilder` pass. It extends the quantization
design in [PR #2588](https://github.com/microsoft/onnxruntime-genai/pull/2588).
Tensor sharing follows the adoption approach in
[PR #2579](https://github.com/microsoft/onnxruntime-genai/pull/2579), and target
gate/up fusion follows
[PR #2585](https://github.com/microsoft/onnxruntime-genai/pull/2585).
The examples describe the proposed API; they are not runnable with the current
builder unchanged. The explicit sharing policies and grouped optimization keys
below are new schema proposals, not existing flags introduced by those PRs.

## 1. Goals

- Separate the target model's export policy from the drafter's export policy.
- Support MTP, DFlash2, and DSpark through one drafter configuration interface.
- Group quantization, attention, KV-cache, and speculative graph options.
- Separate graph construction from runtime configuration and tuning.
- Use the same field names and semantics in Olive recipes and direct builder calls.
- Preserve legacy recipes and their effective defaults through explicit adapters.
- Leave room for vision components without making a text model's policy implicitly
  apply to every component of a multimodal model.

The generated runtime file is `genai_config.json`, not an INI file. Its existing
hierarchy remains unchanged. This proposal changes builder inputs, not the C++
runtime's model configuration schema.

Vision configuration, multiple simultaneous drafters, new DSpark quantization
algorithms, and arbitrary external MTP checkpoints are outside the initial scope.

## 2. Shared Configuration Envelope

These fields are siblings of `type` in an Olive `ModelBuilder` pass. The builder
accepts identically named keyword arguments, with CLI JSON options exposing the
same structure.

| Field | Owner and purpose |
| --- | --- |
| `builder_config_version` | Structured builder schema version, proposed as `2`. Not written to the generated runtime config. |
| `target_options` | Target text-decoder export policy: `quant_config`, `attention`, and `optimizations`. |
| `drafter_options` | Drafter selection, source, quantization, attention, `optimizations`, `shared_weights`, and type-specific export settings. |
| `speculative_options` | Graph requirements connecting the target and drafter. |
| `runtime_config` | Inline runtime JSON fragment or a path/resource referencing one. |
| `precision`, `search`, `extra_options` | Retained compatibility inputs. New recipes prefer the structured fields. |

The target checkpoint remains Olive's `input_model`, or the direct builder's
`model_name`/`input_path`. Do not duplicate it inside `target_options`.

Version `2` denotes the new envelope, not a GenAI release number. Version `1`
names today's flat `extra_options` surface: supplying it explicitly is accepted
and selects legacy normalization, exactly as omitting the field does for
legacy-only input. Structured fields without a version select this new schema;
examples specify the version explicitly. Version `1` combined with structured
fields is an error, as is any other explicit version. Both failures happen
before loading weights.

The execution provider is not part of this envelope and must not be duplicated
into it. It stays where each front end already carries it: Olive's
`systems.<name>.accelerators[].execution_providers`, and the direct builder's
`--execution_provider`. Normalization nonetheless *reads* it, because several
compatibility defaults are provider-dependent today:

| Resolved value | Current provider-dependent rule |
| --- | --- |
| `weights.accuracy_level` | `4` on CPU/WebGPU, else `0` |
| `moe.block_size` | `128` on TRT-RTX, else `32` |
| `moe.type` | `int2`/`mxfp4`/`nvfp4` accepted only on CUDA |
| `moe.fc1_type`, `moe.fc2_type` | Projection overrides require integer QMoE on CUDA; INT2 and mixed widths require block size 64 or 128 |
| `format.matmulnbits_weights_prepacked` | Prepacked layouts are CUDA-only; a block drafter's packing is forced off elsewhere |
| `format.use_qdq` | Required `true` for TRT-RTX integer dense weights |

So the resolver signature is (envelope, execution provider), and rule 1 in
section 9 means "identical given the same provider." Resolve the provider before
normalization and record it with the effective configuration, so a recipe moved
between accelerators reports a changed effective policy instead of silently
producing one.

Olive forwards typed dictionaries, lists, booleans, and numbers. It must not
flatten these objects into legacy strings or implement a second version of the
builder's quantization rules. GenAI owns semantic normalization and validation;
Olive owns workflow resources, caching, and pass integration.

## 3. Per-Model Export Policy

### Quantization and MoE

Both `target_options.quant_config` and `drafter_options.quant_config` reuse the
full quantization configuration from PR #2588:

| Section | Contents |
| --- | --- |
| `io_dtype` | Requested activation/I/O dtype, subject to the component's supported contract. |
| `checkpoint_policy` | Reserved for loaders that implement both paths. Target options currently reject it; MTP supports `preserve` and `requantize`. |
| `weights` | `type`, `block_size`, `symmetric`, `method`, `accuracy_level`, `op_types`, and ordered `overrides`. |
| `moe` | Expert quantization type, optional `fc1_type`/`fc2_type` projection overrides, block size, and packing. |
| `format` | `use_qdq` and `matmulnbits_weights_prepacked`. |

`format` is the proposed replacement name for `quant_config.runtime`: QDQ and
weight packing change the exported graph or its initializers. They are not
runtime-profile overrides. Keep `quant_config.runtime` as a compatibility alias;
conflicting values supplied through both names are errors.

Structured `weights.accuracy_level` must be a JSON integer from `0` to `4`,
`moe.weights_prepacked` must be `-1`, `0`, or `1`, and
`format.matmulnbits_weights_prepacked` must be `0`, `1`, or `2`. Booleans,
floating-point values, and numeric strings are rejected for these fields;
legacy adapters retain their string conversion. `weights.op_types` must be a
non-empty JSON array containing only `"MatMul"` and/or `"Gather"`, not a single
string.

Keep MoE quantization in `quant_config.moe`, separately for each model. Do not add
a duplicate `moe.quant_config` location. Future non-quantization MoE export options
can have their own group when there are concrete supported settings to expose.
`moe.fc1_type` controls the fused gate/up projection (FC1 and FC3), while
`moe.fc2_type` controls the down projection. They inherit `moe.type` when omitted.
The initial mixed-width implementation accepts `int2`, `int4`, and `int8`, uses
symmetric weights without zero points, and requires CUDA with block size 64 or 128.

An omitted `moe` group needs an explicit rule, because today's default derives
from the legacy root `precision` rather than from the dense weight type: `int8`
(or `use_8bits_moe`) gives `int8` experts, a float precision gives `none`, and
anything else gives `int4`. Under version 2 an omitted `moe.type` follows
`weights.type` instead, mapping integer dense types to the same expert type,
`none` to `none`, and leaving FP4 expert formats as an explicit opt-in. That
reproduces the legacy result whenever `precision` and `weights.type` agree, which
is the only shape a migrated recipe should have. Report the derived expert type
with the effective configuration, and require an explicit `moe` group when
`weights.type` is `none` on an MoE checkpoint, so that dropping the legacy
`precision` shorthand from an MoE recipe can never silently change how experts
are quantized.

Preserve the existing ordered, first-match override semantics. Exact-name rules
must match eligible emitted nodes, and unsupported algorithms or numeric formats
must fail rather than be ignored. A shared schema does not mean every exporter
supports every combination.

Preset and exact-name typed weight overrides currently support only `int4` and
`int8`, using the base quantizer's other settings. Float, FP4, and unsigned
override types are rejected rather than reduced to a bit count. INT8 overrides
also require QOperator format; they are not supported with QDQ.

### Attention and KV Cache

Each supported component can have an `attention` group:

| Field | Meaning |
| --- | --- |
| `implementation` | `auto` or `paged`; an explicit `paged` request requires backend/architecture support. |
| `paged.block_size` | Exported page layout; valid only with paged attention. |
| `kv_cache.scheme` | Existing scheme spelling, such as `int4_per_channel`; `none` means unquantized. |
| `kv_cache.scale_file` | Calibration-scale input resource used during export. |
| `kv_cache.windowed` | Existing windowed-cache export behavior (legacy `windowed_kv_cache`), subject to architecture support. |

`auto` retains architecture/provider selection and is the only spelling for
non-paged exports, because `paged` versus not-paged is the only attention choice
the builder exposes today (`use_paged_attention`). Version 2 deliberately does
**not** introduce `gqa` or `mha` values: there is no current option to force
either one, so accepting them would be new capability rather than a rename, and
the implementation phases in section 11 do not cover it. Adding named
implementations later is a compatible extension of this enum, gated on its own
exporter support and acceptance checks.

Explicit selections must not silently fall back. Head counts and other checkpoint
architecture metadata are not tuning options.

Pool size, scheduler limits, utilization targets, and prefill chunk size belong
to `runtime_config`. They remain constrained by the exported graph and supported
runtime behavior even though they are not graph-construction options.

### Gate/Up Projection Fusion

Expose `optimizations.fuse_mlp_gate_up` independently under `target_options` and
`drafter_options`. The default is `false` for each; enabling target fusion does
not enable drafter fusion or apply it to future vision components.

For the target this maps to `fuse_mlp_gate_up` from PR #2585. It combines eligible
MLP gate/up projections before weight quantization, producing a single projection
(`MatMulNBits` for the INT4 example) followed by `Split`. For DFlash2 the same
structured key maps to its existing `dflash2_fuse_gate_up` option. This supersedes
the earlier draft's DFlash2-only `dflash2.fuse_gate_up` placement; reserve
type-specific groups for settings that have no shared model-level meaning.

Fusion requires a supported exporter and compatible projection shapes, biases,
and quantization policy. Current target fusion requires unpacked floating-point
gate/up weights and does not support adapted projections. Reject unsupported
requests, including any per-projection exclusion or precision rule that cannot
be preserved. Resolve exact-name overrides against the final emitted graph;
do not silently drop old gate/up names after fusion or change one projection's
policy to match the other. An unsupported DSpark/MTP fusion request must fail,
not be accepted just because the schema has the field.

Fusion is distinct from selecting the CUDA fpA/intB kernel family. The legacy
`enable_cuda_fpa_intb_gemm` option maps to the runtime decoder session entry
`ep.cuda.fpa_intb_gemm`, whereas offline weight layout remains in
`quant_config.format`. Neither setting implies fusion. DFlash2's raw BF16 body
must retain its own supported session settings rather than inherit target flags.

## 4. Drafter Configuration

`drafter_options.drafter_type` is `mtp`, `dflash2`, `dspark`, or `none`. Initially
only one drafter can be selected. `none` explicitly disables automatic MTP export;
an omitted `drafter_options` retains current automatic discovery for compatibility.

Common model fields are `quant_config`, `attention`, and `optimizations`.
Additional drafter fields are:

| Field | Applicability |
| --- | --- |
| `path` | DFlash2/DSpark checkpoint resource. Omitted for MTP stored in the target checkpoint. |
| `num_draft_tokens` | Exported block proposal width for DFlash2/DSpark, not a generic MTP graph dimension. |
| `shared_weights.embedding` | Cross-model embedding sharing policy: `auto`, `required`, or `off`. |
| `shared_weights.lm_head` | Independent cross-model LM-head sharing policy: `auto`, `required`, or `off`. |
| `dspark.top_k` | DSpark candidate-graph setting; not equivalent to generation `search.top_k`. |

Reject settings for a different drafter type and unsupported separate MTP paths.
Do not guess that a missing local checkpoint directory is a Hugging Face ID.
Explicit HF source/revision support can be added separately.

### Independent Defaults

An explicit new `drafter_options` uses checkpoint and drafter-exporter defaults,
never target quantization defaults. Target node overrides, MoE policy, block
size, packing, and quantized KV settings do not propagate implicitly.

The new default checkpoint policy is `preserve`. Quantizing a dense checkpoint
is still allowed; converting an already quantized native representation requires
an explicitly supported policy. Legacy `mtp_quant_config` retains its existing
explicit-config `requantize` default and implicit inheritance behavior.

Some drafters borrow the target embedding or LM head. That is an architectural
dependency, not inheritance of all target settings. Sharing requires compatible
dtype, layout, and tensor contents, governed by the independent policies below.
Overrides for borrowed tensors must not silently alter the target's policy.

### Target/Drafter Weight Sharing

`drafter_options.shared_weights` controls each borrowed tensor separately:

| Value | Meaning |
| --- | --- |
| `auto` | Default for new structured input. Adopt the target's emitted tensor and share storage when supported; otherwise use a supported private representation and report the reason. |
| `required` | Require exact target-tensor adoption and shared external storage. Fail export if either cannot be satisfied; never silently requantize or fall back. |
| `off` | Keep private drafter storage using a supported representation. Do not deduplicate that tensor with the target, even if its bytes happen to match. |

These policies are independent of drafter body precision. A dense BF16 DFlash2
or DSpark body can still adopt the target's supported quantized head. This follows
the decoupling in the latest PR #2579 revision; do not resurrect the earlier
coupling between `dflash2_precision=bf16` and disabling head adoption.

For an adopted quantized tensor, reuse the saved target weights, scales, any zero
points, and relevant node attributes. Do not infer them from requested precision
or quantize the same source weights twice and assume equality. Quantized embedding
adoption must reproduce the target lookup operator, not emit a dense `Gather`
against the packed table. Verify external-data references/ranges and ensure both
graphs reference one saved payload. Report each tensor's effective sharing or
fallback decision independently.

This is cross-model sharing, not the existing `shared_embeddings` option, which
ties an embedding to the LM head within a model. Do not alias the two. Tied target
graphs, QDQ, unsupported algorithms, or incompatible prepacked head layouts may
prevent adoption. With `auto`, retain a supported fallback and warn about its
storage and numerical implications; with `required`, fail with the exact reason.
An INT8 embedding requires INT8 export **and** a compatible adoption path; PR
#2579's INT4 lookup adoption alone does not supply that capability.

`off` does not give a block drafter an independent checkpoint embedding or head:
those still originate from the target source. A private representation can have
different quantization and is not a claim of bit-identical scoring. Body
`quant_config` overrides cannot target an adopted tensor. For MTP, sharing also
requires checkpoint-declared/equivalent tensors; never replace a distinct learned
MTP head with the target head merely because sharing was requested. Reject a
`required` request when the architecture or adapter cannot honor it.

These are build/save policies, not runtime flags. Emit existing shared-initializer
metadata into the generated config; do not copy `shared_weights` into the runtime
JSON. Preserve legacy defaults via the compatibility adapter when no new policy
is supplied.

### Initial Capability Boundaries

| Drafter | Current exporter boundary to preserve and validate |
| --- | --- |
| MTP | Full `QuantConfig` path with model-specific loader and state constraints. |
| DFlash2 | BF16 body, target-typed boundary tensors, paged attention, unquantized drafter KV; dense or supported symmetric DEFAULT INT4/INT8 matmuls. |
| DSpark | Dense BF16 export, paged attention, unquantized drafter KV; integer quantization is separate future work. |

DFlash2's BF16 body currently uses raw, not prepacked, matmul weights. Its borrowed
LM head has separate layout rules. Reject explicit new options that cannot be
honored; preserve legacy effective behavior through the legacy adapter.

### Two Dtypes in a Block Drafter

`quant_config.io_dtype` names one dtype per component, but a block drafter has
two. DFlash2 runs its body in BF16 because the activations genuinely leave the
FP16 range, while the tensors it shares with the target -- the auxiliary hidden
states, the embedding table, and the LM head -- stay at the *target's* I/O dtype.
Only the body dtype is a component property; the boundary dtype is a consequence
of the target's, and the drafter cannot choose it independently without breaking
the sharing it depends on.

So `drafter_options.quant_config.io_dtype` describes the body only, and for
DFlash2 and DSpark today `bf16` is its single supported value. An explicit
`fp16`/`fp32` body request must be rejected with that reason rather than
silently honored or silently ignored; omitting the field selects the supported
body dtype. The examples below spell `bf16` out to document the exporter's
choice, not to imply an alternative exists. Do not add a second boundary-dtype
field: it is derived, and letting a recipe set it would only create a way to
express an invalid pair.

An MTP graph has no such split: it consumes the decoder hidden state directly,
so its `io_dtype` defaults to the resolved target I/O dtype, and an explicitly
different value is rejected until an exporter inserts the conversion.

## 5. Speculative Graph Contract

`speculative_options` contains build-time coordination settings:

- `aux_hidden_state_layers`: ordered target tap indices. For supported block
  drafters, infer them from checkpoint `target_layer_ids + 1` when omitted.
  An explicit list must match exactly and be in range.
- `state_update_capacity`: target recurrent-state capture/rollback capacity.
  This changes target graph outputs and must not be placed under the drafter
  or changed by a runtime-only overlay.

Resolve checkpoint-dependent requirements before building the target graph.
DFlash2/DSpark require a compatible paged target and shared page size; target
KV quantization does not imply drafter KV quantization.

Keep these quantities separate:

| Quantity | Owner |
| --- | --- |
| Exported proposal width | `drafter_options.num_draft_tokens` |
| Target capture capacity | `speculative_options.state_update_capacity` |
| Runtime proposal-use limit | `runtime_config.speculative.max_draft_tokens` |

DFlash2 adds an anchor row to the exported proposal width; DSpark uses its own
checkpoint block convention. A wider bidirectional drafter can change earlier
predictions, so the runtime limit must not silently resize the graph.

For new structured input, validate the requested usable width against drafter,
state, query, and runtime limits before export. Derive capacity defaults only
when the architecture requires capture, using the actual state contract rather
than a universal off-by-one formula. Legacy runtime clamping remains unchanged.

## 6. Runtime JSON Profile

`runtime_config` is an input fragment in the existing runtime hierarchy. The
builder produces the complete `genai_config.json`; users need not duplicate
generated tensor bindings or graph metadata in their profile.

### Allowed and Protected Settings

Allow supported settings under:

- `search`, including generation parameters and `chunk_size`.
- `speculative`, including `max_draft_tokens` and supported adaptation policy.
- `engine.dynamic_batching`, including batch/token limits and cache allocation.
- `model.<existing-component>.session_options` and `run_options`, including
  supported provider, allocator, threading, and profiling options.

MTP accepts session and run options even when its generated component has no
`session_options` object. MTP, DFlash2, and DSpark provider overlays use decoder
providers as their starting point when the component has no provider list.

Protect generated filenames, bindings, dtypes, geometry, state manifests,
capture capacity, auxiliary taps, shared-initializer descriptors, and drafter
selection. In particular, `engine.dynamic_batching.block_size` is derived from
the graph, not an independent runtime override.

Session/provider settings are not an unrestricted escape hatch: profiles cannot
disable requirements of a prepacked graph or switch to an incompatible provider.
Reject absent components and unknown structural fields. Preserve intentional
string-valued ORT session/run/provider extension entries.

Session-option values must match the types the runtime config parser reads:
numbers for the thread/log-level fields, booleans for the arena/memory-pattern
fields, a supported `ORT_*` name for `graph_optimization_level`, and strings for
everything else, including extension entries.

### Merge Rules

1. Normalize legacy runtime options and Olive `search` into a low-priority fragment.
2. Overlay explicit `runtime_config`: merge objects recursively, replace scalars,
   and replace arrays atomically. Reject duplicate JSON keys and `null` initially.
3. A supplied `provider_options` array is complete, not a partial array patch.
  Provider entries merge by case-insensitive provider name, but graph-derived
  options such as WebGPU/TRT-RTX rotary-cache offsets cannot be changed. Only
  known runtime-tunable provider options are accepted.
4. Explicit `num_blocks` selects fixed allocation and removes a generated
   `gpu_utilization_factor`; explicit utilization removes generated `num_blocks`.
   Supplying both explicitly is an error.
5. Validate against the complete exported model, then finalize the runtime config
   after all target/drafter components have contributed their sections.

These are the proposed builder-fragment semantics. They are not a claim that
every array in the existing C++ overlay API behaves this way. That API remains
unchanged. Runtime tuning applies before model/session construction, not as
arbitrary live mutation of existing sessions.

The same object/scalar/array semantics -- rules 2 and 3 -- also describe how a
partial pass fragment in this document composes with a full pass: objects merge
recursively, scalars replace, arrays replace whole. That is a documentation
convention for presenting a variant without repeating a long recipe, not a
second configuration feature. A pass is always supplied whole; nothing in the
implementation merges two pass objects.

Olive must not perform a second independent search merge on the new path.
Updating only a runtime profile may reuse a validated metadata-only path for an
existing artifact; it must never invent or modify graph capabilities. Automatic
separate build/runtime caching is not required in the initial implementation.

## 7. Complete Olive Example

This is the proposed migration of the user-supplied Qwen3.8-27B DFlash2 Olive
recipe, reproduced below without requiring a separate recipe checkout.
It retains the target source, CUDA system, output/cache locations, and requested
target INT4 plus INT8 embedding policy. It additionally opts into target gate/up
fusion and makes embedding/head sharing policy explicit. Fusion is an intentional
graph change, so this extended example is not a claim of byte-identical migration.

**Prerequisite:** INT8 embedding export must be implemented and validated before
this recipe can be migrated. PR #2588 rejects a Gather type override, and the
current DEFAULT quantizer's Gather path is INT4-only. An eight-bit runtime kernel
alone does not make the exporter support the request. Do not substitute a dense
or INT4 embedding and call that a behavior-preserving migration.

The JSON blocks are example file contents embedded in this proposal, not separate
files installed by this documentation change. Place the runtime profile beside
the recipe and run the workflow from that directory for the relative-path
convention shown here. The drafter directory and scale resource must also exist
there, or be supplied as resolved Olive resources.

### Recipe

```json
{
  "input_model": {
    "type": "HfModel",
    "model_path": "Qwen/Qwen3.8-27B",
    "load_kwargs": {
      "torch_dtype": "float16"
    }
  },
  "systems": {
    "local_system": {
      "type": "LocalSystem",
      "accelerators": [
        {
          "device": "gpu",
          "execution_providers": ["CUDAExecutionProvider"]
        }
      ]
    }
  },
  "passes": {
    "builder_int4_int8_embed_int4_per_channel_kv_paged_dflash2_int4_24gb": {
      "type": "ModelBuilder",
      "builder_config_version": 2,
      "precision": "int4",
      "target_options": {
        "quant_config": {
          "io_dtype": "fp16",
          "weights": {
            "type": "int4",
            "block_size": 32,
            "method": "default",
            "symmetric": true,
            "op_types": ["MatMul", "Gather"],
            "overrides": [
              {
                "match": {"name": "/model/embed_tokens/Gather"},
                "type": "int8"
              }
            ]
          },
          "format": {
            "use_qdq": false,
            "matmulnbits_weights_prepacked": 1
          }
        },
        "attention": {
          "implementation": "paged",
          "paged": {"block_size": 256},
          "kv_cache": {
            "scheme": "int4_per_channel",
            "scale_file": "kv_scales_int8_per_channel.json"
          }
        },
        "optimizations": {
          "fuse_mlp_gate_up": true
        }
      },
      "drafter_options": {
        "drafter_type": "dflash2",
        "path": "Qwen3.8-27B-DFlash2",
        "num_draft_tokens": 7,
        "shared_weights": {
          "embedding": "auto",
          "lm_head": "auto"
        },
        "optimizations": {
          "fuse_mlp_gate_up": false
        },
        "quant_config": {
          "io_dtype": "bf16",
          "weights": {
            "type": "int4",
            "block_size": 32,
            "method": "default",
            "symmetric": true,
            "op_types": ["MatMul"]
          },
          "format": {
            "use_qdq": false,
            "matmulnbits_weights_prepacked": 0
          }
        }
      },
      "speculative_options": {
        "aux_hidden_state_layers": [6, 20, 34, 48, 62],
        "state_update_capacity": 7
      },
      "runtime_config": "runtime_cuda_24gb.json"
    }
  },
  "target": "local_system",
  "log_severity_level": 0,
  "output_dir": "model_int4_int8_embed_int4_per_channel_kv_paged_dflash2_int4_24gb",
  "cache_dir": "cache",
  "no_artifacts": false
}
```

The root `precision` is retained to illustrate an agreeing legacy shorthand; the
new API can omit it when `target_options.quant_config.weights.type` is explicit.
Qwen3.8-27B is an MoE model and this pass carries no `moe` group, so dropping
`precision` relies on the version-2 rule above that an omitted `moe.type`
follows `weights.type`: `int4` dense weights keep `int4` experts either way.
Check that rule before deleting a root `precision` from any MoE recipe, since
the legacy default derives from `precision` alone.
The scale filename retains the original `int8` label intentionally: validate its
contents for the selected INT4 KV scheme rather than inferring format from its name.

Drafter block size is explicit to reproduce the former target-derived value
without new implicit inheritance. Drafter packing `0` describes the BF16 body;
the target's borrowed head follows its separately validated sharing/layout policy.
Effective shared tensor behavior must be checked during migration, not assumed
from these numeric settings alone.

The two `auto` sharing settings request adoption when compatible, not guaranteed
sharing. In particular, the INT8 embedding adoption path is an additional
capability gate; this example must not be advertised as a one-copy package merely
because it exports successfully. Use `required` below to enforce a memory budget.
Target fusion is enabled; DFlash2 fusion is explicitly disabled independently.
Set the drafter's `optimizations.fuse_mlp_gate_up` to `true` to opt into its own
supported gate/up fusion as well, with separate numerical validation.

### Runtime Profile

Proposed contents of `runtime_cuda_24gb.json`:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "session.use_device_allocator_for_initializers": "1",
        "provider_options": [
          {"CUDA": {"enable_cuda_graph": "1"}}
        ]
      }
    }
  },
  "engine": {
    "dynamic_batching": {
      "max_batch_size": 1,
      "max_scheduled_tokens": 512,
      "num_blocks": 448
    }
  },
  "search": {
    "chunk_size": 512
  },
  "speculative": {
    "max_draft_tokens": 7
  }
}
```

ORT provider options and the allocator config entry above use string values,
not booleans. Other required generated session entries survive recursive object
merging. Provider arrays must repeat every generated provider, and entries merge
by provider name; graph-derived provider values remain unchanged.

The builder supplies the page size, bindings, state groups, and DFlash2 geometry.
The profile does not duplicate them. For an inline profile, replace the recipe's
`runtime_config` filename with this JSON object.

### Require Shared Embedding and Head

For a package that must store one target/drafter copy of each tensor, change both
policies to `required`. The fragment below is a variant of the pass above, shown
without repeating it. Compose it using the section 6 convention: objects merge
recursively, scalars replace, arrays replace whole. So its
`target_options.quant_config.format` changes only
`matmulnbits_weights_prepacked`, and the base pass's `use_qdq: false` survives;
its `runtime_config` object first replaces the recipe's profile filename with
that file's loaded contents, then merges the session entries shown here on top,
leaving the base profile's `provider_options` array and `engine`, `search`, and
`speculative` sections intact. The result is one complete pass.

```json
{
  "target_options": {
    "quant_config": {
      "format": {"matmulnbits_weights_prepacked": 0}
    }
  },
  "drafter_options": {
    "shared_weights": {
      "embedding": "required",
      "lm_head": "required"
    }
  },
  "runtime_config": {
    "model": {
      "decoder": {
        "session_options": {"ep.cuda.fpa_intb_gemm": "1"}
      },
      "dflash2": {
        "session_options": {"ep.cuda.fpa_intb_gemm": "0"}
      }
    }
  }
}
```

Raw target layout plus target-only fpA/intB session selection follows PR #2585
and avoids one offline-prepacked-layout obstacle to sharing. It does not by
itself guarantee adoption: the actual head format, boundary dtype, target tying,
and INT8 embedding adoption must still pass validation. Keep the target's INT8
embedding requirement; fail rather than downgrade it to satisfy sharing. Some
compatible prepacked exports can share already and need not switch to raw layout.
The runtime flags above select kernels independently of the still-enabled target
fusion; DFlash2's session does not inherit the target's fpA/intB setting.

### Other Drafter Selections

MTP using the target checkpoint's native head and independent INT4 policy:

```json
{
  "drafter_options": {
    "drafter_type": "mtp",
    "quant_config": {
      "checkpoint_policy": "preserve",
      "weights": {"type": "int4", "block_size": 32}
    }
  }
}
```

This requires a compatible source checkpoint. A conflicting native quantization
format is rejected under `preserve`; use an explicitly supported `requantize`
conversion when that is intended. Legacy explicit MTP settings may need that
policy written explicitly when migrated.

DSpark with its currently supported dense body:

```json
{
  "drafter_options": {
    "drafter_type": "dspark",
    "path": "Qwen3.8-27B-DSpark",
    "num_draft_tokens": 7,
    "shared_weights": {
      "embedding": "auto",
      "lm_head": "auto"
    },
    "quant_config": {
      "io_dtype": "bf16",
      "weights": {"type": "none"}
    },
    "dspark": {"top_k": 16}
  }
}
```

These are replacement fragments, not additional simultaneous drafters. Checkpoint
limits, target compatibility, and supported exporter capabilities still apply.

## 8. Future Vision Configuration

The target is potentially a multimodal model, not just its text decoder. Reserve
`target_options.vision` as a future component group. Do not introduce an unrelated
global `vision_options` that loses target ownership or accidentally applies to
the drafter.

Existing `target_options.quant_config` and `target_options.attention` retain
text-decoder scope. They must not implicitly quantize the vision encoder,
projector, or image preprocessing graph. Component-specific settings should use
the same quantization and attention types where the component supports them.

An illustrative future extension is:

```json
{
  "target_options": {
    "quant_config": {
      "weights": {"type": "int4", "block_size": 32}
    },
    "vision": {
      "encoder": {
        "quant_config": {
          "io_dtype": "fp16",
          "weights": {"type": "none"}
        },
        "attention": {"implementation": "auto"}
      },
      "projector": {
        "quant_config": {
          "io_dtype": "fp16",
          "weights": {"type": "none"}
        }
      }
    }
  }
}
```

This is not part of initial version-2 support. Until a version/capability advertises
vision options, supplying this group must produce an unsupported-feature error,
not be ignored. The exact encoder/projector selectors require validation against
supported multimodal architectures before implementation.

Future vision work should preserve these boundaries:

- Keep omitted vision policy compatible with the existing multimodal exporter;
  a new text policy must not silently change vision precision.
- Scope override node names and calibration resources to the selected component.
- Derive normalization, patch size, token layout, and default preprocessing from
  checkpoint/processor metadata. Any supported preprocessing overrides need their
  own validation, not placement in a generic runtime overlay.
- Treat image/crop limits, dynamic shapes, and encoder-to-projector-to-decoder
  dtypes as graph/interface contracts. A setting serialized to JSON is not
  automatically safe to tune after export.
- Route vision session options to the existing emitted runtime component, such
  as `model.vision.session_options` where supported. A logical projector need
  not be a separate ONNX session; do not invent a runtime section for it.
- Keep drafter inputs based on the verified target hidden-state contract.
  Do not presume a text drafter independently consumes image tensors or supports
  every multimodal target.

The same component ownership can later accommodate audio without changing the
target/drafter/runtime envelope.

## 9. Legacy Migration and Precedence

| Legacy input | Canonical destination |
| --- | --- |
| `precision` | Target weight-type fallback only |
| `quant_config` | `target_options.quant_config` |
| List-form quantization override | `target_options.quant_config.weights.overrides` |
| `block_size`, `op_types_to_quantize` | Target `quant_config.weights` fields |
| `is_symmetric`, `accuracy_level` | Target `quant_config.weights.symmetric` and `weights.accuracy_level` |
| `algo_config`, `nodes_to_exclude` | Target `quant_config.weights.method` plus generated `weights.overrides` entries; exclusions precede generated preset rules so they stay unconditional |
| `matmulnbits_weights_prepacked`, `use_qdq` | Target `quant_config.format` fields |
| `moe_quant_type`, `qmoe_fc1_type`, `qmoe_fc2_type`, `qmoe_block_size`, `qmoe_weights_prepacked` | Target `quant_config.moe` fields |
| `use_8bits_moe` | Deprecated `moe_quant_type` alias; unchanged |
| `use_paged_attention`, `paged_block_size` | Target `attention.implementation` and `attention.paged.block_size` |
| `kv_cache_quant_scheme`, `kv_cache_scale_file` | Target `attention.kv_cache` fields |
| `windowed_kv_cache` | Target `attention.kv_cache.windowed` |
| `mtp_quant_config` | MTP quantization, retaining legacy defaults through the adapter |
| `dflash2_path`, `dspark_path` | Drafter selection and `path` |
| `dflash2_precision` | Drafter weight policy plus explicit legacy-derived settings |
| `dflash2_num_draft_tokens`, `dspark_num_draft_tokens` | `drafter_options.num_draft_tokens` |
| `fuse_mlp_gate_up` | `target_options.optimizations.fuse_mlp_gate_up` |
| `dflash2_fuse_gate_up` | `drafter_options.optimizations.fuse_mlp_gate_up` |
| `dspark_top_k` | `drafter_options.dspark.top_k` |
| Existing automatic target/drafter tensor adoption | New `drafter_options.shared_weights` policies; preserve existing decisions for legacy-only calls |
| `shared_embeddings` | Existing within-model tying; not an alias for cross-model `shared_weights` |
| `aux_hidden_state_layers`, `state_update_capacity` | `speculative_options` fields |
| `max_draft_tokens` | Runtime `speculative.max_draft_tokens` |
| `max_batch_size`, `max_scheduled_tokens`, `num_blocks`, `gpu_utilization_factor` | Runtime `engine.dynamic_batching` fields |
| `paged_chunk_size` | Runtime `search.chunk_size` |
| `enable_cuda_graph`, `use_device_allocator_for_initializers` | Runtime decoder session/provider settings |
| `enable_cuda_fpa_intb_gemm` | Runtime decoder session entry `ep.cuda.fpa_intb_gemm` |
| Olive `search` | Runtime `search` |
| Any other `extra_options` key | No canonical destination yet; see rule 8 |

Normalization rules:

1. Legacy-only calls preserve current precedence, defaults, and behavior, for a
   given execution provider. Several defaults are provider-dependent (see
   section 2), so "unchanged" is only meaningful against a fixed provider. In
   Olive, legacy `extra_options` continues to override legacy pass-level knobs.
2. Structured leaves override legacy aliases with a warning naming both paths.
   Omitted target leaves retain compatibility defaults. An explicit new drafter
   never receives implicit target quantization defaults.
3. Conflicting canonical declarations/aliases fail, as do mutually exclusive
   drafter selections and subtype options for the wrong drafter.
4. Retain full-object and older list-form quantization inputs through adapters.
   Syntax compatibility does not bypass exporter capability checks.
5. Preserve omission versus explicit values during normalization, including
   checkpoint policy. Do not merge defaults too early and lose their provenance.
6. Validate effective target precision after resolution, rather than rejecting a
   structured configuration based on an unrelated Olive default `precision`.
7. New Olive with old GenAI can still execute legacy recipes. Structured recipes
   require a supported schema/capability and fail early with upgrade guidance;
   never drop unknown groups or guess a lossy flattening.
8. The table above is not exhaustive, and the remaining `extra_options` keys --
   `exclude_embeds`, `exclude_lm_head`, `prune_lm_head`, `state_window`,
   `hf_token`, and the rest -- keep working unchanged under version 2. They pass
   through to the same handling they have today, alongside the structured
   groups, until a later version gives each one a canonical home. This is
   deliberate: it keeps the migration incremental instead of requiring the whole
   flat surface to be redesigned first. Two consequences. A structured group and
   a passthrough key that govern the same graph property still conflict, and
   must be detected by rule 3 rather than silently resolved by ordering. And a
   key that the installed builder does not recognize at all must fail, not be
   dropped, so that a recipe written for a newer builder cannot quietly export a
   different model.

## 10. Olive Resources and Caching

Add first-class typed fields to Olive's pass configuration. GenAI provides a
versioned schema/capability description and semantic resolver; Olive's recipe
schema can reference that contract for editor validation.

Reuse Olive's recursive resource discovery and rewriting for nested drafter
directories and calibration files. Accept existing `ResourcePath` representations
on the Olive side, materializing local paths before calling GenAI. Resource
transport metadata is not part of the backend-independent export policy.

Preserve existing path-base conventions for legacy recipes. Document the working
directory for relative examples, and resolve resources before remote execution.
Do not hide resources in JSON-encoded strings or infer an HF download from a
missing path.

Load an external runtime JSON into the normalized pass configuration before
computing its cache identity. Changing the profile contents at the same pathname
must invalidate the corresponding result. Keep the original resource identity
for reproducibility and remote materialization. A profile change need not imply
re-export in a future split cache, but the first implementation must not return
stale configuration merely because the filename is unchanged.

Preserve multiple ONNX output files, shared initializers, processor metadata,
and any later vision artifacts when packaging the result.

## 11. Implementation and Validation

### Rollout

1. Define the shared envelope, aliases, provenance, and capability validation.
   Reuse the existing quantization resolver and per-model attribute dictionaries.
2. Adapt MTP/DFlash2/DSpark to isolated policies and resolve cross-model graph
   requirements before export. Reject capabilities not implemented by each adapter.
  Reuse PR #2579's tensor-adoption mechanism with per-tensor sharing policies,
  and PR #2585's target fusion path with independent per-model optimization flags.
3. Add runtime fragment loading, validation, and finalization after composite
   model configuration is complete. Keep the existing runtime hierarchy.
4. Integrate Olive fields, resources, cache identity, and typed forwarding.
   Avoid a second independent normalization or post-export search merge.
5. Separately implement and validate the INT8 Gather export prerequisite at the
   quantizer, then enable the builder override behind capability checks.
6. Migrate the example recipe and representative MTP/DSpark recipes only after
   their capability and behavior checks pass.
7. Add vision support later without changing the envelope or text-policy scope.

No implementation, release version requirement, or performance improvement is
claimed by this document. The INT8 embedding prerequisite is a numeric export
feature, not merely a rename of configuration keys.

### Acceptance Checks

- Parser tests for legacy equivalence, ordered overrides, explicit/omitted
  defaults, aliases, explicit version `1`, unsupported versions, version `1`
  combined with structured fields, unrecognized passthrough keys, and
  independent drafter policy.
- Provider-dependent default tests: resolve the same envelope against CPU,
  WebGPU, CUDA, and TRT-RTX and check `weights.accuracy_level`, `moe.block_size`,
  FP4 expert acceptance, prepacked layout, and `use_qdq` each match today's
  values for that provider, and that the effective configuration records which
  provider produced them.
- MoE derivation tests: an omitted `moe` group follows `weights.type`, matches
  the legacy `precision`-derived expert type for every agreeing pair, and fails
  rather than defaulting when `weights.type` is `none` on an MoE checkpoint.
- Drafter body-dtype tests: an omitted `io_dtype` selects BF16, an explicit
  `bf16` is accepted, an explicit `fp16`/`fp32` is rejected with the body-range
  reason, and boundary tensors stay at the target's dtype in every case. MTP
  instead follows the resolved target I/O dtype and rejects a different one.
- Graph tests for exact tap order, page-size agreement, draft/state limits,
  per-model KV policy, borrowed tensors, and unsupported drafter settings.
- Sharing tests for each tensor and policy, including dense-body/quantized-head
  independence, actual node attributes, exact tensor payloads, external-data
  range/identity checks, `required` failures, `auto` diagnostics, and `off` keeping
  private storage. Cover INT8 embeddings, target tying, QDQ, native FP8 heads,
  prepacked incompatibility, and distinct learned MTP heads without assuming
  every combination supports adoption.
- Fusion tests for independent target/drafter flags, legacy mappings, one fused
  projection plus split, invalid shapes/adapters, and exclusion/override conflicts.
  Fusion can change quantization or kernel accumulation; compare numerical outputs
  and validate quality/acceptance rather than assuming bit-identical generation.
- Runtime tests for object/array merge semantics, protected fields, absent
  components, provider requirements, fixed/utilization allocation, and applying
  the profile only after all component sections exist. Check target-only fpA/intB
  selection does not overwrite the drafter's BF16 session requirements.
- Olive tests for typed forwarding, effective precision, legacy compatibility,
  nested local/remote resources, profile-content cache invalidation, and no
  duplicate runtime merge.
- Real INT8 embedding graph validation and CPU/CUDA numerical checks, including
  an INT4 body and incompatible tied head. Older dependencies must fail rather
  than silently skip the requested quantization.
- Small paired exports for supported legacy/new target-only, MTP, DFlash2, and
  DSpark configurations. Compare graph attributes, tensor contents, sharing,
  and generated runtime settings, not only schema parsing.
- End-to-end CUDA export/load and speculative prefill/decode/rollback using the
  supplied recipe once dependencies are available. Since the current PR rejects
  its INT8 Gather request, use an independent reference or a verified older
  exporter rather than assuming a runnable current legacy baseline.
- Future vision gates: encoder/projector precision isolation, processor metadata,
  image shape/token alignment, mixed text/image inference, and runtime session
  isolation for architectures that actually support them.

### Existing Integration Points

- [builder.py](../src/python/py/models/builder.py): input normalization, CLI,
  orchestration, and final runtime configuration.
- [quant_config.py](../src/python/py/models/quantization/quant_config.py): shared
  quantization parsing, validation, aliases, and compatibility defaults.
- [base.py](../src/python/py/models/builders/base.py),
  [mtp.py](../src/python/py/models/builders/mtp.py), and
  [qwen.py](../src/python/py/models/builders/qwen.py): graph and drafter contracts.
- [Olive ModelBuilder](https://github.com/microsoft/Olive/blob/main/olive/passes/onnx/model_builder.py) and
  [resource handling](https://github.com/microsoft/Olive/blob/main/olive/resource_path.py): pass fields and staging.
- [ORT quantizer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py):
  the INT8 Gather export prerequisite.
- [config.cpp](../src/config.cpp): existing runtime parsing/overlay semantics,
  retained without a new runtime hierarchy.
