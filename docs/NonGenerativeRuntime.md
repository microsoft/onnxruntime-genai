# Non-generative component runtime

`RankingSession` (CLM) and `DecisionSession` (KEV) are semantic Python facades
over one native non-generative session. They execute exported directory
packages without constructing a `Generator`. Each runtime object owns an ORT
session for each named component and bounded, session-local non-generative
caches.

```python
import json
import onnxruntime_genai as og

ranking = og.RankingSession("/models/clm-v0.1-8b-fp32", providers=["cuda"])
ranked = ranking.rank(json.load(open("free-form-rank-request.json")))

decision = og.DecisionSession("/models/kev-4b-fp32")
answers = decision.decide(json.load(open("request.json")))
```

`NonGenerativeSession` is the single public C++ package entry point. It
discovers the workflow from the package and accepts/returns UTF-8 JSON:

```cpp
auto ranking_config = OgaConfig::Create("/models/clm-v0.1-8b-fp32");
ranking_config->ClearProviders();
ranking_config->AppendProvider("cuda");
NonGenerativeSession ranking{*ranking_config};
auto ranked = ranking.Run(
    R"({"state":{"weather":"rain"},"instructions":"Choose","candidates":{"picnic":"Outside","museum":"Inside"}})");

auto decision_config = OgaConfig::Create("/models/kev-4b-fp32");
NonGenerativeSession decision{*decision_config};
auto decisions = decision.Run(
    R"({"state":{"weather":"rain"},"questions":{"umbrella":{"type":"noul","instructions":"Take an umbrella?"}}})");
```

Session creation reuses `OgaConfig`, including provider selection, provider
options, hardware-device filters, and overlays. The session copies the
effective configuration during construction, so the caller may destroy or
reuse its `OgaConfig` afterward. Packages must contain `genai_config.json`.

Internally these workflows reuse the existing `Config`, `Tokenizer`, `Tensor`,
and `NamedTensors` infrastructure. Named component execution and package pad
token resolution are private implementation details rather than additional
public handle families.

The C ABI exposes one opaque session plus create, destroy, and `RunJson`.
Returned JSON strings use the existing
`OgaDestroyString` ownership convention.

## Managed bindings

This change establishes the C ABI used by managed bindings. C#, Java,
and JavaScript wrappers are intentionally delivered by the dependent managed
bindings change rather than this native-runtime change.

CLM caches projected action embeddings after the encoder and action head.
Split-head packages also cache projected state embeddings, allowing an
identical repeated request to skip the encoder and both projection heads while
still executing the scorer with the request temperature. Action and state keys
include the canonical package identity, head layout, `float32` projection
dtype, and length-prefixed rendered text. Combined-head compatibility packages
continue to cache actions only because their state and action projections are
produced by one component invocation. Package identity is session-local and
includes the canonical package path and provider list; in-place model reload is not supported. State
and action projections share the same bounded LRU. The
default is 256 entries and 64 MiB. KEV caches escaped/rendered tokenized state
prefixes and question branches
(including branch-local option readout indices), not probabilities. Its token
cache default is 512 entries and 16 MiB. Compatible stateful backbones also use
a separate 32-entry, 512 MiB byte-bounded prefix-state LRU. Both caches use thread-safe LRU
eviction and are isolated per session and provider configuration. Cache policy
and diagnostics are internal implementation details rather than public API.

KEV prefix reuse is an internal optimization. The runtime executes the tokenized
state once with batch 1, retains every `present.*` tensor, and executes only the
single-question branch against those owned tensors. Attention
masks cover the cached prefix plus the current branch, position IDs continue at
the prefix length, and pointer indices are local to branch hidden states.
Full-attention key/value tensors may grow along their past-sequence dimension;
convolution and recurrent tensors retain fixed shapes. FP16, FP32, and BF16
state storage is copied byte-for-byte without conversion.

An input named `past_key_values.<layer>.<type>` or `past.<layer>.<type>` must have exactly one
corresponding `present.<layer>.<type>` (or `present_key_values.*`) output with
matching dtype, rank, batch dimension, and fixed dimensions. A `position_ids`
input is also required. Graphs without that complete contract use the original
full-row path. A malformed graph that advertises compatible state I/O raises
rather than silently falling back. CPU
GroupQueryAttention currently supports cached context only for batch one, so
multi-question CPU requests use the full-row path; single-question CPU
requests use prefix reuse. CUDA currently uses the full-row path until
device-native state repetition is qualified.

Cached tensors are immutable values, while session execution and cache
mutation are serialized.

### KEV component thread pools

Component sessions apply `intra_op_num_threads`, `inter_op_num_threads`, and
string-valued ORT config entries from `model.decoder.session_options` in an
optional package `genai_config.json`. KEV CPU packages use these settings to
select 16 intra-op threads, one inter-op thread, and non-spinning idle pools
instead of creating an unrestricted pool for every component. Package builders
can tune this policy for their target hardware without adding model-specific
behavior to the runtime. CUDA KEV packages specify one host thread because their
serialized GPU execution needs little host-side parallelism.

Mobius ORT GenAI export emits these defaults automatically for packages with a
`backbone` and `pointer_head`: 16/1 threads for CPU and 1/1 for accelerator
providers. Exporters may edit the generated standard session options for
hardware-specific tuning.

### Runtime requirements

CLM requires the standard operators available in the repository's default ORT
package. KEV exports containing `com.microsoft::LinearAttentionGate` require
ONNX Runtime 1.30 or newer. Use an ORT package with that operator enabled when
building or loading KEV; older packages fail model loading with a missing
operator diagnostic. Native ORT GenAI CUDA qualification, device-resident
cross-component tensors, and explicit workflow-controlled CUDA graph capture
remain follow-up work and are not claimed by this first API.

Creation and execution functions return `OgaResult*` (`nullptr` on success).
Destroy functions accept `nullptr`; response strings are caller-owned and must
be released with `OgaDestroyString`.

## Directory schema

A package contains `tokenizer.json`, `tokenizer_config.json`, and component
subdirectories containing `model.onnx` (and optional external data). CLM uses
`encoder`, `state_head`, `action_head`, and `scorer`; KEV uses `backbone` and
`pointer_head`. The required `genai_config.json` is the source of truth for
`pad_token_id`, `context_length`, tokenizer construction, and component
session configuration.

Builders should also emit `component_manifest.json`:

```json
{
  "schema_version": 1,
  "model_type": "clm-v0.1-8b",
  "components": {
    "encoder": {"role": "backbone", "filename": "encoder/model.onnx"},
    "state_head": {"role": "head", "filename": "state_head/model.onnx"},
    "action_head": {"role": "head", "filename": "action_head/model.onnx"},
    "scorer": {"role": "scorer", "filename": "scorer/model.onnx"}
  }
}
```

Version 1 requires an integer `schema_version` equal to 1, a string
`model_type`, and a non-empty object of named component objects. Every component
requires a package-relative, traversal-free `filename` naming an existing file.
Unknown metadata is preserved for forward compatibility. For compatibility
with the already exported CLM and KEV directories, the runtime also recognizes
the exact layouts above when the manifest is absent. This does not change
single-component `OrtModelPackage` detection or validation. Manifest mappings
are authoritative and may use arbitrary safe names and flat filenames such as
`backbone.onnx`, `clm_heads.onnx`, or `kev_head.onnx`. Absolute paths and `..`
traversal are rejected before ORT session creation.

The native C++ sessions, and the thin Python dictionary adapters over them,
accept both exported schemas:

* Mobius CLM: `encoder`, `state_head`, `action_head`, `scorer`
* Model Builder CLM: `backbone`, `clm_heads` (scoring is performed by the runtime)
* Mobius KEV: `backbone`, `pointer_head`
* Model Builder KEV: `backbone`, `kev_head`

CLM applies reference structured rendering, no-special-token tokenization,
right padding, padding-aware last-token pooling, encoder L2 normalization,
state/action projection, calibrated scoring, request temperature, and typed
`noul`/`choice`/`score` answers. KEV escapes forged delimiters, constructs one
causal row per question, records option-end and decide readouts, right-pads
rows, runs its pointer head, and returns the reference four-decimal typed
answers.
