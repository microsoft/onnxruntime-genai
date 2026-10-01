# Non-generative component runtime

`RankingSession` (CLM) and `DecisionSession` (KEV) execute exported directory
packages without constructing a `Generator`. Each runtime object owns an ORT
session for each named component and bounded, session-local non-generative
caches.

```python
import json
import onnxruntime_genai as og

ranking = og.RankingSession("/models/clm-v0.1-8b-fp32", providers=["cuda"])
answers = ranking.rank(json.load(open("request.json")))

decision = og.DecisionSession("/models/kev-4b-fp32")
answers = decision.decide(json.load(open("request.json")))
print(decision.prefix_reuse_status, decision.prefix_cache_stats)
# Force the reference full-row path for parity/debugging.
decision.prefix_reuse_enabled = False

print(ranking.cache_stats)
ranking.clear_cache()
# Zero entries or zero bytes disables the cache.
ranking.set_cache_capacity(0, 0)
```

`ComponentSession(package_path, component)` is the lower-level named graph
primitive. Its `input_names`, `output_names`, and `input_info` properties expose
the graph contract; `run({name: numpy_array}, outputs)` returns a name-to-array
dictionary. A session serializes calls to `run`, while distinct sessions may run
concurrently.

The same primitive is public C++ API:

```cpp
#include "ort_genai.h"

ComponentSession session{"/models/package", "custom_encoder", {"cuda"}};
std::vector<float> values(8);
OgaComponentInput input{"input", values.data(), values.size() * sizeof(float),
                        {2, 4}, OgaElementType_float32};
auto outputs = session.Run({input}, {"output"});
```

`RankingSession` and `DecisionSession` are public typed C++ package entry points.
They own the tokenizer and component sessions and perform the same rendering,
tokenization, pooling, scoring, and answer shaping as Python:

```cpp
OgaStructuredRequest request;
request.state = OgaStructuredValue::Object{{"weather", "heavy rain"}};
request.questions.emplace_back(
    "umbrella", OgaQuestion{"noul", "Should I take an umbrella?", {}});

RankingSession ranking{"/models/clm-v0.1-8b-fp32", {"cuda"}};
OgaModelResult answers = ranking.Run(request);

OgaFreeFormRankRequest free_form;
free_form.state = request.state;
free_form.instructions = "Which activity is more suitable?";
free_form.candidates = {{"picnic", "Have a picnic outdoors"},
                        {"museum", "Visit an indoor museum"}};
OgaRankingResult ranked = ranking.Rank(free_form);  // stable on ties

DecisionSession decision{"/models/kev-4b-fp32"};
OgaModelResult decisions = decision.Decide(request);
```

`OgaStructuredValue` is a JSON-independent recursive variant supporting null,
booleans, integers, doubles, strings, ordered arrays, and ordered objects.
`OgaAnswer` exposes typed `noul`, `choice`, `score`, `confidence`,
`probabilities`, and `legend` fields. `OgaModelResult::model` identifies CLM or
KEV. `Component(name)` remains available and creates a provider-configured
named component.
`DirectoryTokenizer(package_path)` is the public movable C++ tokenizer primitive
for these packages; `Encode(text)` performs no-special-token tokenization and
`PadTokenId()` exposes the configured pad id. These C++ classes are inline RAII
wrappers; the shared-library boundary contains no STL types.

The stable C ABI in `ort_genai_c.h` exposes opaque handles for the same
functionality:

* `OgaComponentSession`, `OgaComponentInputs`, and `OgaComponentTensors` cover
  named graph execution and tensor name/type/shape/data access.
  `OgaComponentInputsAdd` copies the input name, shape, and bytes, so callers
  may immediately release or mutate every supplied buffer.
* `OgaDirectoryTokenizer` and `OgaTokenIds` cover directory tokenization.
* `OgaStructuredValueHandle`, request, and question builders preserve ordered
  recursive values while cloning values supplied by callers.
* `OgaRankingSessionHandle` and `OgaDecisionSessionHandle` run typed requests;
  model-result and ranking-result accessors expose all answer and ranked-item
  fields without transferring borrowed string/value storage.
* Session cache controls set entry/byte capacity, return hit/miss/eviction and
  occupancy statistics, clear entries, or invalidate entries while refreshing
  package identity.

## Managed bindings

This change establishes the stable C ABI used by managed bindings. C#, Java,
and JavaScript wrappers are intentionally delivered by the dependent managed
bindings change rather than this native-runtime change.

CLM caches projected action embeddings after the encoder and action head.
Split-head packages also cache projected state embeddings, allowing an
identical repeated request to skip the encoder and both projection heads while
still executing the scorer with the request temperature. Action and state keys
include the canonical package identity, head layout, `float32` projection
dtype, and length-prefixed rendered text. Combined-head compatibility packages
continue to cache actions only because their state and action projections are
produced by one component invocation. Package identity includes the canonical
package path, provider list, and size/mtime fingerprints for the fixed known
manifest, tokenizer, backbone, head, scorer, and external-data filenames
supported by the runtime; arbitrary manifest-declared filenames are not
fingerprinted. State and action projections share the same bounded LRU. The
default is 256 entries and 64 MiB. KEV caches escaped/rendered tokenized state
prefixes and question branches
(including branch-local option readout indices), not probabilities. Its token
cache default is 512 entries and 16 MiB. Compatible stateful backbones also use
a separate 32-entry, 512 MiB byte-bounded prefix-state LRU. Both caches use thread-safe LRU
eviction; either capacity set to zero disables caching. Caches are isolated per
session and provider configuration.

KEV prefix reuse is enabled by default. The runtime executes the tokenized
state once with batch 1, caches every `present.*` tensor, deep-copies each state
along the batch dimension, and executes only the question branches. Attention
masks cover the cached prefix plus the current branch, position IDs continue at
the prefix length, and pointer indices are local to branch hidden states.
Full-attention key/value tensors may grow along their past-sequence dimension;
convolution and recurrent tensors retain fixed shapes. FP16, FP32, and BF16
state storage is copied byte-for-byte without conversion.

An input named `past_key_values.<layer>.<type>` must have exactly one
corresponding `present.<layer>.<type>` (or `present_key_values.*`) output with
matching dtype, rank, batch dimension, and fixed dimensions. A `position_ids`
input is also required. Graphs without that complete contract use the original
full-row path; `prefix_reuse_status` reports the explicit reason and
`fallback_runs` is incremented. A malformed graph that advertises compatible
state I/O raises rather than silently falling back. Setting
`prefix_reuse_enabled=False` selects the full-row parity/debug path. CUDA
sessions also use that path automatically for state prefixes shorter than 128
tokens, where recomputation is cheaper than transferring and repeating the
cached component state. CPU sessions and longer CUDA prefixes continue to use
prefix reuse.

`cache_stats` describes token/branch caching. `prefix_cache_stats` describes
model-state hits, misses, eviction/occupancy, prefix executions, batched branch
executions, and full-row fallbacks. `set_prefix_cache_capacity()` and the
matching C/C++ APIs configure it. Clearing or invalidating a session clears
both caches. Cached tensors are immutable values and every branch feed owns a
deep copy, while session execution and cache mutation are serialized.

### Optional KEV CUDA graph capture

Set `ORT_GENAI_KEV_CUDA_GRAPH=1` before constructing a CUDA
`DecisionSession` to specialize and capture the KEV backbone for the first
observed input shape. The runtime fixes the backbone's symbolic dimensions,
which allows ORT to constant-fold host-side shape operations and place every
remaining backbone node on CUDA. Inputs and outputs use stable device buffers
and I/O binding, and subsequent requests with the same tensor shape replay the
captured graph.

Capture is opt-in because specialization adds work to the first result and is
appropriate only for shape-stable workloads. If a later request has a different
backbone signature, the captured graph and its buffers are released, the
generic session is restored, and graph capture stays disabled for that session.
The pointer head is intentionally excluded because its scalar/index inputs have
CPU memory requirements that are not safe to capture. The default value is
`0`; any value other than `0` or `1` is rejected.

Creation, mutation, execution, and accessor functions return `OgaResult*`
(`nullptr` on success). Destroy functions accept `nullptr`. Strings, tensor
buffers, and borrowed structured values returned by accessors remain valid
until their owning handle is mutated or destroyed.
The legacy prefix-reuse status pointer is immutable for the session lifetime;
new code should use `OgaDecisionSessionCopyPrefixReuseStatus`, which supports a
size query followed by a caller-owned buffer copy.

## Directory schema

A package contains `tokenizer.json`, `tokenizer_config.json`, and component
subdirectories containing `model.onnx` (and optional external data). CLM uses
`encoder`, `state_head`, `action_head`, and `scorer`; KEV uses `backbone` and
`pointer_head`.

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
