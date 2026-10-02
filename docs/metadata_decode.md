# Metadata Decoding

`TokenizerStream` owns the Extensions decoder cache. It creates and retains one
`MetadataCoreState` per stream; the caller receives a shared handle. These are
internal C++ interfaces. Public C/C++/C#/Python bindings expose generic decode and
finalize operations returning typed metadata. The state handle remains internal.

For plain text (`timestamp_level: "off"`), the stream needs no metadata state. The
generator returns token IDs; decode and append each incremental text fragment:

```cpp
auto stream = OgaTokenizerStream::Create(*tokenizer);
while (!generator->IsDone()) {
  generator->GenerateNextToken();
  for (int32_t token_id : generator->GetNextTokens()) {
    ProcessText(stream->Decode(token_id));
  }
}
```

For timestamps (`word`, `segment`, or `all`), initialize metadata state *before*
decoding. The generator returns each ID with an acoustic frame interval; Extensions
returns decoded text and completed word token spans. The stream joins the spans
to buffered intervals and returns word/segment events:

```cpp
auto stream = OgaTokenizerStream::Create(*tokenizer);
stream->CreateMetadataCoreStateUsingTokenizerConfig();
while (!generator->IsDone()) {
  generator->GenerateNextToken();
  const auto tokens = generator->GetNextTokensWithMetadata();
  for (const auto& token : tokens) {
    const OgaTokenMetadataOutput& metadata = stream->DecodeWithMetadata(token);
    ProcessText(metadata.text);
    if (metadata.timestampMetadata) {
      const auto& timestamps = *metadata.timestampMetadata;
      for (size_t word_index = 0; word_index < timestamps.word_count; ++word_index)
        ProcessWord(timestamps.words[word_index]);
    }
  }
}
ProcessTrailingMetadata(stream->FinalizeMetadata());
```

`ProcessText` may display partial text as it arrives, while `ProcessWord` handles
completed timestamped words; they should not both be appended to the same transcript.
`FinalizeMetadata()` flushes a trailing word or segment without generating another token.

The loop assumes one decoding sequence, such as a Nemotron transducer stream.
For batches or beams, route each independent sequence to its own tokenizer stream.
`GetNextTokensWithMetadata()` reads the latest generated tokens without advancing
generation. Pass each returned record directly to `DecodeWithMetadata(token)`;
callers need not unpack or interpret the metadata fields. Each record contains a
`token_id` and optional `token_acoustic_frame_interval`, a pair of 64-bit integers
ordered `(start, stop)`. The absolute acoustic frame interval is half-open
`[start, stop)`; it is not restricted to one frame. When the generator's model
timestamp level is `off`, the getter does not read or validate acoustic timing and
marks it absent. Models without acoustic timing also return records without it.
Timestamp-enabled decoding rejects absent timing before consuming the token;
timestamp-disabled decoding accepts the record and skips timing retention.
Plain `GetNextTokens()` still returns IDs only and is unchanged.

The C getter `OgaGenerator_GetNextTokensWithMetadata` returns a borrowed array of
records valid until the next generator operation or destruction. Each record stores
its interval inline; `has_token_acoustic_frame_interval == 0` means timing is absent.
Copying a record by value also copies its interval, so the copy survives subsequent
generator operations and generator destruction.

The C++ wrapper returns a vector of copied `OgaTokenMetadataInput` records. Python
binds the same value type as `TokenMetadataInput` and exposes timing as a tuple or
`None`. C# copies the record into a managed `TokenMetadataInput` with a nullable
`(long start, long stop)` tuple. The C decoder takes a pointer to a whole
`OgaTokenMetadataInput`; the wrappers pass their copied values for the decode call.
Callers still pass the returned token object directly to decoding. Reading result
fields does not advance decoding.

`OgaTokenMetadataOutput` contains decoded `text` and nullable `timestampMetadata`. Timestamp
data contains `words`/`word_count` and `segments`/`segment_count`, with text, frame,
and second intervals in each record. No per-field native getter calls are required.
The C entry points are `OgaTokenizerStreamDecodeWithMetadata` and
`OgaTokenizerStreamFinalizeMetadata`. C# returns `TokenMetadataOutput` with `Text` and nullable
`TokenMetadataTimestamp`; Python returns `text` and `timestamp_metadata` (None when disabled).
Words and segments are nested under the timestamp field.

`Tokenizer` stores an internal `MetadataCoreConfig` with timestamp settings nested under
`timestamps`, leaving room for future feature configurations.
`CreateMetadataCoreStateUsingTokenizerConfig()` explicitly copies that tokenizer
configuration into a new state. For overrides, `CreateMetadataCoreState(config)`
still requires an argument; it has no default argument or no-argument overload.
`Tokenizer::GetMetadataCoreConfig()` returns an independent copy of the model-derived
settings, including timestamp level, frame duration parameters, and segment settings.
Callers may modify this copy without affecting the tokenizer or other streams, or
supply a manually populated internal config. Passing an all-disabled config
disables all consumers. The tokenizer-config factory also leaves consumers disabled
when they are disabled in the tokenizer configuration.
`Tokenizer::UpdateOptions()` forwards Extensions string options; it does not
change these GenAI model-derived settings. Per-cache producer configuration is
determined by the config passed to the factory.
Enabled timestamps require a positive sample rate, hop length, and subsampling
factor. Configuration is copied into the state and stays fixed until reset.

Creation configures the stream's Extensions cache through
`OrtxSetDetokenizerCacheMetadataConfig`. This overrides shared tokenizer metadata
options only for that cache. Streams sharing a tokenizer may select different
features without toggling shared options. Disabled timestamps allocate no
calculator or grouping state, produce a null `timestampMetadata`, and impose no
consumption requirement.

## Decode and Process

The public `DecodeWithMetadata(token)`
retains the supplied ID and enabled feature data, decodes once through Extensions, and runs enabled
post-processing before returning. Generator-specific retrieval belongs to
`GetNextTokensWithMetadata()`, not the tokenizer. Callers pass the records to decoding
in order, keeping independent sequences in separate streams.
Creating metadata state does not enable timing collection in an existing generator;
configure the model's timestamp level before creating the generator.

The stream has one metadata decoder and one finalizer, both using its owned state.
Decode validates the selected token and timing, calls Extensions once, retains the
result with `SetDecoded`, and calls `ProcessMetadata` before returning. Feature
processing remains separate inside `MetadataCoreState`, but callers do not need a
second operation to consume the decoded step. Invalid/missing timing is rejected
before advancing the decoder. When timestamp production is enabled, the generator
getter checks that model-produced timing records match the emitted token count and
IDs before returning them; missing timing for emitted transducer tokens is an error.

An enabled timestamp consumer receives each token's timing, including steps that
complete no words. Repeated processing of the same step returns the cached result.
Results contain only events completed by the current call, not cumulative history.

Finalization flushes Extensions, retains the trailing spans with `SetFinalized`,
and calls `ProcessMetadata` to complete pending words and segments. Repeating stream
finalization returns no duplicate words or segments. Finalization does not inject
a token. Decoding may resume after finalization; use reset for an independent sequence.

## Ownership and Lifecycle

Text and calculated timestamp records are GenAI-owned. `Metadata()` exposes a
read-only borrowed Extensions result for the current step, not an independently
owned snapshot. Do not retain its pointers across decoding, finalization, or reset.
Copy any output that must survive the next step. Reads and operations on the same
stream/state require external serialization.

A state cannot be used with another stream. Creating a second active state or
mixing text-only and metadata decoding requires `Reset()`. Reset may deliberately
discard pending work, recreates the decoder cache, and invalidates old state
handles. Stream destruction also invalidates them, even if a caller retains a
shared handle. State methods accessing decode results then throw instead of
dereferencing released cache storage.

Generic decode and finalization require explicit state creation, even for an empty
stream. Neither creates state implicitly. Disabled features return null data, while
ordinary decoded text remains available. After `Reset()`, initialize again.

Public native `OgaTokenMetadataOutput` and all nested pointers are stream-owned until the next
decode/finalize/reset or destruction. Python and C# copy them into owned snapshots.
The C-compatible field layouts are the contract; use compatible headers and native
packages. Never free borrowed result pointers.

## Public Binding Initialization

In C++ and C#, call `stream.CreateMetadataCoreStateUsingTokenizerConfig()` before
the first metadata decode. In Python use
`stream.create_metadata_core_state_using_tokenizer_config()`. The C entry point is
`OgaTokenizerStreamCreateMetadataCoreStateUsingTokenizerConfig`. These create state
owned by the stream, rather than returning the internal shared state handle.

For explicit settings, use `CreateMetadataCoreState(config)` in C++/C#,
`create_metadata_core_state(config)` in Python, or
`OgaTokenizerStreamCreateMetadataCoreState` in C. C/C++ use `OgaTokenMetadataCoreConfig`;
Python/C# expose `TokenMetadataCoreConfig`. Like the existing `Config`, this owns an
opaque native handle and accepts JSON overlays. No managed/native config struct
mirrors or per-field setters are needed.

```csharp
using var config = new TokenMetadataCoreConfig();
config.Overlay(@"{
  ""timestamps"": {
    ""level"": ""all"",
    ""segment_separators"": [""."", ""!"", ""?""],
    ""segment_gap_threshold_seconds"": 0.26
  }
}");
stream.CreateMetadataCoreState(config);
```

The tokenizer reads `sample_rate`, `hop_length`, and `subsampling_factor` from
`model` in the package's `genai_config.json`; do not repeat these fields in the
metadata overlay. Levels are `off`, `word`, `segment`, and `all`; the default is
`off`. The optional gap is specified in non-negative seconds (or `null` to disable
it), then rounded to the nearest acoustic frame using the model's frame duration.
Zero and positive gaps below half a frame split each word into its own segment.
Separators default to an empty list for an explicit config. A model that requests
timestamps without positive timing values, or an explicit metadata state that
enables them without those values, fails rather than silently disabling timestamps.

Overlays preserve omitted fields and replace supplied separator arrays; failed
parsing leaves the config unchanged. Strings are decoded by the native JSON parser.
The stream copies settings at creation, so later overlays or disposal of the config
do not affect it. An all-disabled config returns text with null timestamp metadata.

In C, use `OgaCreateTokenMetadataCoreConfig`, `OgaTokenMetadataCoreConfigOverlay`, and
`OgaDestroyTokenMetadataCoreConfig`. C++ uses `OgaTokenMetadataCoreConfig::Create()` and
`config->Overlay(json)`, with `unique_ptr` ownership. Python uses
`og.TokenMetadataCoreConfig()` and `config.overlay(json)` with automatic ownership.
The native result metadata remains a directly readable typed structure; only the
configuration uses an opaque handle.

## Adding a Feature

Add typed settings to `MetadataCoreConfig`, configure the corresponding Extensions
producer per cache, and keep the feature's calculator and consumption method in
`MetadataCoreState` (or an owned helper). Clear its per-step outputs when a new
result is installed. Stateful consumers add a pending check before advancement;
stateless optional readers need not. Finalization and invalidation must include
the feature. No additional feature-specific stream decode method or runtime
registry is required.

This workflow requires Extensions with the per-cache configuration API. The local
integration build uses the modified Extensions checkout; the packaged dependency
revision must be updated once that API is committed and available.