# Metadata Decoding

`TokenizerStream` owns the Extensions decoder cache and its private
`MetadataCoreState`. Public C/C++/C#/Python bindings expose decode and finalize
operations returning typed metadata, never a state handle.

For plain text (including when a timestamp-configured tokenizer is used for text
only), the generator returns token IDs; decode and append each incremental text
fragment. A stream that selects text decoding drops its unused default metadata state:

```cpp
auto stream = OgaTokenizerStream::Create(*tokenizer);
while (!generator->IsDone()) {
  generator->GenerateNextToken();
  for (int32_t token_id : generator->GetNextTokens()) {
    ProcessText(stream->Decode(token_id));
  }
}
```

For timestamps (`word`, `segment`, or `all`), the stream initializes metadata state
from the tokenizer configuration when created. The generator returns each ID with
an acoustic frame interval; Extensions
returns decoded text and completed word token spans. The stream joins the spans
to buffered intervals and returns word/segment events:

```cpp
auto stream = OgaTokenizerStream::Create(*tokenizer);
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

`Tokenizer` stores internal model-derived metadata settings, including timestamp
level, frame duration, and segment rules. Each stream automatically copies those
settings when metadata is enabled. Configure them on the model before creating the
tokenizer; the stream has no public metadata-state creation or override methods.
When all features are disabled, metadata state is created only if metadata
decoding or finalization is requested.
`Tokenizer::UpdateOptions()` forwards Extensions string options; it does not
change these GenAI model-derived settings. Per-cache producer configuration is
determined by the tokenizer configuration.
Enabled timestamps require a positive sample rate, hop length, and subsampling
factor. Configuration is copied into the state and stays fixed until reset.

Metadata state creation configures the stream's Extensions cache through
`OrtxSetDetokenizerCacheMetadataConfig`. This overrides shared tokenizer metadata
options only for that cache. Streams sharing a tokenizer have independent
accumulators and decoder caches. Disabled timestamps do not buffer
token intervals or accumulate words and segments, return a null
`timestampMetadata`, and impose no timing requirement.

## Decode and Process

The public `DecodeWithMetadata(token)`
retains the supplied ID and enabled feature data, decodes once through Extensions, and runs enabled
post-processing before returning. Generator-specific retrieval belongs to
`GetNextTokensWithMetadata()`, not the tokenizer. Callers pass the records to decoding
in order, keeping independent sequences in separate streams.
Creating metadata state does not enable timing collection in an existing generator;
configure the model's timestamp level before creating the generator.

The stream has one metadata decoder and one finalizer, both using its owned state.
Decode validates the selected token and timing before advancing Extensions, then
`MetadataCoreState` buffers the token interval, matches completed-word token spans
to their first and last intervals, and directly publishes per-call word/segment
records. It retains intervals until Extensions' pending-token watermark releases
them; unfinished segments persist between calls. When timestamp production is enabled,
the generator getter checks that model-produced interval counts match emitted token
counts and pairs them by position; missing intervals are an error. IDs can repeat
and cannot be used to look up timing.

An enabled timestamp consumer receives each token's timing, including steps that
complete no words. Results contain only events completed by the current call, not
cumulative history. Completed text is owned by the state and the C records borrow
it until the next stream operation.

Finalization flushes Extensions, completes trailing words and segments, and publishes
the same result shape without adding a token. Repeating stream
finalization returns no duplicate words or segments. Finalization does not inject
a token. Decoding may resume after finalization; use reset for an independent sequence.

## Ownership and Lifecycle

Text and calculated timestamp records are GenAI-owned. Native result pointers are
borrowed from the stream: do not retain them across decoding, finalization, reset,
or destruction. Copy any output that must survive the next operation. Reads and
operations on the same stream require external serialization.

Each stream has independent state. Mixing text-only and metadata decoding requires
`Reset()`, which discards pending work, recreates the decoder cache and restores
fresh model-derived metadata state if enabled. Destruction releases that state.

Metadata decode and finalization use the stream's model-derived state automatically.
If all features are off, they create a disabled state on first use. Disabled features
return null data, while ordinary decoded text remains available. `Reset()` restores
the tokenizer defaults. The first decode or
metadata finalization chooses a mode; switching modes afterward requires another reset.

Public native `OgaTokenMetadataOutput` and all nested pointers are stream-owned until the next
decode/finalize/reset or destruction. Python and C# copy them into owned snapshots.
The C-compatible field layouts are the contract; use compatible headers and native
packages. Never free borrowed result pointers.

## Configuration and Public Bindings

In C/C++/C#/Python, configure the model, create a tokenizer and a tokenizer stream,
then call metadata decode directly. Model-enabled features are initialized on stream
creation and restored on reset. Neither the state nor its configuration is exposed
through the stream's public API.

```csharp
using var config = new Config(modelPath);
config.Overlay(@"{""model"":{""timestamp_level"":""all"",
  ""segment_separators"":[""."",""!"",""?""],
  ""segment_gap_threshold_seconds"":0.26}}");
using var tokenizer = new Tokenizer(config);
using var stream = tokenizer.CreateStream();
```

The tokenizer reads `sample_rate`, `hop_length`, and `subsampling_factor` from
`model` in the package's `genai_config.json`. Levels are `off`, `word`, `segment`,
and `all`; the default is `off`. The optional gap is specified in non-negative
seconds (or `null` to disable it), then rounded to the nearest acoustic frame.
Zero and positive gaps below half a frame split each word into its own segment.
The model's default separators are `.`, `?`, and `!`. A model that requests timestamps
without positive timing values fails rather than silently disabling timestamps.
The native result metadata remains a directly readable typed structure.

## Adding a Feature

Add model-derived settings to the stream's internal metadata configuration, configure the corresponding Extensions
producer per cache, and process its per-step data in `MetadataCoreState` (or an owned
helper). Keep unfinished work across calls and clear only completed per-call outputs
at the start of each decode/finalize operation. Validate inputs before advancing
Extensions. Finalization and invalidation must include the feature. No additional
feature-specific stream decode method or runtime registry is required.

This workflow requires Extensions with the per-cache configuration API. The local
integration build uses the modified Extensions checkout; the packaged dependency
revision must be updated once that API is committed and available.