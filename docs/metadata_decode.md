# ASR Post-Processing

Native postprocessors and their timestamp/word-grouping helpers live in
[`src/models/postprocessing/`](../src/models/postprocessing/), separate from
input preprocessing in
[`src/models/preprocessing/`](../src/models/preprocessing/).

`PostProcessor` is the native base class for generator-facing postprocessors. It
holds the borrowed generator reference shared by output-specific wrappers and
declares virtual `ProcessNext()`, `Finalize()`, and `Reset()` methods.
`ProcessNext()` advances the concrete processor using the generator's current
output without returning a result. The base has no result getter or shared
result contract.

`ASRPostProcessor(generator)` derives from that base. It retrieves token and
optional timing views from the generator and delegates to its private
`ProcessNext(const TokenOutputs& outputs)` overload. That overload processes
the supplied views without reading the generator. The processor directly owns
a tokenizer, an ordinary tokenizer stream,
`MetadataCoreState`, grouping/timing state, and the current results. Configuration
comes from the generator's model and its `genai_config.json`. Callers do not pass a
model or create token metadata records.

Callers explicitly select `ASRPostProcessor` to obtain incremental text and
optional speech timestamps. Its non-virtual `GetNextResults()` returns
ASR-specific `ASRPostProcessorResult` values; other subclasses can expose their
own result types. This processor also supports untimed text when timestamps
are disabled.

## Processing Loop

Configure the model's timestamp level before creating the generator, then create
the processor from that generator alone. The generator retains its native model,
which supplies the processor configuration and tokenizer; no public model getter
or separate model ownership is required. The generator must outlive the processor. Only
`batch_size=1` and `num_beams=1` are supported; other values are rejected.

```cpp
auto postprocessor = OgaASRPostProcessor::Create(*generator);
while (!generator->IsDone()) {
  generator->GenerateNextToken();
  postprocessor->ProcessNext();
  for (const auto* result : postprocessor->GetNextResults()) {
    ProcessText(result->GetText());
    if (const auto* timestamps = result->GetTimestampResult()) {
      for (size_t index = 0; index < timestamps->GetWordCount(); ++index) {
        const auto* word = timestamps->GetTimestampRecord(OgaTimestampRecordKind_Words, index);
        ProcessWord(word->GetText(), word->GetFrameInterval(), word->GetTimestampInterval());
      }
      for (size_t index = 0; index < timestamps->GetSegmentCount(); ++index) {
        const auto* segment = timestamps->GetTimestampRecord(OgaTimestampRecordKind_Segments, index);
        ProcessSegment(segment->GetText(), segment->GetFrameInterval(), segment->GetTimestampInterval());
      }
    }
  }
}
postprocessor->Finalize();
for (const auto* result : postprocessor->GetNextResults()) {
  if (const auto* timestamps = result->GetTimestampResult()) {
    for (size_t index = 0; index < timestamps->GetWordCount(); ++index) {
      const auto* word = timestamps->GetTimestampRecord(OgaTimestampRecordKind_Words, index);
      ProcessWord(word->GetText(), word->GetFrameInterval(), word->GetTimestampInterval());
    }
    for (size_t index = 0; index < timestamps->GetSegmentCount(); ++index) {
      const auto* segment = timestamps->GetTimestampRecord(OgaTimestampRecordKind_Segments, index);
      ProcessSegment(segment->GetText(), segment->GetFrameInterval(), segment->GetTimestampInterval());
    }
  }
}
```

`ProcessNext()` consumes all emissions from the current generation step in
order. A step can emit zero tokens or multiple sequential tokens. Call it once
after each generation step, before advancing the generator again. There is no
generation-step counter or duplicate-processing detection.

`GetNextResults()` only reads the latest results. Repeated reads do not
decode, flush, or otherwise change state. Results are per-call events, not
cumulative history; retain any history the application needs. Incremental text
and completed words/segments are alternative transcript representations: do not
append both to the same transcript.

`Finalize()` flushes the final word and unfinished segment without inserting a
token. Read its results through the same getter. Repeated finalization emits no
duplicate words or segments. `Reset()` discards pending decoding, grouping,
intervals, segments, and results together, using the processor's copied model
configuration for a fresh sequence. It does not reset the generator.

## Internal Flow

The following native method flow starts after generation. It shows timestamps
enabled; the private overload handles the supplied tokens in order, including empty
decoded fragments. The language bindings forward processing to this same native path.

```text
ASRPostProcessor.ProcessNext()
  -> Generator.GetTokenOutputs(): view existing output without advancing generation
      -> transducer: GetStepTokens() and optional GetStepTokenIntervals()
          -> validate token/interval counts
      -> search: GetNextTokens().CopyDeviceToCpu()
  -> ASRPostProcessor.ProcessNext(outputs): private overload
      -> clear previous processing results
      -> TokenOutputs.GetTokens() / GetAcousticFrameIntervals()
      -> validate token/interval counts
      -> for each emission, in order:
          -> MetadataCoreState.ValidateInput(token)
          -> if timestamps enabled: Tokenizer.GetWordPieceInfo(token_id)
              -> OrtxGetTokenWordPieceInfo(...)
          -> TokenizerStream.Decode(token_id)
              -> OrtxDetokenizeCached(...)
          -> store exact decoded text in a result
          -> MetadataCoreState.ProcessDecoded(token, lexical facts, result)
  -> return to application

ASRPostProcessor.GetNextResults()
  -> read the processor's current results without mutation
  -> GetText() / nullable GetTimestampResult()
```

The private generator getter borrows existing transducer token and interval storage,
or reads the existing search-selected token buffer. It does not change generation
or token selection. Nemotron continues to supply zero or one token per step, without
copying its token or interval buffers. Optional timing is selected from the model's
timestamp configuration. `GenerateNextToken()` and the decoding strategies are
unchanged.

The private getter returns an internal `TokenOutputs` with inline private spans,
not a heap-allocated PImpl. Token IDs and optional timing are separate collections;
`GetAcousticFrameIntervals()` returns null when timing is absent. The spans borrow their
producer's storage and must remain valid for `ProcessNext(outputs)`. Native timing uses
`std::pair<int64_t, int64_t>` and `std::optional`, not a public C interval layout or
an integer presence flag. Timestamp records store frame bounds as an integer pair
and rounded second bounds as a `std::pair<double, double>`. The first element is
the start and the second is the exclusive stop. Public C declarations and result
handles remain field-based and opaque.

### Batch and Offline Input

The private `ProcessNext(outputs)` overload consumes only the supplied token/timing
views and the processor's tokenizer and copied configuration; it does not query
the generator or model state. It can process one token, a batch, or all supplied
tokens, followed by `Finalize()` to flush trailing records. Batch and incremental
processing preserve the same text and timestamps for identical ordered inputs.

This does not add a public offline factory or another producer API. The public
wrapper remains `ASRPostProcessor(generator)`. A different offline producer
would need an adapter and must supply timing to obtain timestamps. Stateful
grouping does not imply an online-only input source.

Extensions owns vocabulary-specific facts: `is_special`, `starts_word`,
`ends_word`, and `boundary_style`. Its standalone lookup is independent of decoder
history and runs before ordinary decoding only when timestamps are enabled. It neither groups words nor owns
timestamp state. The tokenizer and stream have no timestamp configuration or
special metadata decode mode.

GenAI consumes every token, including empty decoded fragments caused by buffered
UTF-8 or skipped special tokens. Grouping preserves exact whitespace and
punctuation. Half-open token spans map to the first and last contributing acoustic
intervals by position, never by token ID. Intervals remain buffered until no
pending word or decoder output needs them. Unfinished segments persist across
processing calls.

### MetadataCoreState Calls

`ProcessDecoded()` writes into the result created by the processor; it does not
construct another output representation. `PublishWord()` evaluates word output
first, then segment output. The conditional branches below show its possible
helper calls, not concurrent processing.

```mermaid
flowchart TD
  decoded["MetadataCoreState::ProcessDecoded()"] --> enabled["TimestampsEnabled()"]
  enabled --> retain["Create result.timestamps<br/>Append this token's acoustic interval to pending buffer"]
  retain --> group["TokenWordGroupingState::Process(piece, result.text)"]
  group -.->|For nonempty text parts| parts["ProcessTextPart()<br/>IsWhitespace() / IsPunctuation()<br/>Append() / CompleteWord() as needed"]
  group --> completed["MetadataCoreState::ProcessCompletedWords()"]
  completed --> words["TokenWordGroupingState::CompletedWords()"]
  words --> align["For each completed word:<br/>validate token span; take first.start and last.stop"]
  align --> publish["MetadataCoreState::PublishWord(text, interval, timestamps)"]
  publish -->|word or all| word["AddRecord(timestamps.words)<br/>RoundTimestamp(start and stop seconds)"]
  publish -->|segment or all| gap["Check gap against pending segment"]
  gap -->|Gap completes prior segment| prior["CompleteSegment()<br/>AddRecord(timestamps.segments)<br/>RoundTimestamp()"]
  gap --> pending["Start or extend pending segment text and interval"]
  prior --> pending
  pending --> separator["EndsWithSeparator() for configured suffixes"]
  separator -->|Suffix completes segment| finish["CompleteSegment()<br/>AddRecord(timestamps.segments)<br/>RoundTimestamp()"]
  completed --> release["After all completed words:<br/>TokenWordGroupingState::FirstPendingTokenIndex()<br/>Release intervals before that watermark"]
  release --> returned["Return to ASRPostProcessor::ProcessNext(outputs) with result populated"]
```

The grouper buffers incomplete words, whitespace, and UTF-8 output. It may
complete zero or multiple words from one token. `AddRecord()` preserves text and
frame bounds and converts each bound to seconds with `RoundTimestamp()`. No
completed-word event means no word/segment publication, but pending intervals and
grouping state still advance. A suffix-BPE end marker can also call
`CompleteWord()` directly from `Process()`.

After the last generation step, flush pending work without decoding a token:

```mermaid
flowchart TD
  finalize["ASRPostProcessor::Finalize()<br/>Clear previous results; create final result"] --> state["MetadataCoreState::ProcessFinalized()"]
  state --> enabled["TimestampsEnabled()"]
  enabled --> group["TokenWordGroupingState::Finalize()<br/>Append() pending delimiter; CompleteWord()"]
  group --> words["MetadataCoreState::ProcessCompletedWords()<br/>Same alignment/publication path as above"]
  words --> segment["MetadataCoreState::CompleteSegment()<br/>Flush any remaining segment with AddRecord()"]
  segment -.->|Caller reads| results["GetNextResults()"]
```

## Configuration

Timestamp levels are `off` (default), `word`, `segment`, and `all`. Enabled
timestamps require a model that produces acoustic intervals, currently
`nemotron_speech`, with positive `sample_rate`, `hop_length`, and
`subsampling_factor`. Being an ASR model alone is not sufficient. Missing or
invalid intervals fail before decoding the affected token.

The processor constructor reads these existing model fields once and derives
`seconds_per_frame` and the rounded segment-gap frame threshold. Its copied
configuration stores that optional timebase separately from word/segment output
settings. Neither the processor nor `MetadataCoreState` stores another set of sample rate,
hop length, and subsampling factor values. Missing or non-positive/non-finite
timebases are rejected when timestamps are enabled.

`segment_separators` defaults to `.`, `?`, and `!`. The optional
`segment_gap_threshold_seconds` is non-negative or `null`; it is rounded to the
nearest acoustic frame. Zero and positive gaps below half a frame split each word
into its own segment. When timestamps are off, ordinary text is still returned,
without timestamp records, interval buffering, or word/segment accumulation.

See [Nemotron speech timestamps](nemotron_speech_timestamps.md) for acoustic
origins and interval semantics. Diarization is not implemented by this processor.

## Bindings and Lifetime

C exposes opaque `OgaASRPostProcessor`, `OgaASRPostProcessorResult`,
`OgaASRPostProcessorTimestampResult`, and
`OgaASRPostProcessorTimestampRecord` handles, with typed getters for text,
timestamp presence, word/segment counts, records, frame bounds, and second bounds.
Result and record layouts are private. An out-of-range C result or record lookup
returns null. `OgaASRPostProcessorTimestampResultGetTimestampRecord()` selects
words or segments through `OgaTimestampRecordKind`; it also returns null for an
unknown kind. Word and segment counts have separate field accessors.

Call `OgaASRPostProcessorAsPostProcessor()` to obtain a borrowed `OgaPostProcessor`
base handle, then use `OgaPostProcessorProcessNext()` to dispatch processing
through the native virtual method. The base handle does not transfer ownership
and is valid until the ASR processor is destroyed; do not free it separately.
C++ `OgaASRPostProcessor::ProcessNext()` and the managed bindings perform this
conversion automatically. Results remain ASR-specific.

Each C getter reads one field or returns one opaque child handle. Adding a native
record field does not change existing getter signatures; expose a new field through
a new accessor rather than extending a combined output-parameter list. The C#
P/Invoke declarations mirror these C functions directly. Public language properties
hide those accessor calls without exposing native layouts.

Results, nested timestamp records, and native strings are borrowed until the next processor mutation
or destruction, not the next getter or generator operation. Never free them;
copy anything needed afterward. Operations and borrowed reads on one processor
must be externally serialized. Independent processors have independent state.

`MetadataCoreState` populates the processor's canonical results directly. There
is no intermediate metadata output or conversion to a second result model. The
processor owns one immutable configuration; metadata state reads that same object.

Python uses `ASRPostProcessor(generator)`, `process_next()`,
`get_next_results()`, `finalize()`, and `reset()`. Borrowed
`ASRPostProcessorResult` objects expose read-only `text` and nullable
`timestamp_result` properties. `ASRPostProcessorTimestampResult` objects expose
`words` and `segments`;
records expose `text`, `frame_interval`, and `time_interval`. Interval properties
return two-element tuples ordered as `(start, stop)`. These pybind11 properties
call the C++ wrapper getters, which delegate to the per-field C API. The binding
constructs Python strings, tuples, and lists without exposing native layouts.

C# uses `ASRPostProcessor(generator)`, `ProcessNext()`,
`GetNextResults()`, `FinalizeProcessing()`, and `Reset()`. Borrowed
`ASRPostProcessorResult` objects expose read-only `Text` and nullable `TimestampResult`
properties. `ASRPostProcessorTimestampResult` objects expose `Words` and `Segments`; records expose `Text`,
`FrameInterval`, and `TimeInterval`. Interval properties return named `(Start, Stop)`
value tuples. For example, read `record.Text` and `record.TimeInterval.Start`.

C++ follows the global, method-only opaque wrapper pattern. `GetNextResults()`
returns borrowed `const OgaASRPostProcessorResult*` pointers, not copied payloads.
`GetText()` returns borrowed native text; `GetTimestampResult()` returns a nullable
`const OgaASRPostProcessorTimestampResult*`. The container exposes `GetWordCount()`,
`GetSegmentCount()`, and `GetTimestampRecord(kind, index)`. Records expose `GetText()`,
`GetFrameInterval()`, and `GetTimestampInterval()`; interval getters return `std::pair`
values through the per-field C accessors. No public payload fields or separate C++
value-model types are exposed. Native C++ callers must respect the borrowed lifetime
themselves and copy text needed beyond the next processor mutation.

Python and C# wrap the same native results without copying their payload. They
reject access to results or nested records invalidated by processing, finalization,
reset, or disposal. Repeated getters do not invalidate results. Read the result,
append needed strings and scalar values to application-owned transcript storage,
and discard the borrowed object before processing more tokens. Returned strings
and scalar values are independent values; the result wrappers are not snapshots.

Bindings retain the generator against garbage collection. C# rejects explicitly
disposed generators before entering native processing. Native C/C++ callers must
keep the generator alive until after processor destruction; no native lifetime
sentinel or generation-step counter is used.

Ordinary `TokenizerStream.Decode()` and `Reset()` remain available and unchanged
for callers needing only text. No public generator-output getter, metadata input
type, tokenizer metadata decode method, or separate metadata state handle is
required.
