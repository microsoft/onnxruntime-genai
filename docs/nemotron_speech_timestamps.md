# Nemotron speech streaming timestamps

Nemotron streaming speech models can emit word and segment timestamps while preserving the same
incremental transcript fragments returned by ordinary tokenizer stream decoding.

## Model configuration

Add timestamp options under `model` in `genai_config.json`:

```json
{
  "model": {
    "timestamp_level": "all",
    "segment_separators": [".", "?", "!"],
    "segment_gap_threshold_seconds": 1.0
  }
}
```

`timestamp_level` controls timestamp generation:

| Value | Completed records |
| --- | --- |
| `off` | None (default) |
| `word` | Words |
| `segment` | Segments |
| `all` | Words and segments |

`segment_separators` is an optional array of decoded text suffixes that complete a segment. When
present, it replaces the default separators (`.`, `?`, and `!`). An empty array disables
separator-based completion.

`segment_gap_threshold_seconds` is an optional non-negative number measured in seconds. Zero splits
each word into its own segment; positive values shorter than half a frame round to the same behavior.
It is converted to the nearest encoder-frame count using half-up rounding. For a model with 80 ms frames, `1.0`
second becomes $1.0 / 0.08 = 12.5$, which rounds to 13 frames. Values below the half-frame boundary
round down. Set it to `null` or omit it to disable gap-based completion.

The frame duration comes from `sample_rate`, `hop_length`, and `subsampling_factor` under
`model` in the same `genai_config.json`; do not repeat them in metadata overrides.
If timestamps are requested for a non-Nemotron model, or these values are missing or non-positive,
config creation or overlay fails. Unknown timestamp levels and negative or non-finite gap
thresholds are rejected while loading the configuration.

## Decoding flows

Both paths start with audio processed into features, followed by generated token IDs. Model
configuration determines whether acoustic timing is produced. `TokenPostProcessor`
forwards borrowed token and optional timing views to its internal `TokenPostProcessorInternal`, which
owns decoding and timestamp processing. Ordinary tokenizer decoding is unchanged.
`timestamp_level` controls timestamp generation:

```mermaid
flowchart TB
  audio[Audio chunks] --> processor[StreamingProcessor]
  processor --> features[Mel features and optional chunk origin]
  features --> generator[Generator / Nemotron]

  subgraph plain[Without timestamps: timestamp_level = off]
    ids[GetNextTokens: token IDs] --> decode[TokenizerStream.Decode: incremental text]
    decode --> transcript[Plain-text transcript]
  end

  subgraph timed[With timestamps: word, segment, or all]
    setup[Processor copies model configuration] --> stream
    timed_tokens[Private Generator.GetTokenOutputs: IDs + frame intervals] --> stream[TokenPostProcessor.ProcessNextTokens]
    stream --> internal[Internal TokenPostProcessorInternal.Process]
    internal --> ortx[Extensions: vocabulary facts + ordinary cached decode]
    ortx --> align[GenAI MetadataCoreState: group words and align buffered frames]
    align --> events[Per-call word / segment events with frame and second bounds]
    events --> read[GetNextTokenResults: read only]
    stream --> final[Finalize: flush trailing word / segment]
    final --> trailing[Trailing word / segment events]
  end

  generator --> ids
  generator --> timed_tokens
```

When timestamps are enabled, the streaming processor includes a scalar int64
`chunk_start_sample` tensor alongside `audio_features` in `NamedTensors`.
The origin counts samples in chunks dropped by VAD; `Flush()` pads the final chunk
for feature extraction but advances the origin by only the actual input samples.
This is a processor-to-generator signal, not an ONNX model input. Pass the complete
processor result to the generator: omitting or renaming `chunk_start_sample` is an
error when timestamps are enabled.
For a locally installed Nemotron model outside the test fixtures, set
`NEMOTRON_STREAMING_MODEL_PATH` when running
`AudioSpeechValidationTests.NemotronProcessorEmitsChunkStartSample`.
That test uses CPU session options, even when the model package specifies CUDA.

For the timed path, create `TokenPostProcessor(generator)` after configuring
the model and creating the generator. The processor obtains its model from the
generator rather than taking a separate model argument. The generator supplies
*when* each token occurred; Extensions supplies
vocabulary-derived lexical facts and ordinary decoded text. GenAI groups the words.
Each processor result also contains the ordinary incremental text fragment. Use it for live text, or collect
completed word/segment events for stable timestamped output; do not append both to one transcript.
The plain path needs no metadata state or finalization call.

Internally, GenAI buffers the frame intervals in token order until its word grouper reports
completed word spans and a pending-token watermark. It uses the first and last token
intervals of each word, accumulates unfinished segments across calls, and owns the
completed event text for the lifetime of the returned per-call result. The generator
pairs tokens and intervals by emission position, not by token ID.

The private generator getter borrows the existing per-step transducer output;
Nemotron continues to supply zero or one token per step. The internal stream accepts
ordered token/timing values and can process a whole offline batch or incremental
batches with the same finalization rules. This does not add a public offline producer API.

At setup, the generator-facing adapter calculates `seconds_per_frame` and the
rounded gap threshold once from the existing model configuration. The internal
timestamp code receives those derived values, not another copy of sample rate,
hop length, and subsampling factor. Native token input uses an optional
`AcousticFrameInterval`; public C output handles and layouts are unchanged.

## Decoding

Configure timestamp level and segment rules before creating the generator and
`TokenPostProcessor`. After each generation step, call `ProcessNextTokens()`
(C++/C#) or `process_next_tokens()` (Python), then read `GetNextTokenResults()` or
`get_next_token_results()`. All sequential emissions are processed in order,
including steps that emit zero or multiple tokens. Reading results repeatedly
does not change them. The private generator getter is not exposed to applications.
Enabled timestamps require present, valid acoustic timing. Intervals are half-open
`[start, stop)` and may span multiple frames. A processor always uses its
generator's model and requires `batch_size=1` and `num_beams=1`.

`GetText()` / `get_text()` returns the ordinary incremental decoded fragment for partial
display. Applications that need only stable output can instead append completed word or segment
event text as it arrives.

Completed words retain their original leading whitespace and attached punctuation. Segment text is
the exact concatenation of its words; no whitespace is trimmed or synthesized. After finalization,
concatenating all emitted words or all emitted segments reproduces the decoded transcript exactly.
In every binding, read and copy needed text and scalar values into the application's
transcript before the next processor mutation. Result wrappers do not retain old native output.

C/C++ expose opaque result and timestamp-record handles with typed accessors.
C# and Python expose the same borrowed timestamp output through nullable
`GetTimestamps()` and `get_timestamps()`. Read records using `GetWords()` / `GetSegments()`
or `get_words()` / `get_segments()`, then the record's text/frame/time accessors.
The returned objects are opaque wrappers, not dictionaries or copied metadata objects.
They are per-call event lists,
not cumulative history. Each list is
empty when that call completes no records, and it may contain multiple records when one decoded
token spans multiple word or segment boundaries. Callers retain any history they need.

After the final audio chunk, call `Finalize()` (C++), `FinalizeProcessing()` (C#),
or `finalize()` (Python), then read the results through the usual getter. It
completes pending words and segments without adding transcript text. Repeated
finalization emits no duplicate events. `Reset()` / `reset()` clears all processor
state together; it does not reset the generator. The generator must outlive its
processor. See [Token post-processing](metadata_decode.md) for the full lifecycle.

The Nemotron examples demonstrate the complete flow:

- `examples/c/src/model_asr.cpp`
- `examples/csharp/ModelASR/Program.cs`
- `examples/python/model-asr.py`

For segment-only output in the C# ModelASR sample, configure `model.timestamp_level` as `segment`.
The sample builds output only from completed segment events, including the trailing segment emitted
by finalization. It prints each segment as `[StartTime - StopTime] SegmentText`; it does not
concatenate token fragments.

The general Nemotron examples render word mode as `[StartTime - StopTime]WordText` followed by one
space, with no space after the interval. Segment mode renders
`[StartTime - StopTime] SegmentText`. All mode reports both views separately: the segment-timestamp
transcript followed by the per-word timestamp transcript.

### Building the examples

These examples require headers, language bindings, and a native GenAI library from a build containing
the processor APIs, together with the corresponding ONNX Runtime Extensions lexical lookup API.
The implementation is available on `main` after [PR #2591](https://github.com/microsoft/onnxruntime-genai/pull/2591).
Existing packages such as GenAI `0.17.0` do not provide those APIs. Until a feature-containing release
is published:

- For Python, use a wheel built from the same `main` checkout as the native library. Call `process_next_tokens()`
  after generation and `finalize()` at the end, then read typed objects from
  `get_next_token_results()`: use nullable `result.get_timestamps()`, then
  `get_words()` or `get_segments()`, and each record's `get_text()`,
  `get_start_time()`, and `get_stop_time()` accessors. Append the needed values to
  application-owned transcript storage before the next processing call; the
  returned objects are borrowed, not independent snapshots.
- For C++, build the `model_asr` target with `MODEL_ASR=ON` and set `OGA_INCLUDE_DIR` and `OGA_LIB_DIR`
  to the matching headers and native library.
- For C#, pass `-p:UseLocalGenAI=true` to build against the local managed project. Run with matching
  native libraries on the platform library search path; the project reference alone does not supply
  the Linux native GenAI library. See the [ASR sample](../examples/csharp/ModelASR/README.md), including
  its segment-timestamp configuration instructions.

The C# package references remain at the common `0.17.0` baseline and must be updated to a release
containing these APIs before using package-based builds. Selecting `timestamp_level: "off"` does not
remove the timestamp API references from the C++ or C# example source.

With a feature-enabled Python package installed, run the example regressions from the repository root:

```bash
python -m pytest test/python/test_onnxruntime_genai_api.py --test_models test/models -k asr_example
```

The real-speech cases use `test/models/nemotron-speech-streaming` and `test/audios/jfk.flac`. Set
`NEMOTRON_STREAMING_MODEL_PATH` to use a model outside the test fixtures. They exercise every timestamp
level, trailing finalization, and VAD-skipped silence; they skip when the model or audio is absent.
The VAD case requires a model with a configured VAD component. These tests run inference on CPU.

## Interval semantics

Frame and time intervals are half-open: `[start, stop)`. Each RNNT token initially covers one
encoder frame. Frame positions are absolute across all submitted audio, including chunks discarded
by VAD. Time values are seconds rounded to two decimal places.

Punctuation remains attached to its word. Segments complete on configured suffixes, configured
frame gaps, or finalization.

When `timestamp_level` is `off`, the examples use the processor's ordinary text results.
Applications needing only text may also use `GetNextTokens()` and tokenizer stream `Decode()`.
The streaming processor does not create timestamp origin metadata, the transducer does not append
token intervals, and the tokenizer stream does not buffer intervals or build timestamp events.
