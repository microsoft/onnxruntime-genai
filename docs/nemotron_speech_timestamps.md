# Nemotron speech streaming timestamps

Nemotron streaming speech models can emit word and segment timestamps while preserving the same
incremental transcript fragments returned by ordinary tokenizer stream decoding.

## Model configuration

Add timestamp options under `model` in `genai_config.json`:

```json
{
  "model": {
    "timestamp_level": "segment",
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

`all` is not supported. Diarization also uses these three levels; see below.

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

Audio is processed into aligned central chunks and optional diarization right-context.
One generator drives ASR and, only when configured, diarization inference.
`TokenPostProcessor` derives from `PostProcessor` and forwards borrowed output views
to its internal `TokenPostProcessorInternal`, which owns decoding and speech metadata.
Ordinary tokenizer stream decoding is unchanged.

```mermaid
flowchart TB
  audio[Audio chunks] --> processor[StreamingProcessor]
  processor --> features[Shared VAD, aligned features, optional chunk origin]
  features --> generator[One Generator: ASR and optional diarization]
  generator --> outputs[Private GetTokenOutputs: IDs, optional frame intervals and speaker evidence]
  outputs --> stream[TokenPostProcessor.ProcessNextTokens]
  stream --> internal[TokenPostProcessorInternal.Process]
  internal --> ortx[Extensions: lexical facts when needed and ordinary cached decode]
  ortx --> align[GenAI MetadataCoreState: optional word grouping and alignment]
  align --> speaker[Optional word speaker aggregation and speaker-turn segments]
  speaker --> events[Canonical per-call text and optional metadata]
  events --> read[GetNextTokenResults: read only]
  stream --> final[Finalize: flush trailing word or turn]
  final --> events
```

When timestamps or diarization are enabled, the streaming processor includes a scalar int64
`chunk_start_sample` tensor alongside `audio_features` in `NamedTensors`.
The origin counts samples in chunks dropped by VAD; `Flush()` pads the final chunk
for feature extraction but advances the origin by only the actual input samples.
This is a processor-to-generator signal, not an ONNX model input. Pass the complete
processor result to the generator: omitting or renaming `chunk_start_sample` is an
error when timestamps or diarization are enabled.
For a locally installed Nemotron model outside the test fixtures, set
`NEMOTRON_STREAMING_MODEL_PATH` when running
`AudioSpeechValidationTests.NemotronProcessorEmitsChunkStartSample`.
That test uses CPU session options, even when the model package specifies CUDA.

For the timed path, create `TokenPostProcessor(generator)` after configuring
the model. The generator supplies *when* each token occurred; Extensions supplies
vocabulary-derived lexical facts and ordinary decoded text. GenAI groups the words.
Each processor result also contains the ordinary incremental text fragment. Use it for live text, or collect
completed word/segment events for stable timestamped output; do not append both to one transcript.
When both speech features are off, processor results contain only ordinary text,
without grouping or metadata accumulation. Text-only applications may alternatively
use `GetNextTokens()` and `TokenizerStream.Decode()` without a postprocessor.

Internally, GenAI buffers the frame intervals in token order until its word grouper reports
completed word spans and a pending-token watermark. It uses the first and last token
intervals of each word, accumulates unfinished segments across calls, and owns the
completed event text for the lifetime of the returned per-call result. The generator
pairs tokens and intervals by emission position, not by token ID.

The internal processor accepts ordered token, interval, and optional speaker-evidence
spans, allowing batch and incremental processing with the same alignment/finalization
rules. This does not add a public offline producer API. At setup, the generator-facing
adapter derives `seconds_per_frame` and the rounded segment gap threshold once from
the model configuration. The internal timestamp code receives those derived values,
not another copy of sample rate, hop length, and subsampling factor. Native token
input uses an optional `AcousticFrameInterval`; public results remain opaque handles.

## Decoding

Configure timestamp level and segment rules before creating the generator and
`TokenPostProcessor`. After each generation step, call `ProcessNextTokens()`
(C++/C#) or `process_next_tokens()` (Python), then read `GetNextTokenResults()` or
`get_next_token_results()`. All sequential emissions are processed in order,
including steps that emit zero or multiple tokens. Reading results repeatedly
does not change them. The private generator getter is not exposed to applications.
Enabled timestamps require present, valid acoustic timing. Intervals are half-open
`[start, stop)` and may span multiple frames. A processor requires the same model
as its generator, `batch_size=1`, and `num_beams=1`.

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
- `examples/csharp/NemotronSpeechTimestamps/Program.cs`
- `examples/python/model-asr.py`

The timestamp-focused C# sample enables segment mode and builds output only from completed segment
events. It prints each segment as `[StartTime - StopTime] SegmentText`; it does not concatenate token
fragments.

The general Nemotron examples render word mode as `[StartTime - StopTime]WordText` followed by one
space, with no space after the interval. Segment mode renders
`[StartTime - StopTime] SegmentText`. When diarization is enabled, records also
include `[speaker_N]`; diarization-only records omit the time interval. The final
summary is the plain transcript. `all` is not supported.

### Building the examples

These examples require headers, language bindings, and a native GenAI library from a build containing
the processor APIs, together with the corresponding ONNX Runtime Extensions lexical lookup API.
Existing packages such as GenAI `0.17.0` do not provide those APIs. Until a feature-containing release
is published:

The timestamp implementation is on `main` after
[PR #2591](https://github.com/microsoft/onnxruntime-genai/pull/2591).
Diarization additionally requires [PR #2719](https://github.com/microsoft/onnxruntime-genai/pull/2719);
use matching bindings and native binaries from that implementation checkout.

- For Python, use the wheel built from the implementation branch. Call `process_next_tokens()`
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
  the Linux native GenAI library. See the [general ASR sample](../examples/csharp/ModelASR/README.md)
  and [segment timestamp sample](../examples/csharp/NemotronSpeechTimestamps/README.md).

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

When `timestamp_level` is `off` and diarization is disabled, the examples use the processor's ordinary text results.
Applications needing only text may also use `GetNextTokens()` and tokenizer stream `Decode()`.
The streaming processor does not create timestamp origin metadata, the transducer does not append
token intervals, and the tokenizer stream does not buffer intervals or build timestamp events.

## Streaming diarization

Configure the ASR package with a stateful streaming export and its external weights:

```json
{
  "model": {
    "timestamp_level": "segment",
    "diarization": {
      "filename": "diarization/model.onnx",
      "right_context": 4,
      "session_options": {"provider_options": []}
    }
  }
}
```

The nonempty filename enables diarization. Paths are relative to the ASR package.
The graph must expose `input_features`, `num_lookahead_frames`, six `past_*` inputs,
`speaker_probs`, and six matching `present_*` outputs. The offline single-input
export is not supported. Cache updates/compression remain in ONNX; capacities and
embedding dimensions are derived from graph I/O, with eight speaker channels.

The diarization block declares these export-specific defaults:

| Field | Default |
| --- | --- |
| `sample_rate` | 16000 |
| `num_mels` | 128 |
| `fft_size` | 512 |
| `win_length` | 400 |
| `hop_length` | 160 |
| `subsampling_factor` | 8 |
| `preemph` | 0.97 |
| `log_eps` | 5.96046448e-8 |
| `right_context` | 4 |

Sample rate must match ASR. Each ASR acoustic frame must cover an integer number
of diarization feature/probability frames, and central chunks must align to both
encoder strides. Other values require a compatible trained/exported model, not
just editing this configuration. Preprocessing/time parameters are declared
because they cannot be inferred from ONNX tensor dimensions alone.

Right-context counts additional diarization encoder frames:
`duration = right_context * hop_length * subsampling_factor / sample_rate`.
Thus four default frames mean **320 ms total**, not 80 ms. With 560 ms central
chunks, initial buffering is 880 ms plus 16 ms STFT context, excluding compute.
Future context is not committed twice or used to advance ASR feature state.
EOF uses only available complete encoder frames for look-ahead.

One generator runs both models serially for each retained chunk before exposing
tokens/evidence to `TokenPostProcessor`. VAD gates both central inference paths,
preserves their caches, and advances original audio time through skipped chunks.
Future audio may influence look-ahead even if VAD later drops its central chunk.
Skipped intervals contribute no probability sums or evaluated-frame count.

`TokenPostProcessor` averages each completed word's evaluated probabilities using
internal boundary sums/counts, includes one ASR frame of left context, clamps to
at least 0.01, normalizes, and chooses one maximum label (`speaker_0`..`speaker_7`).
Multiple speakers may be active; the transcript determines emitted words rather
than a diarization threshold. Labels are recording-local anonymous identities.

| Timestamp level | Diarization enabled |
| --- | --- |
| `off` | Speaker-turn segments, no timing |
| `word` | Timestamped labeled words |
| `segment` | Timestamped labeled speaker turns |

`GetTimestamps()` remains non-null in diarization-only mode. Record accessors
expose optional times and speaker labels, with the same borrowed-result lifetime
as timestamp-only output. Speaker turns split only on speaker change/finalization;
timestamp-only segments retain punctuation/gap rules.

Flush all remaining audio through inference before calling the postprocessor's
single finalizer. Resetting it clears word/turn/evidence state, not inference.
See [Token post-processing](metadata_decode.md) for binding and lifetime details.
