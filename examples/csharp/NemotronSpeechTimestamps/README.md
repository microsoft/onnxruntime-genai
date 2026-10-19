# Nemotron Speech Segment Timestamps

This sample streams audio through Nemotron Speech and builds its output only from completed segment
events. It enables `timestamp_level: "segment"` with a configuration overlay and prints each segment
as `[StartTime - StopTime] SegmentText` when the segment completes.

`TokenPostProcessor(generator)` owns decoding and timestamp state, using the model
configuration. `Reset()` clears all processor state together; no explicit metadata-state
initialization is needed. The generator must outlive its processor.

## Build and run

This sample requires managed bindings and a native GenAI library containing the timestamp APIs,
including the corresponding ONNX Runtime Extensions lexical lookup API. The current `0.17.0` package
references are aligned with the other C# examples but do not contain these APIs. Until a
feature-containing package is published, use the local project reference and matching native build.

From the repository root on Linux, point `LD_LIBRARY_PATH` at the directories containing the matching
GenAI, ONNX Runtime, and execution-provider libraries, then run:

```bash
dotnet run --project examples/csharp/NemotronSpeechTimestamps -c Release -p:UseLocalGenAI=true -- \
  /path/to/model /path/to/audio.wav cuda
```

On Windows, make the corresponding native libraries available on `PATH`. Update both the CPU and
CUDA package references to a feature-containing release before using the package-based build.

After each `GenerateNextToken()`, the sample calls `ProcessNextTokens()` and iterates
`GetNextTokenResults()`. After checking `result.GetTimestamps()` for null,
`GetSegments()` returns the completed records for the current call,
not cumulative history. A step can emit zero or multiple tokens and complete zero or
multiple segments, so every result and record is visited. Repeated getter calls do
not change state. Enabled timestamps require valid model-provided acoustic timing.
`FinalizeProcessing()` completes the trailing segment; read it through the same
getter. Results and nested records borrow processor-owned data and expire on the
next process, finalize, reset, or disposal. The sample reads each record's
`GetText()`, `GetStartTime()`, and `GetStopTime()` and appends the formatted string
to its own transcript before advancing. The processor supports one sequence and
one beam and must use its generator's model.