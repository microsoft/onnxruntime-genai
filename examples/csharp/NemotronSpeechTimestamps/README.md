# Nemotron Speech Segment Timestamps

This sample streams audio through Nemotron Speech and builds its output only from completed segment
events. It enables `timestamp_level: "segment"` with a configuration overlay and prints each segment
as `[StartTime - StopTime] SegmentText` when the segment completes.

The tokenizer stream automatically initializes metadata from the model configuration.
`Reset()` restores those settings; no explicit metadata-state initialization is needed.

## Build and run

This sample requires managed bindings and a native GenAI library containing the timestamp APIs,
including the corresponding ONNX Runtime Extensions metadata support. The current `0.17.0` package
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

`TokenMetadataOutput.TimestampMetadata.Segments` is a per-call event list, not cumulative history. It is usually
empty, but one decoded token can complete multiple segments, so the sample iterates every returned
record. Pass each record from `GetNextTokensWithMetadata()` directly to
`DecodeWithMetadata(token)`. Acoustic timing is optional and is not supplied when
generator timestamps are disabled; an enabled timestamp consumer requires it.
`FinalizeMetadata()` emits the final trailing segment through the same metadata wrapper.