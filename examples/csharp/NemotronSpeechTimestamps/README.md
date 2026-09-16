# Nemotron Speech Segment Timestamps

This sample streams audio through Nemotron Speech and builds its output only from completed segment
events. It enables `timestamp_level: "segment"` with a configuration overlay and prints each segment
as `[StartTime]SegmentText` when the segment completes.

```bash
dotnet run --project examples/csharp/NemotronSpeechTimestamps -- \
  /path/to/model /path/to/audio.wav cuda
```

`TimestampDecodeResult.Segments` is a per-call event list, not cumulative history. It is usually
empty, but one decoded token can complete multiple segments, so the sample iterates every returned
record. `FinalizeTimestamps()` emits the final trailing segment.