# Nemotron Speech Segment Timestamps

This sample streams audio through Nemotron Speech and builds its output only from completed segment
events. It enables `timestamp_level: "segment"` with a configuration overlay and prints each segment
as `[StartTime - StopTime] SegmentText` when the segment completes.

The sample calls `CreateMetadataCoreStateUsingTokenizerConfig()` after creating the
tokenizer stream. Timestamp decoding and finalization require this explicit setup;
after resetting a stream, initialize its metadata state again.

```bash
dotnet run --project examples/csharp/NemotronSpeechTimestamps -- \
  /path/to/model /path/to/audio.wav cuda
```

`TokenMetadataOutput.TimestampMetadata.Segments` is a per-call event list, not cumulative history. It is usually
empty, but one decoded token can complete multiple segments, so the sample iterates every returned
record. Pass each record from `GetNextTokensWithMetadata()` directly to
`DecodeWithMetadata(token)`. Acoustic timing is optional and is not supplied when
generator timestamps are disabled; an enabled timestamp consumer requires it.
`FinalizeMetadata()` emits the final trailing segment through the same metadata wrapper.