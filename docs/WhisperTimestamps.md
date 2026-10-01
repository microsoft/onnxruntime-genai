# Whisper timestamp decoding

ONNX Runtime GenAI can apply Whisper's token-level timestamp rules during generation.
The rules run before token selection and cover timestamp pairing and monotonicity,
the initial timestamp boundary, timestamp-versus-text probability mass, and Whisper
control-token suppression.

Timestamp decoding is opt-in:

```python
params = og.GeneratorParams(model)
params.set_search_options(
    whisper_timestamps=True,
    whisper_max_initial_timestamp_index=50,
)
```

Use a timestamp-compatible Whisper decoder prompt and do not include the
`<|notimestamps|>` token. The runtime rejects timestamp-enabled prompts containing that
token.

`whisper_max_initial_timestamp_index` is measured in 20 ms timestamp-token intervals.
The default value, 50, limits the first timestamp to the first second of the current
audio window. Set it to `-1` to disable the initial boundary.

Models must provide `model.timestamp_begin_token_id` and
`model.no_timestamps_token_id` in `genai_config.json`. The Whisper model builder emits
these values when the tokenizer contains the standard contiguous timestamp-token suffix.

## Consuming timestamp tokens

Generated sequences retain timestamp token IDs so a transcription component can inspect
them, but tokenizer decoding and streaming decoding omit timestamp tokens from visible
text. Use the tokenizer timestamp APIs to distinguish text from timestamps and convert a
timestamp to seconds relative to the current audio window:

```python
if tokenizer.has_timestamp_tokens:
    prompt_length = len(decoder_prompt)
    for token_id in generator.get_sequence(0)[prompt_length:]:
        if tokenizer.is_timestamp_token(token_id):
            relative_seconds = tokenizer.timestamp_to_seconds(token_id)
```

The example inspects a completed generated suffix. Prompt timestamps used for
conditioning are not current-window output.

`timestamp_begin_token_id` raises an error when timestamp metadata is unavailable.
`timestamp_to_seconds` raises an error when passed a token outside the model's timestamp
token range.

Equivalent capability, classification, and conversion APIs are available in C, C++,
C#, and Java.

## Ownership boundary

These APIs expose low-level, relative-window decoding state. A higher-level
transcription component remains responsible for audio chunking, media duration probing,
VAD policy, seek/redecode behavior, adding each window's absolute offset, and final
segment assembly. Word-level cross-attention/DTW timestamps are not provided by this
feature. Create a fresh generator for each new audio window or redecode attempt; appending
another window after timestamp generation begins is rejected.
