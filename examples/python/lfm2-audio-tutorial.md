# Build your LFM2-Audio / LFM2.5-Audio ONNX models for ONNX Runtime GenAI

LFM2-Audio pairs a FastConformer speech encoder (based on
[nvidia/canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash)) with an LFM2 decoder
(the hybrid conv/attention model described in
[the LFM2 support PR](https://github.com/microsoft/onnxruntime-genai/pull/1979)). ONNX Runtime GenAI
runs the speech-to-text half of it — ASR and spoken-prompt chat — as three ONNX models:

| Model | Inputs | Outputs |
| --- | --- | --- |
| speech encoder (`audio_encoder.onnx`) | `mel_spectrogram`, `mel_lengths` | `audio_embeddings`, `audio_lengths` |
| `embeddings.onnx` | `input_ids`, `audio_features` | `inputs_embeds` |
| `model.onnx` (decoder) | `inputs_embeds`, `attention_mask`, KV + conv cache | `logits` |

Only the decoder is produced by the model builder in this repository; the speech encoder is
published as ONNX by LiquidAI, and the embedding model is a small graph you build once.

The model can also *speak*. Two more graphs, both published by LiquidAI, give it a voice here: a
depthformer that turns a decoder hidden state into a frame of audio codes, and the audio embedding that
feeds each frame back to the decoder. [Speech output](#6-speech-output) adds them; without them the
model answers in text only.

## Steps

1. [Build the decoder](#1-build-the-decoder)
2. [Get the speech encoder and build the embedding model](#2-get-the-speech-encoder-and-build-the-embedding-model)
3. [Write `genai_config.json`](#3-write-genai_configjson)
4. [Choose the precisions](#4-choose-the-precisions)
5. [Run the model](#5-run-the-model)
6. [Speech output](#6-speech-output)
7. [Known limitations](#7-known-limitations)

## 1. Build the decoder

```bash
# Download the PyTorch model
$ huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B --local-dir ./lfm2.5-audio/pytorch

# Build the decoder as INT4 with FP32 inputs/outputs for CPU
$ python3 -m onnxruntime_genai.models.builder \
    -i ./lfm2.5-audio/pytorch \
    -o ./lfm2.5-audio/cpu \
    -p int4 \
    -e cpu \
    --extra_options exclude_embeds=true
```

`-m LiquidAI/LFM2.5-Audio-1.5B` instead of `-i` works too, and downloads the checkpoint itself.

The three checkpoints LiquidAI publishes as PyTorch — `LFM2-Audio-1.5B`, `LFM2.5-Audio-1.5B` and
`LFM2.5-Audio-1.5B-JP` — build the same way. Their `"lfm"`, `"encoder"` and `"preprocessor"` settings
are identical, so the front-end defaults below fit all three; only the weights differ. The GGUF
repositories are for llama.cpp and are not inputs to this builder.

The checkpoint has no `model_type` and no transformers model class: its `config.json` nests the LFM2
decoder config under `"lfm"`, next to the speech encoder, depthformer and mel front-end settings, and
the checkpoint stores the decoder under the `lfm.` prefix. The builder recognizes
`Lfm2AudioForConditionalGeneration`, reads that nested config and loads only the decoder, whose
logits are tied to the token embeddings.

`exclude_embeds=true` is what makes this a speech pipeline stage: the decoder then takes
`inputs_embeds` instead of `input_ids`, so the embedding model can splice the encoder output into the
token embeddings before the decoder runs. The builder writes `"type": "lfm2_audio"` into
`genai_config.json` for this case, and `"type": "lfm2_audio_text"` when the flag is omitted — that
second form is a plain text-only LFM2 model that happens to come from an audio checkpoint, and it
runs through the normal `AppendTokens` path with no speech or embedding model.

## 2. Get the speech encoder and build the embedding model

The speech encoder is `onnx/audio_encoder.onnx` in
[LiquidAI/LFM2.5-Audio-1.5B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX)
(`audio_encoder_fp16.onnx` and `audio_encoder_q4.onnx` are there too). Download the graph together
with every `*.onnx_data*` file next to it. That repository is the export of `LFM2.5-Audio-1.5B`, so
its encoder belongs to that checkpoint only; `LFM2-Audio-1.5B` and `LFM2.5-Audio-1.5B-JP` have their
own encoder weights and need their own export, from `conformer.*` and `audio_adapter.*` in the
checkpoint (`liquid_audio.model.conformer.ConformerEncoder` plus its adapter MLP, traced for one
clip). It has the signature ONNX Runtime GenAI expects, for a
single clip at a time:

| Name | Shape | Type |
| --- | --- | --- |
| `mel_spectrogram` (input) | `[1, num_frames, 128]` | float |
| `mel_lengths` (input) | `[1]` | int64 |
| `audio_embeddings` (output) | `[1, ceil(num_frames / 8), hidden_size]` | float |
| `audio_lengths` (output) | `[1]` | int64 |

The encoder subsamples the mel frames by 8 (three stride-2 convolutions) and its adapter projects the
result to the decoder's hidden size, so one output frame covers 80 ms of audio. `mel_lengths` gives
it the clip's real length; the runtime always passes the clip's own frames, so nothing is padded.
The `audio_lengths` output is not read — the runtime derives the same counts itself, before the
encoder runs, so that it can size the prompt — but the graph produces it.

The published export only handles one clip per run: its subsampling mask is traced for a batch of
one, and a wider `mel_spectrogram` fails inside the graph. The runtime therefore runs it once per
clip. An encoder exported to take a real batch would work here too, one clip at a time.

The embedding model looks up `input_ids` in the decoder's embedding table and scatters
`audio_features` into the positions holding the audio placeholder token. Build it from
`lfm.embed_tokens.weight` in the checkpoint:

```python
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import load_file

pytorch_dir = Path("./lfm2.5-audio/pytorch")
output_dir = Path("./lfm2.5-audio/cpu")

config = json.loads((pytorch_dir / "config.json").read_text())["lfm"]
hidden_size, vocab_size = config["hidden_size"], config["vocab_size"]
# Pick any id the text never uses; it must match model.audio_token_id in genai_config.json.
audio_token_id = 133  # <|reserved_123|>

embed_weight = load_file(pytorch_dir / "model.safetensors")["lfm.embed_tokens.weight"]
table = numpy_helper.from_array(embed_weight.float().numpy(), name="embed_tokens.weight")

graph = helper.make_graph(
    [
        helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["text_embeds"], axis=0),
        helper.make_node("Shape", ["text_embeds"], ["embeds_shape"]),
        helper.make_node("Reshape", ["text_embeds", "flat_rows"], ["flat_embeds"]),
        helper.make_node("Reshape", ["input_ids", "flat"], ["flat_ids"]),
        helper.make_node("Equal", ["flat_ids", "audio_token_id"], ["is_audio"]),
        helper.make_node("NonZero", ["is_audio"], ["audio_positions_t"]),
        helper.make_node("Transpose", ["audio_positions_t"], ["audio_positions"], perm=[1, 0]),
        helper.make_node("ScatterND", ["flat_embeds", "audio_positions", "audio_features"], ["merged"]),
        helper.make_node("Reshape", ["merged", "embeds_shape"], ["inputs_embeds"]),
    ],
    "embedding",
    [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, ["num_audio_tokens", hidden_size]),
    ],
    [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])],
    initializer=[
        table,
        numpy_helper.from_array(np.array([-1, hidden_size], np.int64), name="flat_rows"),
        numpy_helper.from_array(np.array([-1], np.int64), name="flat"),
        numpy_helper.from_array(np.array(audio_token_id, np.int64), name="audio_token_id"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, output_dir / "embeddings.onnx", save_as_external_data=True, location="embeddings.onnx.data")
```

## 3. Write `genai_config.json`

Add `embedding` and `speech` sections and an `audio_token_id` to the `genai_config.json` the builder
produced. Leave the `decoder` and `search` sections it wrote alone:

```json
{
    "model": {
        "audio_token_id": 133,
        "bos_token_id": 1,
        "context_length": 32768,
        "decoder": { "...": "written by the model builder" },
        "embedding": {
            "filename": "embeddings.onnx",
            "inputs": {
                "input_ids": "input_ids",
                "audio_features": "audio_features"
            },
            "outputs": {
                "inputs_embeds": "inputs_embeds"
            },
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            }
        },
        "eos_token_id": [7, 128, 130],
        "pad_token_id": 0,
        "speech": {
            "filename": "audio_encoder.onnx",
            "inputs": {
                "audio_embeds": "mel_spectrogram",
                "audio_lengths": "mel_lengths",
                "audio_sizes": "audio_sizes"
            },
            "outputs": {
                "audio_features": "audio_embeddings"
            },
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": []
            }
        },
        "type": "lfm2_audio",
        "vocab_size": 65536
    },
    "search": { "...": "written by the model builder" }
}
```

Keep the `eos_token_id` list the builder wrote. Besides `<|im_end|>` (7) it holds `<|audio_start|>`
(128) and `<|text_end|>` (130), where the model turns from text to speech; without them generation
runs on into positions meant for the audio head and returns fluent nonsense. See
[the modes](#the-models-modes).

`filename` is whatever the encoder file is called on disk — keep the name it was published under,
for the reason in [Choose the precisions](#4-choose-the-precisions).

`audio_sizes` is not an encoder input; the name only tells the runtime what to call the per-clip
token counts it computes, so leave it as is unless it collides with a real input of your graph.
`audio_token_id` must be the id the embedding model scatters over, and must not be an id the
tokenizer can emit for ordinary text — the reserved range of the LFM2 tokenizer is the natural home
for it.

The audio front end needs no configuration file: it defaults to the settings every published
LFM2-Audio checkpoint shares (`"preprocessor"` in `config.json`), which are NeMo's
`AudioToMelSpectrogramPreprocessor`:

| Setting | Value | `genai_config.json` override |
| --- | --- | --- |
| Sample rate | 16000 | `model.sample_rate` |
| Mel bins | 128 | `model.num_mels` |
| FFT size | 512 | `model.fft_size` |
| Window | 400 samples (25 ms), symmetric Hann | `model.win_length` |
| Hop | 160 samples (10 ms) | `model.hop_length` |
| Pre-emphasis | 0.97 | `model.preemph` |
| Log guard | 2⁻²⁴ | `model.log_eps` |
| Normalization epsilon | 1e-5 | `model.norm_eps` |
| Encoder subsampling | 8 | `model.subsampling_factor` |

Set the overrides only for a fine-tune that changed them; the defaults already match all the
published models.

## 4. Choose the precisions

`-p int4` quantizes the decoder, and the decoder alone. The embedding model and the speech encoder
are assembled by hand, so they stay at whatever precision you built or downloaded them at — and
together they are bigger than the quantized decoder. Measured on LFM2.5-Audio-1.5B, transcribing
three clips (16 kHz, 44.1 kHz stereo, and an mp3):

| Decoder | Embedding table | Encoder | Size | Transcripts |
| --- | --- | --- | --- | --- |
| fp32 | fp32 | fp32 | 5736 MB | baseline |
| int4 | fp32 | fp32 | 1787 MB | correct; one comma differs from fp32 |
| int4 | fp16 | fp32 | 1518 MB | identical to the row above |
| int4 | fp16 | fp16 | 1278 MB | identical to the row above |
| **int4** | **fp16** | **q4** | **1178 MB** | **identical to the row above** |

Quantizing the other two components is free here: every int4 row produced the same tokens as every
other, so the last row is the one to build. The only difference anywhere is the int4 decoder against
fp32 — `And so, my fellow Americans` became `And so my fellow Americans` on one clip, with the words
otherwise unchanged.

For the embedding table, `astype(np.float16)` on the weight in the snippet above is the whole change;
keep the graph's output fp32 by casting after the `Gather`, so the decoder still receives what it
declares. A 4-bit table would be smaller still — the builder's own text-only export gets one, through
`GatherBlockQuantized` over the quantized `lm_head` weights — but `exclude_embeds` is what puts the
table outside the decoder in the first place, so the pipeline cannot share that copy.

For the encoder, take `audio_encoder_q4.onnx` instead of `audio_encoder.onnx`. **Keep the published
file names.** Each variant's graph refers to its own `*.onnx_data` by name, so renaming the pair to
`speech.onnx` breaks the lookup with `External data path validation failed`. Leave both files as they
are and point `model.speech.filename` at the name you downloaded.

## 5. Run the model

[`model-mm.py`](model-mm.py) drives any multi-modal model in this repository:

```bash
$ python3 model-mm.py -m ./lfm2.5-audio/cpu -e cpu --audio_paths ./question.wav
```

The prompt must hold exactly one `<|audio|>` marker per clip, in clip order; the processor rejects
any other count rather than guessing, and it accepts one prompt at a time (a single-entry list is
fine, batching is not). The C++ audio processor replaces each marker with one `audio_token_id` per
encoder frame, so the number of placeholders always matches the number of features the encoder
produced. Text on either side of a marker is tokenized on its own, which is what
`liquid_audio.ChatState` does.

LFM2-Audio has no audio token of its own and no chat-template entry for audio: `<|audio|>` is this
runtime's marker, the same role `<image>` plays for LFM2-VL.

```
<|startoftext|><|im_start|>system
Perform ASR.<|im_end|>
<|im_start|>user
<|audio|><|im_end|>
<|im_start|>assistant
```

### The model's modes

The system prompt picks the task, and the reference implementation pairs each with a generation
routine of its own:

| Mode | System prompt | Reference routine | Output | Text-only build | With [speech output](#6-speech-output) |
| --- | --- | --- | --- | --- | --- |
| Chat | none | `generate_sequential` | text | works | works |
| ASR | `Perform ASR.` | `generate_sequential` | text | works | works |
| TTS | `Perform TTS. Use the UK male voice.` (also US male, US female, UK female) | `generate_sequential` | speech | stops immediately | works |
| Interleaved | `Respond with interleaved text and audio.` | `generate_interleaved` | text and speech, alternating | do not use | works, with `audio_interleaved=True` |

**ASR** and **chat** need nothing more than the three models above. A spoken or typed question with
no system prompt gets a text answer, and both match the reference step for step: on the reference's
own `question.wav` and three typed questions the text logits agree to within 6e-4 and every token is
the same.

`LFM2.5-Audio-1.5B-JP` takes its own prompts, `Perform ASR in japanese.` and
`Perform TTS in japanese.`; the rest of this section applies to it unchanged.

Two tokens mark the turn passing from text to speech: `<|audio_start|>` when the rest of the answer is
spoken, and `<|text_end|>` when the text half of an interleaved answer is done. In a text-only build
the builder puts both in `eos_token_id`, because the positions after either are audio codes meant for
the depthformer, and read off the text head they decode to fluent, plausible nonsense.

**TTS** in a text-only build produces nothing: the whole text stream the model emits is
`<|audio_start|>` followed by `<|im_end|>`, everything in between being audio. Generation stops on
that first token, and the stop token is not added to the sequence, so there is nothing after the
prompt to decode: expect an empty string.

**Interleaved** cannot be used in a text-only build. The reference writes six text tokens, then
`interleaved_n_audio` audio frames (12, or 9 for the JP checkpoint) whose embeddings go back into the
context, then six more text tokens, and so on. Without audio frames to feed, the text head is read at
positions the model means for speech: the first six tokens are right and the rest is mostly the
non-breaking-space token. For a text answer, leave the system prompt out and use chat mode.

Sampling: the reference generates text greedily in every mode, so `do_sample=False` is right here.
The temperatures and `top_k` values quoted for the model apply to the audio codes only; see
[Speech output](#6-speech-output) for where they go.

## 6. Speech output

Speech needs the decoder's hidden states, two more graphs and a few lines of `genai_config.json`.

**Build the decoder with its hidden states.** The depthformer reads the hidden state the logits are
projected from, so add `include_hidden_states=true`:

```bash
$ python3 -m onnxruntime_genai.models.builder \
    -i ./lfm2.5-audio/pytorch -o ./lfm2.5-audio/cpu -p int4 -e cpu \
    --extra_options exclude_embeds=true include_hidden_states=true
```

**Get the two graphs** from
[LiquidAI/LFM2.5-Audio-1.5B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX):
`onnx/vocoder_depthformer.onnx` and `onnx/audio_embedding.onnx` (with their `*.onnx_data` files for
the `fp16` and `q4` variants), and `onnx/audio_detokenizer.onnx` for turning the codes into sound
afterwards. Like the encoder they belong to `LFM2.5-Audio-1.5B`; the other checkpoints need their own
export, which LiquidAI's [onnx-export](https://github.com/Liquid4All/onnx-export) produces.

**Fix the depthformer's rotary embedding.** As published, the graph's attention nodes carry rotary
tables but do not apply them (`do_rotary` is unset), and the tables are computed for `theta` 10000
where the model uses 1,000,000. The first codebook of each frame sits at position zero and is
unaffected; the other seven come out with logits up to 2.5 away from the PyTorch model's. The speech
is intelligible either way, since the first codebook carries most of what is said, but only the
corrected graph gives the model's own codes: with both fixes every codebook agrees to 1e-5.

```python
import numpy as np
import onnx
from onnx import helper, numpy_helper

model = onnx.load("vocoder_depthformer.onnx")
for node in model.graph.node:
    if node.op_type == "GroupQueryAttention" and not any(a.name == "do_rotary" for a in node.attribute):
        node.attribute.append(helper.make_attribute("do_rotary", 1))
for table in model.graph.initializer:
    if table.name.endswith((".gqa_cos", ".gqa_sin")):
        positions, half = numpy_helper.to_array(table).shape
        angles = np.outer(np.arange(positions), 1.0 / 1_000_000 ** (np.arange(0, 2 * half, 2) / (2 * half)))
        values = np.cos(angles) if table.name.endswith("cos") else np.sin(angles)
        table.CopyFrom(numpy_helper.from_array(values.astype(np.float32), table.name))
onnx.save(model, "vocoder_depthformer.onnx")
```

**Add `audio_output` to `genai_config.json`, and take the two switch tokens out of `eos_token_id`.**
They end a text-only answer; here they are where speech begins, and the model is refused if they are
still stop tokens:

```json
{
    "model": {
        "eos_token_id": 7,
        "audio_output": {
            "depthformer": { "filename": "vocoder_depthformer.onnx" },
            "embedding": { "filename": "audio_embedding.onnx" }
        }
    }
}
```

`audio_output` also takes `num_codebooks` (8), `codebook_size` (2049, the last entry being the
end-of-audio code), `audio_start_token_id` (128), `text_end_token_id` (130), `interleaved_n_text` (6)
and `interleaved_n_audio` (12). The defaults are the published checkpoints'; set
`interleaved_n_audio` to 9 for `LFM2.5-Audio-1.5B-JP`.

**Generate.** Three search options steer the speech, next to the usual ones:

| Option | Default | Meaning |
| --- | --- | --- |
| `audio_interleaved` | `false` | Alternate text and speech by count, as `generate_interleaved` does. Leave it off for TTS, ASR and chat, which switch on `<|audio_start|>` alone. |
| `audio_temperature` | `1.0` | Temperature of the audio codes. `0` takes the likeliest code. The model card uses 0.8 for TTS and 1.0 for interleaved. |
| `audio_top_k` | `4` | Audio codes kept when sampling. `1` takes the likeliest code. The model card uses 64 for TTS and 4 for interleaved. |

The text is still decoded greedily, and `random_seed` seeds the audio codes as well.

```python
import numpy as np
import onnxruntime_genai as og

model = og.Model("./lfm2.5-audio/cpu")
processor = model.create_multimodal_processor()
tokenizer = og.Tokenizer(model)

prompt = (
    "<|startoftext|><|im_start|>system\nRespond with interleaved text and audio.<|im_end|>\n"
    "<|im_start|>user\n<|audio|><|im_end|>\n<|im_start|>assistant\n"
)
inputs = processor(prompt, audios=og.Audios.open("question.wav"))
prompt_length = inputs["input_ids"].as_numpy().shape[1]

params = og.GeneratorParams(model)
params.set_search_options(
    do_sample=False, max_length=prompt_length + 512, audio_interleaved=True, audio_temperature=1.0, audio_top_k=4
)
generator = og.Generator(model, params)
generator.set_inputs(inputs)
while not generator.is_done():
    generator.generate_next_token()

answer = generator.get_sequence(0)[prompt_length:]
audio_token_id, audio_start, text_end = 133, 128, 130
text = tokenizer.decode(np.array([t for t in answer if t not in (audio_token_id, audio_start, text_end)], np.int32))
codes = generator.get_output("audio_codes")  # [num_frames, 8] int64, 80 ms of speech per frame
```

Each audio frame takes one place in the sequence, held by `audio_token_id`, so `max_length` counts
frames as well as text tokens; twelve and a half frames make a second of speech. The frames themselves
come from `get_output("audio_codes")`, at any point during generation or after it, with the
end-of-audio frames left out.

**Turn the codes into sound** with `audio_detokenizer.onnx`, which gives the log-magnitude and phase
of a short-time Fourier transform, six columns per frame, and an inverse transform with the
reference's "same" padding. This graph needs no correction: it matches the PyTorch detokenizer to
1e-5.

```python
import onnxruntime as ort
import soundfile as sf

n_fft, hop = 1280, 320
detokenizer = ort.InferenceSession("audio_detokenizer.onnx")
# Only the first codebook carries the end-of-audio code on purpose; the detokenizer takes 0..2047.
features = detokenizer.run(None, {"audio_codes": np.minimum(codes, 2047).T[None]})[0][0]
spectrum = (np.exp(features[:, : n_fft // 2 + 1]) * np.exp(1j * features[:, n_fft // 2 + 1 :])).T

window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # torch.hann_window: periodic
frames = np.fft.irfft(spectrum, n_fft, axis=0) * window[:, None]
wave = np.zeros((frames.shape[1] - 1) * hop + n_fft)
envelope = np.zeros_like(wave)
for t in range(frames.shape[1]):
    wave[t * hop : t * hop + n_fft] += frames[:, t]
    envelope[t * hop : t * hop + n_fft] += window**2
pad = (n_fft - hop) // 2
sf.write("answer.wav", (wave[pad:-pad] / envelope[pad:-pad]).astype(np.float32), 24000)
```

**How it compares with the reference.** With the audio codes taken greedily on both sides
(`audio_top_k=1`), so that the two can be compared at all, this runtime reproduces liquid-audio
exactly: every text token and every audio frame of an interleaved answer to the reference's
`question.wav` (200 items, 149 of them frames), of an interleaved answer to a typed question, and of a
TTS sentence. With the model card's sampling the speech reads back through this runtime's own ASR as
the text the model wrote: *Red, blue, and yellow are the three primary colors. Would you like to hear
how they’re used in art?* comes back word for word, and *The quick brown fox jumps over the lazy dog.*
comes back exactly in TTS.

## 7. Known limitations

**The waveform is made outside the runtime.** Generation gives audio codes; the detokenizer and the
inverse transform that turn them into sound are the few lines of Python in
[Speech output](#6-speech-output), not part of the generation loop.

**Speech output is for one sequence.** A batch size of 1 and no beam search, as for speech input.
After an answer that ended in speech, a new turn appended to the same generator starts in text.

**One prompt at a time, decoded greedily.** Several clips in one prompt work, but batched prompts
and beam search (`num_beams` above 1) are refused; the reference decodes its text greedily too. The
clips do not share an encoder run: the published encoder export is traced for a single clip, so each
one is encoded on its own frames and the results are concatenated in prompt order.

**In ASR mode, two clips in one turn give one transcript.** Asked to transcribe a turn holding two
clips, the model writes out the last one and leaves the first, whichever order they come in. That is
the model and not this runtime: the reference implementation, given the same two clips through
`ChatState.add_audio`, produces the same tokens exactly, in both orders. Transcribe one clip per
request.

**LoRA adapters are not supported.** The builder refuses `adapter_path` for these checkpoints: an
adapter trained on them names the decoder `lfm.*`, which does not match the decoder the builder
loads. Merge the adapter into the checkpoint first.

**More than two channels has to be downmixed first.** Mono and stereo are handled: a stereo clip is
mixed down and reaches the front end at its true length. Wider audio is not — the decoder
deinterleaves it as though it were stereo, so a six channel clip arrives three times too long and
garbled, and transcribes to nonsense. Nothing in the decoder's interface reports how many channels a
file had, so this cannot be caught and refused here. Mix down to mono or stereo before passing the
file in.

**Audio is not chunked.** The whole clip goes through the encoder in one run, and one hour of audio
is 450k mel frames, so memory grows with the clip length. The reference implementation has the same
shape; for long-form transcription, split the audio yourself.

**The mel front end differs from the reference by about 1e-4.** It computes the same NeMo pipeline in
float32 where the reference uses float64 intermediates in places, and the FFT is a different
implementation. The difference is far below the audio's own quantization noise.

**Resampling is the reference's.** A clip at any other rate is decoded at its own rate and brought to
16 kHz with `torchaudio`'s resampler (Hann-windowed sinc), upwards as well as downwards, which is what
`ChatState.add_audio` does. The audio decoder's own resampler is a different filter and changes the
log-mel frames by up to about 2, enough to flip a token: on the reference's 24 kHz `asr_jp.wav` it
turned one homophone, and resampled this way the transcript matches token for token.

**Dither is off, as in the reference's eval mode.** NeMo adds 1e-5 of white noise to the samples
during training only; the runtime never does.
