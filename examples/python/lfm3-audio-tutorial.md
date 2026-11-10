# Build and run LFM3-Audio with ONNX Runtime GenAI

[LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) pairs an audio encoder with an LFM3 decoder. It transcribes speech and answers spoken questions in text. The model is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/main/LICENSE) (`lfm1.0`), not under this repository's MIT license; read it before you download or ship the model.

ONNX Runtime GenAI runs it as three ONNX models:

| Model | Made by | Inputs | Outputs |
| --- | --- | --- | --- |
| `audio_encoder.onnx` | the model builder | `input_features` `[1, 128, num_frames]` | `audio_features` `[ceil(num_frames / 8), 2048]` |
| `embeddings.onnx` | the model builder | `input_ids`, `audio_features` | `inputs_embeds` |
| `model.onnx` (decoder) | the model builder | `inputs_embeds`, `attention_mask`, KV and conv caches | `logits` |

The runtime computes the log-mel input itself (128 mel bands, a 400-sample window every 160 samples at 16 kHz) and runs the encoder once per clip, so one decoder token stands for 80 ms of audio.

## 1. Set up

onnxruntime-genai 0.19.0 and earlier cannot load `lfm3_audio` models. Until the next release, build the Python package from this repository ([build instructions](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)) and install the wheel it writes.

```bash
python3 -m venv lfm3-audio-venv
source lfm3-audio-venv/bin/activate
pip install onnxruntime_genai*.whl  # the wheel you built
pip install torch transformers==5.20.0 onnx onnx_ir huggingface_hub
git clone https://github.com/microsoft/onnxruntime-genai
```

No transformers release has the LFM3-Audio classes yet, so the builder loads the checkpoint's own Python code with `hf_remote=true`. That code does not load with transformers 5.17; 5.20.0 works.

## 2. Build the model

```bash
hf download LiquidAI/LFM3-Audio-2B --local-dir ./lfm3-audio-2b/pytorch

# CPU, fp32
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-audio-2b/pytorch -o ./lfm3-audio-2b/cpu -p fp32 -e cpu --extra_options exclude_embeds=true hf_remote=true

# CUDA, fp16
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-audio-2b/pytorch -o ./lfm3-audio-2b/cuda -p fp16 -e cuda --extra_options exclude_embeds=true hf_remote=true

# WebGPU, fp32
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-audio-2b/pytorch -o ./lfm3-audio-2b/webgpu -p fp32 -e webgpu --extra_options exclude_embeds=true hf_remote=true
```

`exclude_embeds=true` makes the builder write the speech encoder, the embedding model and the `embedding` and `speech` sections of `genai_config.json`, with `"type": "lfm3_audio"`. Without it you get a text-only LFM3 model. The decoder can be quantized (for example `-p int4`); the speech encoder and the embedding model never are. An int4 decoder changes most answers: on 20 prompts, 6 greedy answers matched PyTorch fp32 on CPU and on CUDA, and the mean KL divergence over the prefill reached 0.54.

The speech encoder is FP16 on CUDA and FP32 on CPU and WebGPU, where an FP16 encoder drifts from FP32.

On an H100 with the fp16 decoder and the fp16 encoder, 19 of 20 greedy answers matched PyTorch fp32: all 13 transcripts, while one chat answer picks another word at token 57. The builder writes `"cudnn_conv_algo_search": "HEURISTIC"` into the `cuda` provider options of the `speech` section: with cuDNN's default search, the encoder searches for convolution algorithms whenever the clip length changes, which delays the first token by about 0.1 s per clip and changes the tokens of 1 of 23 prompts.

## 3. Run the model

```python
import json

import onnxruntime_genai as og

model = og.Model("./lfm3-audio-2b/cpu")
tokenizer = og.Tokenizer(model)
processor = model.create_multimodal_processor()


def answer(messages, audio_paths, prefill="", max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True) + prefill
    inputs = processor(prompt, audios=og.Audios.open(*audio_paths))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=inputs["input_ids"].as_numpy().shape[1] + max_new_tokens)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    stream = tokenizer.create_stream()
    while not generator.is_done():
        generator.generate_next_token()
        print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
    print()


clip = ["onnxruntime-genai/test/audios/jfk.flac"]
# Speech to text
answer([{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [{"type": "audio"}]}], clip)
# A question about the clip, answered directly
question = [{"type": "audio"}, {"type": "text", "text": "What does the speaker ask people to do?"}]
answer([{"role": "user", "content": question}], clip, prefill="<think></think>")
```

The chat template writes `<|audio|>` for each `{"type": "audio"}` item, and the processor repeats it once per encoder output. The prompt must hold exactly one `<|audio|>` per clip, in clip order.

By default the model reasons inside `<think>...</think>` before it answers; the empty `<think></think>` after the prompt makes it answer directly.

To run the WebGPU folder, `pip install onnxruntime-ep-webgpu` and register the plug-in before you load the model: `import onnxruntime_ep_webgpu as webgpu` and `og.register_execution_provider_library(webgpu.get_ep_name(), webgpu.get_library_path())`.

Audio files are decoded by onnxruntime-extensions (WAV, FLAC, MP3), mixed down to mono and resampled to 16 kHz by the runtime. Its resampler differs from the soxr resampler of the Hugging Face example. On 9 clips not at 16 kHz, 5 transcripts equal the soxr reference's. The other 4 are copies of one recording. On the 3 copies at 44.1 and 48 kHz the first 32 tokens match, then the runtime writes "flour-fattened", as both do for the 16 kHz copy, where the reference writes "flour fatten". On the 8 kHz copy the transcripts differ from the first word, and the runtime's keeps the words of the 16 kHz copy.

## 4. What the runtime does not do

- Speech output. A prompt with `<|audio_out|>` is refused, and the tokens that switch to speech end the text answer.
- Audio after the first `set_inputs`. A later `set_inputs` with a clip fails inside ONNX Runtime and leaves the generator unusable, so discard it. Later turns can add text that starts with `<|im_end|>`, since the generator does not add the token that ended the previous answer.
- Batches above 1 and beam search. Both are refused when the generator is created.
- `rewind_to`. It is refused, as for the other LFM2 and LFM3 models.
- A limit on clip length. Memory grows with the clip.
- Long prompts in one WebGPU run. The prompt's logits must fit in about 4 GiB: about 8,400 tokens (11 minutes of audio) in fp32 and 16,800 in fp16. Set the `chunk_size` search option (for example 4096) to run longer prompts in parts.
- More than two channels. They cannot be detected, so such a file is not mixed down correctly.
- MP3 files that start with an ID3 tag. The audio decoder does not recognize them; remove the tag or convert the file.
- Execution providers other than CPU, CUDA and WebGPU, and `use_paged_attention=true`. The builder refuses them.
- Graph capture on WebGPU. ONNX Runtime refuses it for the LFM3 decoder, which keeps 6 attention-mask nodes and `/model/last_real_token/Cast` on the CPU.
- Window-sized caches. Every attention layer, the sliding-window ones included, keeps a KV cache of `search.max_length` tokens, so set `search.max_length` to the length you need.

On WebGPU, fp16 decoders drift from PyTorch: on 10 prompts, 8 or 9 greedy answers matched PyTorch's with fp16, depending on the torch version that built the export, and all 10 with fp32. LFM3 text models drift the same way.
