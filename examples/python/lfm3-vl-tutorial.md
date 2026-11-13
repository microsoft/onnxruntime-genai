# Build and run LFM3-VL with ONNX Runtime GenAI

[LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) pairs a vision tower of 8 depthwise convolution layers and 6 attention layers with an LFM3 decoder. The model is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/LICENSE), not under this repository's MIT license; read it before you download or ship the model.

ONNX Runtime GenAI runs it as three ONNX models, which the model builder writes from the Hugging Face checkpoint: `vision.onnx` turns one image's 16x16 patches into one token per 2x2 block of patches, `embeddings.onnx` places those tokens at the image's `<|image|>` positions, and `model.onnx` is the decoder. The runtime runs the vision model once per image.

## 1. Set up

onnxruntime-genai 0.19.0 and earlier cannot load `lfm3_vl` models. Until the next release, build the Python package from this repository ([build instructions](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)) and install the wheel it writes.

```bash
python3 -m venv lfm3-vl-venv
source lfm3-vl-venv/bin/activate
pip install onnxruntime_genai-*.whl  # the wheel you built; onnxruntime_genai_cuda-*.whl for a CUDA build
pip install torch transformers==5.20.0 onnx onnx_ir huggingface_hub pillow
git clone https://github.com/microsoft/onnxruntime-genai
```

No transformers release has the LFM3-VL classes yet, so the builder loads the checkpoint's own Python code with `hf_remote=true`. That code does not load with transformers 5.15 to 5.17; 5.20.0 works.

## 2. Build the model

```bash
hf download LiquidAI/LFM3-VL-2B --local-dir ./lfm3-vl-2b/pytorch

# CPU, fp32
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-vl-2b/pytorch -o ./lfm3-vl-2b/cpu -p fp32 -e cpu --extra_options exclude_embeds=true hf_remote=true

# CUDA, fp16
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-vl-2b/pytorch -o ./lfm3-vl-2b/cuda -p fp16 -e cuda --extra_options exclude_embeds=true hf_remote=true

# WebGPU, fp32
python3 onnxruntime-genai/src/python/py/models/builder.py -i ./lfm3-vl-2b/pytorch -o ./lfm3-vl-2b/webgpu -p fp32 -e webgpu --extra_options exclude_embeds=true hf_remote=true
```

`exclude_embeds=true` makes the builder write the vision and embedding models, `processor_config.json` and their sections in `genai_config.json`, with `"type": "lfm3_vl"`. Without it you get a text-only LFM3 model. The decoder can be quantized (for example `-p int4`); the vision and embedding models never are. The vision model uses the decoder's precision, except fp16 for a bf16 decoder and fp32 on WebGPU, where fp16 matrix products lose too much precision.

## 3. Run the model

```bash
python3 onnxruntime-genai/examples/python/model-mm.py -m ./lfm3-vl-2b/cpu -e follow_config --image_paths onnxruntime-genai/test/images/australia.jpg --user_prompt "Describe this image." --non_interactive
```

The chat template writes `<|vision_start|><|image|><|vision_end|>` for each image. The image processor resizes each image so both sides are multiples of 32 pixels and it gives 64 to 16,384 tokens, then repeats its `<|image|>` once per token. The prompt must hold exactly one `<|image|>` per image, in image order.

## 4. What the runtime does not do

- Video. A prompt with `<|video|>` is refused.
- Batches above 1 and beam search. Both are refused when the generator is created.
- `rewind_to`. It is refused, as for the other LFM2 and LFM3 models.
- Tiling. A large image is shrunk to at most 16,384 tokens instead.
- Aspect ratios above 200:1. The image processor refuses them.
- Images with a side above 16,384 pixels. onnxruntime-extensions refuses to decode them.
- Execution providers other than CPU, CUDA and WebGPU, and `use_paged_attention=true`. The builder refuses them.
- Window-sized caches. Every attention layer, the sliding-window ones included, keeps a KV cache of `search.max_length` tokens, so set `search.max_length` to the length you need.
- Long prompts in one WebGPU run. The prompt's logits must fit in about 4 GiB: about 8,000 tokens in fp32 and 16,000 in fp16, and one large image can give 16,384. Set the `chunk_size` search option (for example 4096) to run longer prompts in parts.

On WebGPU, fp16 decoders drift from CPU: on 3 image prompts, 88 of 192 greedy tokens matched the CPU's. LFM3 text models drift the same way. fp32 matched all 192.

Images 16 pixels or less on the short side and longer than about 2,048 get fewer tokens than the Hugging Face processor gives them (2590x14: 81 against 109), because onnxruntime-extensions raises the short side to 32 pixels before it checks the area.

JPEG images are decoded by onnxruntime-extensions, not PIL: on a noisy 1280x960 test photo, 43% of the values differ from PIL's, by up to 22 of 255 levels. Use PNG images when you compare with PyTorch. The bicubic resize differs from the Hugging Face processor's by at most 2 of 255 levels.
