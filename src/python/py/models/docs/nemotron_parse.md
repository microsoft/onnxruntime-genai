# Nemotron Parse

[Model builder documentation](../README.md)

Nemotron Parse exports two graphs constructed directly with the common ONNX IR
builder: a static RADIO encoder that also produces cross-attention K/V caches,
and one mBART decoder used for both prompt processing and token generation. The
decoder emits the standard main-domain `TensorScatter` operator from opset 24.
Both graph components reuse `Model` for weight serialization, quantization,
MatMul/bias construction, and shape/operator wrappers. Their linear helpers only
compose the shared MatMul and bias builders. Standalone LayerNormalization keeps
each source module's epsilon and tensor shape without the base decoder's residual
state bookkeeping. RADIO's packed QKV, patch geometry, and compression neck, plus
mBART's absolute positions, cross-attention, and fixed TensorScatter cache, remain
model-specific; they do not use the base decoder's fused-attention/cache pipeline.

The exported package declares `model.default_user_prompt` in `genai_config.json`
as `</s><s><predict_bbox><predict_classes><output_markdown>`. The multimodal examples
use this optional string when `--user_prompt` is omitted in non-interactive mode.
Explicit nonempty prompts are preserved; interactive empty input asks again.
Packages without the setting retain the examples' conversational default.
The exported `chat_template.jinja` passes through only the
current user message's text, leaving special-token handling to the processor.
Re-export existing packages to use these metadata-driven example defaults.

## Model Configuration

Both `encoder.onnx` and `decoder.onnx` are always exported. No model-specific
extra options are needed. Image dimensions come from the checkpoint's
`image_size`, and cache capacity comes from its
`max_sequence_length`.

The resolved settings travel with the package:

- `vision_processing.json`: `image_height`, `image_width`, and normalization settings. The native processor validates the dimensions against the encoder's `pixel_values` shape.
- `genai_config.json`: `model.context_length` records cache capacity, and `model.decoder.prefill_sequence_length` records the default task prompt's token count, including decoder-start and tokenizer BOS/EOS tokens. The latter selects the static TRT-RTX fast path; other prompt lengths use dynamic prefill.

Checkpoint loading precision follows the shared `-p` export precision:
FP32, FP16, and BF16 exports load at that precision, while INT4 exports use
`auto` to preserve the checkpoint dtype. No separate loading-dtype option is needed.

Image resolution and cache capacity are baked into the ONNX graphs. Changing
them requires re-export, not just editing the package's JSON files.

Supported sampling defaults, including `repetition_penalty`, are preserved from
the checkpoint's generation configuration. Batch size and beam count remain one.
The native image processor supports the checkpoint's bilinear aspect-preserving
resize, centered constant white padding, rescaling by 1/255, and CLIP normalization.
Export rejects incompatible source settings or transforms, including padding
library versions that silently change white padding to black.

INT4 export uses the model builder's standard quantization options. The shared
builder automatically enables Q/DQ for TRT-RTX, and Nemotron Parse defaults to
block size 32, emitting the `DequantizeLinear -> MatMul` weight-only pattern.

## Export

For models that provide custom Hugging Face code, explicitly set
`hf_remote=true` only after verifying and trusting that code. For example, the
following exports an INT4 package using the checkpoint's resolution and cache
capacity. Run the source export command from `src/python/py/models`; run the
example and validation commands below from the repository root.

```bash
# From wheel:
python -m onnxruntime_genai.models.builder -i path_to_nemotron_parse_model -o path_to_output_folder -p int4 -e NvTensorRtRtx --extra_options hf_remote=true

# From source:
python builder.py -i path_to_nemotron_parse_model -o path_to_output_folder -p int4 -e NvTensorRtRtx --extra_options hf_remote=true
```

Remove `image_height`, `image_width`, `cache_sequence_length`,
`prefill_sequence_length`, `export_components`, and `torch_dtype` from older
`--extra_options` commands; these model-specific overrides are no longer accepted.
Exports now use the checkpoint's full cache capacity rather than an example's
reduced capacity, so memory usage can increase compared with a previous
1032-token export. A lower runtime `--max_length` limits generation, not the
exported static cache allocation.

## Run

Run the exported package through the shared multimodal example:

```bash
python examples/python/model-mm.py -m path_to_output_folder --image_paths document.png --non_interactive
```

When `--max_length` is omitted, the Python multimodal example uses the smaller of
`7680` and the package's configured search maximum. Explicit values are passed
through to the runtime's normal validation.

## Prompt Lengths

For Nemotron Parse, omitting `--user_prompt` in non-interactive mode uses
`</s><s><predict_bbox><predict_classes><output_markdown>`. The processor enables
the tokenizer's `add_special_tokens` option and prepends the decoder-start token,
making the default task eight input tokens. Custom task prompts are tokenized the same way; count tokens,
not characters or words. All prompts must be shorter than `context_length` to
leave room for generation.

The examples retain their existing empty-input retry behavior: do not pass
`--user_prompt ""` in non-interactive mode. Direct processor API calls still
accept empty text to select the default task.

CPU/CUDA sessions accept shorter and longer prompts, independently of
`prefill_sequence_length`. TRT-RTX eagerly creates three sessions from the same
decoder ONNX file, with batch size fixed to 1 and cross-attention image-token
count fixed to the encoder output size: static prefill at `prefill_sequence_length`, dynamic
prefill for other lengths, and static one-token decode. The dynamic profile spans
`1` through `context_length - 1`, with `prefill_sequence_length` as its optimum.
The processor rejects prompts outside this range before image preprocessing;
the runtime also validates callers that bypass the processor. No automatic
padding or truncation is performed. A package configured with
`prefill_sequence_length=16` still accepts the default eight-token task through
dynamic prefill, while 16-token prompts use the static fast path.

All sessions are created during model loading, so custom prompts do not trigger
session creation on the first request. The extra session increases model startup
time and memory usage. Custom prompts accept dynamic-prefill latency; the static
prefill and one-token decode fast paths retain their shape specialization. This
supports varying prompt length, not arbitrary batch or image shapes. Manually
specialized ONNX files with a fixed sequence dimension remain restricted to that
dimension and must be re-exported to support varying lengths.

## Preprocessing

Nemotron Parse exports `vision_processing.json`, referenced by
`model.vision.config_filename`. It contains the resolved image dimensions, RGB
decoding pipeline, and validated `image_mean` and `image_std` arrays used by native preprocessing,
including normalization of white padding. Each array must contain three finite
numbers, and standard deviations must be positive. Packages exported before
these settings were included must regenerate their processing config; the
runtime does not fall back to hardcoded normalization values.

## Runtime Limits

The processor accepts a string or a singleton prompt list. An empty string,
`[""]`, or `[]` selects the default task; multi-item lists are rejected.
Rewind, adapters, and `nv_multi_profile_enable=1` are not
supported. Runtime options are forwarded to both encoder and decoder sessions.
After an inference error or interrupted inference, discard the generator and
create a new one; partially written in-place caches cannot safely be reused.

## Validation

From a source checkout with the native Python bindings built and importable:

```bash
python -m pytest test/python/models/test_nemotron_parse_prompts.py test/python/builder/test_nemotron_parse.py -q
# Require CUDA runtime coverage on a CUDA-enabled build:
NEMOTRON_PARSE_REQUIRE_CUDA=1 python -m pytest test/python/models/test_nemotron_parse_prompts.py -k in_place_cache -q
# Require TRT-RTX INT4 compilation, placement, and numerical coverage:
NEMOTRON_PARSE_REQUIRE_TRT_RTX=1 python -m pytest test/python/models/test_nemotron_parse_prompts.py test/python/builder/test_nemotron_parse.py -k trt_rtx -q
```

For a plugin TRT-RTX build, set `ORT_TRT_RTX_EP_LIBRARY` to its library before
running the last command. The target-provider tests disable CPU fallback; missing
required providers fail instead of silently passing on CPU. Pixel-level tests
also require Pillow and OpenCV. GPU-specific tests skip on ordinary CPU runs.
The cache regression covers FP16 and FP32 on CPU/CUDA, checking both past/present
buffers and untouched cache slots after prefill and five decode steps. It also
checks profiling evidence for all twelve key/value TensorScatter executions on
the requested provider. CUDA validation requires a CUDA-enabled GenAI build and
CUDA-enabled ONNX Runtime with drivers compatible with its compiled kernels.
The TRT-RTX tests bind each past/present cache pair to the same GPU allocation
and carry the target's static or dynamic prefill cache into decode; both logits
and caches are compared against the FP32 reference. Native runtime tests alternate
static and dynamic prompts on one model, cover profile boundaries and non-default
fast-path lengths, and verify cache contents through decode. On GB10 with the legacy TRT-RTX 1.6.1.75
bundle, also set `__LUNOWUD=-l2cm:enable=off` for the test process: that bundle's
L2CM promotion kernel lacks an sm_121 image. This is a runtime workaround, not
a change to the exported model.
