# Run d1 decisions with ONNX Runtime GenAI

Sections 1 to 7 are about d1-3B. [Section 8](#8-d1-2b) covers text decisions with d1-2B.

## What d1-3B is

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) is a decision model. You give it a state (text,
JSON or images) and named questions, and it answers each question with probabilities. It writes no
text: each answer comes from one forward pass.

The model is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE).
Read it before you use the model. This tutorial and its script do not change its terms.

d1-3B is a stock `Lfm2VlForConditionalGeneration`, so the model builder exports it and GenAI runs it.
[`d1-decisions.py`](d1-decisions.py) adds the d1 part: it builds d1's prompt, runs one prefill per
question, reads the logits at the last position and turns them into option probabilities.

## 1. Install

```bash
pip install "onnxruntime-genai>=0.18.0"
pip install torch "transformers>=5.14" onnx onnx_ir huggingface_hub   # for the model builder
```

For an NVIDIA GPU, install `onnxruntime-genai-cuda` instead, in its own virtual environment (each
package brings its own ONNX Runtime). Version 0.18.0 needs the CUDA 13 runtime libraries and an NVIDIA
driver 580 or newer.

onnxruntime-genai 0.18.0 cannot read d1-3B's `tokenizer.json`, and it resizes some images one 32-pixel
step differently than d1 (for example a 400-pixel side). With 0.18.0, patch the file after each export
as in [step 3 of an earlier version of this tutorial](https://github.com/microsoft/onnxruntime-genai/blob/7f090db1731887b6a1b3f2715642df4ffe0e7863/examples/python/d1-tutorial.md#3-fix-the-tokenizer).

`d1-decisions.py` imports `common.py` from the same folder. Take both from the same commit of
[`examples/python`](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python):

```bash
curl -O https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/d1-decisions.py
curl -O https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/common.py
```

## 2. Build the model

```bash
hf download LiquidAI/d1-3B --local-dir ./d1-3b/pytorch

# CPU
python -m onnxruntime_genai.models.builder -i ./d1-3b/pytorch -o ./d1-3b/cpu -p int8 -e cpu \
    --extra_options prune_lm_head=true

# NVIDIA GPU (with onnxruntime-genai-cuda installed)
python -m onnxruntime_genai.models.builder -i ./d1-3b/pytorch -o ./d1-3b/cuda -p fp16 -e cuda \
    --extra_options prune_lm_head=true
```

`prune_lm_head=true` makes the model return logits for the last position only. Without it, a long
prompt needs gigabytes for logits that a decision never reads.

The int8 export logs `[ERROR] Gather only supports 4 bits quantization.` a few times; the export still
succeeds. int4 also works but is not recommended: it can flip the answer when two options are close.

## 3. Run a request

`request.json`:

```json
{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "App or site faults",
        "fraud": "Suspected unauthorised use"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Can wait", "Today", "Blocking the customer now"]
    }
  }
}
```

```bash
python d1-decisions.py -m ./d1-3b/cpu -r request.json
```

stdout holds only the JSON response; setup messages go to stderr. `-e` picks the execution provider, as
in the other examples. Output from the int8 CPU model (rounded):

```json
{
  "answers": {
    "refund": {"type": "noul", "noul": 0.9897},
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9834,
             "probabilities": {"billing": 0.9834, "technical": 0.0107, "fraud": 0.0059}},
    "urgency": {"type": "score", "score": 0.7519, "confidence": 0.5607,
                "probabilities": {"0": 0.3437, "1": 0.5607, "2": 0.0956},
                "legend": {"0": "Can wait", "1": "Today", "2": "Blocking the customer now"}}
  },
  "usage": {"input_tokens": 116, "output_tokens": 0}
}
```

## 4. Request and response format

The request and the answers have the same shape as d1's own `system_one()`.

| Field | Value |
| --- | --- |
| `state` | required: a string (used as is), any other JSON value (written as indented JSON), or `null` |
| `questions` | `{name: question}`, at least one |
| `type` | `noul` (yes or no), `choice` (the default) or `score` |
| `instructions` | the question text, a string |
| `criteria` | `noul`: optional `{"true": "...", "false": "..."}`; `choice`: `{name: description}`, 2 or more; `score`: a list of 2 to 10 levels, lowest first |

| Answer | Fields |
| --- | --- |
| `noul` | `noul`: P(yes) |
| `choice` | `choice`, `confidence` (its probability), `probabilities` `{name: p}` |
| `score` | `score` (the expected level), `confidence` (the top probability), `probabilities` `{"0": p0, ...}`, `legend` |

`usage.input_tokens` counts the shared part of the prompt once, plus each question, as d1 does.

Before it runs the model, the script refuses a request with other fields in the request or a question, an unknown `type`, a typed answer such as `box`, a choice with `"multiple": true`, missing `instructions`, bad `criteria`, a NUL character, a prompt longer than the model's context, or images for a model without a vision model. The command line then prints one `error: ...` line and exits with code 2.

## 5. Images

To answer questions about images, build d1-3B with `exclude_embeds=true`. The builder then also writes
the vision model, the embedding model and their configs; the [LFM2-VL tutorial](lfm2-vl-tutorial.md)
explains the parts. This is newer than onnxruntime-genai 0.18.0, so until the next release run the
builder from a clone of this repository (with a newer wheel, run `python -m onnxruntime_genai.models.builder`
with the same arguments):

```bash
git clone --depth 1 https://github.com/microsoft/onnxruntime-genai

# CPU
python onnxruntime-genai/src/python/py/models/builder.py -i ./d1-3b/pytorch -o ./d1-3b/cpu-vl -p int8 -e cpu \
    --extra_options exclude_embeds=true prune_lm_head=true

# NVIDIA GPU
python onnxruntime-genai/src/python/py/models/builder.py -i ./d1-3b/pytorch -o ./d1-3b/cuda-vl -p fp16 -e cuda \
    --extra_options exclude_embeds=true prune_lm_head=true
```

The embedding model uses d1-3B's own embedding table (d1-3B retrained it) and puts the image features
where the image token (id 124907) is. `processor_config.json` uses d1-3B's bicubic resize. The vision
and embedding models are not quantized: they are float32 in the CPU export and float16 in the CUDA one,
and they run on the decoder's execution provider. The same steps work for a WebGPU export
(`-p fp32 -e webgpu`, with `onnxruntime-ep-webgpu` installed). Use fp32 on WebGPU: the plug-in's fp16
kernels drift more. With the fp16 vision model, the image features of `cats.jpg` below have a smallest
per-token cosine similarity to transformers of 0.832 on WebGPU, against 0.990 for the same file on the CPU.

Then pass images with `--image`. Use `"state": null` when the images are the whole state. For example,
with the two-cats picture from the model card and a `cats.json` that asks one `choice`, "How many cats
are there?" (one, two, three or more):

```bash
curl -o cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
python d1-decisions.py -m ./d1-3b/cpu-vl -r cats.json --image cats.jpg
```

The answer is `two` (transformers gives it a probability of 0.986).

## 6. Limitations

- One prefill per question: with several questions, the state is processed again for each question.
- No image tiling: GenAI resizes each image once, while d1 cuts images above about 0.5 megapixels into
  tiles, so answers on large images differ.
- No escaping: special tokens such as `<|im_end|>` in caller text are read as special tokens, as in d1
  and llama.cpp.
- A few rare whitespace runs (for example a line of only spaces) tokenize differently than in
  transformers.
- The script refuses some requests that d1's own code answers, for example an unknown `type`, a choice
  with one option, or a score with more than 10 levels.

## 7. Accuracy and speed

Measured with `d1-decisions.py` (onnxruntime-genai 0.18.0, ONNX Runtime 1.31.0) against d1-3B in
transformers fp32, on 32 hand-written decisions (all three types; text, JSON, log and code states).
Times are for one question, with the default number of threads. The fp32 rows are `-p fp32` exports
(the CPU one without `prune_lm_head`); the WebGPU row used the `onnxruntime-ep-webgpu` plug-in.

| Machine, export | same top answer | largest probability difference | 39 / 101 / 467 tokens |
| --- | --- | --- | --- |
| Linux, CUDA (H100), fp16 | 32/32 | 0.0033 | 5.5 / 6.0 / 9.0 ms |
| Linux, CPU (Xeon 8480+, 26 vCPU), int8 | 32/32 | 0.0088 | 200 / 458 / 2039 ms |
| Linux, CPU, fp32 | 32/32 | 0.0000025 | 844 / 2037 / 9219 ms |
| macOS, WebGPU (M3 Ultra), fp32 | 32/32 | 0.0000068 | 88 / 110 / 296 ms |

On the first 1000 items of public sets, GenAI gave the same top answer as transformers on:

- BoolQ: 998 (fp16, CUDA) and 995 (int8, CPU);
- MNLI (matched): 997 (fp16, CUDA) and 995 (int8, CPU);
- POPE (images, built as in [Images](#5-images)): 1000 (fp16, CUDA) and 999 (int8, CPU).

Use fp16 on CUDA, not bf16: bf16 gave fewer same top answers (BoolQ 994, MNLI 988) and more exact ties.

## 8. d1-2B

[LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) is a d1 model built on LFM3-VL. It is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-2B/blob/main/LICENSE). Read it before you use the model. This tutorial and its script do not change its terms.

GenAI runs its text questions: `noul`, `choice` and `score`, each with one forward pass over its own prompt. Running the export needs onnxruntime-genai 0.19 or newer with ONNX Runtime 1.30 or newer. The builder needs `torchvision` and transformers 5.18 or newer, because it loads the model's own code, which fails on older transformers. The builder is newer than onnxruntime-genai 0.19.0, so until the next release run it from a clone of this repository (with a newer wheel, run `python -m onnxruntime_genai.models.builder` with the same arguments):

```bash
pip install "onnxruntime-genai>=0.19" torchvision "transformers>=5.18"
git clone --depth 1 https://github.com/microsoft/onnxruntime-genai
hf download LiquidAI/d1-2B --local-dir ./d1-2b/pytorch

# CPU
python onnxruntime-genai/src/python/py/models/builder.py -i ./d1-2b/pytorch -o ./d1-2b/cpu -p int8 -e cpu \
    --extra_options hf_remote=true prune_lm_head=true
cp ./d1-2b/pytorch/calibration.json ./d1-2b/cpu/

# NVIDIA GPU (with onnxruntime-genai-cuda installed)
python onnxruntime-genai/src/python/py/models/builder.py -i ./d1-2b/pytorch -o ./d1-2b/cuda -p fp16 -e cuda \
    --extra_options hf_remote=true prune_lm_head=true
cp ./d1-2b/pytorch/calibration.json ./d1-2b/cuda/

python d1-decisions.py -m ./d1-2b/cpu -r request.json
```

`hf_remote=true` runs Python code from the checkpoint, because only that code registers the model. With `request.json` from [section 3](#3-run-a-request), the int8 CPU model answers on the Xeon 8480+ (rounded; other CPUs differ in the third decimal):

```json
{
  "answers": {
    "refund": {"type": "noul", "noul": 0.9974},
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9639,
             "probabilities": {"billing": 0.9639, "technical": 0.0192, "fraud": 0.0168}},
    "urgency": {"type": "score", "score": 0.0, "confidence": 0.7769,
                "probabilities": {"0": 0.7769, "1": 0.204, "2": 0.0191},
                "legend": {"0": "Can wait", "1": "Today", "2": "Blocking the customer now"}}
  },
  "usage": {"input_tokens": 160, "output_tokens": 0}
}
```

The script picks these from the model type, `lfm3`, to match d1-2B's own code:

- Each question is a whole prompt. On d1-2B's 6 full-attention layers the state and the question read each other in both directions (GroupQueryAttention with `causal=0`), so a state is processed again for each question and `usage.input_tokens` counts it each time. The other 18 layers stay causal.
- The prompt ends in an empty reasoning block, `<think></think>`.
- Special tokens in caller text are escaped with a zero-width space, so `<|im_end|>` in a state cannot close the turn.
- Option scores are divided by the temperature in `calibration.json` (1.337 for d1-2B). The script refuses to run without that file.
- `score` is the first level whose cumulative probability reaches one half, not the expected level.
- A choice with more than 26 options uses the codes AA, AB and so on.

What this does not do:

- A choice with `"multiple": true` is refused: it needs one answer slot per option, and this script reads one position per question. The `text`, `number`, `date`, `box` and `boxes` types are refused for the same reason.
- No images. They need the `lfm3_vl` pipeline, which GenAI does not have yet.
- One question per pass. d1's own code reads up to 16 questions in one pass (`d1_set_size` in its `config.json`), and its model card says larger sets are faster and cost some accuracy on text; only one question per pass was compared here.
- WebGPU, JS and DirectML are refused by the builder, and `d1-decisions.py` refuses `-e WebGpuExecutionProvider`, because `causal=0` runs on CPU and CUDA only. ONNX Runtime 1.29 and older fail to load the model.
- Do not generate text with this export: its full-attention layers read later tokens, so it is for one forward pass over a whole prompt.

Measured with `d1-decisions.py` (ONNX Runtime 1.31.0) against d1-2B's own `system_one(state, questions, set_size=1)` in transformers fp32, at revision `97c57047`, on 49 questions: the 32 hand-written decisions of section 7, 10 questions asked together over 3 states, 3 states or questions that hold special tokens, one with 30 options, one with no state, and 2 over a state of 2,600 tokens. Token ids were equal on all 49. States with long runs of whitespace are a known gap: GenAI's tokenizer can split them differently from Hugging Face's. On 60 such questions the ids differed on 12, probabilities moved by up to 0.085 and 2 top answers changed. [onnxruntime-extensions #1138](https://github.com/microsoft/onnxruntime-extensions/pull/1138) fixes most of these; 12 or more newlines in a row and U+0085 next to punctuation still differ with it. Times are for one question at a length already seen, with the default number of threads.

| Machine, export | same top answer | largest probability difference | 42 / 289 / 4189 tokens |
| --- | --- | --- | --- |
| Linux, CUDA (H100), fp16 | 48/49 | 0.0015 | 5.0 / 6.6 / 37.7 ms |
| Linux, CPU (Xeon 8480+, 26 vCPU), fp32 | 49/49 | 0.0000046 | 179 / 856 / 14437 ms |
| Linux, CPU, int8 | 49/49 | 0.030 | 98 / 562 / 12065 ms |
| macOS, CPU (M3 Ultra), fp32 | 49/49 | 0.0000014 | 132 / 637 / 11316 ms |

The one different answer on fp16 is a `noul` at 0.4999 in transformers and 0.5015 here. On CUDA the first prompt at a new length takes about 170 ms, because cuDNN plans its attention for that length. If your prompts vary in length, set `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0`: a first prompt then takes 14 to 44 ms, the times above stay within 1 ms, and the same top answers hold on 48/49 (largest difference 0.0045).

An export without `causal=0` gives the same top answer on only 44 of the 49, with probabilities up to 0.60 apart.
