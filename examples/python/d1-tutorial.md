# Run d1-3B decisions with ONNX Runtime GenAI

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

## 3. Fix the tokenizer

The tokenizer in GenAI cannot parse the pre-tokenizer pattern in d1-3B's `tokenizer.json` yet. Replace
it with the equivalent pattern, as for LFM2.5-VL-3B. Run this once for every export folder:

```python
import json
from pathlib import Path

export = Path("./d1-3b/cpu")  # also ./d1-3b/cuda, ./d1-3b/cpu-vl and ./d1-3b/cuda-vl
path = export / "tokenizer.json"
tokenizer = json.loads(path.read_text(encoding="utf-8"))
old = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"
new = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
split = tokenizer["pre_tokenizer"]["pretokenizers"][0]
assert split["pattern"]["Regex"] == old
split["pattern"]["Regex"] = new
path.write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
```

## 4. Run a request

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

## 5. Request and response format

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

Before it runs the model, the script refuses a request with other fields in the request or a question,
an unknown `type`, missing `instructions`, bad `criteria`, a NUL character, a prompt longer than the
model's context, or images for a model without a vision model. The command line then prints one
`error: ...` line and exits with code 2.

## 6. Images

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

Then fix the tokenizer of each export as in [step 3](#3-fix-the-tokenizer).

The embedding model uses d1-3B's own embedding table (d1-3B retrained it) and puts the image features
where the image token (id 124907) is. `processor_config.json` uses d1-3B's bicubic resize. The vision
and embedding models are not quantized: they are float32 in the CPU export and float16 in the CUDA one,
and they run on the decoder's execution provider. The same steps work for a WebGPU export
(`-p fp32 -e webgpu`, with `onnxruntime-ep-webgpu` installed).

Then pass images with `--image`. Use `"state": null` when the images are the whole state. For example,
with the two-cats picture from the model card and a `cats.json` that asks one `choice`, "How many cats
are there?" (one, two, three or more):

```bash
curl -o cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
python d1-decisions.py -m ./d1-3b/cpu-vl -r cats.json --image cats.jpg
```

The answer is `two` (transformers gives it a probability of 0.985).

## 7. Limitations

- One prefill per question: with several questions, the state is processed again for each question.
- No image tiling: GenAI resizes each image once, while d1 cuts images above about 0.5 megapixels into
  tiles, so answers on large images differ. When an image side falls exactly halfway between two
  32-pixel steps (for example 400 pixels), GenAI rounds up and d1 rounds to the even step.
- No escaping: special tokens such as `<|im_end|>` in caller text are read as special tokens, as in d1
  and llama.cpp.
- A few rare whitespace runs (for example a line of only spaces) tokenize differently than in
  transformers.
- The script refuses some requests that d1's own code answers, for example an unknown `type`, a choice
  with one option, or a score with more than 10 levels.

## 8. Accuracy and speed

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
- POPE (images, int8 CPU pipeline from [Images](#6-images)): 998.

Use fp16 on CUDA, not bf16: bf16 gave fewer same top answers (BoolQ 994, MNLI 988) and more exact ties.
