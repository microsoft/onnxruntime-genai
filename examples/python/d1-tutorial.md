# Run d1-3B decisions with ONNX Runtime GenAI

## What d1-3B is

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) is a decision model. You give it a state (text,
JSON or images) and named questions. It answers each question with probabilities. It does not write
text: each answer comes from one forward pass with zero output tokens.

d1-3B is a stock `Lfm2VlForConditionalGeneration`, the same architecture as LFM2.5-VL-3B. So the model
builder already exports it and ONNX Runtime GenAI already runs it. [`d1-decisions.py`](d1-decisions.py)
adds the d1 part. For each question it:

1. renders d1's prompt;
2. runs one prefill;
3. reads the logits at the last position;
4. takes, for each option, the best log-probability among the option's single-token forms (for
   example `A` and ` A`);
5. applies a softmax over the options.

There are three question types:

| `type` | Meaning | `criteria` |
| --- | --- | --- |
| `noul` | yes or no | optional: `{"true": "...", "false": "..."}` |
| `choice` | one of named options (the default type) | `{name: description}`, 2 or more options |
| `score` | one of 2 to 10 ordered levels | a list of level descriptions, lowest first |

## Steps

1. [Install](#1-install)
2. [Build the model](#2-build-the-model)
3. [Fix the tokenizer](#3-fix-the-tokenizer)
4. [Run a request](#4-run-a-request)
5. [Request and response format](#5-request-and-response-format)
6. [Images](#6-images)
7. [Limitations](#7-limitations)
8. [Accuracy and speed](#8-accuracy-and-speed)

## 1. Install

```bash
pip install "onnxruntime-genai>=0.18.0"
# The model builder also needs these. huggingface_hub gives the `hf` command.
pip install torch "transformers>=5.14" onnx onnx_ir huggingface_hub
```

For an NVIDIA GPU, install `onnxruntime-genai-cuda` instead of `onnxruntime-genai`, in its own virtual
environment. Do not install both in one environment: each one brings its own ONNX Runtime package. The
CUDA package of onnxruntime-genai 0.18.0 needs the CUDA 13 runtime libraries (for example `libcudart`
and `libcublas`) and an NVIDIA driver 580 or newer. `og.is_cuda_available()` can return `True` even
when these libraries cannot be loaded.

```bash
pip install "onnxruntime-genai-cuda>=0.18.0"
pip install torch "transformers>=5.14" onnx onnx_ir huggingface_hub
```

`d1-decisions.py` imports `common.py` from the same folder. Both are in
[`examples/python`](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python). Clone the
repository, or download the two files into one folder:

```bash
curl -O https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/d1-decisions.py
curl -O https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/common.py
```

The examples can use API from newer onnxruntime-genai versions than the one you have. Replace `main`
with a release tag or a commit that matches your installed version, or upgrade the package. The script
was tested with onnxruntime-genai 0.18.0.

## 2. Build the model

```bash
hf download LiquidAI/d1-3B --local-dir ./d1-3b/pytorch

# Recommended on CPU
python -m onnxruntime_genai.models.builder \
    -i ./d1-3b/pytorch \
    -o ./d1-3b/cpu \
    -p int8 \
    -e cpu \
    --extra_options prune_lm_head=true

# On an NVIDIA GPU
python -m onnxruntime_genai.models.builder \
    -i ./d1-3b/pytorch \
    -o ./d1-3b/cuda \
    -p fp16 \
    -e cuda \
    --extra_options prune_lm_head=true
```

The builder writes `"type": "lfm2_vl_text"` into `genai_config.json`: a text-only model. See
[Images](#6-images) for the vision model.

Both exports were measured with `d1-decisions.py` against transformers; see
[Accuracy and speed](#8-accuracy-and-speed). Build the CUDA export with the CUDA package installed (see
[Install](#1-install)).

The int8 export logs `[ERROR] Gather only supports 4 bits quantization.` a few times. You can ignore
it: the export still succeeds and keeps the embedding table in fp32.

int4 (`-p int4`) also works, but it is not recommended. It changes the probabilities more than int8, and
it can flip the answer when two options are close.

**Why `prune_lm_head=true`.** Without it, the model returns logits for every prompt position. GenAI
stores them as fp32 `[batch, prompt length, vocabulary]`. d1-3B has a 128,000-token vocabulary and a
32,768-token context, so a full-length prompt needs about 16.8 GB for the logits alone. With
`prune_lm_head=true` the model returns only the last position, which is all a decision needs.

**Set `max_length` for each request.** The builder sets the default `max_length` to the full context,
32,768 tokens. On CPU, GenAI allocates the KV cache for `max_length` up front: about 1 GiB per sequence
for d1-3B. `d1-decisions.py` sets `max_length` to the exact prompt length of each question.

## 3. Fix the tokenizer

d1-3B's `tokenizer.json` uses the `'(?i:[sdmt]|ll|ve|re)|...` pre-tokenizer pattern. The tokenizer in
onnxruntime-extensions cannot parse it yet ("Invalid regex"). Replace it with the equivalent pattern,
as for LFM2.5-VL-3B (see [Known limitations](lfm2-vl-tutorial.md#5-known-limitations) in the LFM2-VL
tutorial).

Run this once for every export folder (`./d1-3b/cpu`, `./d1-3b/cuda`, `./d1-3b/cpu-vl`): set `export`
to the folder.

```python
import json
from pathlib import Path

export = Path("./d1-3b/cpu")
path = export / "tokenizer.json"
tokenizer = json.loads(path.read_text(encoding="utf-8"))
old = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"
new = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
split = tokenizer["pre_tokenizer"]["pretokenizers"][0]
assert split["pattern"]["Regex"] == old
split["pattern"]["Regex"] = new
path.write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
```

To check that GenAI gives the same token ids as transformers on your own data:

```python
import onnxruntime_genai as og
from transformers import AutoTokenizer

text = "<|startoftext|><|im_start|>user\nWE'LL be there, it's   fine.\n\n\nQUESTION:\n"
reference = AutoTokenizer.from_pretrained("./d1-3b/pytorch")
tokenizer = og.Tokenizer(og.Model("./d1-3b/cpu"))
assert tokenizer.encode(text).tolist() == reference.encode(text, add_special_tokens=False)
```

The ids match on d1's JSON prompts. A few rare whitespace runs still tokenize differently in GenAI, with
either pattern: for example a line that holds only spaces (`"\n  \n  x"`), or U+0085 or U+2028 next to
a newline. Check free-text states of this kind before you rely on them.

## 4. Run a request

Write a request file, `request.json`:

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

Run it:

```bash
python d1-decisions.py -m ./d1-3b/cpu -r request.json
```

stdout holds only the JSON response, so you can pipe it (for example to `jq`). Setup messages go to
stderr. When the script refuses a request (see [Request and response format](#5-request-and-response-format)),
it prints one `error: ...` line to stderr and exits with code 2.

Output from the int8 CPU model:

```json
{
  "answers": {
    "refund": {
      "type": "noul",
      "noul": 0.9897252074365641
    },
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9834051677559257,
      "probabilities": {
        "billing": 0.9834051677559257,
        "technical": 0.010658739574589455,
        "fraud": 0.005936092669484753
      }
    },
    "urgency": {
      "type": "score",
      "score": 0.7519013517205196,
      "confidence": 0.5606897534658334,
      "probabilities": {
        "0": 0.34370444740682343,
        "1": 0.5606897534658334,
        "2": 0.09560579912734314
      },
      "legend": {
        "0": "Can wait",
        "1": "Today",
        "2": "Blocking the customer now"
      }
    }
  },
  "usage": {
    "input_tokens": 116,
    "output_tokens": 0
  }
}
```

The numbers change a little with the precision and the hardware. `-e` picks the execution provider,
as in the other examples.

You can also call it from Python. `d1-decisions.py` has a dash in its name, so load it by path (give
the real path to the file if you run from another folder):

```python
import importlib.util

import onnxruntime_genai as og

spec = importlib.util.spec_from_file_location("d1_decisions", "d1-decisions.py")
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)

decisions = d1.D1Decisions(og.Model("./d1-3b/cpu"), d1.model_context_length("./d1-3b/cpu"))
refund = {"type": "noul", "instructions": "Is this a refund request?"}
print(decisions.decide({"state": "Where is my parcel?", "questions": {"refund": refund}}))
```

## 5. Request and response format

The request and the answers have the same shape as d1's own `system_one()`. Other servers that run d1
(for example llama.cpp) use a close but not identical format: they can take images in the request,
they may require `type`, and llama.cpp reports a different `confidence` (see below).

| Request field | Value |
| --- | --- |
| `state` | required. A string (used as is), any other JSON value (written as indented JSON), or `null` (no state, for example when the images are the whole state) |
| `questions` | `{name: question}`, at least one question |
| `questions.<name>.type` | `noul`, `choice` or `score`; `choice` when missing |
| `questions.<name>.instructions` | the question text, a string |
| `questions.<name>.criteria` | see the table in [What d1-3B is](#what-d1-3b-is) |

A question has no other fields.

| Answer `type` | Fields |
| --- | --- |
| `noul` | `noul`: P(yes) |
| `choice` | `choice`: the most likely name; `confidence`: its probability; `probabilities`: `{name: p}` |
| `score` | `score`: the expected level, sum of `i * p_i`; `confidence`: the top probability; `probabilities`: `{"0": p0, ...}`; `legend`: `{"0": text0, ...}` |

A request has no other fields. Images come from `--image` (or `image_paths` in Python), not from the
request.

`confidence` is the top probability, as in d1. llama.cpp reports a normalized value instead, so the
two differ for the same probabilities. For example, three options with a top probability of 0.5 give
0.5 here and 0.25 in llama.cpp.

`usage.input_tokens` counts tokens the way d1 does: the shared part of the prompt (images and state)
once, plus each question. `usage.output_tokens` is always 0. With several questions, GenAI does more
work than `input_tokens` shows, because it reads the state again for each question.

The script refuses a request before it runs the model, with a `ValueError`, when:

- the request is not an object, or has no `state` field;
- the request has a field other than `state` and `questions`, for example `images`, `files`, `videos`
  or a misspelled `State`;
- the state is a list of chat messages with an `image_url`, `input_audio` or `video_url` part;
- `questions` is not an object with at least one question;
- a question is not an object, has a field other than `type`, `instructions` and `criteria`, has an
  unknown `type`, or has no string `instructions`;
- a `choice` has fewer than 2 options, or a `score` does not have 2 to 10 levels;
- `noul` criteria are not an object, or have keys other than `true` and `false` (one of the two may be
  left out; it is then shown as `None`, as in d1);
- the state or a question holds a NUL character (`\x00`), which would cut the prompt short (a JSON
  state is safe: it writes NUL as `\u0000`);
- the tokenizer has no single-token `yes` or `no`, no single-token digit for a score level, or no
  single-token code left for a choice option;
- the prompt does not start with exactly one `<|startoftext|>` token (a tokenizer that adds its own);
- the prompt is longer than the model's `context_length`;
- images are given but the model has no vision model, or `image_paths` is one string instead of a
  list.

Where the rules come from:

- a score has 2 to 10 levels: the d1-3B model card;
- a choice has at least 2 options: llama.cpp's d1 server.

Some refusals are stricter than d1's own code, which answers these requests:

- an unknown `type` (d1 treats it as a choice);
- `instructions` that are not a string (d1 formats any value into the prompt);
- a choice with one option;
- a score with 1 level, or with more than 10 levels (d1 accepts up to 1000);
- unknown question fields and unknown `noul` criteria keys (d1 ignores them);
- a NUL character (d1's tokenizer reads the whole prompt).

Caller text is not escaped, the same as in d1 and llama.cpp. If a state or a question contains special
tokens such as `<|im_end|>`, the tokenizer reads them as special tokens.

## 6. Images

To answer questions about images, run d1-3B as an `lfm2_vl` pipeline with a vision model and an
embedding model. The [LFM2-VL tutorial](lfm2-vl-tutorial.md) explains the parts. These steps are for
the CPU export: the embedding graph below takes and returns float32 tensors, which matches a CPU
decoder but not an fp16 one such as the CUDA build above. For d1-3B:

1. Build the decoder with `exclude_embeds=true`:

   ```bash
   python -m onnxruntime_genai.models.builder \
       -i ./d1-3b/pytorch \
       -o ./d1-3b/cpu-vl \
       -p int8 \
       -e cpu \
       --extra_options exclude_embeds=true prune_lm_head=true
   ```

   Then fix `tokenizer.json` in `./d1-3b/cpu-vl` as in [step 3](#3-fix-the-tokenizer).

2. Reuse the vision model of LFM2.5-VL-3B. d1-3B's vision tower and projector are bit-identical to
   LFM2.5-VL-3B's. Download `onnx/vision_encoder.onnx` and its four `onnx/vision_encoder.onnx_data*`
   files from [LiquidAI/LFM2.5-VL-3B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-ONNX) into
   `./d1-3b/cpu-vl`. The command pins the revision that was checked:

   ```bash
   hf download LiquidAI/LFM2.5-VL-3B-ONNX --revision 020f9311343b8f3c6d9789b4dd300e749ec52cd1 \
       --include "onnx/vision_encoder.onnx*" --local-dir ./lfm2.5-vl-3b-onnx
   cp ./lfm2.5-vl-3b-onnx/onnx/vision_encoder.onnx* ./d1-3b/cpu-vl/
   ```

3. Build `embeddings.onnx` from d1-3B's own embedding table. Do not reuse the LFM2.5-VL-3B one: d1-3B
   retrained it. The graph looks up `input_ids` and writes `image_features` into the positions that
   hold the image token (id 124907):

   ```python
   import numpy as np
   import onnx
   from onnx import TensorProto, helper, numpy_helper
   from safetensors import safe_open

   hidden_size, image_token_id = 2048, 124907
   with safe_open("./d1-3b/pytorch/model.safetensors", "pt") as f:
       table = f.get_tensor("model.language_model.embed_tokens.weight").float().numpy()


   def const(name, value):
       return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(value, np.int64)))


   nodes = [
       helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["embeds"], axis=0),
       helper.make_node("Shape", ["embeds"], ["embeds_shape"]),
       const("rows", [-1, hidden_size]),
       helper.make_node("Reshape", ["embeds", "rows"], ["flat_embeds"]),
       const("image_token_id", image_token_id),
       helper.make_node("Equal", ["input_ids", "image_token_id"], ["is_image"]),
       const("flat", [-1]),
       helper.make_node("Reshape", ["is_image", "flat"], ["flat_is_image"]),
       helper.make_node("NonZero", ["flat_is_image"], ["positions"]),
       helper.make_node("Transpose", ["positions"], ["indices"], perm=[1, 0]),
       helper.make_node("ScatterND", ["flat_embeds", "indices", "image_features"], ["flat_out"]),
       helper.make_node("Reshape", ["flat_out", "embeds_shape"], ["inputs_embeds"]),
   ]
   graph = helper.make_graph(
       nodes,
       "d1_embedding",
       [
           helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
           helper.make_tensor_value_info("image_features", TensorProto.FLOAT, ["num_image_tokens", hidden_size]),
       ],
       [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])],
       initializer=[numpy_helper.from_array(table, "embed_tokens.weight")],
   )
   model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
   onnx.save(model, "./d1-3b/cpu-vl/embeddings.onnx", save_as_external_data=True, location="embeddings.onnx.data")
   ```

4. Add the `embedding` and `vision` sections to the `genai_config.json` the builder wrote, and write
   `processor_config.json`. Keep everything else the builder wrote. The values are the LFM2.5-VL-3B ones
   from [step 3 of the LFM2-VL tutorial](lfm2-vl-tutorial.md#3-write-genai_configjson-and-processor_configjson),
   with the `CUBIC` resize that d1-3B uses:

   ```python
   import json
   from pathlib import Path

   model = Path("./d1-3b/cpu-vl")

   # Add the embedding and vision models to the genai_config.json the builder wrote.
   config = json.loads((model / "genai_config.json").read_text(encoding="utf-8"))
   session_options = {"log_id": "onnxruntime-genai", "provider_options": []}
   config["model"]["embedding"] = {
       "filename": "embeddings.onnx",
       "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
       "outputs": {"inputs_embeds": "inputs_embeds"},
       "session_options": session_options,
   }
   config["model"]["vision"] = {
       "filename": "vision_encoder.onnx",
       "config_filename": "processor_config.json",
       "patch_size": 16,
       "spatial_merge_size": 2,
       "max_num_patches": 1024,
       "inputs": {"pixel_values": "pixel_values", "attention_mask": "pixel_attention_mask", "image_sizes": "spatial_shapes"},
       "outputs": {"image_features": "image_features"},
       "session_options": session_options,
   }
   (model / "genai_config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")


   # The image processor: one resize (bicubic, like d1-3B), then the SigLIP2 normalization.
   def step(name, kind, **attrs):
       return {"operation": {"name": name, "type": kind, **({"attrs": attrs} if attrs else {})}}


   transforms = [
       step("decode_image", "DecodeImage", color_space="RGB"),
       step("resize", "Resize", height=512, width=512, interpolation="CUBIC", smart_resize=1,
            min_pixels=65536, max_pixels=262144, patch_size=16, merge_size=2),
       step("rescale", "Rescale", rescale_factor=1 / 255),
       step("normalize", "Normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
       step("to_channel_first", "Permute3D", dims=[2, 0, 1]),
       step("image_sizes", "PixtralImageSizes"),
   ]
   processor = {"processor": {"name": "lfm2_vl_image_processor", "transforms": transforms}}
   (model / "processor_config.json").write_text(json.dumps(processor, indent=2), encoding="utf-8")
   ```

Then pass the images with `--image`. Each image gets one `<image>` marker at the start of the user turn,
as in d1. For example, with the two-cats picture from the d1-3B model card and this `cats.json`:

```json
{
  "state": null,
  "questions": {
    "cats": {
      "type": "choice",
      "instructions": "How many cats are there?",
      "criteria": {"one": "One", "two": "Two", "more": "Three or more"}
    }
  }
}
```

```bash
curl -o cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
python d1-decisions.py -m ./d1-3b/cpu-vl -r cats.json --image cats.jpg
```

The answer is `two` with a probability of about 0.985, as in transformers.

With `"state": null` the images are the whole state. The same model also answers text-only requests.

## 7. Limitations

- **One prefill per question.** A request with several questions runs the whole prompt again for each
  question. d1 reads the state once and branches.
- **Batch size 1.** Each question runs alone.
- **No image tiling.** GenAI resizes each image once, to at most 256 image tokens. d1 cuts images above
  about 0.52 megapixels into 512×512 tiles plus a thumbnail. So answers on larger images differ from d1.
  d1 also first shrinks any image larger than 1024×1024 pixels in area (it keeps the aspect ratio). The
  script leaves all resizing to the GenAI image processor.
- **Some image sizes round differently.** The resize picks a size in steps of 32 pixels. When a side
  falls exactly halfway between two steps (for example 400 pixels), GenAI rounds up and d1 rounds to
  the even step, so GenAI uses one more row or column of image tokens. In a POPE test of 1000 questions
  over 167 images, the one 500×400 image (6 questions) got 16 more image tokens, and its answers moved
  by up to 0.05.
- **The vision model runs again for each question** when a request has images and several questions.
- **Some whitespace tokenizes differently** in GenAI (see [Fix the tokenizer](#3-fix-the-tokenizer)).
- **No NUL characters.** GenAI's tokenizer stops at `\x00`, so the script refuses text that holds one.
- **No escaping** of caller text (see [Request and response format](#5-request-and-response-format)).

## 8. Accuracy and speed

Measured with `d1-decisions.py`, onnxruntime-genai 0.18.0 and ONNX Runtime 1.31.0, against d1-3B in
transformers (fp32, d1's own `system_one`). Times are for one `decide()` call with one question,
including everything the script does, with the default number of threads.

32 hand-written decisions (all three question types; text, JSON, log and code states):

| Machine, execution provider, export | same top answer | largest probability difference | 39 / 101 / 467 tokens |
| --- | --- | --- | --- |
| Linux, CUDA (H100), fp16 | 32/32 | 0.0033 | 5.5 / 6.0 / 9.0 ms |
| Linux, CPU (Xeon 8480+, 26 vCPU), int8 | 32/32 | 0.0088 | 200 / 458 / 2039 ms |
| Linux, CPU, fp32 | 32/32 | 0.0000025 | 844 / 2037 / 9219 ms |
| macOS, CPU (M3 Ultra), int8 | 32/32 | 0.0105 | 70 / 161 / 725 ms |
| macOS, WebGPU, fp32 | 32/32 | 0.0000068 | 88 / 110 / 296 ms |
| macOS, WebGPU, fp16 | 32/32 | 0.094 | 80 / 89 / 166 ms |

All exports above use `prune_lm_head=true`, except the CPU fp32 one. WebGPU fp16 is the fastest
option on that machine but moves some probabilities by up to 0.09; use fp32 on WebGPU when the exact
probabilities matter. The WebGPU rows used the `onnxruntime-ep-webgpu` plug-in from Python.

Public benchmarks, one question per item, first 1000 items:

| Set | Export | transformers accuracy | GenAI accuracy | same top answer |
| --- | --- | --- | --- | --- |
| BoolQ | fp16, CUDA | 87.0% | 86.9% | 998 / 1000 |
| MNLI (matched) | fp16, CUDA | 86.2% | 86.1% | 997 / 1000 |
| BoolQ | int8, CPU | 87.0% | 86.7% | 995 / 1000 |
| MNLI (matched) | int8, CPU | 86.2% | 86.1% | 995 / 1000 |
| POPE (images) | int8, CPU, image pipeline from [Images](#6-images) | 87.4% | 87.6% | 998 / 1000 |

The answers that changed were all close calls: in transformers the top two options were within 0.09
of each other. The BoolQ accuracy counts P(yes) above 0.5 as yes, so an exact 0.5 counts as no.

**fp16 on CUDA.** The fp16 export returns fp16 logits, so two close options can get exactly the same
probability (for a `noul`, P(yes) is then exactly 0.5). For a `choice` or a `score`, a tie goes to the
first option, as in d1. All 5 changed answers in the CUDA fp16 rows above were such exact ties in GenAI.
An export with `-p bf16` returns fp32 logits instead.

Long states (one question, Linux):

| Prompt tokens | CUDA fp16: time, peak GPU memory | CPU int8: time, peak memory | CPU int8 without `prune_lm_head`: peak memory |
| --- | --- | --- | --- |
| 4,042 | 48 ms, 7.8 GiB | 18.8 s, 5.8 GiB | 7.7 GiB |
| 16,079 | 206 ms, 11.2 GiB | 82.9 s, 12.3 GiB | 19.8 GiB |
| 30,110 | 403 ms, 15.7 GiB | 168.2 s, 19.6 GiB | 33.7 GiB |

Every long-state answer was on the same side of 0.5 as transformers.
