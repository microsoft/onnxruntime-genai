# Qwen3.6 Multi-Token Prediction (MTP) self-speculative decoding

This document describes how ONNX Runtime GenAI accelerates Qwen3.6-A3B decoding with the
model's built-in **multi-token-prediction (MTP)** head, and the design of the export,
config, and runtime pieces that make it work.

## Contents

1. [What MTP is and why it helps](#what-mtp-is-and-why-it-helps)
2. [The MTP head architecture](#the-mtp-head-architecture)
3. [Exporting the model](#exporting-the-model)
4. [genai_config.json layout](#genai_configjson-layout)
5. [Runtime: the draft / verify loop](#runtime-the-draft--verify-loop)
6. [The hybrid-model rollback problem](#the-hybrid-model-rollback-problem)
7. [API additions](#api-additions)
8. [Running the example](#running-the-example)
9. [Measured results](#measured-results)
10. [Limitations and future work](#limitations-and-future-work)

## What MTP is and why it helps

Plain autoregressive decoding produces **one token per model forward pass**, and at batch
size 1 each pass is memory-bandwidth bound (it streams the whole weight set to emit a single
token). Qwen3.6 ships a small **MTP head** — one extra decoder layer — trained to predict the
*next-next* token from the main model's current hidden state. Used as a draft model, it lets
us:

1. draft one speculative token `d` with the cheap MTP head, then
2. **verify** it by running the main model on `[t, d]` in a single forward, and
3. **accept** `d` when it equals the token the main model would have produced anyway.

Because acceptance is checked against the main model's own output, the result is **lossless**:
the generated text is identical to greedy decoding (modulo floating-point non-determinism — see
below). When the draft is accepted, the step yields two tokens (plus a third "free" prediction
from the verify pass) for the cost of one main forward, so effective throughput scales with the
acceptance rate.

The Qwen3.6 MTP head's measured greedy acceptance on its own generations is **~88%**,
independently corroborated by vLLM (~96%) and SGLang (~95%) on the same model.

## The MTP head architecture

The head is a standard DeepSeek/Qwen-style MTP module (verified from the HF `mtp.*` weights):

```
h'_i   = fc( concat[ pre_fc_norm_embedding(embed(t_{i+1})),
                     pre_fc_norm_hidden(h_i) ] )
h''_i  = MtpDecoderLayer(h'_i)          # one full-attention + MoE layer
logits = lm_head( mtp.norm(h''_i) )     # predicts t_{i+2}
```

where `h_i` is the main model's **post-final-norm** hidden state at position `i` and `t_{i+1}`
is the token the main model just produced. The MTP layer is a `full_attention` GQA + MoE layer
(with its own q/k-norm and a shared expert), so it needs rotary embeddings and a small KV cache,
but — unlike the main model's 30 GatedDeltaNet layers — **no recurrent state**. The head reuses
the main model's token embedding and `lm_head`; the two pre-FC norms use the `(1 + weight)`
offset RMSNorm convention.

## Exporting the model

The model builder emits the MTP head as a separate `mtp.onnx` when `enable_mtp=true`. The main
model must also expose its hidden state (`include_hidden_states=true`) so the head has an input
to consume:

```bash
python -m onnxruntime_genai.models.builder \
    -i <path-to-qwen3.6-hf-checkpoint> \
    -o <output-dir> \
    -p fp16 -e cuda \
    --extra_options enable_mtp=true include_hidden_states=true exclude_embeds=false
```

This produces, in `<output-dir>`:

* `model.onnx` (+ `.data`) — the main decoder, now with a `hidden_states` graph output;
* `mtp.onnx` (+ `.data`) — the MTP head (one full-attention + MoE layer, `fc`, the pre/post
  RMSNorms, and a copy of the embedding + `lm_head`);
* `genai_config.json` — carrying an `mtp` section and the decoder's `hidden_states` output.

`Qwen35MtpHead` (in `src/python/py/models/builders/qwen.py`) builds the head by reusing the
parent `Qwen35MoeTextModel` machinery (`_make_full_attention`, `make_moe`, mRoPE, the residual
chain) for the single layer, and loads the `mtp.*` + shared embedding/`lm_head` weights directly
from the source safetensors — HF `transformers` discards the `mtp.*` weights on load, so they
cannot be read through `from_pretrained`.

> The exported `mtp.onnx` was validated to reproduce the PyTorch reference's **88.3% greedy
> acceptance, bit-for-bit**, confirming the export (mRoPE, offset RMSNorm, fc/concat, MoE +
> shared expert, lm_head) is numerically correct.

## genai_config.json layout

The builder adds an `mtp` section alongside `decoder`, and exposes the decoder's hidden state:

```jsonc
{
  "model": {
    "type": "qwen3_5_moe_text",
    "decoder": {
      "filename": "model.onnx",
      "outputs": {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value",
        "hidden_states": "hidden_states"      // <-- exposed for the MTP head
      }
    },
    "mtp": {
      "filename": "mtp.onnx",
      "num_hidden_layers": 1,
      "main_hidden_states": "hidden_states",
      "inputs": {
        "input_ids": "input_ids",
        "hidden_states": "hidden_states",     // <-- fed the main model's hidden state
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value"
      },
      "outputs": { "logits": "logits", "present_key_names": "present.%d.key",
                   "present_value_names": "present.%d.value" }
    }
  }
}
```

The `Config::Model::Mtp` struct and its JSON parsing live in `src/config.h` / `src/config.cpp`
(mirroring the `Encoder`/`Decoder` pattern). Configs without an `mtp` section parse unchanged.

To run the MTP head as a standalone `og.Model`, point a small `genai_config.json` at `mtp.onnx`
with `type: qwen3_5_moe_text`, `num_hidden_layers: 1`, and a `decoder.inputs.hidden_states`
entry. (A first-class loader that reads the `mtp` section of the main config directly is listed
under [future work](#limitations-and-future-work).)

## Runtime: the draft / verify loop

Qwen3.6 MTP is the **unified single-graph, next-N** case: one shared single-stream KV cache,
serial base → MTP, a flat (non-tree) speculation chain, and partial rollback on rejection.

The loop maintains an invariant: *on entry to each iteration the cache holds positions
`0 .. L-1`; `t` is the token predicted for position `L` (not yet in the cache); and `h` is the
hidden state of position `L-1` that produced `t`.*

```text
emit t
d        = MTP(accumulated (h_i, t_i))      # 1 draft token
snapshot = recurrent-state snapshot         # for rollback on reject
verify [t, d] at positions L, L+1           # ONE main forward
m        = argmax(logits @ L)               # main model's real token after t
if d == m:                                  # ACCEPT
    emit d;  t = argmax(logits @ L+1);  h = hidden @ L+1;  L += 2
else:                                       # REJECT
    rewind to L (KV crop + recurrent restore)
    re-run [t];  t = argmax;  h = hidden;  L += 1
```

On accept the verify pass yields a free third prediction (`logits @ L+1`), so the next iteration
starts without an extra forward. On reject, the speculative forward must be undone (next
section).

## The hybrid-model rollback problem

Qwen3.6 is a **hybrid** architecture: 30 GatedDeltaNet (linear-attention) layers + 10 GQA
layers. The GQA layers keep a normal KV cache that can be cropped to any length, but the
GatedDeltaNet layers keep a **recurrent state** (a fixed-size conv + linear-attention state)
that is updated in place every step and **cannot be partially cropped** — once the speculative
`[t, d]` forward folds `d` into the recurrent state, there is no "remove the last token" operation.

So a rejected draft cannot be undone with the existing `RewindTo`. The fix is **snapshot /
restore**:

* `RecurrentState::Snapshot()` copies the conv + recurrent buffers into a lazily-allocated
  shadow copy *before* the speculative forward;
* `RecurrentState::RestoreSnapshot()` copies them back **in place** (preserving buffer addresses,
  which CUDA-graph replay requires) when the draft is rejected;
* `RecurrentState::RewindTo(index != 0)` now restores from the snapshot instead of throwing, so a
  "rewind by 1" rolls back both the KV cache (crop) and the recurrent state (snapshot restore).

This is the same constraint that forces SGLang's Qwen3.6 MTP to use
`--mamba-scheduler-strategy extra_buffer`. It also caps the achievable speedup: a pure-attention
model reaches `1 + accept` tokens/forward, while the hybrid model pays one extra forward on each
reject, giving `(1 + a) / (2 - a)` (≈1.68× at 88% accept).

## API additions

The runtime pieces are exposed through every binding layer:

| Capability | C++ `State` | `Generator` | C API | Python |
|---|---|---|---|---|
| Read the hidden state | (auto via `ExtraOutputs`) | — | `OgaGenerator_GetOutput` | `get_output("hidden_states")` |
| Feed the MTP head's hidden input | `SetHiddenStates` | `SetHiddenStates` | `OgaGenerator_SetHiddenStates` | `set_hidden_states` |
| Snapshot recurrent state | `SnapshotState` | `SnapshotState` | `OgaGenerator_SnapshotState` | `snapshot_state` |
| Roll back on reject | `RewindTo` | `RewindToLength` | `OgaGenerator_RewindTo` | `rewind_to` |

`get_output("hidden_states")` needs no special handling: `State::Run` auto-registers any graph
output not otherwise managed by GenAI, so once the main model is exported with
`include_hidden_states=true` the hidden state is retrievable. `get_output("logits")` returns the
full per-position logits tensor, so the 2-token verify reads both `logits @ L` and `logits @ L+1`
from one forward.

`set_hidden_states` is backed by a `HiddenStatesInputs` feeder (`src/models/hidden_states_inputs.*`):
a resizable `[batch, seq, hidden]` device tensor refreshed each step from the staged value. It is
created by `DecoderOnly_State` only when `config.model.decoder.inputs.hidden_states` is set, so
models without an MTP head are unaffected.

## Running the example

[`qwen-3.6-mtp.py`](qwen-3.6-mtp.py) drives the whole loop through the Python API with both
models loaded as `og.Model` (no raw onnxruntime):

```bash
python qwen-3.6-mtp.py \
    -m <output-dir> \              # main model folder (model.onnx + genai_config.json)
    -d <mtp-model-folder> \        # mtp.onnx + a genai_config.json declaring its hidden_states input
    -n 128 \
    -p "Describe the main causes of the fall of the Roman Empire."
```

The `MtpGenerator` class encapsulates the draft/verify loop; `generate(prompt_tokens,
max_new_tokens)` returns the tokens plus stats (accept rate, tokens/forward).

## Measured results

Single-stream, batch 1, greedy, on one H200 (fp16):

| Metric | Value |
|---|---|
| MTP greedy acceptance (offline, PyTorch & ONNX) | ~88% |
| Acceptance, all-in-engine draft/verify | ~82–88% |
| Effective tokens per main forward | ~1.5–1.7× |

The output matches plain greedy decoding except for occasional late divergences caused by fp16
near-ties: the batched 2-token verify forward and the 1-token baseline can round a near-tie
logit differently and pick a different argmax. This is a documented property of greedy
speculative decoding, not a correctness bug — every emitted token is still one the main model
produced.

## Limitations and future work

* **First-class `MtpGenerator`.** Today the draft/verify orchestration lives in the Python
  example. Folding it into a C++ `MtpGenerator` (loading the `mtp` section of the main config
  directly, so a single `generate_next_token` does the draft/verify internally) would remove the
  per-step Python overhead and give a clean C/C#/Java surface.
* **Wall-clock benchmark.** The current numbers report effective *tokens per main forward*; an
  in-engine generator is needed for a true end-to-end tok/s comparison against the ~158 tok/s
  INT4 baseline.
* **INT4.** This example uses fp16 for fast iteration. The same export/runtime path works for the
  INT4 (QMoE) model; only the build precision changes.
* **Sampling.** The loop shown is greedy. Speculative sampling (accept/reject with the draft and
  target distributions) would extend it to `do_sample=true`.
* **Weight sharing.** `mtp.onnx` currently duplicates the embedding + `lm_head` (~2 GB in fp16);
  sharing them with the main model would shrink the head.
