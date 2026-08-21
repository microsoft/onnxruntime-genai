# Gemma 4 assistant-head speculative decoding

Gemma 4 pairs its decoder with a small **assistant** head that drafts tokens for the target to
verify. `og.MtpGenerator` drives the draft/verify loop in-engine, so the embedding and hidden-state
handoff between the two graphs stays on-device.

Build the pair with [gemma-4-mtp-build.py](gemma-4-mtp-build.py), then run it with
[gemma-4-mtp.py](gemma-4-mtp.py):

```bash
python gemma-4-mtp-build.py all --out-dir models

python gemma-4-mtp.py \
  -m models/target-mtp-int4 \
  -d models/assistant-fp16 \
  --baseline
```

Greedy speculative decoding is lossless, so `--baseline` should print identical text and a speedup
greater than 1. Divergence means the pairing is misconfigured.

## How it differs from the Qwen3.6 MTP head

Both are driven through the same `og.MtpGenerator` API, but the graphs are wired differently. The
runtime picks the Gemma path when the draft model's `model.type` is `gemma4_assistant`.

| | Qwen3.6 MTP head | Gemma 4 assistant |
| --- | --- | --- |
| Token input | `input_ids`; the head embeds internally | the target's embedding tensor, concatenated with the carried hidden state |
| KV cache | its own | reads the target's present KV for a few layers |
| Feedback | `hidden_states_out` | `projected_state` |
| Verify width | any | bounded by `model.decoder.max_logits_sequence_length` |

## Target contract

The target must expose three things beyond a normal decoder:

1. **Full per-token logits over the verify window.** A pruned decoder that only emits last-token
   logits cannot verify a multi-token draft. If the graph bounds its logits sequence dimension,
   declare that bound as `model.decoder.max_logits_sequence_length`; the runtime then expects one
   logits row per input token up to that bound, and last-token logits only for longer prefill
   forwards.
2. **Its final hidden state** as an output, named by `model.decoder.outputs.hidden_states`. This is
   the same `include_hidden_states` export the Qwen MTP head needs.
3. **Its token embeddings.** `model.mtp.main_inputs_embeds` must name an output the runtime
   *binds*, not merely one the ONNX graph declares. For a multi-modal Gemma package that is the
   embedding stage's output, `model.embedding.outputs.embeddings` (`inputs_embeds` by default);
   adding a separate pass-through output to the decoder graph will not resolve.

## Config

Everything is declared in the **target's** `genai_config.json`. The assistant is loaded as an
ordinary `og.Model` from its own folder.

```jsonc
{
  "model": {
    "decoder": {
      "filename": "model.onnx",
      "hidden_size": 2560,
      "max_logits_sequence_length": 6,
      "outputs": {
        "logits": "logits",
        "hidden_states": "final_hidden_state"
      }
    },
    "mtp": {
      "filename": "assistant.onnx",
      "main_hidden_states": "final_hidden_state",
      "main_inputs_embeds": "inputs_embeds",
      "shared_kv_layers": [22, 23],
      "inputs": {
        "hidden_states": "inputs_embeds",
        "attention_mask": "attention_mask",
        "shared_key_names": [
          "shared_kv.sliding_attention.key",
          "shared_kv.full_attention.key"
        ],
        "shared_value_names": [
          "shared_kv.sliding_attention.value",
          "shared_kv.full_attention.value"
        ]
      },
      "outputs": {
        "logits": "logits",
        "hidden_states": "projected_state"
      }
    }
  }
}
```

Notes:

- `shared_key_names[i]` and `shared_value_names[i]` are bound to the target's present key/value for
  `shared_kv_layers[i]`. The target-side names are composed from
  `model.decoder.outputs.present_key_names` / `present_value_names`, so layer indices are all you
  supply. All three arrays must be the same length.
- Buffer widths come from `model.decoder.hidden_size`. The head's `inputs.hidden_states` input is
  `[1, 1, 2 * hidden_size]`: the token embedding followed by the carried hidden state.
- `inputs.hidden_states` and `outputs.hidden_states` name the head's own tensors, so they can be
  called whatever the export produced (`inputs_embeds` and `projected_state` above).
- The assistant folder's `genai_config.json` must set `"type": "gemma4_assistant"` and declare the
  same vocabulary size as the target.

## Build

`builder.py` has no Gemma 4 support — it stops at `Gemma3ForConditionalGeneration` — so every
Gemma 4 package, MTP or not, comes from an external exporter. [gemma-4-mtp-build.py](gemma-4-mtp-build.py)
drives [mobius](https://github.com/onnxruntime/mobius) and applies the post-processing this pairing
needs on top of it.

Save the two Hugging Face `config.json` files first, then run the whole pipeline:

```bash
mkdir -p models
hf download google/gemma-4-E4B-it config.json --local-dir /tmp/t \
  && cp /tmp/t/config.json models/target-config.json
hf download google/gemma-4-E4B-it-assistant config.json --local-dir /tmp/a \
  && cp /tmp/a/config.json models/assistant-config.json

python gemma-4-mtp-build.py all --out-dir models
```

That produces `models/target-mtp-int4` and `models/assistant-fp16`, which is what
[gemma-4-mtp.py](gemma-4-mtp.py) expects. `uv` must be on `PATH`; the exporter and `transformers`
versions are pinned in the script because the graph and node names below depend on them. The
`target` and `assistant` stages need `onnx`; `quantize` also needs `onnxruntime` and `onnx_ir`.

The stages can also be run individually — `target`, `assistant`, `quantize` — if you already have
mobius output, or want a different export configuration:

```bash
python gemma-4-mtp-build.py target models/target-config.json models/target-fp16 models/target-mtp-fp16
python gemma-4-mtp-build.py assistant models/assistant-config.json models/assistant-ordered \
  models/assistant-fp16 --target models/target-mtp-fp16
python gemma-4-mtp-build.py quantize models/target-mtp-fp16 models/target-mtp-int4
```

What each stage does:

| Stage | Work |
| --- | --- |
| `target` | Slices the LM head's input at the last position into a `final_hidden_state` output; gates the LM head so inputs longer than `--verify-window` produce last-token logits only (otherwise a long prefill materializes `[1, prompt_len, vocab]`); derives `shared_kv_layers` from the HF config's `layer_types` and `num_kv_shared_layers`; writes the `mtp` block |
| `assistant` | Replaces the exporter's ordered (sampled-vocab) head with a dense LM projection, since verification compares full-vocabulary argmaxes; prunes the now-unreachable nodes and initializers; writes the head's `genai_config.json` with `"type": "gemma4_assistant"` |
| `quantize` | INT4 weight-only quantization of the prepared target |

Both `target` and `assistant` validate the result: every name in the `mtp` block must exist in the
graph that has to provide it. At runtime the generator reports the first name that does not, for
example `Gemma4AssistantGenerator: missing target output 'present.22.key'`. A name that exists in
the graph but is never bound by the runtime is reported separately as
`target output '...' is not bound by the target model's state` — that is what you get by adding a
pass-through output to the decoder graph instead of pointing `main_inputs_embeds` at the embedding
stage.

## Limitations

- Greedy only, batch size 1, no guidance — the same restrictions `ValidateMtpPair` applies to the
  Qwen MTP path.
- `max_draft_tokens` must not exceed `max_logits_sequence_length`, since the verify forward is that
  many tokens wide.
