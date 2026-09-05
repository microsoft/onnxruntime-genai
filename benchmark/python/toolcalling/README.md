# Tool-calling benchmark

Measures whether a GenAI-supported ONNX chat model emits correct tool calls.

Tool definitions are passed to `apply_chat_template(tools=...)` as ordinary OpenAI
function specs, so a run exercises the whole serving path: rendering the tool block
into the prompt, generating, stopping, and the emitted call itself.

## Running

```
python benchmark_toolcalling.py -m {model folder} -e cuda -o toolcalling.json
```

The runner defaults to the Engine for paged models and the Generator otherwise;
force one with `--runner`. `-o` writes the full report including every generation,
while stdout carries just the summary.

## What the metrics mean

| Metric | Meaning |
|---|---|
| `correct` | function, required arguments, enums, and values all right |
| `correct_function` | picked the right tool, or correctly called nothing |
| `required_present` | every required parameter was supplied |
| `enum_valid` | enum-constrained arguments used an allowed value |
| `no_unknown_params` | no arguments outside the declared schema |
| `args_exact` | argument values match the expected ones |
| `clean_stop` | the model ended its turn instead of writing the next one |

`enum_valid`, `no_unknown_params` and `clean_stop` usually fail for configuration
reasons rather than model quality, so check them before blaming the weights:

- **Low `enum_valid` or `no_unknown_params`** — the prompt likely advertised a
  degraded tool schema. Run with `--check_prompt_fidelity` (requires `transformers`)
  to diff the rendered prompt against the HuggingFace reference; a mismatch means
  the tool block reaching the model is not the one it was trained on.
- **Low `clean_stop`** — the chat template's end-of-turn token is probably missing
  from `eos_token_id` in `genai_config.json`. The model then keeps generating past
  its own turn and invents the tool's result instead of yielding to the caller.
  Compare against `eos_token` in `tokenizer_config.json`.

## Cases

`toolcall_cases.json` holds the tool library and the cases. Each case names the
tools to offer (or `"all"`) and pins an expected function and arguments, so scoring
needs no judge model. Cases with `"expected_function": null` must be answered
directly, which catches over-eager calling. Point `--cases` at your own file to
benchmark a different tool set.

Reasoning models can spend the whole token budget thinking. Raise
`--max_new_tokens`, or pass `--chat_template_file` with a template that pins a lower
reasoning effort.
