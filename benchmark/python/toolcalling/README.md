# Tool-calling benchmark

Measures whether a GenAI-supported ONNX chat model calls tools correctly and answers
from their results.

Each case runs two turns. The first hands the model ordinary OpenAI function specs via
`apply_chat_template(tools=...)` and grades the call it emits. The second replays that
call back as an assistant turn, appends the tool's result as a `tool` message, and
grades the answer built from it. The second turn matters because it is the only thing
that exercises the template's tool-result and assistant-`tool_calls` rendering, and the
only thing that shows whether the model actually grounds its answer in what the tool
returned.

No tools are really executed: each case ships a canned `tool_result`, which keeps the
benchmark deterministic and offline. Use `--mode tool_call` to grade only the first
turn.

## Running

```
python benchmark_toolcalling.py -m {model folder} -e cuda -o toolcalling.json
```

The runner defaults to the Engine for paged models and the Generator otherwise;
force one with `--runner`. `-o` writes the full report including every generation,
while stdout carries just the summary.

## What the metrics mean

First turn, the emitted call:

| Metric | Meaning |
|---|---|
| `correct` | function, required arguments, enums, and values all right |
| `correct_function` | picked the right tool, or correctly called nothing |
| `required_present` | every required parameter was supplied |
| `enum_valid` | enum-constrained arguments used an allowed value |
| `no_unknown_params` | no arguments outside the declared schema |
| `args_exact` | argument values match the expected ones |
| `clean_stop` | the model ended its turn instead of writing the next one |

Second turn, the answer built from the tool result (`--mode end_to_end`):

| Metric | Meaning |
|---|---|
| `end_to_end_correct` | headline: the call was right and so was the answer |
| `answer_uses_result` | the answer quotes values only the tool result supplied |
| `no_repeat_call` | the model answered instead of calling the tool again |
| `final_clean_stop` | the answer turn ended cleanly |

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

A case's `tool_result` is what the tool would have returned, and
`expected_answer_contains` lists values the model can only know from that result, so
`answer_uses_result` separates a grounded answer from an invented one. Matching
ignores case, spacing and markdown, so `UA482` still matches `**UA 482**`.

Reasoning models can spend the whole token budget thinking. Raise
`--max_new_tokens`, or pass `--chat_template_file` with a template that pins a lower
reasoning effort.
