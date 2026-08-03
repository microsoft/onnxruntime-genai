"""Run the exported DeepSeek-V4-Flash-0731 across its tensor-parallel ranks.

A thin CLI over `dsv4_engine.DSV4Engine`, which owns the process launch, the
per-GPU KV cache and the lockstep decode loop.  Prompts go through the
checkpoint's own message encoder unless `--raw` or a token-id list is given.

    # 8 GPUs, plain continuation
    python run.py --model ~/dsv4_onnx --tokenizer /path/to/DeepSeek-V4-Flash-0731 \
        --prompt "The capital city of France is called" --raw --max-new-tokens 8

    # a question, with the reasoning block
    python run.py --model ~/dsv4_onnx --tokenizer /path/to/DeepSeek-V4-Flash-0731 \
        --prompt "Why is the sky blue?" --thinking-mode thinking --max-new-tokens 512

    # single-rank smoke test of a `--world 1` export
    python run.py --model /tmp/dsv4_w1 --world 1 --max-new-tokens 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsv4_engine import DEFAULT_PORT, DSV4Engine  # noqa: E402

DEFAULT_PROMPT_IDS = "[0, 671, 6102, 4593, 294, 8760, 344, 3252]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="directory holding rank_<r>/")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="NCCL rendezvous port")
    ap.add_argument("--tokenizer", help="checkpoint dir; enables text prompts and decoding")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT_IDS,
                    help="prompt text, or a JSON list of token ids")
    ap.add_argument("--system", help="optional system message")
    ap.add_argument("--raw", action="store_true",
                    help="tokenize the prompt as-is instead of encoding it as a user turn")
    ap.add_argument("--thinking-mode", default="chat", choices=("chat", "thinking"))
    ap.add_argument("--reasoning-effort", default="low", choices=("low", "high", "max"))
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--log-dir", default="/tmp")
    a = ap.parse_args()

    prompt = None
    if a.tokenizer:
        from dsv4_prompt import DSV4Prompt
        prompt = DSV4Prompt(a.tokenizer, a.thinking_mode, a.reasoning_effort)

    if a.prompt.lstrip().startswith("["):
        ids = json.loads(a.prompt)
    elif prompt is None:
        raise SystemExit("--tokenizer is required for a text prompt")
    elif a.raw:
        ids = [prompt.tokenizer.bos_token_id or 0] + prompt.encode(a.prompt)
    else:
        messages = [{"role": "system", "content": a.system}] if a.system else []
        messages.append({"role": "user", "content": a.prompt})
        ids = prompt.encode_messages(messages)

    eos = prompt.eos_token_ids if prompt else [1]
    with DSV4Engine(a.model, world=a.world, port=a.port, log_dir=a.log_dir) as engine:
        out = engine.generate(ids, max_new_tokens=a.max_new_tokens, eos_token_ids=eos)

    print(f"prompt {out['prompt_len']} tokens, generated {len(out['tokens'])} "
          f"({out['stop_reason']}); prefill {out['prefill_s']:.2f}s, "
          f"decode {out['decode_tok_s']:.2f} tok/s")
    print("tokens:", out["tokens"])
    if prompt is not None:
        text = prompt.decode(out["tokens"])
        print("text:", repr(text))
        if a.thinking_mode == "thinking":
            parsed = prompt.parse(text)
            print("reasoning:", repr(parsed["reasoning_content"][:400]))
            print("answer:", repr(parsed["content"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
