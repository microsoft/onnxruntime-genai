# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Qwen3.6 MTP (multi-token prediction) self-speculative decoding example.

The Qwen3.6-A3B model ships a built-in MTP head: a single extra decoder layer that,
given the main model's last hidden state and the just-emitted token, predicts the
*next-next* token. Used as a draft model, it lets the main model verify two tokens
per forward pass and accept the draft when it matches greedy decoding -- a lossless
speedup over plain autoregressive decoding.

Two ways to run it are shown:

  * the built-in ``og.MtpGenerator`` (default) -- a first-class C++ generator that runs
    the whole draft/verify loop in-engine, keeping the hidden-state handoff on-device
    (no per-step host round-trip). This is the recommended, fastest path.
  * ``--reference`` -- an equivalent hand-rolled Python loop (``ReferenceMtpGenerator``)
    that drives two ``og.Model`` instances through the public API. It documents the
    algorithm and is useful for experimentation, but is slower (per-step Python +
    host round-trips).

Both require:
  * the main Qwen3.6 decoder exported with ``include_hidden_states=true`` (so it exposes
    a ``hidden_states`` output), and
  * the MTP head (``mtp.onnx``, exported with ``enable_mtp=true``), loaded as a standalone
    model whose ``hidden_states`` input is fed the main model's last hidden state.

See ``qwen-3.6-mtp.md`` for the design and export instructions.
"""

import argparse
import time

import numpy as np
import onnxruntime_genai as og


class ReferenceMtpGenerator:
    """Reference (educational) MTP self-speculative decoder in pure Python (greedy, 1 draft
    token). It mirrors what the built-in ``og.MtpGenerator`` does in C++, but drives two
    ``og.Model`` generators through the public API. Prefer ``og.MtpGenerator`` in production.

    Wraps a main-model generator and an MTP-head generator and exposes a simple
    ``generate(prompt_tokens, max_new_tokens)`` that returns the decoded tokens.
    """

    def __init__(self, main_model: og.Model, mtp_model: og.Model, max_length: int = 4096):
        self.main_model = main_model
        self.mtp_model = mtp_model
        self.max_length = max_length

    def _new_main(self) -> og.Generator:
        params = og.GeneratorParams(self.main_model)
        params.set_search_options(max_length=self.max_length, do_sample=False)
        return og.Generator(self.main_model, params)

    def _mtp_draft(self, hidden_context, token_context) -> int:
        """Run the MTP head over the accumulated (hidden_state, token) context and
        return its predicted next token. ``hidden_context`` holds, for each position
        i, the main model's hidden state h_i; ``token_context`` holds t_{i+1}."""
        params = og.GeneratorParams(self.mtp_model)
        params.set_search_options(max_length=self.max_length, do_sample=False)
        draft = og.Generator(self.mtp_model, params)
        hidden = np.stack(hidden_context).astype(np.float16)[None, :, :]
        draft.set_hidden_states(hidden)
        draft.append_tokens(np.asarray(token_context, dtype=np.int32))
        logits = np.asarray(draft.get_output("logits"))[0]
        return int(logits[-1].argmax(-1))

    def generate(self, prompt_tokens, max_new_tokens):
        """Greedy self-speculative generation. Returns (tokens, stats)."""
        gen = self._new_main()
        gen.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))
        length = len(prompt_tokens)  # number of tokens already committed to the KV cache

        logits = np.asarray(gen.get_output("logits"))
        hidden = np.asarray(gen.get_output("hidden_states"))
        token = int(logits[0, -1].argmax(-1))            # token predicted for position `length`
        h = hidden[0, -1].astype(np.float16)             # hidden state that produced `token`

        out_tokens = []
        forwards = 1
        accepts = trials = 0
        hidden_context, token_context = [], []

        while len(out_tokens) < max_new_tokens:
            out_tokens.append(token)
            hidden_context.append(h)
            token_context.append(token)
            if len(out_tokens) >= max_new_tokens:
                break

            # 1. Draft the next token with the MTP head.
            draft = self._mtp_draft(hidden_context, token_context)

            # 2. Snapshot the recurrent state, then verify [token, draft] in one forward.
            gen.snapshot_state()
            gen.append_tokens(np.array([token, draft], dtype=np.int32))
            forwards += 1
            v_logits = np.asarray(gen.get_output("logits"))      # [1, 2, V]
            v_hidden = np.asarray(gen.get_output("hidden_states"))  # [1, 2, H]
            main_next = int(v_logits[0, 0].argmax(-1))           # main model's real token after `token`

            trials += 1
            if draft == main_next:
                # 3a. Accept: the draft was correct -> two tokens this step, plus a
                #     free third prediction from the verify pass.
                accepts += 1
                out_tokens.append(draft)
                hidden_context.append(v_hidden[0, 0].astype(np.float16))
                token_context.append(draft)
                token = int(v_logits[0, 1].argmax(-1))
                h = v_hidden[0, 1].astype(np.float16)
                length += 2
            else:
                # 3b. Reject: roll back the speculative forward (KV crop + recurrent
                #     snapshot restore), then commit only the correct token.
                gen.rewind_to(length)
                gen.append_tokens(np.array([token], dtype=np.int32))
                forwards += 1
                r_logits = np.asarray(gen.get_output("logits"))
                r_hidden = np.asarray(gen.get_output("hidden_states"))
                token = int(r_logits[0, -1].argmax(-1))
                h = r_hidden[0, -1].astype(np.float16)
                length += 1

        out_tokens = out_tokens[:max_new_tokens]
        stats = {
            "forwards": forwards,
            "accepts": accepts,
            "trials": trials,
            "accept_rate": accepts / max(trials, 1),
            "tokens_per_forward": len(out_tokens) / forwards,
        }
        return out_tokens, stats


def run_builtin(main_model, mtp_model, tokenizer, prompt_tokens, args):
    """Run the built-in in-engine og.MtpGenerator (recommended path)."""
    params = og.GeneratorParams(main_model)
    params.set_search_options(max_length=args.max_length, do_sample=False)
    gen = og.MtpGenerator(main_model, mtp_model, params)
    n_prompt = len(prompt_tokens)

    gen.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))
    start = time.perf_counter()
    while not gen.is_done() and len(gen.get_sequence()) < n_prompt + args.max_new_tokens:
        gen.generate_next_token()
    elapsed = time.perf_counter() - start

    tokens = gen.get_sequence().tolist()[n_prompt:]
    s = gen.get_stats()
    stats = {
        "forwards": s["forwards"], "accepts": s["accepts"], "trials": s["trials"],
        "accept_rate": s["accepts"] / max(s["trials"], 1),
        "tokens_per_forward": len(tokens) / max(s["forwards"], 1),
    }
    return tokens, stats, elapsed


def main(args):
    print("Loading main model...")
    main_model = og.Model(args.main_model_path)
    tokenizer = og.Tokenizer(main_model)
    print("Loading MTP head...")
    mtp_model = og.Model(args.mtp_model_path)

    reference = ReferenceMtpGenerator(main_model, mtp_model, max_length=args.max_length) if args.reference else None

    prompts = args.prompts or [
        "Explain how photosynthesis works in plants, step by step.",
    ]
    for prompt in prompts:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenizer.encode(text)

        if args.reference:
            start = time.perf_counter()
            tokens, stats = reference.generate(prompt_tokens, args.max_new_tokens)
            elapsed = time.perf_counter() - start
        else:
            tokens, stats, elapsed = run_builtin(main_model, mtp_model, tokenizer, prompt_tokens, args)

        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print(tokenizer.decode(tokens))
        print("-" * 80)
        print(
            f"accept rate: {stats['accept_rate']:.1%} "
            f"({stats['accepts']}/{stats['trials']})  |  "
            f"tokens/forward: {stats['tokens_per_forward']:.2f}  |  "
            f"{len(tokens)} tokens in {stats['forwards']} forwards, "
            f"{len(tokens) / elapsed:.1f} tok/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.6 MTP self-speculative decoding")
    parser.add_argument("-m", "--main_model_path", required=True,
                        help="Path to the main model folder (exported with include_hidden_states=true)")
    parser.add_argument("-d", "--mtp_model_path", required=True,
                        help="Path to the MTP head model folder (mtp.onnx + a genai_config.json declaring its hidden_states input)")
    parser.add_argument("-n", "--max_new_tokens", type=int, default=128,
                        help="Number of tokens to generate per prompt")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("-p", "--prompts", nargs="*", default=None, help="Prompt(s) to run")
    parser.add_argument("--reference", action="store_true",
                        help="Use the pure-Python ReferenceMtpGenerator instead of the built-in og.MtpGenerator")
    main(parser.parse_args())
