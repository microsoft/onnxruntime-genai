# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Typed decisions with LiquidAI/d1-3B.

Each answer is read from the logits of one batch-1 prefill per question; no tokens are generated.

    python d1-decisions.py -m ./d1-3b/cpu -r request.json [--image photo.jpg]

Request:  {"state": ..., "questions": {name: {"type", "instructions", "criteria"}}}
           The state is a string, any other JSON value, or null.
Response: {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}

The prompt and the readout use the same format as llama.cpp's d1 support (ggml-org/llama.cpp#30110) and
were checked against d1's reference code; names such as option_codes and readout_ids match d1's. Caller
text is not escaped: "<|im_end|>" in a state is read as a special token. See d1-tutorial.md.

The d1-3B model is released under the LFM Open License v1.0, not under this repo's license; see the
model card.
"""

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Mapping

import numpy as np
import onnxruntime_genai as og

BOS, IM_START, IM_END, IMAGE = "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<image>"
QUESTION_KEYS = ("type", "instructions", "criteria")
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
# Option codes to try, in order, when a code is not a single token.
FALLBACK_CODES = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)


def check_request(request):
    """The state and the questions of a request, each question with its type filled in."""
    if not isinstance(request, Mapping) or set(request) != {"state", "questions"}:
        got = sorted(request) if isinstance(request, Mapping) else type(request).__name__
        raise ValueError(f'a request has exactly "state" and "questions"; got {got}')
    questions = request["questions"]
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be {name: question} with at least one question")
    return request["state"], {name: check_question(name, q) for name, q in questions.items()}


def check_question(name, question):
    if not isinstance(question, Mapping) or any(key not in QUESTION_KEYS for key in question):
        raise ValueError(f"question {name!r} must be an object with only type, instructions and criteria")
    kind, criteria = question.get("type", "choice"), question.get("criteria")
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"question {name!r} has unknown type {kind!r}; use noul, choice or score")
    if not isinstance(question.get("instructions"), str):
        raise ValueError(f"question {name!r} needs instructions: a string")
    if kind == "choice" and (not isinstance(criteria, Mapping) or len(criteria) < 2):
        raise ValueError(f"choice {name!r} needs criteria {{name: description}} with at least 2 options")
    if kind == "score" and (not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10):
        raise ValueError(f"score {name!r} needs criteria: a list of 2 to 10 levels, lowest first")
    # A noul may leave one side out; it prints as "None", as in d1.
    if kind == "noul" and not (
        criteria is None or (isinstance(criteria, Mapping) and set(criteria) <= {"true", "false"})
    ):
        raise ValueError(f'noul {name!r} criteria must be {{"true": "...", "false": "..."}} if given')
    return {"type": kind, "instructions": question["instructions"], "criteria": criteria}


def single_token_ids(tokenizer, texts):
    """The ids of the texts that encode to exactly one token, in order, without repeats."""
    ids = []
    for text in texts:
        encoded = [int(i) for i in tokenizer.encode(text)]
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def option_codes(labels):
    """The labels when they are all single letters, else A..Z, else 00, 01, ..."""
    labels = [str(label).strip() for label in labels]
    if labels and all(len(label) == 1 and label.isalpha() for label in labels):
        return labels
    if len(labels) <= 26:
        return [chr(ord("A") + i) for i in range(len(labels))]
    return [f"{i:02d}" for i in range(len(labels))]


def option_aliases(tokenizer, labels):
    """A distinct single-token code for every label: [(code, token_id)]."""
    used, aliases = [], []
    for code in option_codes(labels):
        for candidate in [code, *FALLBACK_CODES]:
            ids = [int(i) for i in tokenizer.encode(candidate)]
            if len(ids) == 1 and ids[0] not in used:
                used.append(ids[0])
                aliases.append((candidate, ids[0]))
                break
        else:
            raise ValueError(f"no single-token alias left for {len(labels)} options")
    return aliases


def choice_codes(tokenizer, question):
    """The [(code, token_id)] of a choice question, None for other types."""
    return option_aliases(tokenizer, list(question["criteria"])) if question["type"] == "choice" else None


def readout_ids(tokenizer, question, codes):
    """One group of token ids per option: [yes, no] for a noul, the digits for a score, and for a
    choice its code and the code after a space."""
    if question["type"] == "noul":
        yes, no = single_token_ids(tokenizer, YES_FORMS), single_token_ids(tokenizer, NO_FORMS)
        if not yes or not no:
            raise ValueError("the tokenizer has no single-token yes or no")
        return [yes, no]
    if question["type"] == "score":
        groups = [single_token_ids(tokenizer, [str(i)]) for i in range(len(question["criteria"]))]
        if not all(groups):
            raise ValueError("the tokenizer has no single-token digit for every score level")
        return groups
    groups = []
    for code, token_id in codes:
        spaced = single_token_ids(tokenizer, [f" {code}"])
        groups.append([token_id] + [i for i in spaced if i != token_id])
    return groups


def probabilities(logits, groups):
    """Each option scores the best log-probability among its ids; a softmax over the scores."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.exp(shifted).sum())
    scores = np.array([log_probs[group].max() for group in groups])
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()).tolist()


def question_block(question, codes):
    instructions, criteria = question["instructions"], question["criteria"]
    if question["type"] == "choice":
        lines = "\n".join(
            f"{code} {criteria[label] or label.replace('_', ' ')}"
            for (code, _), label in zip(codes, list(criteria), strict=True)
        )
        return f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if question["type"] == "noul":
        extra = f"\nYes: {criteria.get('true')}\nNo: {criteria.get('false')}" if criteria else ""
        return f"{instructions}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {text}" for i, text in enumerate(criteria))
    return f"{instructions}\n\n{legend}\n\nReply with a single digit 0-{len(criteria) - 1} only."


def prompt_prefix(state, num_images=0):
    """The part of the prompt that all questions share: the images, then the state."""
    if state is None:
        body = ""
    else:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
        body = f"{text}\n\n\nQUESTION:\n"
    return f"{BOS}{IM_START}user\n{IMAGE * num_images}{body}"


def prompt_suffix(question, codes):
    """The question and the assistant header, up to the answer."""
    return f"{question_block(question, codes)}{IM_END}\n{IM_START}assistant\n"


def answer(question, probs):
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probs[0]}
    names = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(probs))]
    table = dict(zip(names, probs, strict=True))
    top = max(table, key=table.get)  # a tie goes to the first option, as in d1
    if kind == "choice":
        return {"type": "choice", "choice": top, "confidence": table[top], "probabilities": table}
    expected = float(np.dot(np.arange(len(probs)), probs))
    legend = dict(zip(names, question["criteria"], strict=True))
    return {"type": "score", "score": expected, "confidence": table[top], "probabilities": table, "legend": legend}


class D1Decisions:
    """Runs d1 requests on an og.Model built from d1-3B."""

    def __init__(self, model, context_length):
        self.model = model
        self.tokenizer = og.Tokenizer(model)
        self.context_length = context_length
        self.processor = model.create_multimodal_processor() if model.type == "lfm2_vl" else None

    def encode(self, text):
        return [int(i) for i in self.tokenizer.encode(text)]

    def decide(self, request, image_paths=()):
        """The response for one request: {"answers": {...}, "usage": {...}}."""
        state, questions = check_request(request)
        if image_paths and self.processor is None:
            raise ValueError(f"images need an lfm2_vl model with a vision model; this model is {self.model.type!r}")
        codes = [choice_codes(self.tokenizer, q) for q in questions.values()]
        groups = [readout_ids(self.tokenizer, q, c) for q, c in zip(questions.values(), codes, strict=True)]
        prefix = prompt_prefix(state, len(image_paths))
        suffixes = [prompt_suffix(q, c) for q, c in zip(questions.values(), codes, strict=True)]
        # The tokenizer stops at a NUL character, so the rest of the prompt would be lost.
        if "\x00" in prefix + "".join(suffixes):
            raise ValueError("the request holds a NUL character (\\x00)")

        # As in d1, one question is encoded as one string; several encode the shared prefix once.
        if len(suffixes) == 1:
            head, tails = prefix + suffixes[0], [[]]
        else:
            head, tails = prefix, [self.encode(suffix) for suffix in suffixes]
        inputs = None
        if image_paths:
            images = og.Images.open(*[os.path.abspath(path) for path in image_paths])
            inputs = self.processor(head, images=images)
            head_ids = inputs["input_ids"].as_numpy()[0].tolist()
        else:
            head_ids = self.encode(head)
        longest = len(head_ids) + max(len(tail) for tail in tails)
        if longest > self.context_length:
            raise ValueError(f"the prompt has {longest} tokens; the model allows {self.context_length}")

        answers = {}
        for (name, question), group, tail in zip(questions.items(), groups, tails, strict=True):
            answers[name] = answer(question, self.prefill(head_ids, inputs, tail, group))
        input_tokens = len(head_ids) + sum(len(tail) for tail in tails)
        return {"answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}

    def prefill(self, head_ids, inputs, tail, group):
        """The option probabilities of one question. Its generator is freed before the next question."""
        params = og.GeneratorParams(self.model)
        # The exact prompt length: the cache holds the prompt and nothing is generated.
        params.set_search_options(max_length=len(head_ids) + len(tail))
        generator = og.Generator(self.model, params)
        if inputs is None:
            generator.append_tokens(head_ids + tail)
        else:
            generator.set_inputs(inputs)  # only reads the processor output, so every question reuses it
            if tail:
                generator.append_tokens(tail)
        return probabilities(generator.get_logits()[0, -1], group)


def model_context_length(model_path):
    with open(os.path.join(model_path, "genai_config.json"), encoding="utf-8") as f:
        return json.load(f)["model"]["context_length"]


def main():
    from common import get_config, get_ep_args, register_ep  # noqa: PLC0415  (only the CLI needs it)

    parser = argparse.ArgumentParser(description="Answer a d1 request with d1-3B")
    parser.add_argument("-m", "--model_path", required=True, help="ONNX model folder with genai_config.json")
    parser.add_argument("-r", "--request", required=True, help="Request JSON file: {state, questions}")
    parser.add_argument("--image", nargs="*", default=[], help="Image paths. Needs an lfm2_vl model")
    get_ep_args(parser)
    args = parser.parse_args()

    # stdout holds only the JSON response.
    with contextlib.redirect_stdout(sys.stderr):
        register_ep(args.execution_provider, args.ep_path, args.use_winml)
        model = og.Model(get_config(args.model_path, args.execution_provider, args.ep_path))
    try:
        with open(args.request, encoding="utf-8-sig") as f:
            request = json.load(f)
        response = D1Decisions(model, model_context_length(args.model_path)).decide(request, args.image)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
