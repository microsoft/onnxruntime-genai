# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Typed decisions with LiquidAI/d1-3B.

d1-3B answers questions about a state in one forward pass. It generates no tokens: each answer is
read from the logits of the last prompt token.

    python d1-decisions.py -m ./d1-3b/cpu -r request.json

A request has the System One shape:

    {"state": "...", "questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}}

The script prints {"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}.

The script has two parts:
- a generic readout: logits and one token-id group per option in, option probabilities out;
- the d1 rules: the prompt, the option codes, the refusals and the answer shapes.

The d1 rules are a re-implementation of the d1-3B prompt and answer readout, the same format that
llama.cpp uses for d1 (ggml-org/llama.cpp#30110). They were checked against d1's reference code, which
is not part of this repo. Names such as option_codes and readout_ids match d1's, so the two are easy
to compare. The d1-3B model itself is released under the LFM Open License v1.0, not under this repo's
license; see the model card.

Caller text is not escaped, the same as in d1 and llama.cpp. For example, "<|im_end|>" in a state
is read as a special token.

See d1-tutorial.md for how to build the model.
"""

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Mapping

import numpy as np
import onnxruntime_genai as og

# ---------------------------------------------------------------------------
# Generic readout
# ---------------------------------------------------------------------------


def single_token_ids(tokenizer, texts):
    """The ids of the texts that encode to exactly one token, in order, without repeats."""
    ids = []
    for text in texts:
        encoded = [int(i) for i in tokenizer.encode(text)]
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def probabilities(logits, groups):
    """Option probabilities from one row of logits.

    Each option scores the best log-probability among its token ids. A softmax over these scores
    gives the probabilities. The log-softmax over the whole vocabulary does not change the result
    (the last softmax cancels it); it is kept so that the scores are real log-probabilities.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.exp(shifted).sum())
    scores = np.array([log_probs[group].max() for group in groups])
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()).tolist()


# ---------------------------------------------------------------------------
# d1 rules
# ---------------------------------------------------------------------------

BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IMAGE = "<image>"
QUESTION_TYPES = ("noul", "choice", "score")
REQUEST_KEYS = ("state", "questions")
QUESTION_KEYS = ("type", "instructions", "criteria")
NOUL_CRITERIA_KEYS = ("true", "false")
# Request fields that other servers read media from. This script takes images from --image only.
MEDIA_KEYS = ("images", "files", "videos")
# Chat-message content parts that hold media instead of text.
MEDIA_PARTS = ("image_url", "input_audio", "video_url")
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")
# Option codes to try when a code is not a single token, in order.
FALLBACK_CODES = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)


def check_question(name, question):
    """The question with its type filled in. Raises ValueError for a question d1 does not define."""
    if not isinstance(question, Mapping):
        raise ValueError(f"question {name!r} must be an object")
    unknown = [key for key in question if key not in QUESTION_KEYS]
    if unknown:
        raise ValueError(f"question {name!r} has unknown field {unknown[0]!r}; use type, instructions and criteria")
    kind = question.get("type", "choice")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {name!r} has unknown type {kind!r}; use noul, choice or score")
    if not isinstance(question.get("instructions"), str):
        raise ValueError(f"question {name!r} needs instructions: a string")
    criteria = question.get("criteria")
    if kind == "choice" and (not isinstance(criteria, Mapping) or len(criteria) < 2):
        raise ValueError(f"choice {name!r} needs criteria {{name: description}} with at least 2 options")
    if kind == "score" and (not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10):
        raise ValueError(f"score {name!r} needs criteria: a list of 2 to 10 levels, lowest first")
    # A noul may leave one side out: d1 prints it as "None". Other keys would be dropped, so they are refused.
    if (
        kind == "noul"
        and criteria is not None
        and (not isinstance(criteria, Mapping) or any(key not in NOUL_CRITERIA_KEYS for key in criteria))
    ):
        raise ValueError(f'noul {name!r} criteria must be {{"true": "...", "false": "..."}} if given')
    return {"type": kind, "instructions": question["instructions"], "criteria": criteria}


def has_media_part(state):
    """True when a chat-message state ([{"role": ..., "content": [parts]}]) holds a media part."""
    messages = state if isinstance(state, list) else []
    contents = [message.get("content") for message in messages if isinstance(message, Mapping)]
    parts = [part for content in contents if isinstance(content, list) for part in content]
    parts = [part for part in parts if isinstance(part, Mapping)]
    return any(part.get("type") in MEDIA_PARTS or any(key in part for key in MEDIA_PARTS) for part in parts)


def check_request(request):
    """The state and the checked questions of a request.

    Fields the script cannot honor are refused, not dropped, so that a request is never answered
    from less than the caller sent.
    """
    if not isinstance(request, Mapping):
        raise ValueError("the request must be an object")
    media = [key for key in MEDIA_KEYS if key in request]
    if media:
        raise ValueError(f"request field {media[0]!r} is not read; pass images with --image (image_paths in Python)")
    unknown = [key for key in request if key not in REQUEST_KEYS]
    if unknown:
        raise ValueError(f"unknown request field {unknown[0]!r}; a request has only state and questions")
    if "state" not in request:
        raise ValueError('the request needs a "state" field; use null for no state')
    if has_media_part(request["state"]):
        raise ValueError("the state holds an image, audio or video part; pass images with --image instead")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("the request needs questions: {name: question} with at least one question")
    return request["state"], {name: check_question(name, q) for name, q in questions.items()}


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
    used = []
    aliases = []
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
    """The [(code, token_id)] of a choice question, None for other types.

    The prompt and the readout both take these codes, so they always agree.
    """
    return option_aliases(tokenizer, list(question["criteria"])) if question["type"] == "choice" else None


def readout_ids(tokenizer, question, codes):
    """The token ids that stand for each option, one group per option.

    A noul has the groups [yes, no]. A choice reads its code and the code after a space. A score
    reads each level's digit. codes comes from choice_codes.
    """
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


def state_block(state):
    """A string state as it is; any other state as indented JSON."""
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def question_block(question, codes):
    """The question text, its options or levels, and the reply rule."""
    instructions, criteria = question["instructions"], question["criteria"]
    if question["type"] == "choice":
        labels = list(criteria)
        lines = "\n".join(
            f"{code} {criteria[label] or label.replace('_', ' ')}"
            for (code, _), label in zip(codes, labels, strict=True)
        )
        return f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if question["type"] == "noul":
        # A missing side prints as "None", as in d1.
        extra = f"\nYes: {criteria.get('true')}\nNo: {criteria.get('false')}" if criteria else ""
        return f"{instructions}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {text}" for i, text in enumerate(criteria))
    return f"{instructions}\n\n{legend}\n\nReply with a single digit 0-{len(criteria) - 1} only."


def prompt_prefix(state, num_images=0):
    """The part of the prompt that all questions share: the images, then the state."""
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{BOS}{IM_START}user\n{IMAGE * num_images}{body}"


def prompt_suffix(question, codes):
    """The question and the assistant header, up to the answer."""
    return f"{question_block(question, codes)}{IM_END}\n{IM_START}assistant\n"


def answer(question, probs):
    """The answer object for one question and its option probabilities."""
    kind = question["type"]
    if kind == "noul":
        # The groups are [yes, no], so the first probability is P(yes).
        return {"type": "noul", "noul": probs[0]}
    if kind == "choice":
        names = list(question["criteria"])
    else:
        names = [str(level) for level in range(len(probs))]
    table = dict(zip(names, probs, strict=True))
    top = max(table, key=table.get)
    if kind == "choice":
        return {"type": "choice", "choice": top, "confidence": table[top], "probabilities": table}
    expected = float(np.dot(np.arange(len(probs)), probs))
    legend = dict(zip(names, question["criteria"], strict=True))
    return {"type": "score", "score": expected, "confidence": table[top], "probabilities": table, "legend": legend}


# ---------------------------------------------------------------------------
# Running a request
# ---------------------------------------------------------------------------


class D1Decisions:
    """Runs System One requests on an og.Model built from d1-3B.

    Each question runs its own batch-1 prefill. With several questions, the state is processed again
    for each question.
    """

    def __init__(self, model, context_length):
        self.model = model
        self.tokenizer = og.Tokenizer(model)
        self.context_length = context_length
        # Only the lfm2_vl pipeline has a vision model.
        self.processor = model.create_multimodal_processor() if model.type == "lfm2_vl" else None

    def encode(self, text):
        return [int(i) for i in self.tokenizer.encode(text)]

    def decide(self, request, image_paths=()):
        """The response for one request: {"answers": {...}, "usage": {...}}."""
        if isinstance(image_paths, str):
            raise ValueError("image_paths must be a list of paths, not one string")
        state, questions = check_request(request)
        if image_paths and self.processor is None:
            raise ValueError(f"images need an lfm2_vl model with a vision model; this model is {self.model.type!r}")
        codes = [choice_codes(self.tokenizer, q) for q in questions.values()]
        groups = [readout_ids(self.tokenizer, q, c) for q, c in zip(questions.values(), codes, strict=True)]
        prefix = prompt_prefix(state, len(image_paths))
        suffixes = [prompt_suffix(q, c) for q, c in zip(questions.values(), codes, strict=True)]
        # The tokenizer stops at a NUL character, so the rest of the prompt would be lost.
        if "\x00" in prefix:
            raise ValueError("the state holds a NUL character (\\x00)")
        for name, text in zip(questions, suffixes, strict=True):
            if "\x00" in text:
                raise ValueError(f"question {name!r} holds a NUL character (\\x00)")

        # Like d1, one question is encoded as one string. Several questions encode the shared
        # prefix once and each suffix on its own.
        if len(suffixes) == 1:
            head, tails = prefix + suffixes[0], [[]]
        else:
            head, tails = prefix, [self.encode(suffix) for suffix in suffixes]
        inputs = None
        if image_paths:
            paths = [os.path.abspath(path) for path in image_paths]
            images = og.Images.open(*paths)
            inputs = self.processor(head, images=images)
            head_ids = inputs["input_ids"].as_numpy()[0].tolist()
        else:
            head_ids = self.encode(head)
        # The prompt text starts with BOS. A tokenizer that also adds BOS on encode would put a
        # second one at the start of the head or of a question.
        bos = self.tokenizer.bos_token_id
        if head_ids[:1] != [bos] or head_ids[1:2] == [bos] or any(tail[:1] == [bos] for tail in tails):
            raise ValueError(f"the prompt must start with exactly one {BOS} token; check the tokenizer")
        lengths = [len(head_ids) + len(tail) for tail in tails]
        if max(lengths) > self.context_length:
            raise ValueError(f"the prompt has {max(lengths)} tokens; the model allows {self.context_length}")

        answers = {}
        for (name, question), group, tail in zip(questions.items(), groups, tails, strict=True):
            answers[name] = answer(question, self.prefill(head_ids, inputs, tail, group))
        input_tokens = len(head_ids) + sum(len(tail) for tail in tails)
        return {"answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}

    def prefill(self, head_ids, inputs, tail, group):
        """The option probabilities of one question.

        The generator and its cache are freed when this returns, so only one question holds memory
        at a time.
        """
        params = og.GeneratorParams(self.model)
        # max_length is the exact prompt length: the prefill fills the cache and nothing is generated.
        params.set_search_options(max_length=len(head_ids) + len(tail))
        generator = og.Generator(self.model, params)
        if inputs is None:
            generator.append_tokens(head_ids + tail)
        else:
            # set_inputs only reads the processor output, so every question can reuse it.
            generator.set_inputs(inputs)
            if tail:
                generator.append_tokens(tail)
        return probabilities(generator.get_logits()[0, -1], group)


def model_context_length(model_path):
    with open(os.path.join(model_path, "genai_config.json"), encoding="utf-8") as f:
        return json.load(f)["model"]["context_length"]


def refuse_repeated_fields(pairs):
    """A json.load hook. A field given twice would silently keep only its last value, so it is refused."""
    names = [name for name, _ in pairs]
    repeated = next((name for name in names if names.count(name) > 1), None)
    if repeated is not None:
        raise ValueError(f"the request repeats the field {repeated!r}")
    return dict(pairs)


def main():
    # common.py sits next to this script. It is imported here, not at the top, so that the classes
    # above can be loaded from anywhere.
    from common import get_config, get_ep_args, register_ep  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Answer a System One request with d1-3B")
    parser.add_argument("-m", "--model_path", required=True, help="ONNX model folder with genai_config.json")
    parser.add_argument("-r", "--request", required=True, help="Request JSON file: {state, questions}")
    parser.add_argument("--image", nargs="*", default=[], help="Image paths. Needs an lfm2_vl model")
    get_ep_args(parser)
    args = parser.parse_args()

    # stdout holds only the JSON response, so the setup messages from common.py go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        register_ep(args.execution_provider, args.ep_path, args.use_winml)
        model = og.Model(get_config(args.model_path, args.execution_provider, args.ep_path))
    try:
        # utf-8-sig also reads files that start with a byte order mark (Windows Notepad, PowerShell).
        with open(args.request, encoding="utf-8-sig") as f:
            request = json.load(f, object_pairs_hook=refuse_repeated_fields)
        decisions = D1Decisions(model, model_context_length(args.model_path))
        response = decisions.decide(request, args.image)
    except ValueError as error:
        # A bad request (including bad JSON): one line on stderr and exit code 2, no traceback.
        parser.exit(2, f"error: {error}\n")
    else:
        # UTF-8 on every platform: on Windows a redirected stdout uses the ANSI code page otherwise.
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
