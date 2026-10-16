# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""
Tests for examples/python/d1-decisions.py against the tiny model in test/models/lfm2-vl.

The fixture is not d1-3B, but it is an LFM2-VL pipeline with an LFM2 tokenizer, so the prompt, the
readout and the image path all run. Its tokenizer has yes, Yes, no, No, NO, the digits, A and ' A' as
single tokens, but not YES.

The dummy decoder only computes logits for ids below 1024 and only looks at the last three tokens. So
every prompt gives the same logits, and yes/no sit in the padding (P(yes) is 0.5). The tests therefore
also check the exact ids each generator receives.

Run with:
    python -m pytest test/python/models/test_d1_decisions.py --test_models test/models
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "python"
# The file name has a dash, so it is loaded by path. Only its main() imports common.py.
_SPEC = importlib.util.spec_from_file_location("d1_decisions", EXAMPLES_DIR / "d1-decisions.py")
d1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d1
_SPEC.loader.exec_module(d1)

# Fixture token ids.
YES, YES_TITLE = 12184, 12948
NO, NO_TITLE, NO_UPPER = 2744, 5048, 20099
A, B, C, SPACE_A, SPACE_B, SPACE_C = 542, 543, 544, 835, 879, 841
DIGITS = list(range(525, 535))
GENERATOR = og.Generator

# Logits for test_answers_follow_the_logits. Every other id gets -10.
CRAFTED = {YES_TITLE: 3.0, NO: 1.0, A: 0.0, B: 2.0, SPACE_C: 4.0, DIGITS[0]: 0.0, DIGITS[1]: 1.0}

HEAD = "<|startoftext|><|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"

STATE = "I was charged twice this month."
REFUND = {"type": "noul", "instructions": "Is this a refund request?"}
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges, refunds", "technical": "", "fraud_team": None},
}
URGENCY = {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today"]}


@pytest.fixture(scope="module")
def model_path(request):
    data = request.config.getoption("--test_models")
    path = (Path(data or "") / "lfm2-vl").resolve()
    if not data or not (path / "genai_config.json").exists():
        pytest.skip(f"LFM2-VL test model not found at {path}")
    return str(path)


@pytest.fixture(scope="module")
def model(model_path):
    return og.Model(model_path)


@pytest.fixture(scope="module")
def tokenizer(model):
    return og.Tokenizer(model)


@pytest.fixture(scope="module")
def decisions(model, model_path):
    return d1.D1Decisions(model, d1.model_context_length(model_path))


@pytest.fixture
def fed(monkeypatch):
    """The token ids each generator receives, one list per generator."""
    fed = []

    class Recorder:
        def __init__(self, model, params):
            self.generator = GENERATOR(model, params)
            self.ids = []
            fed.append(self.ids)

        def set_inputs(self, inputs):
            self.ids += inputs["input_ids"].as_numpy()[0].tolist()
            self.generator.set_inputs(inputs)

        def append_tokens(self, tokens):
            self.ids += [int(token) for token in tokens]
            self.generator.append_tokens(tokens)

        def get_logits(self):
            return self.generator.get_logits()

    monkeypatch.setattr(og, "Generator", Recorder)
    return fed


@pytest.fixture
def crafted(monkeypatch):
    """Makes every generator return the CRAFTED logits at the last position."""

    class Crafted:
        def __init__(self, model, params):
            self.generator = GENERATOR(model, params)

        def __getattr__(self, name):
            return getattr(self.generator, name)

        def get_logits(self):
            logits = np.full_like(self.generator.get_logits(), -10.0)
            for token, value in CRAFTED.items():
                logits[0, -1, token] = value
            return logits

    monkeypatch.setattr(og, "Generator", Crafted)


@pytest.fixture
def no_prefill(monkeypatch):
    """Fails the test if anything creates a generator."""

    def fail(*args):
        raise AssertionError("a refused request must not run the model")

    monkeypatch.setattr(og, "Generator", fail)


def image_path(model_path, name="australia.jpg"):
    path = (Path(model_path).parent.parent / "images" / name).resolve()
    if not path.exists():
        pytest.skip(f"Test image not found at {path}")
    return str(path)


def suffix(tokenizer, question):
    return d1.prompt_suffix(question, d1.choice_codes(tokenizer, question))


def readout(tokenizer, question):
    return d1.readout_ids(tokenizer, question, d1.choice_codes(tokenizer, question))


def render(tokenizer, state, question, num_images=0):
    _, questions = d1.check_request({"state": state, "questions": {"q": question}})
    return d1.prompt_prefix(state, num_images) + suffix(tokenizer, questions["q"])


def reference_probabilities(model, groups, ids, inputs=None):
    """Group-max softmax over the logits of a separate generator, written out in numpy.

    With processor inputs, ids starts with their input_ids and the rest is appended.
    """
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(ids))
    generator = GENERATOR(model, params)
    if inputs is None:
        generator.append_tokens(np.array(ids, dtype=np.int32))
    else:
        generator.set_inputs(inputs)
        tail = ids[inputs["input_ids"].as_numpy().shape[1] :]
        if tail:
            generator.append_tokens(np.array(tail, dtype=np.int32))
    logits = generator.get_logits()[0, -1].astype(np.float64)
    log_probs = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
    scores = np.array([log_probs[group].max() for group in groups])
    return np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()


# Prompts


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            REFUND,
            "Is this a refund request?\n\nReply with yes or no only.",
        ),
        (
            {**REFUND, "criteria": {"true": "Money back", "false": "Anything else"}},
            "Is this a refund request?\nYes: Money back\nNo: Anything else\n\nReply with yes or no only.",
        ),
        (
            {**REFUND, "criteria": {"true": "Money back"}},
            "Is this a refund request?\nYes: Money back\nNo: None\n\nReply with yes or no only.",
        ),
        (
            TEAM,
            (
                "Which team should handle this?\n\nOptions:\nA Charges, refunds\nB technical\nC fraud team\n\n"
                "Reply with the option code only."
            ),
        ),
        (
            {"instructions": "Pick one.", "criteria": {"y": "Yes, go", "x": "No, stop", "z": ""}},
            "Pick one.\n\nOptions:\ny Yes, go\nx No, stop\nz z\n\nReply with the option code only.",
        ),
        (
            {"type": "choice", "instructions": "Intent?", "criteria": {f"intent_{i}": "" for i in range(12)}},
            (
                "Intent?\n\nOptions:\nA intent 0\nB intent 1\nC intent 2\nD intent 3\nE intent 4\nF intent 5\n"
                "G intent 6\nH intent 7\nI intent 8\nJ intent 9\nK intent 10\nL intent 11\n\n"
                "Reply with the option code only."
            ),
        ),
        (
            URGENCY,
            "How urgent is this?\n\n0 Can wait\n1 Today\n\nReply with a single digit 0-1 only.",
        ),
        (
            {"type": "score", "instructions": "Rate it.", "criteria": [f"level {i}" for i in range(10)]},
            (
                "Rate it.\n\n0 level 0\n1 level 1\n2 level 2\n3 level 3\n4 level 4\n5 level 5\n6 level 6\n"
                "7 level 7\n8 level 8\n9 level 9\n\nReply with a single digit 0-9 only."
            ),
        ),
    ],
    ids=[
        "noul",
        "noul_criteria",
        "noul_one_side",
        "choice_letters",
        "choice_native_letters",
        "choice_12",
        "score_2",
        "score_10",
    ],
)
def test_question_prompt(tokenizer, question, expected):
    prompt = HEAD + STATE + "\n\n\nQUESTION:\n" + expected + TAIL
    assert render(tokenizer, STATE, question) == prompt


@pytest.mark.parametrize(
    "state,expected",
    [
        ("Line one.\nLine two.", "Line one.\nLine two.\n\n\nQUESTION:\n"),
        (
            {"order": 42, "items": ["tea", "cup"]},
            '{\n  "order": 42,\n  "items": [\n    "tea",\n    "cup"\n  ]\n}\n\n\nQUESTION:\n',
        ),
        (
            {"note": "Remboursez-moi, s'il vous plaît ☕ 返金"},
            '{\n  "note": "Remboursez-moi, s\'il vous plaît ☕ 返金"\n}\n\n\nQUESTION:\n',
        ),
        ([1, None, True], "[\n  1,\n  null,\n  true\n]\n\n\nQUESTION:\n"),
        (None, ""),
    ],
    ids=["str", "dict", "non_ascii", "list", "none"],
)
def test_state_prompt(tokenizer, state, expected):
    question = "Is this a refund request?\n\nReply with yes or no only."
    assert render(tokenizer, state, REFUND) == HEAD + expected + question + TAIL


def test_image_markers_come_before_the_state():
    assert d1.prompt_prefix(None, 2) == HEAD + "<image><image>"
    assert d1.prompt_prefix("Hi", 1) == HEAD + "<image>Hi\n\n\nQUESTION:\n"


# Readout ids


def test_readout_ids(tokenizer):
    _, questions = d1.check_request({"state": None, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}})
    # YES is not one token in the fixture, so it is dropped.
    assert readout(tokenizer, questions["refund"]) == [[YES, YES_TITLE], [NO, NO_TITLE, NO_UPPER]]
    assert readout(tokenizer, questions["team"]) == [[A, SPACE_A], [B, SPACE_B], [C, SPACE_C]]
    assert readout(tokenizer, questions["urgency"]) == [[DIGITS[0]], [DIGITS[1]]]
    ten = d1.check_question("ten", {"type": "score", "instructions": "x", "criteria": list("abcdefghij")})
    assert readout(tokenizer, ten) == [[digit] for digit in DIGITS]


# Probabilities


def test_probabilities_take_the_best_token_per_option():
    logits = np.random.default_rng(0).normal(0, 3, 64).astype(np.float32)
    groups = [[3, 7], [5], [10, 11, 12]]

    log_probs = logits.astype(np.float64) - np.log(np.exp(logits.astype(np.float64)).sum())
    scores = [max(log_probs[i] for i in group) for group in groups]
    expected = np.exp(scores) / np.exp(scores).sum()

    np.testing.assert_allclose(d1.probabilities(logits, groups), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "question,groups",
    [
        (REFUND, [[YES, YES_TITLE], [NO, NO_TITLE, NO_UPPER]]),
        (TEAM, [[A, SPACE_A], [B, SPACE_B], [C, SPACE_C]]),
        (URGENCY, [[DIGITS[0]], [DIGITS[1]]]),
    ],
    ids=["noul", "choice", "score"],
)
def test_one_question_matches_numpy(model, tokenizer, decisions, fed, question, groups):
    ids = tokenizer.encode(render(tokenizer, STATE, question)).tolist()
    expected = reference_probabilities(model, groups, ids)

    response = decisions.decide({"state": STATE, "questions": {"q": question}})

    assert fed == [ids]
    assert response["usage"] == {"input_tokens": len(ids), "output_tokens": 0}
    result = response["answers"]["q"]
    assert result["type"] == question["type"]
    if question["type"] == "noul":
        np.testing.assert_allclose(result["noul"], expected[0], rtol=0, atol=1e-12)
        return
    probs = list(result["probabilities"].values())
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)
    assert result["confidence"] == max(probs)
    if question["type"] == "choice":
        assert list(result["probabilities"]) == list(question["criteria"])
        assert result["choice"] == list(question["criteria"])[int(np.argmax(probs))]
    else:
        assert list(result["probabilities"]) == ["0", "1"]
        assert result["score"] == pytest.approx(sum(i * p for i, p in enumerate(probs)))
        assert result["legend"] == {"0": "Can wait", "1": "Today"}


def test_answers_follow_the_logits(decisions, crafted):
    # The fixture's own logits give P(yes) = 0.5, so these logits pin which side is yes, that the
    # spaced code counts for a choice, and the expected score.
    response = decisions.decide({"state": STATE, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}})

    answers = response["answers"]
    assert answers["refund"]["noul"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    team = np.exp([0.0, 2.0, 4.0]) / np.exp([0.0, 2.0, 4.0]).sum()
    assert answers["team"]["choice"] == "fraud_team"
    assert answers["team"]["confidence"] == pytest.approx(team[2])
    np.testing.assert_allclose(list(answers["team"]["probabilities"].values()), team)
    assert answers["urgency"]["score"] == pytest.approx(np.e / (1 + np.e))
    assert answers["urgency"]["confidence"] == pytest.approx(np.e / (1 + np.e))


def test_several_questions_encode_the_prefix_once(model, tokenizer, decisions, fed):
    # A leading newline in the instructions tokenizes differently after "QUESTION:\n" than on its
    # own. d1 encodes the shared prefix and each question separately, so this example does too.
    split = {"type": "noul", "instructions": "\nIs this a refund request?"}
    request = {"state": STATE, "questions": {"split": split, "team": TEAM}}
    _, questions = d1.check_request(request)
    prefix = tokenizer.encode(d1.prompt_prefix(STATE)).tolist()
    suffixes = [tokenizer.encode(suffix(tokenizer, q)).tolist() for q in questions.values()]
    joint = tokenizer.encode(d1.prompt_prefix(STATE) + suffix(tokenizer, questions["split"])).tolist()
    assert joint != prefix + suffixes[0]

    response = decisions.decide(request)

    assert fed == [prefix + suffix for suffix in suffixes]
    assert response["usage"] == {"input_tokens": len(prefix) + sum(map(len, suffixes)), "output_tokens": 0}
    choice = reference_probabilities(model, [[A, SPACE_A], [B, SPACE_B], [C, SPACE_C]], prefix + suffixes[1])
    np.testing.assert_allclose(list(response["answers"]["team"]["probabilities"].values()), choice, rtol=0, atol=1e-12)


def test_missing_type_means_choice(decisions):
    question = {key: value for key, value in TEAM.items() if key != "type"}
    response = decisions.decide({"state": STATE, "questions": {"team": question}})
    assert response["answers"]["team"]["type"] == "choice"


# Images


def test_one_image_matches_numpy(model, model_path, tokenizer, decisions, fed):
    path = image_path(model_path)
    prompt = render(tokenizer, None, URGENCY, num_images=1)
    inputs = model.create_multimodal_processor()(prompt, images=og.Images.open(path))
    ids = inputs["input_ids"].as_numpy()[0].tolist()
    expected = reference_probabilities(model, [[DIGITS[0]], [DIGITS[1]]], ids, inputs=inputs)

    response = decisions.decide({"state": None, "questions": {"urgency": URGENCY}}, [path])

    assert fed == [ids]
    assert response["usage"]["input_tokens"] == len(ids)
    probs = list(response["answers"]["urgency"]["probabilities"].values())
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)
    assert sum(probs) == pytest.approx(1.0)


def test_one_image_with_several_questions(model, model_path, tokenizer, decisions, fed):
    path = image_path(model_path)
    inputs = model.create_multimodal_processor()(d1.prompt_prefix(STATE, 1), images=og.Images.open(path))
    prefix = inputs["input_ids"].as_numpy()[0].tolist()
    _, questions = d1.check_request({"state": STATE, "questions": {"refund": REFUND, "urgency": URGENCY}})
    suffixes = [tokenizer.encode(suffix(tokenizer, q)).tolist() for q in questions.values()]
    expected = reference_probabilities(model, [[DIGITS[0]], [DIGITS[1]]], prefix + suffixes[1], inputs=inputs)

    response = decisions.decide({"state": STATE, "questions": {"refund": REFUND, "urgency": URGENCY}}, [path])

    assert fed == [prefix + suffix for suffix in suffixes]
    assert response["usage"]["input_tokens"] == len(prefix) + sum(map(len, suffixes))
    probs = list(response["answers"]["urgency"]["probabilities"].values())
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)


# Refusals


@pytest.mark.parametrize(
    "questions,message",
    [
        ({"q": {"type": "bool", "instructions": "x"}}, "unknown type"),
        ({"q": {"type": "noul"}}, "needs instructions"),
        ({"q": {"type": "noul", "instructions": None}}, "needs instructions"),
        ({"q": {"type": "noul", "instructions": 42}}, "needs instructions"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": {"only": ""}}}, "at least 2 options"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}, "at least 2 options"),
        ({"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}, "2 to 10 levels"),
        ({"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}, "2 to 10 levels"),
        ({"q": {"type": "score", "instructions": "x", "criteria": {"0": "low", "1": "high"}}}, "2 to 10 levels"),
        ({"q": {"type": "noul", "instructions": "x", "criteria": "money back"}}, "noul 'q' criteria"),
        ({"q": "Is it?"}, "must be an object"),
        ({}, "at least one question"),
        (["q"], "at least one question"),
    ],
    ids=[
        "unknown_type",
        "no_instructions",
        "null_instructions",
        "number_instructions",
        "choice_one_option",
        "choice_list",
        "score_one_level",
        "score_11_levels",
        "score_mapping",
        "noul_criteria_text",
        "question_text",
        "no_questions",
        "questions_list",
    ],
)
def test_refuses_bad_questions(decisions, no_prefill, questions, message):
    with pytest.raises(ValueError, match=message):
        decisions.decide({"state": STATE, "questions": questions})


IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
AUDIO_PART = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}


@pytest.mark.parametrize(
    "body,message",
    [
        (
            {"state": STATE, "questions": {"q": REFUND}, "images": ["data:image/jpeg;base64,AAAA"]},
            "'images' is not read",
        ),
        ({"state": STATE, "questions": {"q": REFUND}, "files": ["cats.jpg"]}, "'files' is not read"),
        ({"state": STATE, "questions": {"q": REFUND}, "videos": ["cats.mp4"]}, "'videos' is not read"),
        ({"State": STATE, "questions": {"q": REFUND}}, "unknown request field 'State'"),
        ({"state": STATE, "questions": {"q": REFUND}, "model": "d1-3B"}, "unknown request field 'model'"),
        ({"questions": {"q": REFUND}}, 'needs a "state" field'),
        ({"state": [{"role": "user", "content": [IMAGE_PART]}], "questions": {"q": REFUND}}, "image, audio or video"),
        ({"state": [{"role": "user", "content": [AUDIO_PART]}], "questions": {"q": REFUND}}, "image, audio or video"),
        ("Is it?", "must be an object"),
    ],
    ids=[
        "images",
        "files",
        "videos",
        "misspelled_state",
        "model",
        "no_state",
        "image_part",
        "audio_part",
        "request_text",
    ],
)
def test_refuses_fields_it_cannot_honor(decisions, no_prefill, body, message):
    with pytest.raises(ValueError, match=message):
        decisions.decide(body)


def test_accepts_a_chat_message_state_with_text_parts():
    state = [{"role": "user", "content": [{"type": "text", "text": "Refund me."}]}]
    assert d1.check_request({"state": state, "questions": {"q": REFUND}})[0] == state


class SplitTokenizer:
    """The fixture tokenizer, except that the listed texts encode to two tokens."""

    def __init__(self, tokenizer, split):
        self.tokenizer = tokenizer
        self.split = split
        self.bos_token_id = tokenizer.bos_token_id

    def encode(self, text):
        if self.split is None or text in self.split:
            return np.array([YES, NO], dtype=np.int32)
        return self.tokenizer.encode(text)


def test_refuses_a_tokenizer_without_single_token_yes(monkeypatch, decisions, tokenizer, no_prefill):
    monkeypatch.setattr(decisions, "tokenizer", SplitTokenizer(tokenizer, {"yes", "Yes", "YES"}))
    with pytest.raises(ValueError, match="no single-token yes or no"):
        decisions.decide({"state": STATE, "questions": {"refund": REFUND}})


def test_refuses_when_no_single_token_alias_is_left(monkeypatch, decisions, tokenizer, no_prefill):
    monkeypatch.setattr(decisions, "tokenizer", SplitTokenizer(tokenizer, None))
    with pytest.raises(ValueError, match="no single-token alias left for 3 options"):
        decisions.decide({"state": STATE, "questions": {"team": TEAM}})


def test_refuses_a_prompt_longer_than_the_context(decisions, no_prefill):
    with pytest.raises(ValueError, match="the model allows 4096"):
        decisions.decide({"state": "word " * 5000, "questions": {"refund": REFUND}})


def test_refuses_a_prompt_without_one_bos(monkeypatch, decisions, tokenizer, no_prefill):
    wrong_bos = SplitTokenizer(tokenizer, set())
    wrong_bos.bos_token_id = YES
    monkeypatch.setattr(decisions, "tokenizer", wrong_bos)
    with pytest.raises(ValueError, match="exactly one"):
        decisions.decide({"state": STATE, "questions": {"refund": REFUND}})


class BosTokenizer:
    """The fixture tokenizer, except that encode adds BOS in front of the texts adds_bos picks."""

    def __init__(self, tokenizer, adds_bos):
        self.tokenizer = tokenizer
        self.adds_bos = adds_bos
        self.bos_token_id = tokenizer.bos_token_id

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        return np.concatenate([[self.bos_token_id], ids]).astype(np.int32) if self.adds_bos(text) else ids


@pytest.mark.parametrize(
    "adds_bos,questions",
    [
        (lambda text: text.startswith(d1.BOS), {"refund": REFUND}),
        (lambda text: text.startswith(d1.BOS), {"refund": REFUND, "team": TEAM}),
        (lambda text: text.endswith(f"{d1.IM_START}assistant\n"), {"refund": REFUND, "team": TEAM}),
    ],
    ids=["head_one_question", "head_several_questions", "question"],
)
def test_refuses_a_tokenizer_that_adds_bos(monkeypatch, decisions, tokenizer, no_prefill, adds_bos, questions):
    monkeypatch.setattr(decisions, "tokenizer", BosTokenizer(tokenizer, adds_bos))
    with pytest.raises(ValueError, match="exactly one"):
        decisions.decide({"state": STATE, "questions": questions})


def test_refuses_images_without_a_vision_model(model_path, tmp_path, no_prefill):
    text_model = tmp_path / "lfm2-vl-text"
    shutil.copytree(model_path, text_model)
    config = json.loads((text_model / "genai_config.json").read_text(encoding="utf-8"))
    config["model"]["type"] = "lfm2_vl_text"
    del config["model"]["vision"], config["model"]["embedding"]
    (text_model / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")
    decisions = d1.D1Decisions(og.Model(str(text_model)), 4096)

    with pytest.raises(ValueError, match="images need an lfm2_vl model"):
        decisions.decide({"state": None, "questions": {"refund": REFUND}}, [image_path(model_path)])


# Command line


def test_command_line_prints_only_the_response(model_path, tmp_path):
    # The default execution provider (follow_config) is the path where common.py prints setup
    # messages; they must not reach stdout. The non-ASCII option name checks the UTF-8 output.
    team = {**TEAM, "criteria": {"billing": "Charges", "返金": "Refunds ☕"}}
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": STATE, "questions": {"refund": REFUND, "team": team}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "d1-decisions.py"), "-m", model_path, "-r", str(request)],
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=EXAMPLES_DIR,
        timeout=300,
    )

    response = json.loads(result.stdout)
    assert list(response["answers"]) == ["refund", "team"]
    assert list(response["answers"]["team"]["probabilities"]) == ["billing", "返金"]
    assert response["usage"]["output_tokens"] == 0
