# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for examples/python/d1-decisions.py on the tiny LFM2-VL model in test/models/lfm2-vl.

The fixture is not d1-3B, but it has an LFM2 tokenizer and an image pipeline. Its dummy decoder gives
almost the same logits for every prompt, so the tests also check the ids and max_length of each generator.
"""

import importlib.util
import json
import subprocess
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "python"
# The file name has a dash, so it is loaded by path.
_SPEC = importlib.util.spec_from_file_location("d1_decisions", EXAMPLES_DIR / "d1-decisions.py")
d1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d1
_SPEC.loader.exec_module(d1)

# Fixture token ids. "YES" is not one token in the fixture.
YES, YES_TITLE = 12184, 12948
NO, NO_TITLE, NO_UPPER = 2744, 5048, 20099
A, B, C, SPACE_A, SPACE_B, SPACE_C = 542, 543, 544, 835, 879, 841
DIGITS = list(range(525, 535))
GENERATOR = og.Generator
PARAMS = og.GeneratorParams

HEAD = "<|startoftext|><|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"
STATE = "I was charged twice this month."
REFUND = {"type": "noul", "instructions": "Is this a refund request?"}
TEAM = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "Charges, refunds", "technical": "", "fraud_team": None},
}
URGENCY = {"type": "score", "instructions": "Urgent?", "criteria": ["Can wait", "Today"]}
ONE_SIDE = {**REFUND, "criteria": {"true": "Money back"}}
LETTERS = {"instructions": "Pick.", "criteria": {"y": "Go", "x": ""}}
YES_NO, CODE = "\n\nReply with yes or no only.", "\n\nReply with the option code only."
REFUND_TEXT = REFUND["instructions"] + YES_NO


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


@pytest.fixture
def decisions(model, model_path):
    return d1.D1Decisions(model, d1.model_context_length(model_path))


@pytest.fixture
def runs(monkeypatch):
    """Records the ids and max_length of each generator and the most generators alive at once.
    Set runs.logits to {token: value} to make the last position return those logits (-10 elsewhere)."""
    runs = types.SimpleNamespace(ids=[], max_lengths=[], most_alive=0, logits=None)
    alive = weakref.WeakSet()

    class Params:
        def __init__(self, model):
            self.params = PARAMS(model)

        def set_search_options(self, **options):
            runs.max_lengths.append(options.get("max_length"))
            self.params.set_search_options(**options)

    class Recorder:
        def __init__(self, model, params):
            alive.add(self)
            runs.most_alive = max(runs.most_alive, len(alive))
            self.generator = GENERATOR(model, params.params)
            self.ids = []
            runs.ids.append(self.ids)

        def set_inputs(self, inputs):
            self.ids += inputs["input_ids"].as_numpy()[0].tolist()
            self.generator.set_inputs(inputs)

        def append_tokens(self, tokens):
            self.ids += [int(token) for token in tokens]
            self.generator.append_tokens(tokens)

        def get_logits(self):
            logits = self.generator.get_logits()
            if runs.logits:
                logits = np.full_like(logits, -10.0)
                for token, value in runs.logits.items():
                    logits[0, -1, token] = value
            return logits

    monkeypatch.setattr(og, "GeneratorParams", Params)
    monkeypatch.setattr(og, "Generator", Recorder)
    return runs


def image_path(model_path):
    path = (Path(model_path).parent.parent / "images" / "australia.jpg").resolve()
    if not path.exists():
        pytest.skip(f"Test image not found at {path}")
    return str(path)


def suffix(tokenizer, question):
    return d1.prompt_suffix(question, d1.choice_codes(tokenizer, question))


def render(tokenizer, state, question, num_images=0):
    _, questions = d1.check_request({"state": state, "questions": {"q": question}})
    return d1.prompt_prefix(state, num_images) + suffix(tokenizer, questions["q"])


def reference_probabilities(model, groups, ids, inputs=None):
    """Group-max softmax over the logits of a separate generator, written out in numpy."""
    params = PARAMS(model)
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


S = f"{STATE}\n\n\nQUESTION:\n"
# name: (state, question, number of images, the prompt between HEAD and TAIL)
PROMPTS = {
    "noul": (STATE, REFUND, 0, S + REFUND_TEXT),
    "noul_one_side": (STATE, ONE_SIDE, 0, S + "Is this a refund request?\nYes: Money back\nNo: None" + YES_NO),
    "choice": (STATE, TEAM, 0, S + "Which team?\n\nOptions:\nA Charges, refunds\nB technical\nC fraud team" + CODE),
    "choice_own_letters": (STATE, LETTERS, 0, S + "Pick.\n\nOptions:\ny Go\nx x" + CODE),
    "score": (STATE, URGENCY, 0, S + "Urgent?\n\n0 Can wait\n1 Today\n\nReply with a single digit 0-1 only."),
    "state_json": (
        {"n": 1, "a": "返金 ☕"},
        REFUND,
        0,
        '{\n  "n": 1,\n  "a": "返金 ☕"\n}\n\n\nQUESTION:\n' + REFUND_TEXT,
    ),
    "state_list": ([1, None], REFUND, 0, "[\n  1,\n  null\n]\n\n\nQUESTION:\n" + REFUND_TEXT),
    "images": (None, REFUND, 2, "<image><image>" + REFUND_TEXT),
}


@pytest.mark.parametrize("state,question,num_images,expected", PROMPTS.values(), ids=list(PROMPTS))
def test_prompt(tokenizer, state, question, num_images, expected):
    assert render(tokenizer, state, question, num_images) == HEAD + expected + TAIL


@pytest.mark.parametrize(
    "question,groups",
    [
        (REFUND, [[YES, YES_TITLE], [NO, NO_TITLE, NO_UPPER]]),
        (TEAM, [[A, SPACE_A], [B, SPACE_B], [C, SPACE_C]]),
        (URGENCY, [DIGITS[:1], DIGITS[1:2]]),
    ],
    ids=["noul", "choice", "score"],
)
def test_one_question_matches_numpy(model, tokenizer, decisions, runs, question, groups):
    checked = d1.check_question("q", question)
    assert d1.readout_ids(tokenizer, checked, d1.choice_codes(tokenizer, checked)) == groups
    ids = tokenizer.encode(render(tokenizer, STATE, question)).tolist()
    expected = reference_probabilities(model, groups, ids)

    response = decisions.decide({"state": STATE, "questions": {"q": question}})

    assert runs.ids == [ids]
    assert runs.max_lengths == [len(ids)]
    assert response["usage"] == {"input_tokens": len(ids), "output_tokens": 0}
    result = response["answers"]["q"]
    probs = [result["noul"]] if question["type"] == "noul" else list(result["probabilities"].values())
    np.testing.assert_allclose(probs, expected[: len(probs)], rtol=0, atol=1e-12)


def test_answers_follow_the_logits(decisions, runs):
    runs.logits = {YES_TITLE: 3.0, NO: 1.0, A: 0.0, B: 2.0, SPACE_C: 4.0, DIGITS[0]: 0.0, DIGITS[1]: 1.0}
    request = {"state": STATE, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}}
    answers = decisions.decide(request)["answers"]

    assert answers["refund"] == {"type": "noul", "noul": pytest.approx(1 / (1 + np.exp(-2.0)))}
    team = np.exp([0.0, 2.0, 4.0]) / np.exp([0.0, 2.0, 4.0]).sum()
    assert answers["team"]["choice"] == "fraud_team"
    assert answers["team"]["confidence"] == pytest.approx(team[2])
    np.testing.assert_allclose(list(answers["team"]["probabilities"].values()), team)
    assert answers["urgency"]["score"] == pytest.approx(np.e / (1 + np.e))
    assert answers["urgency"]["confidence"] == pytest.approx(np.e / (1 + np.e))
    assert answers["urgency"]["legend"] == {"0": "Can wait", "1": "Today"}

    # A tie goes to the first option, as in d1.
    runs.logits.update({A: 5.0, B: 5.0, SPACE_C: -10.0})
    tie = decisions.decide({"state": STATE, "questions": {"team": TEAM}})["answers"]["team"]
    assert tie["choice"] == "billing"
    assert tie["probabilities"]["billing"] == tie["probabilities"]["technical"]


class SplitTokenizer:
    """The fixture tokenizer, except that the listed texts (or all texts) encode to two tokens."""

    def __init__(self, tokenizer, split):
        self.tokenizer, self.split = tokenizer, split

    def encode(self, text):
        if self.split == "all" or text in self.split:
            return np.array([YES, NO], dtype=np.int32)
        return self.tokenizer.encode(text)


def test_option_codes(tokenizer):
    # "A" is not one token here, so the first option takes the next free code and the others move on.
    codes = d1.choice_codes(SplitTokenizer(tokenizer, {"A"}), d1.check_question("team", TEAM))
    assert codes == [("B", B), ("C", C), ("D", int(tokenizer.encode("D")[0]))]
    # More than 26 options use two-digit codes.
    many = d1.check_question("q", {"instructions": "x", "criteria": {f"option {i}": "" for i in range(27)}})
    assert [code for code, _ in d1.choice_codes(tokenizer, many)] == [f"{i:02d}" for i in range(27)]


def test_one_question_is_encoded_as_one_string(tokenizer, decisions, runs):
    # A leading newline tokenizes differently after "QUESTION:\n" than on its own.
    split = {"type": "noul", "instructions": "\nIs this a refund request?"}
    prefix, text = d1.prompt_prefix(STATE), suffix(tokenizer, d1.check_question("split", split))
    joint = tokenizer.encode(prefix + text).tolist()
    assert joint != tokenizer.encode(prefix).tolist() + tokenizer.encode(text).tolist()

    response = decisions.decide({"state": STATE, "questions": {"split": split}})

    assert runs.ids == [joint]
    assert response["usage"]["input_tokens"] == len(joint)


def test_several_questions_encode_the_prefix_once(model, tokenizer, decisions, runs):
    split = {"type": "noul", "instructions": "\nIs this a refund request?"}
    request = {"state": STATE, "questions": {"split": split, "team": TEAM}}
    _, questions = d1.check_request(request)
    prefix = tokenizer.encode(d1.prompt_prefix(STATE)).tolist()
    suffixes = [tokenizer.encode(suffix(tokenizer, q)).tolist() for q in questions.values()]

    response = decisions.decide(request)

    assert runs.ids == [prefix + suffix for suffix in suffixes]
    assert runs.max_lengths == [len(prefix) + len(suffix) for suffix in suffixes]
    assert response["usage"] == {"input_tokens": len(prefix) + sum(map(len, suffixes)), "output_tokens": 0}
    choice = reference_probabilities(model, [[A, SPACE_A], [B, SPACE_B], [C, SPACE_C]], prefix + suffixes[1])
    np.testing.assert_allclose(list(response["answers"]["team"]["probabilities"].values()), choice, rtol=0, atol=1e-12)


def test_each_question_frees_its_generator_before_the_next(decisions, runs):
    # Each generator holds a KV cache for its whole prompt. Two alive at once would double that.
    decisions.decide({"state": STATE, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}})

    assert len(runs.ids) == 3
    assert runs.most_alive == 1


@pytest.mark.parametrize("names", [["urgency"], ["refund", "urgency"]], ids=["one_question", "two_questions"])
def test_one_image(model, model_path, tokenizer, decisions, runs, names):
    # One question goes through the processor as one prompt, with nothing appended after it. Several
    # questions share the processed prefix and append their own suffix.
    path = image_path(model_path)
    request = {"state": STATE, "questions": {name: {"refund": REFUND, "urgency": URGENCY}[name] for name in names}}
    _, questions = d1.check_request(request)
    suffixes = [suffix(tokenizer, q) for q in questions.values()]
    processor = model.create_multimodal_processor()
    if len(names) == 1:
        inputs = processor(d1.prompt_prefix(STATE, 1) + suffixes[0], images=og.Images.open(path))
        expected_ids = [inputs["input_ids"].as_numpy()[0].tolist()]
        usage = len(expected_ids[0])
    else:
        inputs = processor(d1.prompt_prefix(STATE, 1), images=og.Images.open(path))
        prefix = inputs["input_ids"].as_numpy()[0].tolist()
        tails = [tokenizer.encode(text).tolist() for text in suffixes]
        expected_ids = [prefix + tail for tail in tails]
        usage = len(prefix) + sum(map(len, tails))
    expected = reference_probabilities(model, [DIGITS[:1], DIGITS[1:2]], expected_ids[-1], inputs=inputs)

    response = decisions.decide(request, [path])

    assert runs.ids == expected_ids
    assert runs.max_lengths == [len(ids) for ids in expected_ids]
    assert response["usage"]["input_tokens"] == usage
    probs = list(response["answers"]["urgency"]["probabilities"].values())
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)


def q(**question):
    return {"state": STATE, "questions": {"q": question}}


# name: (request, texts the tokenizer splits, with an image, error)
REFUSALS = {
    "unknown_type": (q(type="bool", instructions="x"), None, False, "unknown type"),
    "no_instructions": (q(type="noul"), None, False, "needs instructions"),
    "number_instructions": (q(type="noul", instructions=42), None, False, "needs instructions"),
    "unknown_question_field": (q(type="noul", instructions="x", options=[]), None, False, "only type, instructions"),
    "choice_one_option": (q(type="choice", instructions="x", criteria={"a": ""}), None, False, "at least 2 options"),
    "score_one_level": (q(type="score", instructions="x", criteria=["a"]), None, False, "2 to 10 levels"),
    "score_11_levels": (q(type="score", instructions="x", criteria=list("abcdefghijk")), None, False, "2 to 10 levels"),
    "noul_criteria_keys": (q(type="noul", instructions="x", criteria={"yes": "a"}), None, False, "noul 'q' criteria"),
    "unknown_request_field": ({"state": STATE, "questions": {"q": REFUND}, "images": []}, None, False, "got .*images"),
    "no_state": ({"questions": {"q": REFUND}}, None, False, r"got \['questions'\]"),
    "request_text": ("Is it?", None, False, "got str"),
    "no_questions": ({"state": STATE, "questions": {}}, None, False, "at least one question"),
    "nul_state": ({"state": "A\x00B", "questions": {"q": REFUND}}, None, False, "NUL"),
    "nul_question": (q(**{**TEAM, "criteria": {"bill\x00ing": "", "technical": ""}}), None, False, "NUL"),
    "no_yes_token": (q(**REFUND), {"yes", "Yes", "YES"}, False, "single-token yes or no"),
    "no_digit_token": (q(**URGENCY), {"1"}, False, "single-token digit"),
    "no_alias_left": (q(**TEAM), "all", False, "no single-token alias left for 3 options"),
    "too_long": ({"state": "word " * 5000, "questions": {"q": REFUND}}, None, False, "the model allows 4096"),
    "images_without_vision": (q(**REFUND), None, True, "images need an lfm2_vl model"),
}


@pytest.mark.parametrize("request_,split,images,message", REFUSALS.values(), ids=list(REFUSALS))
def test_refusals(monkeypatch, model_path, tokenizer, decisions, request_, split, images, message):
    if split is not None:
        monkeypatch.setattr(decisions, "tokenizer", SplitTokenizer(tokenizer, split))
    if images:
        monkeypatch.setattr(decisions, "processor", None)
    monkeypatch.setattr(og, "Generator", lambda *args: pytest.fail("a refused request must not run the model"))

    with pytest.raises(ValueError, match=message):
        decisions.decide(request_, [image_path(model_path)] if images else [])


@pytest.mark.parametrize("valid", [True, False], ids=["answer", "refusal"])
def test_command_line(model_path, tmp_path, valid):
    # Non-ASCII option names check the UTF-8 output; stdout must hold only the JSON response.
    team = {**TEAM, "criteria": {"billing": "Charges", "返金": "Refunds ☕"}}
    if not valid:
        team["type"] = "bool"
    # utf-8-sig writes a byte order mark, as Windows Notepad does; the script must still read the file.
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"state": STATE, "questions": {"refund": REFUND, "team": team}}), encoding="utf-8-sig"
    )

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "d1-decisions.py"), "-m", model_path, "-r", str(request)],
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )

    if valid:
        assert result.returncode == 0, result.stderr
        response = json.loads(result.stdout)
        assert list(response["answers"]["team"]["probabilities"]) == ["billing", "返金"]
        assert response["usage"]["output_tokens"] == 0
    else:
        assert (result.returncode, result.stdout) == (2, "")
        assert result.stderr.splitlines()[-1].startswith("error: question 'team' has unknown type")
        assert "Traceback" not in result.stderr


def test_setup_messages_stay_off_stdout(monkeypatch, capsys, model_path, tmp_path):
    # common.py prints when it registers an execution provider plug-in. Those lines must go to stderr.
    def get_ep_args(parser):
        parser.add_argument("-e", "--execution_provider", default="follow_config")
        parser.add_argument("--ep_path", default="")
        parser.add_argument("--use_winml", action="store_true")

    def register_ep(*args):
        print("Registered a plug-in")

    def get_config(path, *args):
        print("Setting the model's provider")
        return og.Config(path)

    common = types.ModuleType("common")
    common.get_ep_args, common.register_ep, common.get_config = get_ep_args, register_ep, get_config
    monkeypatch.setitem(sys.modules, "common", common)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": STATE, "questions": {"refund": REFUND}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["d1-decisions.py", "-m", model_path, "-r", str(request)])

    d1.main()

    out, err = capsys.readouterr()
    assert list(json.loads(out)["answers"]) == ["refund"]
    assert "Registered a plug-in" in err and "Setting the model's provider" in err
