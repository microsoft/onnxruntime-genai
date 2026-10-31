# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests of the Engine qualification and benchmark scripts in ``tools/qwen38_graph``.

Only the pure logic is exercised (histogram arithmetic, output-file planning, report comparison, prompt verification control flow);
the Engine, the GPU, ``torch``, the research evaluation libraries and the tokenizers are replaced by small fakes.
"""

from __future__ import annotations

import json
import re
import runpy
import sys
import types

import numpy as np
import pytest
from _qwen38_tools_test_utils import load_tools, make_kv_calibration_stubs


@pytest.fixture(scope="module")
def tools():
    with load_tools(
        "bench_engine_mtp", "calibrate_qwen38_kv", "mmlu_engine_eval", stubs=make_kv_calibration_stubs()
    ) as (bench, calibrate, mmlu):
        yield types.SimpleNamespace(bench=bench, calibrate=calibrate, mmlu=mmlu)


# ---------------------------------------------------------------------------
# bench_engine_mtp.py: acceptance histogram
# ---------------------------------------------------------------------------


def speculative_stats(histogram, rounds=0):
    return {"acceptance_length_histogram": list(histogram), "rounds": rounds}


def test_acceptance_stats_keep_bins_of_rounds_with_different_draft_widths(tools):
    # Two fully accepted rounds that proposed 3 and 7 drafts: the average width is 5, but bin 7 is still occupied.
    before = speculative_stats([0] * 8)
    after = speculative_stats([0, 0, 0, 1, 0, 0, 0, 1], rounds=2)
    hist, positional, conditional = tools.bench.acceptance_stats(before, after, rounds=2)
    assert hist == [0, 0, 0, 1, 0, 0, 0, 1]
    assert positional == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    assert conditional == [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]


def test_acceptance_stats_report_the_difference_of_the_two_snapshots(tools):
    before = speculative_stats([4, 3, 2, 1, 0, 0, 0, 0], rounds=10)
    after = speculative_stats([5, 3, 2, 2, 0, 0, 0, 1], rounds=13)
    hist, positional, conditional = tools.bench.acceptance_stats(before, after, rounds=3)
    assert hist == [1, 0, 0, 1, 0, 0, 0, 1]
    assert positional == pytest.approx([2 / 3] * 3 + [1 / 3] * 4)
    assert conditional == pytest.approx([2 / 3, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])


def test_acceptance_stats_without_rounds_are_zero(tools):
    snapshot = speculative_stats([0] * 8)
    hist, positional, conditional = tools.bench.acceptance_stats(snapshot, snapshot, rounds=0)
    assert hist == [0] * 8
    assert positional == [0.0] * 7
    assert conditional == [0.0] * 7


@pytest.mark.parametrize(
    "before,after,rounds",
    [
        ({"rounds": 0}, speculative_stats([0, 0, 1, 0]), 1),
        (speculative_stats([0, 0, 0, 0]), {"rounds": 1}, 1),
        (speculative_stats([]), speculative_stats([]), 0),
        (speculative_stats([0, 0, 0, 0]), speculative_stats([0, 0, 1, 0, 0, 0, 0, 0]), 1),
        (speculative_stats([0, 2, 0, 0]), speculative_stats([0, 1, 1, 0]), 1),
        (speculative_stats([0, 0, 0, 0]), speculative_stats([0, 1, 1, 0]), 1),
    ],
    ids=["no-before", "no-after", "empty", "length-mismatch", "regressed-bin", "rounds-not-accounted"],
)
def test_acceptance_stats_reject_inconsistent_snapshots(tools, before, after, rounds):
    with pytest.raises(RuntimeError, match="histogram"):
        tools.bench.acceptance_stats(before, after, rounds)


def test_bench_records_the_complete_histogram_of_mixed_width_rounds(tools, tmp_path, monkeypatch):
    bench = tools.bench
    snapshots = [
        {
            "rounds": 0,
            "draft_tokens_accepted": 0,
            "draft_tokens_proposed": 0,
            "draft_forward_passes": 0,
            "acceptance_length_histogram": [0] * 8,
        },
        {
            "rounds": 2,
            "draft_tokens_accepted": 10,
            "draft_tokens_proposed": 10,
            "draft_forward_passes": 2,
            "acceptance_length_histogram": [0, 0, 0, 1, 0, 0, 0, 1],
        },
    ]

    class FakeEngine:
        def __init__(self, model):
            pass

        def create_event_buffer(self, size):
            return object()

        def get_speculative_stats(self):
            return snapshots.pop(0)

        def create_request(self):
            return types.SimpleNamespace(close=lambda: None)

    class FakePrompts:
        @classmethod
        def from_args(cls, model, args):
            return cls()

        def make(self, kind, n, seed):
            return list(range(n))

    def fake_run_turn(engine, buffer, request, prompt, max_new):
        turn = types.SimpleNamespace(
            tokens=list(range(max_new)),
            times=[1.0 + 0.1 * i for i in range(max_new)],
            start=0.5,
            finish_reason="length",
            runs=2,
        )
        return turn, 0

    monkeypatch.setattr(bench.og, "Engine", FakeEngine, raising=False)
    monkeypatch.setattr(bench, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(bench, "Prompts", FakePrompts)
    monkeypatch.setattr(bench, "run_turn", fake_run_turn)
    out = tmp_path / "bench.json"
    argv = ["bench_engine_mtp.py", "package", "--out", str(out), "--contexts", "16", "--reps", "1", "--new", "8"]
    monkeypatch.setattr(sys, "argv", [*argv, "--no-warmup"])

    bench.main()

    runs = json.loads(out.read_text())["contexts"]["16"]["runs"]
    assert [run["label"] for run in runs] == ["run1"]
    measured = runs[0]
    assert measured["accept_hist"] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert measured["pos_accept_rates"][0] == 1.0
    assert measured["cond_accept_rates"][3] == 0.5


# ---------------------------------------------------------------------------
# calibrate_qwen38_kv.py: one output file per quantization
# ---------------------------------------------------------------------------

QUANT_TYPES = ("int8_per_channel", "fp8_per_channel")


def synthetic_capture(layer_ids=(3, 7), channels=8, tokens=24, seed=0):
    """K/V statistics shaped like the capture loop's (one entry per target layer plus one for the MTP head)."""
    rng = np.random.default_rng(seed)
    entries = len(layer_ids) + 1

    def captured():
        return [[rng.normal(size=(tokens, channels)).astype(np.float32)] for _ in range(entries)]

    k_raw, v_raw = captured(), captured()
    return {
        "k_amax": np.stack([np.abs(entry[0]).max(axis=0) for entry in k_raw]),
        "v_amax": np.stack([np.abs(entry[0]).max(axis=0) for entry in v_raw]),
        "k_env_buf": [[np.abs(a).astype(np.float16) for a in entry] for entry in k_raw],
        "v_env_buf": [[a.astype(np.float16) for a in entry] for entry in v_raw],
        "k_raw_buf": k_raw,
        "v_raw_buf": v_raw,
        "layer_ids": list(layer_ids),
        "method": "percentile",
        "percentile": 99.9,
        "rotary_dim": 4,
    }


@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize(
    "out_name",
    [
        "scales.json",
        "kv_int8_scales.json",
        "kv_fp8_scales.json",
        "kv_scales_int8_per_channel.json",
        "kv_scales_fp8_per_channel.json",
    ],
)
def test_calibration_writes_one_distinct_file_per_quantization(tools, tmp_path, out_name, quant_type):
    out = tmp_path / out_name
    output_paths = tools.calibrate.plan_output_paths(out, quant_type)
    assert tuple(output_paths) == QUANT_TYPES
    assert output_paths[quant_type] == out
    assert len(set(output_paths.values())) == 2

    results = tools.calibrate.calibrate_quant_types(output_paths, **synthetic_capture())

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(path.name for path in output_paths.values())
    for qtype, path in output_paths.items():
        payload = json.loads(path.read_text())
        assert payload["quant_type"] == qtype
        assert payload["qmax"] == (448.0 if qtype.startswith("fp8") else 128.0)
        assert payload["layer_ids"] == [3, 7]
        assert results[qtype]["quant_type"] == qtype
    assert json.loads(out.read_text())["quant_type"] == quant_type


@pytest.mark.parametrize(
    "out_name,quant_type,companion",
    [
        ("scales.json", "int8_per_channel", "scales.fp8_per_channel.json"),
        ("scales.json", "fp8_per_channel", "scales.int8_per_channel.json"),
        ("kv_scales_int8_per_channel.json", "int8_per_channel", "kv_scales_fp8_per_channel.json"),
        ("kv_scales_fp8_per_channel.json", "fp8_per_channel", "kv_scales_int8_per_channel.json"),
        ("kv_int8_scales.json", "int8_per_channel", "kv_int8_scales.fp8_per_channel.json"),
        ("kv_int8_scales.json", "fp8_per_channel", "kv_int8_scales.int8_per_channel.json"),
        ("scales", "int8_per_channel", "scales.fp8_per_channel"),
    ],
)
def test_companion_output_name_is_derived_from_the_quantization_names(tools, tmp_path, out_name, quant_type, companion):
    output_paths = tools.calibrate.plan_output_paths(tmp_path / out_name, quant_type)
    other = next(qtype for qtype in QUANT_TYPES if qtype != quant_type)
    assert output_paths[quant_type] == tmp_path / out_name
    assert output_paths[other] == tmp_path / companion


def test_unsupported_quant_type_is_rejected_before_any_calibration_work(tools, tmp_path):
    with pytest.raises(ValueError, match="quant_type"):
        tools.calibrate.plan_output_paths(tmp_path / "scales.json", "int4_per_channel")
    # calibrate_package validates first: the missing package, checkpoint, onnxruntime and mobius are never reached.
    with pytest.raises(ValueError, match="quant_type"):
        tools.calibrate.calibrate_package(
            tmp_path / "package", tmp_path / "checkpoint", tmp_path / "scales.json", quant_type="int8"
        )
    assert list(tmp_path.iterdir()) == []


def test_command_line_only_accepts_the_calibrated_quantizations(tools, tmp_path, monkeypatch, capsys):
    argv = ["calibrate_qwen38_kv.py", "--package", "p", "--checkpoint", "c", "--out", str(tmp_path / "scales.json")]
    monkeypatch.setattr(sys, "argv", [*argv, "--quant-type", "int4_per_channel"])
    with pytest.raises(SystemExit) as excinfo:
        tools.calibrate.main()
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

    calls = []
    monkeypatch.setattr(tools.calibrate, "calibrate_package", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", [*argv, "--quant-type", "fp8_per_channel"])
    tools.calibrate.main()
    assert calls[0][1]["quant_type"] == "fp8_per_channel"


def test_colliding_outputs_are_rejected(tools, tmp_path):
    same = tmp_path / "scales.json"
    with pytest.raises(ValueError, match="both be written"):
        tools.calibrate.ensure_distinct_outputs({"int8_per_channel": same, "fp8_per_channel": same})
    # A different spelling of the same file is the same output.
    respelled = tmp_path / "sub" / ".." / "scales.json"
    with pytest.raises(ValueError, match="both be written"):
        tools.calibrate.ensure_distinct_outputs({"int8_per_channel": same, "fp8_per_channel": respelled})
    tools.calibrate.ensure_distinct_outputs({"int8_per_channel": same, "fp8_per_channel": tmp_path / "other.json"})


# ---------------------------------------------------------------------------
# mmlu_engine_eval.py: --compare
# ---------------------------------------------------------------------------


def check_item(index, stream, prompt_tokens=100):
    return {
        "index": index,
        "prompt_tokens": prompt_tokens,
        "tokens": len(stream),
        "sha256": "0" * 64,
        "cycles": 4,
        "accepted": 2,
        "accept_rate": 0.5,
        "decode_s": 1.0,
        "ttft_s": 0.1,
        "ended_with_eos": False,
        "stream": stream,
    }


def write_check_run(path, items, force_reject, check_new=8):
    path.write_text(json.dumps({"force_reject": force_reject, "check_new": check_new, "items": items}))
    return str(path)


def compare_main(mmlu, monkeypatch, a, b):
    monkeypatch.setattr(sys, "argv", ["mmlu_engine_eval.py", "--compare", a, b])
    return mmlu.main()


STREAMS = {1: [5, 6, 7, 8], 2: [9, 10, 11, 12]}


def mtp_and_forced(tmp_path, mtp_items, forced_items, **forced_options):
    mtp = write_check_run(tmp_path / "mtp.json", mtp_items, force_reject=False)
    forced = write_check_run(tmp_path / "forced.json", forced_items, force_reject=True, **forced_options)
    return mtp, forced


def test_compare_passes_identical_streams(tools, tmp_path, monkeypatch, capsys):
    items = [check_item(index, stream) for index, stream in STREAMS.items()]
    mtp, forced = mtp_and_forced(tmp_path, items, items)
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["all_identical"] is True
    assert report["problems"] == []
    assert [item["index"] for item in report["items"]] == [1, 2]
    assert all(item["identical"] and item["first_diff"] is None for item in report["items"])


def test_compare_fails_a_token_mismatch(tools, tmp_path, monkeypatch, capsys):
    mismatched = [check_item(1, STREAMS[1]), check_item(2, [9, 10, 99, 12])]
    mtp, forced = mtp_and_forced(tmp_path, [check_item(i, s) for i, s in STREAMS.items()], mismatched)
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["all_identical"] is False
    assert [item["identical"] for item in report["items"]] == [True, False]
    assert report["items"][1]["first_diff"] == 2
    assert "item 2: streams differ at token 2" in report["problems"][0]
    assert "comparison failed" in captured.err


def test_compare_fails_when_one_stream_is_a_prefix_of_the_other(tools, tmp_path, monkeypatch, capsys):
    truncated = [check_item(1, STREAMS[1][:3]), check_item(2, STREAMS[2])]
    mtp, forced = mtp_and_forced(tmp_path, [check_item(i, s) for i, s in STREAMS.items()], truncated)
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["items"][0]["identical"] is False
    assert report["items"][0]["first_diff"] == 3


def test_compare_fails_a_missing_item(tools, tmp_path, monkeypatch, capsys):
    mtp, forced = mtp_and_forced(
        tmp_path,
        [check_item(i, s) for i, s in STREAMS.items()],
        [check_item(1, STREAMS[1])],
    )
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["all_identical"] is False
    assert report["items"][1] == {"index": 2, "identical": False, "missing_from": "b"}
    assert report["problems"] == ["item 2 is only in a"]


def test_compare_fails_disjoint_item_sets(tools, tmp_path, monkeypatch, capsys):
    mtp, forced = mtp_and_forced(
        tmp_path,
        [check_item(1, STREAMS[1]), check_item(2, STREAMS[2])],
        [check_item(3, STREAMS[1]), check_item(4, STREAMS[2])],
    )
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["all_identical"] is False
    assert [item["index"] for item in report["items"]] == [1, 2, 3, 4]
    assert not any(item["identical"] for item in report["items"])


def test_compare_fails_empty_inputs(tools, tmp_path, monkeypatch, capsys):
    mtp, forced = mtp_and_forced(tmp_path, [], [])
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["all_identical"] is False
    assert report["items"] == []
    assert report["problems"] == ["a has no items", "b has no items"]


def test_compare_fails_empty_streams(tools, tmp_path, monkeypatch, capsys):
    mtp, forced = mtp_and_forced(tmp_path, [check_item(1, [])], [check_item(1, [])])
    assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["items"][0]["identical"] is False
    assert "empty token stream" in report["problems"][0]


def test_compare_fails_inconsistent_runs(tools, tmp_path, monkeypatch, capsys):
    items = [check_item(1, STREAMS[1])]
    cases = {
        "prompt length": (
            [check_item(1, STREAMS[1], prompt_tokens=100)],
            [check_item(1, STREAMS[1], prompt_tokens=101)],
            {},
        ),
        "check_new": (items, items, {"check_new": 16}),
        "duplicate": (items, items + items, {}),
    }
    for name, (mtp_items, forced_items, options) in cases.items():
        mtp, forced = mtp_and_forced(tmp_path, mtp_items, forced_items, **options)
        assert compare_main(tools.mmlu, monkeypatch, mtp, forced) == 1, name
        assert json.loads(capsys.readouterr().out)["all_identical"] is False, name


def test_compare_requires_an_mtp_run_and_a_force_reject_run(tools, tmp_path, monkeypatch, capsys):
    items = [check_item(1, STREAMS[1])]
    first = write_check_run(tmp_path / "first.json", items, force_reject=False)
    second = write_check_run(tmp_path / "second.json", items, force_reject=False)
    reject = write_check_run(tmp_path / "reject.json", items, force_reject=True)
    for a, b in ((first, second), (reject, first), (reject, reject)):
        assert compare_main(tools.mmlu, monkeypatch, a, b) == 1
        assert "expected an MTP run" in json.loads(capsys.readouterr().out)["problems"][0]


def test_compare_command_exits_with_the_comparison_status(tools, tmp_path, monkeypatch, capsys):
    items = [check_item(1, STREAMS[1])]
    mtp, forced = mtp_and_forced(tmp_path, items, items)
    monkeypatch.setattr(sys, "argv", ["mmlu_engine_eval.py", "--compare", mtp, forced])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(tools.mmlu.__file__, run_name="__main__")
    assert excinfo.value.code == 0
    capsys.readouterr()

    mtp, forced = mtp_and_forced(tmp_path, items, [check_item(1, [5, 6, 7, 0])])
    monkeypatch.setattr(sys, "argv", ["mmlu_engine_eval.py", "--compare", mtp, forced])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(tools.mmlu.__file__, run_name="__main__")
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# mmlu_engine_eval.py: canonical prompt verification gates the run
# ---------------------------------------------------------------------------

NUM_ITEMS = 800
TEMPLATE = "non-thinking"


class FakeTokenizer:
    """Chat template = "<template>question"; encode = UTF-8 bytes."""

    def __init__(self, model):
        pass

    def apply_chat_template(self, messages, template_str=None, add_generation_prompt=True):
        return f"<{template_str or 'default'}>{json.loads(messages)[0]['content']}"

    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return "A"


def canonical_prompt(index):
    text = f"<{TEMPLATE}>question {index}"
    return text, list(text.encode())


@pytest.fixture
def mmlu_run(tools, tmp_path, monkeypatch):
    """Runs ``mmlu_engine_eval.run`` on fake prompts; returns the callable, the prompt rows and the generate-call log."""
    mmlu = tools.mmlu
    generated = []

    def fake_generate(engine, buffer, prompt, max_new):
        generated.append(len(prompt))
        return {
            "tokens": [1, 2, 3],
            "stop_token_appended": False,
            "usage_ok": True,
            "ttft_s": 0.1,
            "decode_s": 0.2,
            "wall_s": 0.3,
            "cycles": 1,
            "accepted": 1,
        }

    eval_lib = types.ModuleType("eval_lib")
    eval_lib.extract_choice_letter = lambda text, letters: "A"
    eval_lib.extract_final_channel = lambda text: text
    eval_lib.load_samples = lambda name, count, offset, step: [(i, {"question": f"question {i}"}) for i in range(count)]
    eval_lib.messages_of = lambda sample: [{"role": "user", "content": sample["question"]}]
    eval_mcq_genai = types.ModuleType("eval_mcq_genai")
    eval_mcq_genai.non_thinking_template = lambda checkpoint: TEMPLATE
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda path: types.SimpleNamespace(decode=lambda tokens, skip_special_tokens=False: "A")
    )
    for name, module in (("eval_lib", eval_lib), ("eval_mcq_genai", eval_mcq_genai), ("transformers", transformers)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mmlu.og, "Tokenizer", FakeTokenizer, raising=False)
    monkeypatch.setattr(mmlu, "open_engine", lambda args, prefill: (object(), object(), object(), 0.0))
    monkeypatch.setattr(mmlu, "generate", fake_generate)

    rows = []
    for index in range(NUM_ITEMS):
        text, ids = canonical_prompt(index)
        rows.append({"index": index, "prompt_ids": ids, "prompt_text": text, "prompt_sha256": "0" * 64, "ideal": "A"})

    def run(mutate=None, shard=0, shards=8):
        if mutate:
            mutate(rows)
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
        args = types.SimpleNamespace(
            prompts=prompts,
            out=tmp_path / "mmlu.jsonl",
            shard=shard,
            shards=shards,
            max_new=8,
            model=tmp_path / "package",
            provider=None,
            checkpoint=tmp_path / "checkpoint",
            force_reject=False,
        )
        mmlu.run(args)

    return run, generated


def test_run_scores_canonical_prompts(mmlu_run, tmp_path):
    run, generated = mmlu_run
    run(shard=1)
    items = NUM_ITEMS // 8
    assert len(generated) == items + 1  # the warm-up turn and every item of the shard
    summary = json.loads((tmp_path / "mmlu.summary.json").read_text())
    assert summary["n"] == items
    assert summary["prompt_check"]["items"] == items
    assert json.loads((tmp_path / "mmlu.promptcheck.json").read_text())["mismatch_items"] == []


def test_run_stops_before_generation_when_a_prompt_token_differs(mmlu_run, tmp_path):
    run, generated = mmlu_run

    def flip_one_token(rows):
        rows[9]["prompt_ids"][3] += 1  # shard 1 of 8 holds items 1, 9, 17, ...

    with pytest.raises(SystemExit) as excinfo:
        run(flip_one_token, shard=1)
    assert excinfo.value.code not in (0, None)
    assert "indices 9)" in str(excinfo.value.code)
    assert generated == []
    assert json.loads((tmp_path / "mmlu.promptcheck.json").read_text())["mismatch_items"] == [9]
    assert not (tmp_path / "mmlu.jsonl").exists()
    assert not (tmp_path / "mmlu.summary.json").exists()


def test_run_stops_before_generation_when_the_prompt_text_differs(mmlu_run, tmp_path):
    run, generated = mmlu_run

    def stale_text(rows):
        rows[17]["prompt_text"] += " "  # stored text differs from the canonical rendering ...
        rows[17]["prompt_ids"] = list(rows[17]["prompt_text"].encode())  # ... although its ids are self-consistent

    with pytest.raises(SystemExit) as excinfo:
        run(stale_text, shard=1)
    assert excinfo.value.code not in (0, None)
    assert "indices 17)" in str(excinfo.value.code)
    assert generated == []
    assert json.loads((tmp_path / "mmlu.promptcheck.json").read_text())["mismatch_items"] == [17]


def test_run_names_every_mismatching_item_up_to_a_limit(mmlu_run, tmp_path):
    run, generated = mmlu_run

    def break_every_prompt(rows):
        for row in rows:
            row["prompt_ids"] = row["prompt_ids"][:-1]

    with pytest.raises(SystemExit) as excinfo:
        run(break_every_prompt, shard=0, shards=1)
    message = str(excinfo.value.code)
    assert "800 of 800 items" in message
    shown = re.search(r"\(indices ([^)]*)\)", message).group(1)
    assert shown.split(", ") == [str(index) for index in range(20)] + ["..."]
    assert generated == []
    assert len(json.loads((tmp_path / "mmlu.promptcheck.json").read_text())["mismatch_items"]) == NUM_ITEMS
