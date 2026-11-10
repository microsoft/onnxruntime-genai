# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests of the Engine qualification and benchmark scripts in ``tools/python/qwen38_flash_next``.

Only the pure logic is exercised (histogram arithmetic, output-file planning, report comparison, prompt verification control flow,
the config overlays and the force-reject hook); the Engine, the GPU, ``torch``, the research evaluation libraries and the tokenizers
are replaced by small fakes.
"""

from __future__ import annotations

import codecs
import ctypes
import json
import os
import re
import runpy
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from _qwen38_tools_test_utils import (
    QWEN38_GRAPH_KEYS,
    REMOVED_QWEN38_GRAPH_KEYS,
    load_tools,
    make_kv_calibration_stubs,
    prefill_chunk_problem,
    unknown_qwen38_graph_keys,
)


@pytest.fixture(scope="module")
def tools():
    with load_tools(
        "bench_engine_mtp",
        "calibrate_qwen38_kv",
        "engine_gate",
        "mmlu_engine_eval",
        "qwen38_engine_util",
        "engine_prompt_lengths",
        "engine_behaviours",
        "bench_ttft_engine",
        "smoke",
        "lcb_engine_eval",
        "test_plane_kernels",
        stubs=make_kv_calibration_stubs(),
    ) as (bench, calibrate, gate, mmlu, util, lengths, behaviours, ttft, smoke, lcb, plane):
        yield types.SimpleNamespace(
            bench=bench,
            calibrate=calibrate,
            gate=gate,
            mmlu=mmlu,
            util=util,
            lengths=lengths,
            behaviours=behaviours,
            ttft=ttft,
            smoke=smoke,
            lcb=lcb,
            plane=plane,
        )


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


@pytest.mark.parametrize("mtp", [True, False])
@pytest.mark.parametrize("max_new", [2, 8])
def test_bench_records_the_complete_histogram_of_mixed_width_rounds(tools, tmp_path, monkeypatch, mtp, max_new):
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
    if not mtp:
        snapshots[1].update(
            draft_tokens_accepted=0,
            draft_tokens_proposed=0,
            draft_forward_passes=0,
            acceptance_length_histogram=[0] * 8,
        )

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
    argv = [
        "bench_engine_mtp.py",
        "package",
        "--out",
        str(out),
        "--contexts",
        "16",
        "--reps",
        "1",
        "--new",
        str(max_new),
    ]
    if not mtp:
        argv.append("--no-mtp")
    monkeypatch.setattr(sys, "argv", [*argv, "--no-warmup"])

    bench.main()

    runs = json.loads(out.read_text())["contexts"]["16"]["runs"]
    assert [run["label"] for run in runs] == ["run1"]
    measured = runs[0]
    assert measured["cycles"] == 2
    assert measured["tokens"] == max_new
    assert measured["tps"] == pytest.approx(10.0)
    if not mtp:
        assert measured["accept_hist"] is None
        assert measured["pos_accept_rates"] is None
        assert measured["cond_accept_rates"] is None
        assert measured["accept_rate"] is None
        assert measured["tps"] > 0
        return
    assert measured["accept_hist"] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert measured["pos_accept_rates"][0] == 1.0
    assert measured["cond_accept_rates"][3] == 0.5


@pytest.mark.parametrize("max_new", [-1, 0, 1])
def test_decode_bench_requires_two_tokens_before_loading(tools, tmp_path, monkeypatch, capsys, max_new):
    monkeypatch.setattr(tools.bench, "load_model", lambda *args: pytest.fail("model must not be loaded"))
    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", ["bench_engine_mtp.py", "package", "--out", str(out), "--new", str(max_new)])
    with pytest.raises(SystemExit) as exc:
        tools.bench.main()
    assert exc.value.code == 2
    assert "--new must be at least 2" in capsys.readouterr().err
    assert not out.exists()


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

    def run(mutate=None, shard=0, shards=8, force_reject=False):
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
            force_reject=force_reject,
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


# ---------------------------------------------------------------------------
# qwen38_flash_graph config surface: the overlays and the force-reject test hook
# ---------------------------------------------------------------------------

FORCE_REJECT_ENV = "ORT_QWEN38_FORCE_REJECT"
GRAPH_PREFIX = "model.qwen38_graph."


def nested(dotted, value):
    result = value
    for key in reversed(dotted.split(".")):
        result = {key: result}
    return result


def test_config_description_accepts_a_current_package_config():
    section = {
        "plane_library": "libs/libqwen38_plane.so",
        "ple_table": "ple_table.txt",
        "kv_capacity": 262144,
        "max_batch_size": 4,
        "warmup": True,
        "head": {"filename_s1": "a.onnx", "filename_s2": "b.onnx", "session_options": {}},
        "mtp": {"enabled": True, "prefill_head": True},
        "fold": {"enabled": True, "head_r2": {"filename": "mtp.onnx", "session_options": {}}},
        "prefill": {
            "enabled": True,
            "chunk": 512,
            "weights_pool": "weights_pool.txt",
            "constants": "constants.bin",
            "warmup_lengths": [128, 512],
            "dense": {"filename": "dense/model.onnx", "session_options": {}},
            "head_kv": {"filename": "head_kv/model.onnx", "session_options": {}},
        },
    }
    assert set(section) == set(QWEN38_GRAPH_KEYS)
    assert unknown_qwen38_graph_keys(section) == []
    assert prefill_chunk_problem(section) is None


@pytest.mark.parametrize("removed", REMOVED_QWEN38_GRAPH_KEYS)
def test_config_description_rejects_every_removed_key(removed):
    assert unknown_qwen38_graph_keys(nested(removed, 1)) == [GRAPH_PREFIX + removed]


def test_config_description_accepts_legacy_fold_graph_ids():
    assert unknown_qwen38_graph_keys({"fold": {"target_graph_id": 17, "head_r2": {"graph_id": 18}}}) == []


@pytest.mark.parametrize(
    "section, problem",
    [
        ({"kv_capacity": 262144, "prefill": {"enabled": True}}, None),
        ({"kv_capacity": 4096, "prefill": {"enabled": True}}, "exceeds kv_capacity"),
        ({"kv_capacity": 4096, "prefill": {"enabled": True, "chunk": 512}}, None),
        ({"kv_capacity": 4096, "prefill": {"enabled": True, "chunk": 5000}}, "exceeds kv_capacity"),
        ({"kv_capacity": 4096, "prefill": {"enabled": False}}, None),
        ({"kv_capacity": 4096, "prefill": {"enabled": False, "chunk": 511}}, "odd"),
        ({"kv_capacity": 4096, "prefill": {"enabled": True, "chunk": -2}}, "negative"),
    ],
)
def test_prefill_chunk_rules_follow_the_engine(section, problem):
    found = prefill_chunk_problem(section)
    assert found is None if problem is None else problem in found


@pytest.fixture
def hook_unset(monkeypatch):
    """Starts a test without $ORT_QWEN38_FORCE_REJECT; whatever the test or the tool sets is undone afterwards."""
    monkeypatch.setenv(FORCE_REJECT_ENV, "")
    monkeypatch.delenv(FORCE_REJECT_ENV)


def fake_engine_stack(seen):
    """A model loader and an Engine class that record the overlay and the force-reject variable they run with."""

    def load_model(package, provider=None, overlay=None):
        seen.overlays.append(overlay)
        seen.env_at_load.append(os.environ.get(FORCE_REJECT_ENV))
        return object()

    class Engine:
        def __init__(self, model):
            seen.env_at_engine.append(os.environ.get(FORCE_REJECT_ENV))

        def create_event_buffer(self, size):
            return object()

        def get_speculative_stats(self):
            return {"rounds": 0, "draft_tokens_proposed": 0, "draft_tokens_accepted": 0, "tokens_emitted": 0}

        def create_request(self):
            return types.SimpleNamespace(close=lambda: None)

    return load_model, Engine


def run_gate(tools, tmp_path, monkeypatch, *flags):
    """Runs ``engine_gate.main`` on fakes; returns what the loader and the Engine saw, the recorded run and the exit status."""
    gate = tools.gate
    seen = types.SimpleNamespace(overlays=[], env_at_load=[], env_at_engine=[])
    load_model, engine = fake_engine_stack(seen)

    class FakePrompts:
        @classmethod
        def from_args(cls, model, args):
            return cls()

        def make(self, kind, n, seed):
            return list(range(n))

    def fake_run_turn(engine, buffer, request, prompt, max_new):
        turn = types.SimpleNamespace(
            tokens=[5, 6, 7],
            times=[1.0, 1.1, 1.2],
            finish_reason="length",
            usage=types.SimpleNamespace(generated_tokens=3),
            runs=2,
        )
        return turn, 0

    monkeypatch.setattr(gate, "load_model", load_model)
    monkeypatch.setattr(gate.og, "Engine", engine, raising=False)
    monkeypatch.setattr(gate, "Prompts", FakePrompts)
    monkeypatch.setattr(gate, "run_turn", fake_run_turn)
    out = tmp_path / "gate.json"
    argv = ["engine_gate.py", "package", "--out", str(out), "--cases", "text", "--context", "16", "--new", "4", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    status = gate.main()
    return seen, json.loads(out.read_text()), status


def test_gate_force_reject_sets_the_environment_hook_and_no_config_key(tools, tmp_path, monkeypatch, hook_unset):
    seen, run, status = run_gate(tools, tmp_path, monkeypatch, "--force-reject")
    assert status == 0
    assert seen.overlays == [None]  # in particular no fold.force_reject, which the config parser rejects
    assert seen.env_at_load == seen.env_at_engine == ["1"]
    assert run["force_reject"] is True


def test_gate_without_force_reject_leaves_the_hook_unset(tools, tmp_path, monkeypatch, hook_unset):
    seen, run, _ = run_gate(tools, tmp_path, monkeypatch)
    assert seen.overlays == [None]
    assert seen.env_at_load == seen.env_at_engine == [None]
    assert run["force_reject"] is False


def test_gate_drops_an_inherited_hook_value_when_the_flag_is_absent(tools, tmp_path, monkeypatch, hook_unset, capsys):
    monkeypatch.setenv(FORCE_REJECT_ENV, "1")
    seen, run, _ = run_gate(tools, tmp_path, monkeypatch)
    assert seen.env_at_load == seen.env_at_engine == [None]
    assert run["force_reject"] is False
    assert f"ignoring {FORCE_REJECT_ENV}=1" in capsys.readouterr().err


def test_gate_force_reject_replaces_an_inherited_value(tools, tmp_path, monkeypatch, hook_unset):
    monkeypatch.setenv(FORCE_REJECT_ENV, "0")
    seen, run, _ = run_gate(tools, tmp_path, monkeypatch, "--force-reject")
    assert seen.env_at_engine == ["1"]
    assert run["force_reject"] is True


@pytest.mark.parametrize(
    "flags, expected",
    [
        (("--no-mtp",), {"mtp": {"enabled": False}}),
        (("--prefill", "steps"), {"prefill": {"enabled": False}}),
        (("--prefill", "fast"), {"prefill": {"enabled": True}}),
        (
            ("--no-mtp", "--prefill", "steps", "--force-reject"),
            {"mtp": {"enabled": False}, "prefill": {"enabled": False}},
        ),
    ],
)
def test_gate_overlays_only_contain_keys_the_engine_accepts(tools, tmp_path, monkeypatch, hook_unset, flags, expected):
    seen, _, _ = run_gate(tools, tmp_path, monkeypatch, *flags)
    assert seen.overlays == [{"model": {"qwen38_graph": expected}}]
    assert unknown_qwen38_graph_keys(expected) == []


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("force_reject", [False, True])
def test_mmlu_engine_runs_with_the_hook_the_flag_asks_for(tools, monkeypatch, hook_unset, prefill, force_reject):
    mmlu = tools.mmlu
    seen = types.SimpleNamespace(overlays=[], env_at_load=[], env_at_engine=[])
    load_model, engine = fake_engine_stack(seen)
    monkeypatch.setattr(mmlu, "load_model", load_model)
    monkeypatch.setattr(mmlu.og, "Engine", engine, raising=False)
    if not force_reject:
        monkeypatch.setenv(FORCE_REJECT_ENV, "1")  # inherited from the caller

    mmlu.open_engine(types.SimpleNamespace(model="package", provider=None, force_reject=force_reject), prefill)

    assert seen.overlays == [{"model": {"qwen38_graph": {"prefill": {"enabled": prefill}}}}]
    assert unknown_qwen38_graph_keys(seen.overlays[0]["model"]["qwen38_graph"]) == []
    expected = ["1"] if force_reject else [None]
    assert seen.env_at_load == seen.env_at_engine == expected


@pytest.mark.parametrize("force_reject", [False, True])
def test_mmlu_check_records_the_mode_the_engine_ran_in(tools, tmp_path, monkeypatch, hook_unset, force_reject):
    mmlu = tools.mmlu
    seen = types.SimpleNamespace(overlays=[], env_at_load=[], env_at_engine=[])
    load_model, engine = fake_engine_stack(seen)
    monkeypatch.setattr(mmlu, "load_model", load_model)
    monkeypatch.setattr(mmlu.og, "Engine", engine, raising=False)
    monkeypatch.setattr(
        mmlu,
        "generate",
        lambda engine, buffer, prompt, max_new: {
            "tokens": [1, 2],
            "cycles": 1,
            "accepted": 0,
            "decode_s": 0.1,
            "ttft_s": 0.1,
        },
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"index": 4, "prompt_ids": [1, 2, 3]}) + "\n")
    if not force_reject:
        monkeypatch.setenv(FORCE_REJECT_ENV, "1")  # inherited from the caller
    args = types.SimpleNamespace(
        model="package",
        provider=None,
        force_reject=force_reject,
        prompts=prompts,
        out=tmp_path / "check.json",
        check_items="4",
        check_new=2,
    )

    mmlu.check(args)

    recorded = json.loads(args.out.read_text())
    assert recorded["force_reject"] is force_reject
    assert seen.env_at_engine == (["1"] if force_reject else [None])
    assert [item["index"] for item in recorded["items"]] == [4]


@pytest.mark.parametrize("prefill", ["steps", "fast"])
@pytest.mark.parametrize("recording", [True, False])
@pytest.mark.parametrize("hash_mode", ["same", "different", "absent"])
def test_gate_checks_reference_prompt_hash(tools, tmp_path, monkeypatch, hook_unset, prefill, recording, hash_mode):
    reference = tmp_path / "reference.json"
    prompt_hash = {"same": tools.util.token_hash(list(range(16))), "different": "0" * 64}
    entry = {"tokens": [5, 6, 7], "ended_with_eos": False}
    if recording:
        entry.update(spec="text:16:1016", prompt_len=16)
    else:
        entry = {"case": "text:16:1016", "prompt_tokens": 16, "old": entry}
    if hash_mode != "absent":
        entry["prompt_hash"] = prompt_hash[hash_mode]
    reference.write_text(json.dumps({"results" if recording else "cases": [entry]}))

    _, result, status = run_gate(tools, tmp_path, monkeypatch, "--prefill", prefill, "--reference", str(reference))

    assert status == (1 if hash_mode == "different" else 0)
    assert result["all_ok"] == (hash_mode != "different")
    comparison = result["results"][0]["reference"]
    assert comparison["prompt_tokens_equal"]  # equal lengths alone cannot establish prompt identity
    assert comparison["prompt_hash_equal"] == (hash_mode != "different")


def test_gate_reference_requires_explicit_prefill_before_loading(tools, tmp_path, monkeypatch, capsys):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"cases": []}))
    monkeypatch.setattr(tools.gate, "load_model", lambda *args: pytest.fail("model must not be loaded"))
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["engine_gate.py", "package", "--out", str(out), "--reference", str(reference)])
    with pytest.raises(SystemExit) as exc:
        tools.gate.main()
    assert exc.value.code == 2
    assert "--prefill is required with --reference" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "a,b,ok", [([], [], False), ([1], [], False), ([1], [1, 2], False), ([1, 2], [1], False), ([1, 2], [1, 2], True)]
)
def test_prompt_length_comparison_requires_complete_nonempty_sets(tools, tmp_path, capsys, a, b, ok):
    paths = [tmp_path / "reference.json", tmp_path / "candidate.json"]
    for path, lengths in zip(paths, (a, b), strict=True):
        path.write_text(json.dumps({"results": [{"length": n, "tokens": [5], "usage_ok": True} for n in lengths]}))
    args = types.SimpleNamespace(reference=paths[0], candidate=paths[1])
    assert tools.lengths.compare(args) == (0 if ok else 1)
    assert capsys.readouterr().out.rstrip().endswith("PASS" if ok else "FAIL")


def test_prompt_length_comparison_fails_bad_usage(tools, tmp_path):
    path = tmp_path / "lengths.json"
    path.write_text(json.dumps({"results": [{"length": 1, "tokens": [5], "usage_ok": False}]}))
    assert tools.lengths.compare(types.SimpleNamespace(reference=path, candidate=path)) == 1


@pytest.mark.parametrize("force_reject", [True, False])
def test_mmlu_resume_rejects_changed_decoding_mode(mmlu_run, tmp_path, force_reject):
    run, generated = mmlu_run
    run(force_reject=force_reject)
    assert json.loads((tmp_path / "mmlu.protocol.json").read_text())["force_reject"] is force_reject
    assert json.loads((tmp_path / "mmlu.summary.json").read_text())["force_reject"] is force_reject
    previous = len(generated)
    with pytest.raises(SystemExit, match="protocol mismatch"):
        run(force_reject=not force_reject)
    assert len(generated) == previous


def test_mmlu_resume_compares_parsed_environment_overlays(mmlu_run, tmp_path, monkeypatch):
    run, generated = mmlu_run
    overlay = {"model": {"qwen38_graph": {"mtp": {"prefill_head": False}}}}
    monkeypatch.setenv("QWEN38_OVERLAY_JSON", json.dumps(overlay, indent=2))
    run()
    assert json.loads((tmp_path / "mmlu.protocol.json").read_text())["overlay"] == overlay
    monkeypatch.setenv("QWEN38_OVERLAY_JSON", json.dumps(overlay, separators=(",", ":")))
    previous = len(generated)
    run()
    assert len(generated) == previous + 1  # Only the unrecorded warm-up is repeated on resume.
    overlay["model"]["qwen38_graph"]["mtp"]["prefill_head"] = True
    monkeypatch.setenv("QWEN38_OVERLAY_JSON", json.dumps(overlay))
    previous = len(generated)
    with pytest.raises(SystemExit, match="protocol mismatch"):
        run()
    assert len(generated) == previous


def test_calibration_registers_each_sessions_configured_libraries(tools, tmp_path):
    for name in ("parent", "dense", "shared"):
        (tmp_path / f"{name}.so").touch()
    config = {
        "model": {
            "decoder": {"session_options": {"custom_ops_library": "parent.so", "custom_ops_libraries": ["shared.so"]}},
            "qwen38_graph": {
                "prefill": {
                    "dense": {
                        "filename": "dense/custom.onnx",
                        "session_options": {
                            "custom_ops_library": "dense.so",
                            "custom_ops_libraries": ["shared.so", "./dense.so"],
                        },
                    },
                    "head_kv": {"session_options": {"custom_ops_libraries": []}},
                }
            },
        }
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(config))
    sessions = []

    class Options:
        def __init__(self):
            self.libraries = []

        def add_session_config_entry(self, key, value):
            pass

        def register_custom_ops_library(self, path):
            self.libraries.append(path)

    def session(filename, sess_options, providers):
        # Capture at construction: registering on options after InferenceSession would be too late.
        sessions.append((filename, list(sess_options.libraries), sess_options))
        return object()

    ort = types.SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=session,
        GraphOptimizationLevel=types.SimpleNamespace(ORT_DISABLE_ALL=0),
    )
    assert len(tools.calibrate.create_calibration_sessions(ort, tmp_path)) == 2
    assert sessions[0][0] == str(tmp_path / "dense/custom.onnx")
    assert sessions[0][1] == [str(tmp_path / "dense.so"), str(tmp_path / "shared.so")]
    assert sessions[1][1] == [str(tmp_path / "parent.so"), str(tmp_path / "shared.so")]
    assert sessions[0][2] is not sessions[1][2]


def test_calibration_rejects_mse_before_capture_or_output(tools, tmp_path):
    paths = tools.calibrate.plan_output_paths(tmp_path / "scales.json", "int8_per_channel")
    with pytest.raises(ValueError, match="MSE.*FP8"):
        tools.calibrate.calibrate_package(
            tmp_path / "missing", tmp_path / "missing", paths["int8_per_channel"], method="mse"
        )
    capture = synthetic_capture()
    capture["method"] = "mse"
    with pytest.raises(ValueError, match="MSE.*FP8"):
        tools.calibrate.calibrate_quant_types(paths, **capture)
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("names", [[], ["present.3.key"], ["present.3.key", "present.3.value", "present.7.value"]])
def test_calibration_rejects_unpaired_kv_outputs(tools, names):
    with pytest.raises(ValueError, match="paired KV"):
        tools.calibrate.discover_kv_outputs(names)


@pytest.mark.parametrize("field", ["k_amax", "v_amax", "k_env_buf", "v_env_buf", "k_raw_buf", "v_raw_buf"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_calibration_rejects_nonfinite_capture_without_publishing(tools, tmp_path, field, value):
    capture = synthetic_capture()
    array = capture[field] if field.endswith("amax") else capture[field][-1][0]
    array.flat[-1] = value
    paths = tools.calibrate.plan_output_paths(tmp_path / "scales.json", "int8_per_channel")
    with pytest.raises(ValueError, match="non-finite"):
        tools.calibrate.calibrate_quant_types(paths, **capture)
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("method", ["minmax", "percentile"])
def test_calibration_all_zero_captures_have_finite_scales_and_errors(tools, tmp_path, method):
    capture = synthetic_capture()
    capture["method"] = method
    for name in ("k_amax", "v_amax"):
        capture[name].fill(0)
    for name in ("k_env_buf", "v_env_buf", "k_raw_buf", "v_raw_buf"):
        for entry in capture[name]:
            entry[0].fill(0)
    paths = tools.calibrate.plan_output_paths(tmp_path / "scales.json", "int8_per_channel")
    results = tools.calibrate.calibrate_quant_types(paths, **capture)
    for result in results.values():
        assert result["stats"]["head_k_rel_l2"] == 0
        assert np.isfinite(list(result["stats"].values())).all()


def test_calibration_validates_both_quantizations_before_publishing(tools, tmp_path, monkeypatch):
    # A failure deriving the second payload must leave the first one unpublished too.
    monkeypatch.setattr(
        tools.calibrate, "_get_quant_type_max", lambda qtype: np.nan if qtype.startswith("fp8") else 128.0
    )
    paths = tools.calibrate.plan_output_paths(tmp_path / "scales.json", "int8_per_channel")
    with pytest.raises(ValueError, match="non-finite fp8.*scales"):
        tools.calibrate.calibrate_quant_types(paths, **synthetic_capture())
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("source", ["package", "environment", "argument"])
def test_folded_no_mtp_is_rejected_before_model_construction(tools, tmp_path, monkeypatch, source):
    configs = {name: {} for name in ("package", "environment", "argument")}
    configs["package"] = {"model": {"qwen38_graph": {"fold": {"enabled": True}}}}
    configs[source].setdefault("model", {}).setdefault("qwen38_graph", {})["mtp"] = {"enabled": False}
    (tmp_path / "genai_config.json").write_text(json.dumps(configs["package"]))
    monkeypatch.setenv("QWEN38_OVERLAY_JSON", json.dumps(configs["environment"]))
    with pytest.raises(ValueError, match="non-folded.*S=1"):
        tools.util.load_model(tmp_path, overlay=configs["argument"])


@pytest.mark.parametrize("provider", ["explicit", "environment", "automatic"])
def test_windows_loading_resolves_relative_dll_directories(tools, tmp_path, monkeypatch, provider):
    util = tools.util
    package = tmp_path / "package"
    (package / "libs").mkdir(parents=True)
    (package / "provider").mkdir()
    provider_path = package / "provider/onnxruntime_providers_cuda.dll"
    provider_path.touch()
    (package / "genai_config.json").write_text(json.dumps({"model": {"qwen38_graph": {"fold": {"enabled": False}}}}))
    monkeypatch.chdir(tmp_path)
    directories, registrations, overlays = [], [], []
    env = {"QWEN_PROVIDER": "package/provider/onnxruntime_providers_cuda.dll"} if provider == "environment" else {}
    monkeypatch.setattr(
        util,
        "os",
        types.SimpleNamespace(name="nt", environ=env, add_dll_directory=lambda path: directories.append(path)),
    )
    monkeypatch.setattr(
        util.og, "register_execution_provider_library", lambda name, path: registrations.append(path), raising=False
    )
    monkeypatch.setattr(util.og, "Config", lambda path: types.SimpleNamespace(overlay=overlays.append), raising=False)
    monkeypatch.setattr(util.og, "Model", lambda config: config, raising=False)
    util.load_model(
        "package",
        str(provider_path.relative_to(tmp_path)) if provider == "explicit" else None,
        {"model": {"qwen38_graph": {"mtp": {"enabled": False}}}},
    )
    assert directories == [str(package / "libs"), str(provider_path.parent)]
    assert registrations == [str(provider_path)]


def test_host_budget_overrides_package_and_overlay_runtime_profiles(tools, tmp_path, monkeypatch):
    util = tools.util
    entry = util.HOST_EXPERT_BYTES_ENTRY
    budget = 56 * (1 << 30)
    original = {
        "model": {"decoder": {"session_options": {entry: "auto", "keep": "base"}}},
        "runtime_profiles": [
            {
                "id": "large",
                "eligibility": {"minimum_total_device_memory_bytes": 1, "is_integrated": False},
                "overlay": {
                    "model": {"decoder": {"session_options": {entry: str(128 * (1 << 30)), "keep": "profile"}}}
                },
            }
        ],
    }
    (tmp_path / "genai_config.json").write_text(json.dumps(original))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.data").write_bytes(b"asset")
    inherited = {
        "model": {"decoder": {"session_options": {entry: "999"}}},
        "runtime_profiles": [
            {
                "id": "extra",
                "eligibility": {"minimum_total_device_memory_bytes": 1, "is_integrated": True},
                "overlay": {"model": {"decoder": {"session_options": {entry: "888"}}}},
            }
        ],
    }
    argument = {"model": {"decoder": {"session_options": {entry: "777"}}}}
    monkeypatch.setenv("QWEN38_OVERLAY_JSON", json.dumps(inherited))
    monkeypatch.setenv(util.HOST_EXPERT_BYTES_ENV, str(budget))

    class Config:
        def __init__(self, path):
            self.path = Path(path)
            self.document = json.loads((self.path / "genai_config.json").read_text())
            self.overlays = []

        def overlay(self, raw):
            self.overlays.append(json.loads(raw))

    monkeypatch.setattr(util.og, "Config", Config, raising=False)
    monkeypatch.setattr(util.og, "Model", lambda config: config, raising=False)
    config = util.load_model(tmp_path, overlay=argument)
    assert config.path != tmp_path
    assert (config.path / "weights" / "model.data").read_bytes() == b"asset"
    assert json.loads((tmp_path / "genai_config.json").read_text()) == original
    assert config.document["model"]["decoder"]["session_options"] == {
        entry: str(budget),
        "keep": "base",
    }
    profile_options = config.document["runtime_profiles"][0]["overlay"]["model"]["decoder"]["session_options"]
    assert profile_options == {entry: str(budget), "keep": "profile"}
    assert all(overlay["model"]["decoder"]["session_options"][entry] == str(budget) for overlay in config.overlays)
    assert config.overlays[0]["runtime_profiles"][0]["overlay"]["model"]["decoder"]["session_options"][entry] == str(
        budget
    )
    assert argument["model"]["decoder"]["session_options"][entry] == "777"
    assert util.execution_protocol() == {"overlay": inherited, "host_expert_bytes": budget}


@pytest.mark.parametrize("raw", ["0", "-1", "invalid", str(1 << 64)])
def test_host_budget_rejects_invalid_byte_counts_before_native_loading(tools, tmp_path, monkeypatch, raw):
    (tmp_path / "genai_config.json").write_text('{"model": {}}')
    monkeypatch.setenv(tools.util.HOST_EXPERT_BYTES_ENV, raw)
    monkeypatch.setattr(
        tools.util.og, "Config", lambda path: pytest.fail("invalid budget reached Config"), raising=False
    )
    with pytest.raises(ValueError, match="positive integer byte budget"):
        tools.util.load_model(tmp_path)


@pytest.mark.parametrize("stage", ["link", "config", "model"])
def test_budget_package_is_cleaned_up_after_native_load_failure(tools, tmp_path, monkeypatch, stage):
    util = tools.util
    (tmp_path / "genai_config.json").write_text('{"model": {}}')
    (tmp_path / "model.data").write_bytes(b"original asset")
    monkeypatch.setenv(util.HOST_EXPERT_BYTES_ENV, str(56 * (1 << 30)))
    temporary_directory = util.tempfile.TemporaryDirectory
    created = []

    def create_directory(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        created.append(Path(directory.name))
        return directory

    def fail(*args, **kwargs):
        raise RuntimeError("native load failed")

    monkeypatch.setattr(util, "tempfile", types.SimpleNamespace(TemporaryDirectory=create_directory))
    if stage == "link":
        monkeypatch.setattr(util.Path, "symlink_to", fail)
    monkeypatch.setattr(util.og, "Config", fail if stage == "config" else lambda path: object(), raising=False)
    monkeypatch.setattr(util.og, "Model", fail, raising=False)
    with pytest.raises(RuntimeError, match="native load failed"):
        util.load_model(tmp_path)
    assert created and all(not path.exists() for path in created)
    assert (tmp_path / "genai_config.json").exists()
    assert (tmp_path / "model.data").read_bytes() == b"original asset"


@pytest.mark.parametrize("batch,length", [(2, 33024), (1, 8192), (1, 262140)])
def test_behaviours_rejects_unsupported_baselines_before_probes(tools, tmp_path, monkeypatch, batch, length):
    monkeypatch.setattr(tools.behaviours, "load_model", lambda *args: object())
    capabilities = types.SimpleNamespace(configured_max_batch_size=batch, max_request_length=length)
    monkeypatch.setattr(
        tools.behaviours.og,
        "Engine",
        lambda model: types.SimpleNamespace(get_capabilities=lambda: capabilities),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["engine_behaviours.py", "package", "--out", str(tmp_path / "behaviours.json")])
    with pytest.raises(ValueError, match="requires the single-slot baseline"):
        tools.behaviours.main()
    assert not (tmp_path / "behaviours.json").exists()


def fake_event_engine(module, monkeypatch, events):
    """Minimal event loop for the real smoke/TTFT/run_turn control flow."""

    class Engine:
        def __init__(self, model=None):
            self.batches = list(events)
            self.request = types.SimpleNamespace(begin_turn=lambda *args: 0, close=lambda: None)

        def create_request(self):
            return self.request

        def create_event_buffer(self, size):
            return object()

        def has_pending_requests(self):
            return bool(self.batches)

        def run(self, buffer):
            return self.batches.pop(0)

        def get_capabilities(self):
            return types.SimpleNamespace(configured_max_batch_size=1, max_request_length=33024)

        def get_speculative_stats(self):
            return {"rounds": 0, "draft_tokens_accepted": 0}

    monkeypatch.setattr(module.og, "Engine", Engine, raising=False)
    monkeypatch.setattr(
        module.og,
        "TurnOptions",
        lambda request: types.SimpleNamespace(
            set_do_sample=lambda value: None, set_max_generated_tokens=lambda value: None
        ),
        raising=False,
    )
    return Engine()


def event(flags, token=5, finish_reason="eos"):
    return types.SimpleNamespace(
        flags=flags,
        token=token,
        finish_reason=finish_reason,
        error_code=17,
        usage=types.SimpleNamespace(prompt_tokens=1, generated_tokens=0, cached_prompt_tokens=0),
        matched_stop_string_index=None,
    )


def test_turn_usage_is_owned_after_event_buffer_reuse(tools, monkeypatch):
    terminal = event(8)
    terminal.usage = types.SimpleNamespace(prompt_tokens=11, generated_tokens=7, cached_prompt_tokens=3)
    engine = fake_event_engine(tools.util, monkeypatch, [[terminal]])
    first, _ = tools.util.run_turn(engine, object(), engine.request, [1], 8)
    terminal.usage.prompt_tokens = 23
    terminal.usage.generated_tokens = 5
    terminal.usage.cached_prompt_tokens = 2
    engine.batches = [[terminal]]
    second, _ = tools.util.run_turn(engine, object(), engine.request, [1], 8)
    assert (first.usage.prompt_tokens, first.usage.generated_tokens, first.usage.cached_prompt_tokens) == (11, 7, 3)
    assert (second.usage.prompt_tokens, second.usage.generated_tokens, second.usage.cached_prompt_tokens) == (23, 5, 2)


def test_turn_completion_timestamp_is_not_set_by_a_partial_pump(tools, monkeypatch):
    engine = fake_event_engine(tools.util, monkeypatch, [[event(4)], [event(8)]])
    ticks = iter([10.0, 11.0, 15.0])
    monkeypatch.setattr(tools.util, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    result, _ = tools.util.run_turn(engine, object(), engine.request, [1], 8, until=lambda r: bool(r.tokens))
    assert result.end is None and result.finish_reason is None
    tools.util.pump(engine, object(), result)
    assert result.end == 15.0 and result.finish_reason == "eos"


@pytest.mark.parametrize("module", ["lcb", "mmlu"])
@pytest.mark.parametrize(
    "reason,tokens,ttft_s,decode_s",
    [
        ("eos", [], None, 0.0),
        ("eos", [23], 1.0, 4.0),
        ("eos", [23, 24], 1.0, 4.0),
        ("length", [23, 24], 1.0, 1.0),
    ],
)
def test_evaluations_time_terminal_eos_and_completed_wall_time(
    tools, monkeypatch, module, reason, tokens, ttft_s, decode_s
):
    evaluator = getattr(tools, module)
    terminal = event(8, finish_reason=reason)
    terminal.usage.generated_tokens = len(tokens)
    engine = fake_event_engine(evaluator, monkeypatch, [[event(4, token=t)] for t in tokens] + [[terminal]])
    ticks = iter([10.0, *[11.0 + i for i in range(len(tokens))], 15.0])
    monkeypatch.setattr(tools.util, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    result = evaluator.generate(engine, object(), [1], 8)
    assert result["tokens"] == tokens + ([evaluator.EOS_APPENDED] if reason == "eos" else [])
    assert result["ttft_s"] == ttft_s
    assert result["decode_s"] == decode_s
    assert result["wall_s"] == 5.0
    assert result["usage_ok"]


def test_behaviour_error_checks_reject_empty_fragments(tools):
    with pytest.raises(ValueError, match="fragment must not be empty"):
        tools.behaviours.expect_error("residency", lambda: None, "")


@pytest.mark.parametrize(
    "message,ok", [("model state is no longer resident", True), ("incompatible function arguments", False)]
)
def test_behaviour_residency_check_does_not_accept_unrelated_errors(tools, monkeypatch, message, ok):
    monkeypatch.setattr(tools.behaviours, "CHECKS", [])

    def fail():
        raise RuntimeError(message)

    tools.behaviours.expect_error("residency", fail, "no longer resident")
    assert tools.behaviours.CHECKS[0]["ok"] is ok


def test_behaviour_stop_string_chunks_use_fresh_utf8_streams(tools):
    streams = []

    def create_stream():
        decoder = codecs.getincrementaldecoder("utf-8")()
        streams.append(decoder)
        return types.SimpleNamespace(decode=lambda token: decoder.decode(bytes([token])))

    tokenizer = types.SimpleNamespace(create_stream=create_stream)
    tokens = list("\u20acabc".encode())
    expected = ["", "", "\u20ac", "a", "b", "c"]
    assert tools.behaviours.decode_chunks(tokenizer, tokens) == expected
    stopped = tools.behaviours.decode_chunks(tokenizer, tokens[:4])
    assert "".join(stopped) == "\u20aca"
    assert "\u20aca" not in "".join(stopped[:-1])
    assert len(streams) == 2


@pytest.mark.parametrize("status", [0, 2, 17])
def test_plane_abi3_status_is_checked_before_output_comparison(tools, tmp_path, monkeypatch, status):
    plane = tools.plane

    class PleParams(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, type]]] = [("dummy", ctypes.c_int)]

    tensor = types.SimpleNamespace()
    tensor.to = lambda *args, **kwargs: tensor
    tensor.reshape = lambda *args: tensor
    tensor.data_ptr = lambda: 42

    class Lookup:
        _eos_token_id = 248046

        def initial_context(self):
            return np.zeros((1, 2), dtype=np.int64)

        def __call__(self, tokens, context):
            return tensor, context

    def compared():
        raise RuntimeError("output comparison reached")

    lib = types.SimpleNamespace(
        qwen38_plane_ple_rows=lambda *args: status,
        qwen38_plane_pool_keys=lambda *args: pytest.fail("unexpected pooled-key call"),
        qwen38_plane_transpose_f32=lambda *args: pytest.fail("unexpected transpose call"),
        qwen38_plane_sizeof_ple_params=lambda: ctypes.sizeof(PleParams),
    )
    monkeypatch.setattr(plane.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setattr(
        plane,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                init=lambda: None, current_stream=lambda: types.SimpleNamespace(cuda_stream=1), synchronize=compared
            ),
            tensor=lambda *args, **kwargs: tensor,
            zeros=lambda *args, **kwargs: tensor,
            empty=lambda *args, **kwargs: tensor,
            bfloat16=object(),
            int64=object(),
        ),
    )
    modules = {
        "plane": {
            "PleParams": PleParams,
            "PleDevice": lambda *args: types.SimpleNamespace(ptr=1, width=4),
        },
        "mobius._configs": {
            "Qwen4ExpConfig": types.SimpleNamespace(
                from_transformers=lambda config: types.SimpleNamespace(ple_layer_ids=[1])
            )
        },
        "mobius.generation._qwen4_exp_ple": {"Qwen4ExpCPUPLELookup": lambda *args: Lookup()},
        "mobius.integrations.transformers._builder": {
            "_load_transformers_config": lambda *args, **kwargs: (None, None)
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_plane_kernels.py",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path),
            "--runner-dir",
            str(tmp_path),
            "--prefill-dir",
            str(tmp_path),
        ],
    )
    message = f"qwen38_plane_ple_rows failed with CUDA status {status}" if status else "output comparison reached"
    with pytest.raises(RuntimeError, match=message):
        plane.main()
    assert lib.qwen38_plane_ple_rows.restype is ctypes.c_int


def test_ttft_reports_eos_before_first_token_as_a_benchmark_error(tools, monkeypatch):
    engine = fake_event_engine(tools.ttft, monkeypatch, [[event(8)]])
    with pytest.raises(RuntimeError, match="no token.*eos"):
        tools.ttft.ttft(engine, object(), [1])


def test_ttft_records_a_delivered_first_token(tools, monkeypatch):
    engine = fake_event_engine(tools.ttft, monkeypatch, [[], [event(4, token=23)], [event(8)]])
    result = tools.ttft.ttft(engine, object(), [1])
    assert result["first_token"] == 23
    assert result["ttft_s"] >= 0
    assert result["engine_steps"] == 2


@pytest.mark.parametrize("failure", [None, 1, 2])
def test_smoke_handles_eos_first_and_fails_on_engine_errors(tools, monkeypatch, capsys, failure):
    smoke = tools.smoke
    batches = [[event(8)]] if failure is None else [[event(4)], [event(failure)]]
    fake_event_engine(smoke, monkeypatch, batches)
    monkeypatch.setattr(smoke, "load_model", lambda *args: object())
    monkeypatch.setattr(
        smoke.og,
        "Tokenizer",
        lambda model: types.SimpleNamespace(
            encode=lambda text: [1],
            apply_chat_template=lambda *args, **kwargs: "prompt",
            create_stream=lambda: types.SimpleNamespace(decode=lambda token: "x"),
        ),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke.py", "package"])
    if failure is not None:
        with pytest.raises(RuntimeError, match="Engine failure"):
            smoke.main()
    else:
        smoke.main()
        assert "0 tokens; first-token latency and decode throughput unavailable; finish eos" in capsys.readouterr().out


@pytest.mark.parametrize("max_new", [0, -1])
def test_lcb_generate_rejects_nonpositive_budget_before_engine_use(tools, max_new):
    with pytest.raises(ValueError, match="budget must be positive"):
        tools.lcb.generate(object(), object(), [1], max_new)


@pytest.mark.parametrize("length", [16, 17])
def test_lcb_rejects_prompts_that_leave_no_context_for_generation(tools, length):
    with pytest.raises(ValueError, match="no generation budget for item 7"):
        tools.lcb.generation_budget([1] * length, 32, 16, 7)
    assert tools.lcb.generation_budget([1] * 15, 32, 16, 7) == 1


def test_lcb_eos_first_records_no_ttft(tools, monkeypatch):
    engine = fake_event_engine(tools.lcb, monkeypatch, [[event(8)]])
    result = tools.lcb.generate(engine, object(), [1], 8)
    assert result["ttft_s"] is None
    assert result["stop_token_appended"]
    assert result["usage_ok"]


@pytest.mark.parametrize("max_context", [1, 2])
@pytest.mark.parametrize("setting", ["QWEN38_OVERLAY_JSON", FORCE_REJECT_ENV, "QWEN38_HOST_EXPERT_BYTES"])
def test_lcb_shard_handles_eos_first_and_checks_budget_before_loading(
    tools, tmp_path, monkeypatch, capsys, max_context, setting
):
    lcb = tools.lcb
    monkeypatch.delenv(setting, raising=False)
    prompts = tmp_path / "prompts.jsonl"
    samples = tmp_path / "samples.jsonl"
    rows = [
        {
            "index": index,
            "ids": [1],
            "prompt_sha256": lcb.token_hash([1]),
            "question_id": str(index),
            "test_file": "tests.json",
            "func_name": None,
        }
        for index in range(175)
    ]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    samples.write_text("".join(json.dumps({"index": index, "difficulty": "easy"}) + "\n" for index in range(175)))
    eval_lib = types.ModuleType("eval_lib")
    eval_lib.messages_of = lambda sample: []
    eval_lib.qwen38_prompt = lambda messages, thinking: "prompt"
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda path: types.SimpleNamespace(decode=lambda *args, **kwargs: "")
    )
    monkeypatch.setitem(sys.modules, "eval_lib", eval_lib)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        lcb.og,
        "Tokenizer",
        lambda model: types.SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
        raising=False,
    )
    loaded, generated, warmups = [], [], []
    engine = types.SimpleNamespace(create_request=lambda: types.SimpleNamespace(close=lambda: None))

    def open_engine(*args):
        loaded.append(True)
        return object(), engine, object(), 0.0

    def generate(engine, buffer, prompt, budget):
        generated.append(budget)
        return {
            "tokens": [lcb.EOS_APPENDED],
            "stop_token_appended": True,
            "usage_ok": True,
            "ttft_s": None,
            "decode_s": 0.0,
            "wall_s": 0.0,
            "cycles": 0,
            "accepted": 0,
        }

    monkeypatch.setattr(lcb, "open_engine", open_engine)
    monkeypatch.setattr(lcb, "generate", generate)
    monkeypatch.setattr(lcb, "run_turn", lambda engine, buffer, request, prompt, budget: warmups.append(budget))
    args = types.SimpleNamespace(
        prompts=prompts,
        samples=samples,
        shard=0,
        shards=100,
        out=tmp_path / "lcb.jsonl",
        max_new=32,
        max_context=max_context,
        model="package",
        provider=None,
        checkpoint="checkpoint",
    )
    if max_context == 1:
        with pytest.raises(ValueError, match="no generation budget"):
            lcb.run_shard(args)
        assert not loaded and not generated and not warmups
        assert not args.out.exists()
    else:
        lcb.run_shard(args)
        assert generated == [1, 1]
        assert warmups == [1]
        records = [json.loads(line) for line in args.out.read_text().splitlines()]
        assert [r["index"] for r in records] == [0, 100]
        assert all(r["prefill_s"] is None for r in records)
        assert capsys.readouterr().out.count("ttft=n/a (no token)") == 2
        metadata_key = {
            "QWEN38_OVERLAY_JSON": "overlay",
            FORCE_REJECT_ENV: "force_reject",
            "QWEN38_HOST_EXPERT_BYTES": "host_expert_bytes",
        }[setting]
        protocol = json.loads(args.out.with_suffix(".protocol.json").read_text())
        assert protocol[metadata_key] == ({} if setting == "QWEN38_OVERLAY_JSON" else None)
        value = json.dumps({"model": {"qwen38_graph": {"mtp": {"prefill_head": False}}}})
        monkeypatch.setenv(setting, value if setting == "QWEN38_OVERLAY_JSON" else "1")
        with pytest.raises(SystemExit, match="protocol mismatch"):
            lcb.run_shard(args)
        assert generated == [1, 1] and warmups == [1]


@pytest.mark.parametrize("problem", [None, "duplicate", "missing", "extra", "protocol"])
def test_lcb_grading_requires_disjoint_complete_consistent_shards(tools, tmp_path, problem):
    indices = [[0, 1], [2, 3]]
    if problem == "duplicate":
        indices[1].append(0)
    elif problem == "missing":
        indices[1].remove(3)
    elif problem == "extra":
        indices[1].append(4)
    paths = []
    for shard, rows in enumerate(indices):
        path = tmp_path / f"shard{shard}.jsonl"
        path.write_text("".join(json.dumps({"index": index}) + "\n" for index in rows))
        protocol = {"shard": shard, "shards": 2, "max_context": 32 if problem == "protocol" and shard else 16}
        path.with_suffix(".protocol.json").write_text(json.dumps(protocol))
        paths.append(path)
    if problem:
        with pytest.raises(ValueError, match="duplicate|index set|protocol mismatch"):
            tools.lcb.load_arm_records(paths, {0, 1, 2, 3})
    else:
        assert set(tools.lcb.load_arm_records(paths, {0, 1, 2, 3})) == {0, 1, 2, 3}


@pytest.mark.parametrize("eos_first", [False, True])
def test_lcb_grading_persists_accuracy_without_decode_metrics(tools, tmp_path, monkeypatch, eos_first):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "".join(
            json.dumps({"index": index, "difficulty": ("easy", "medium", "hard")[index % 3]}) + "\n"
            for index in range(175)
        )
    )
    records = tmp_path / "shard.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "index": index,
                    "question_id": str(index),
                    "test_file": "tests.json",
                    "tokens": [tools.lcb.EOS_APPENDED if eos_first else 23],
                    "text": "",
                    "generated_tokens": 1,
                    "truncated": not eos_first,
                    "tps": 0.0,
                    "cycles": 0 if eos_first else 1,
                    "accept_rate": 0.0 if eos_first else 0.5,
                    "prefill_s": None if eos_first else 0.1,
                    "usage_ok": True,
                }
            )
            + "\n"
            for index in range(175)
        )
    )
    records.with_suffix(".protocol.json").write_text("{}")
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: object())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    scipy_stats = types.ModuleType("scipy.stats")
    scipy_stats.binomtest = lambda *args: pytest.fail("unpaired arm must not need McNemar")
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stats)
    monkeypatch.setattr(
        tools.lcb,
        "grade_records_in_docker",
        lambda items, *args: {
            (item["arm"], int(item["key"])): {"correct": True, "extracted_code": False} for item in items
        },
    )
    out = tmp_path / "report.json"
    args = types.SimpleNamespace(
        checkpoint=tmp_path,
        samples=samples,
        caps=[16],
        arms=[f"engine={records}"],
        tests=tmp_path,
        lcb_repo=tmp_path,
        grader=tmp_path / "grader.py",
        workers=1,
        timeout=1,
        out=out,
    )
    tools.lcb.grade_suite(args)
    report = json.loads(out.read_text())
    assert report["caps"]["16"]["engine"]["pass_at_1_pct"] == 100.0
    performance = report["performance"]["engine"]
    assert performance["total_generated_tokens"] == 175
    assert performance["decode_tps_median"] is None
    assert performance["decode_tps_mean"] is None
    assert performance["acceptance_rate_mean"] == (None if eos_first else 0.5)
    assert performance["prefill_s_median"] == (None if eos_first else 0.1)
    assert performance["prefill_s_mean"] == (None if eos_first else 0.1)
