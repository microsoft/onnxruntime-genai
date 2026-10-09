# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Model-free tests of the operator prefix-cache QA matrix and failure detection."""

import importlib.util
import json
import subprocess
import sys
import weakref
from enum import Enum, IntFlag
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "python" / "engine" / "prefix-cache-qa.py"
_SPEC = importlib.util.spec_from_file_location("prefix_cache_qa", _SCRIPT)
qa = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qa
_SPEC.loader.exec_module(qa)


def test_standalone_import_initializes_optional_provider_discovery():
    program = """
import importlib
import sys
import types

module = types.ModuleType("prefix_cache_standalone")
module.__file__ = sys.argv[1]
sys.modules[module.__name__] = module
with open(module.__file__, encoding="utf-8") as source:
    exec(compile(source.read(), module.__file__, "exec"), module.__dict__)
assert importlib.util.find_spec("prefix_cache_nonexistent_provider") is None
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", program, str(_SCRIPT)], check=True)


def profile(hybrid=True):
    return qa.Profile(4, 8, 2, 16, 64, hybrid)


def row(cached=8, length=15, tokens=(7, 8), cancelled=False, ttft=0.1):
    return {
        "prompt": "p",
        "prompt_length": length,
        "cached_tokens": cached,
        "tokens": list(tokens),
        "cancelled": cancelled,
        "ttft_s": ttft,
    }


@pytest.mark.parametrize(
    ("hybrid", "length", "adopted", "expected"),
    [
        (True, 15, 0, 8),
        (True, 16, 0, 8),
        (True, 17, 0, 16),
        (True, 23, 8, 16),
        (True, 23, 12, 20),
        (False, 15, 0, 12),
        (False, 16, 0, 12),
        (False, 17, 0, 16),
    ],
)
def test_boundary_oracle_leaves_final_token_and_accounts_for_shifted_chunks(hybrid, length, adopted, expected):
    assert profile(hybrid).warm_boundary(length, adopted) == expected


def test_unaligned_chunk_oracle_does_not_invent_fixed_checkpoints():
    assert qa.Profile(4, 6, 2, 16, 64, True).warm_boundary(25) == 24
    assert qa.Profile(4, 6, 2, 16, 64, True).warm_boundary(23) == 12


@pytest.mark.parametrize("rule", ["warm", "exact-warm"])
def test_detects_short_first_checkpoint_plateau(rule):
    with pytest.raises(RuntimeError, match="cache plateau"):
        qa.check_result(row(8, 23), [7, 8], profile(), rule, row(8, 23), 2)


def test_detects_branch_between_checkpoints_plateau():
    with pytest.raises(RuntimeError, match="cache plateau"):
        qa.check_result(row(8, 25), [7, 8], profile(), "warm", row(8, 25), 2)


def test_allows_lease_deferral_then_requires_progress_after_release():
    qa.check_result(row(8, 25), [7, 8], profile(), "safe", None, 2)
    qa.check_result(row(24, 25), [7, 8], profile(), "warm", row(8, 25), 2)


@pytest.mark.parametrize("cached", [-4, 3, 15, 16])
def test_rejects_unsafe_boundaries(cached):
    with pytest.raises(RuntimeError, match="unsafe cached-token boundary"):
        qa.check_result(row(cached), [7, 8], profile(), "safe", None, 2)


def test_parity_failure_is_not_waived_for_cross_shape_numerical_changes():
    actual = row(tokens=(7, 9))
    with pytest.raises(RuntimeError, match=r"differs.*token 1"):
        qa.check_result(actual, [7, 8], profile(), "safe", None, 2)
    assert actual["parity_difference"] == {
        "index": 1,
        "actual_token": 9,
        "expected_token": 8,
        "actual_length": 2,
        "expected_length": 2,
    }


@pytest.mark.parametrize(
    ("actual", "expected", "index", "actual_token", "expected_token"),
    [
        ([7], [7, 8], 1, None, 8),
        ([7, 8], [7], 1, 8, None),
        ([], [7], 0, None, 7),
        ([7], [], 0, 7, None),
        ([9, 8], [7, 8], 0, 9, 7),
    ],
)
def test_token_difference_records_mismatches_and_length_only_divergence(
    actual, expected, index, actual_token, expected_token
):
    assert qa.token_difference(actual, expected) == {
        "index": index,
        "actual_token": actual_token,
        "expected_token": expected_token,
        "actual_length": len(actual),
        "expected_length": len(expected),
    }


def test_matching_outputs_have_no_parity_difference():
    actual = row()
    qa.check_result(actual, [7, 8], profile(), "safe", None, 2)
    assert actual["parity_difference"] is None


def test_parity_and_reuse_failures_are_reported_independently():
    actual = row(8, 25, (7, 9))
    with pytest.raises(qa.ValidationError) as failure:
        qa.check_result(actual, [7, 8], profile(), "warm", row(8, 25), 2)
    assert "differs" in str(failure.value)
    assert "cache plateau" in str(failure.value)
    assert actual["checks"]["parity"] is False
    assert actual["checks"]["reuse"] is False
    assert actual["checks"]["safety"] is True


def test_cancellation_checks_only_emitted_reference_prefix():
    qa.check_result(row(0, tokens=(7,), cancelled=True), [7, 8], profile(), "cold", None, 2)
    with pytest.raises(RuntimeError, match="differs"):
        qa.check_result(row(0, tokens=(9,), cancelled=True), [7, 8], profile(), "cold", None, 2)


def test_truncated_generation_and_false_cold_hits_fail():
    with pytest.raises(RuntimeError, match="expected 2 generated tokens"):
        qa.check_result(row(tokens=(7,)), [7], profile(), "safe", None, 2)
    with pytest.raises(RuntimeError, match="fresh Engine"):
        qa.check_result(row(), [7, 8], profile(), "cold", None, 2)


def test_pressure_requires_observable_turnover():
    qa.check_result(row(0, 25), [7, 8], profile(), "evicted", None, 2)
    with pytest.raises(RuntimeError, match="did not demonstrate"):
        qa.check_result(row(24, 25), [7, 8], profile(), "evicted", None, 2)


def test_latency_uses_cold_reference_not_previous_warm_request():
    qa.check_result(row(24, 25), [7, 8], profile(), "warm", row(24, 25, ttft=0.1), 2, 0.2, 1.0)
    with pytest.raises(RuntimeError, match="warm TTFT ratio"):
        qa.check_result(row(24, 25, ttft=0.5), [7, 8], profile(), "warm", row(24, 25), 2, 0.2, 1.0)


def test_random_matrix_is_seeded_and_every_prompt_is_referenced():
    first = qa.make_plan(profile(), None, qa.SUITES, 42, 4, 10)
    second = qa.make_plan(profile(), None, qa.SUITES, 42, 4, 10)
    assert first == second
    specs, scenarios = first
    assert all(name in specs for scenario in scenarios for action in scenario.actions for name in action.prompts)
    assert {3, 4, 5, 7, 8, 9, 15, 16, 17} <= {spec.length for spec in specs.values()}
    assert {3, 4, 5, 7, 8, 9, 15, 16, 17} <= {
        spec.mutation for spec in specs.values() if spec.name.startswith("branch-")
    }


def test_pressure_churn_exceeds_entire_configured_block_budget():
    specs, scenarios = qa.make_plan(profile(), [25], ["pressure"], 42, 1, 2, 2, 20)
    pressure = scenarios[0]
    churn = [spec for name, spec in specs.items() if name.startswith("churn-")]
    assert sum(spec.length // 4 for spec in churn) > 20
    assert pressure.actions[-2].check == "evicted"
    assert pressure.actions[-1].check == "warm"


def test_batch_one_rejects_concurrency_instead_of_skipping():
    with pytest.raises(ValueError, match="max_batch_size"):
        qa.make_plan(qa.Profile(4, 8, 1, 16, 64, True), [25], ["leases"], 42, 1, 2)


def test_concurrency_plan_covers_both_admission_orders_and_single_row_slot_isolation():
    _, scenarios = qa.make_plan(profile(), [25], ["concurrency"], 42, 1, 2)
    assert scenarios[0].actions[0].prompts == ("short-seed", "long")
    assert scenarios[1].actions[0].prompts == ("long", "short-seed")
    pinned = scenarios[2]
    assert pinned.actions[0].keep_open
    assert pinned.actions[1].prompts == ("long",)
    assert pinned.actions[2].release
    assert all(scenario.uncached_control for scenario in scenarios)


def test_named_selection_preserves_actions_controls_and_only_required_references():
    specs, scenarios = qa.make_plan(profile(), [25], ["branching", "concurrency"], 42, 1, 2)
    selected_specs, selected = qa.select_scenarios(specs, scenarios, ["branch-16", "simultaneous-long-short"])
    assert [scenario.name for scenario in selected] == ["branch-16", "simultaneous-long-short"]
    assert set(selected_specs) == {"long", "branch-16", "short-seed"}
    assert selected == [scenario for scenario in scenarios if scenario.name in {item.name for item in selected}]
    assert selected[1].uncached_control
    assert qa.select_scenarios(specs, scenarios, None) == (specs, scenarios)


def test_named_selection_rejects_unknown_cases_instead_of_silently_skipping():
    specs, scenarios = qa.make_plan(profile(), [25], ["boundaries"], 42, 1, 2)
    with pytest.raises(ValueError, match=r"Unknown scenario.*branch-16"):
        qa.select_scenarios(specs, scenarios, ["boundary-25", "branch-16"])


@pytest.mark.parametrize("failure", [None, "batch", "unstable", "false-hit"])
def test_uncached_controls_use_fresh_engines_preserve_schedule_and_keep_sequential_parity(
    tmp_path, monkeypatch, failure
):
    drivers = []

    class ControlDriver:
        def __init__(self, *args):
            self.number = len(drivers)
            self.actions = []
            self.retained = set()
            self.closed = False
            drivers.append(self)

        def close(self):
            self.retained.clear()
            self.closed = True

        def run(self, action):
            self.actions.append(action)
            if action.release:
                for name in action.prompts:
                    self.retained.remove(name)
                return [], {}
            if action.keep_open:
                self.retained.update(action.prompts)
            token = 8 if failure == "batch" and len(action.prompts) > 1 else 7
            if failure == "unstable" and self.number == 1:
                token = 9
            cached = 4 if failure == "false-hit" else 0
            return [{**row(cached, 25, (token,)), "prompt": name} for name in reversed(action.prompts)], {}

    monkeypatch.setattr(qa, "Driver", ControlDriver)
    _, scenarios = qa.make_plan(profile(), [25], ["concurrency"], 42, 1, 2)
    args = SimpleNamespace(reference_repeats=2, generated_tokens=1, fail_fast=False, output=tmp_path / "report.json")
    entry = {"uncached_controls": [], "control_stability": {}}
    report = {"profiles": [entry]}
    references = {name: {"tokens": [7]} for name in ("short-seed", "long")}
    qa.run_uncached_controls(None, None, None, None, {}, args, False, profile(), scenarios, references, entry, report)
    assert len(drivers) == 6
    assert all(driver.closed and not driver.retained for driver in drivers)
    assert all(action.check == "cold" for driver in drivers for action in driver.actions)
    assert drivers[0].actions[0].prompts == ("short-seed", "long")
    assert drivers[2].actions[0].prompts == ("long", "short-seed")
    assert drivers[4].actions[0].keep_open
    assert drivers[4].actions[2].release
    controls = entry["uncached_controls"]
    assert all(control["status"] == "passed" for control in controls) == (failure is None)
    if failure == "batch":
        actual = controls[0]["steps"][0]["rows"][0]
        assert actual["cached_tokens"] == 0
        assert actual["checks"]["parity"] is False
        assert controls[1]["steps"][0]["rows"][0]["checks"]["repeat_parity"] is True
        assert entry["control_stability"]["simultaneous-short-long"] is True
    elif failure == "unstable":
        assert entry["control_stability"]["simultaneous-short-long"] is False
        assert controls[1]["steps"][0]["rows"][0]["checks"]["repeat_parity"] is False
        assert controls[0]["steps"][0]["rows"][0]["tokens"] == [7]
    elif failure == "false-hit":
        assert controls[0]["steps"][0]["rows"][0]["checks"]["reuse"] is False
    assert json.loads(args.output.read_text())["profiles"][0] == json.loads(json.dumps(entry))


def test_uncached_control_failure_releases_pinned_owner_and_persists_partial_report(tmp_path, monkeypatch):
    closed = []

    class ControlDriver:
        def __init__(self, *args):
            self.calls = 0

        def close(self):
            closed.append(True)

        def run(self, action):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Execution failed with a pinned owner.")
            return [{**row(0, 25, (7,)), "prompt": "p"}], {}

    monkeypatch.setattr(qa, "Driver", ControlDriver)
    scenario = qa.Scenario("held-slot", (qa.Action(("p",), keep_open=True), qa.Action(("q",))), uncached_control=True)
    args = SimpleNamespace(reference_repeats=2, generated_tokens=1, fail_fast=False, output=tmp_path / "report.json")
    entry = {"uncached_controls": [], "control_stability": {}}
    with pytest.raises(RuntimeError, match="pinned owner"):
        qa.run_uncached_controls(
            None,
            None,
            None,
            None,
            {},
            args,
            False,
            profile(),
            [scenario],
            {"p": {"tokens": [7]}, "q": {"tokens": [7]}},
            entry,
            {"profiles": [entry]},
        )
    assert closed == [True]
    control = json.loads(args.output.read_text())["profiles"][0]["uncached_controls"][0]
    assert control["status"] == "failed"
    assert control["steps"][0]["status"] == "passed"
    assert control["steps"][1]["status"] == "failed"
    assert entry["control_stability"]["held-slot"] is None


def test_one_uncached_control_repeat_leaves_stability_unchecked(tmp_path, monkeypatch):
    class ControlDriver:
        def __init__(self, *args):
            pass

        def close(self):
            pass

        def run(self, action):
            return [row(0, 25)], {}

    monkeypatch.setattr(qa, "Driver", ControlDriver)
    scenario = qa.Scenario("single", (qa.Action(("p",)),), uncached_control=True)
    args = SimpleNamespace(reference_repeats=1, generated_tokens=2, fail_fast=False, output=tmp_path / "report.json")
    entry = {"uncached_controls": [], "control_stability": {}}
    qa.run_uncached_controls(
        None,
        None,
        None,
        None,
        {},
        args,
        False,
        profile(),
        [scenario],
        {"p": {"tokens": [7, 8]}},
        entry,
        {"profiles": [entry]},
    )
    assert entry["control_stability"]["single"] is None
    control = entry["uncached_controls"][0]
    assert control["status"] == "passed"
    assert control["steps"][0]["rows"][0]["checks"]["repeat_parity"] is None


def test_alternation_requires_deep_hits_on_each_switch_not_just_immediate_repeats():
    specs, scenarios = qa.make_plan(profile(), [25], ["alternation"], 42, 1, 2, generated=2)
    assert set(specs) == {"long", "alternating-branch"}
    assert specs["alternating-branch"].mutation == 15
    actions = scenarios[0].actions
    assert [action.check for action in actions[:4]] == ["cold", "exact-warm", "safe", "warm"]
    assert [action.prompts for action in actions[4:]] == [("long",), ("alternating-branch",)] * 3
    assert all(action.check == "warm" for action in actions[4:])
    with pytest.raises(qa.ValidationError, match="cache plateau"):
        qa.check_result(row(8, specs["long"].length), [7, 8], profile(), actions[4].check, row(32, 37), 2)


def test_alternation_accepts_root_divergence_and_records_requested_round_count():
    specs, scenarios = qa.make_plan(
        profile(), [25], ["alternation"], 42, 1, 2, generated=2, alternation_split=0, alternation_rounds=5
    )
    assert specs["alternating-branch"].mutation == 0
    assert len(scenarios[0].actions) == 14


@pytest.mark.parametrize("split", [-1, 37])
def test_alternation_rejects_mutation_outside_the_prompt(split):
    with pytest.raises(ValueError, match="mutation inside"):
        qa.make_plan(profile(), [25], ["alternation"], 42, 1, 2, alternation_split=split)


def test_alternation_refuses_insufficient_retention_budget_instead_of_skipping():
    with pytest.raises(ValueError, match="at least 19"):
        qa.make_plan(qa.Profile(4, 8, 2, 16, 18, True), [25], ["alternation"], 42, 1, 2, generated=2)
    qa.make_plan(qa.Profile(4, 8, 2, 16, 19, True), [25], ["alternation"], 42, 1, 2, generated=2)


def test_alternation_requires_two_checkpoints_only_for_hybrid_models():
    with pytest.raises(ValueError, match="two fixed-state checkpoints"):
        qa.make_plan(qa.Profile(4, 8, 1, 16, 64, True), [25], ["alternation"], 42, 1, 2)
    qa.make_plan(qa.Profile(4, 8, 1, 16, 64, False), [25], ["alternation"], 42, 1, 2)


@pytest.mark.parametrize("draft_alias", ["dflash2", "dspark"])
def test_profile_and_ablation_overlay_preserve_input_config(draft_alias):
    config = {
        "model": {
            "decoder": {"state_groups": [{"kind": "fixed_recurrent"}]},
            draft_alias: {"filename": "draft.onnx", "sliding_window": 16, "block_size": 4},
        },
        "engine": {"dynamic_batching": {"block_size": 4, "num_blocks": 64, "max_batch_size": 2}},
        "search": {"chunk_size": 8},
    }
    original = json.dumps(config)
    resolved = qa.read_profile(config)
    assert resolved.hybrid
    assert resolved.draft_window == 20
    overlay = resolved.overlay(False, "target-only")
    assert overlay["engine"]["dynamic_batching"]["prefix_caching"] is False
    assert overlay["model"]["dflash2"]["filename"] == ""
    assert overlay["model"]["mtp"]["enabled"] is False
    assert json.dumps(config) == original


def test_budget_smaller_than_chunk_is_rejected():
    with pytest.raises(ValueError, match="max_scheduled_tokens"):
        qa.read_profile({"model": {}, "engine": {"dynamic_batching": {"block_size": 4, "max_scheduled_tokens": 4}}}, 8)


def test_plan_only_does_not_import_genai(tmp_path, monkeypatch, capsys):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {}, "engine": {"dynamic_batching": {"block_size": 4, "max_batch_size": 2}}})
    )
    monkeypatch.setattr(qa.importlib, "import_module", lambda name: pytest.fail(f"Unexpected runtime import: {name}"))
    qa.main(["-m", str(tmp_path), "--plan-only", "--suites", "boundaries"])
    plan = json.loads(capsys.readouterr().out)
    assert plan[0]["scenarios"]


def test_plan_only_named_selection_does_not_import_runtime_and_preserves_controls(tmp_path, monkeypatch, capsys):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {}, "engine": {"dynamic_batching": {"block_size": 4, "max_batch_size": 2}}})
    )
    monkeypatch.setattr(qa.importlib, "import_module", lambda name: pytest.fail(f"Unexpected runtime import: {name}"))
    qa.main(
        [
            "-m",
            str(tmp_path),
            "--plan-only",
            "--suites",
            "concurrency",
            "--scenarios",
            "simultaneous-long-short",
        ]
    )
    plan = json.loads(capsys.readouterr().out)[0]
    assert [scenario["name"] for scenario in plan["scenarios"]] == ["simultaneous-long-short"]
    assert plan["scenarios"][0]["uncached_control"]
    assert {spec["name"] for spec in plan["prompts"]} == {"short-seed", "long"}


def test_unknown_named_case_overwrites_stale_success_report(tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {}, "engine": {"dynamic_batching": {"block_size": 4}}})
    )
    output = tmp_path / "report.json"
    output.write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="Unknown scenario"):
        qa.main(["-m", str(tmp_path), "--scenarios", "misspelled", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["arguments"]["scenarios"] == ["misspelled"]


def test_named_case_must_exist_in_every_chunk_profile(tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {}, "engine": {"dynamic_batching": {"block_size": 4}}})
    )
    with pytest.raises(ValueError, match="Unknown scenario"):
        qa.main(
            [
                "-m",
                str(tmp_path),
                "--plan-only",
                "--suites",
                "branching",
                "--chunk-sizes",
                "8",
                "16",
                "--scenarios",
                "branch-8",
            ]
        )


class CorpusTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        roles = [message["role"] for message in json.loads(messages)]
        assert roles == ["system", "user"]
        assert add_generation_prompt
        return "H" + "PREFIX_CACHE_QA_SYSTEM_BODY" + "assistant<think>"

    def encode(self, text):
        if text == "H":
            return [10]
        if text == "assistant<think>":
            return [11, 12, 13]
        if text.startswith(" alternative"):
            return [201, 202, 203]
        return list(range(20, 150))


def test_named_selection_preserves_exact_prompt_tokens_and_hashes():
    specs, scenarios = qa.make_plan(profile(), [25], ["branching", "concurrency"], 42, 1, 2)
    original, _ = qa.materialize_prompts(CorpusTokenizer(), specs, "body", "task")
    selected_specs, _ = qa.select_scenarios(specs, scenarios, ["branch-16", "simultaneous-long-short"])
    selected, _ = qa.materialize_prompts(CorpusTokenizer(), selected_specs, "body", "task")
    assert selected == {name: original[name] for name in selected_specs}
    assert {name: qa.token_digest(tokens) for name, tokens in selected.items()} == {
        name: qa.token_digest(original[name]) for name in selected_specs
    }


def test_materialized_prompts_have_exact_lengths_and_exact_branch_position():
    specs = {"base": qa.PromptSpec("base", 25), "branch": qa.PromptSpec("branch", 25, 7)}
    prompts, thinking = qa.materialize_prompts(CorpusTokenizer(), specs, "body", "task")
    assert thinking
    assert all(len(tokens) == 25 for tokens in prompts.values())
    assert prompts["base"][:7] == prompts["branch"][:7]
    assert prompts["base"][7] != prompts["branch"][7]
    assert prompts["base"][8:] == prompts["branch"][8:]
    assert prompts["branch"][-3:] == (11, 12, 13)


def test_pressure_prompts_materialize_distinct_root_blocks():
    specs, _ = qa.make_plan(profile(), [25], ["pressure"], 42, 1, 2, 2, 20)
    prompts, _ = qa.materialize_prompts(CorpusTokenizer(), specs, "body", "task")
    churn = [tokens[:4] for name, tokens in prompts.items() if name.startswith("churn-")]
    assert len(set(churn)) == len(churn)


def test_pressure_mutations_can_span_multiple_tokens_without_duplicate_roots():
    specs = {f"churn-{index}": qa.PromptSpec(f"churn-{index}", 25, 0, index + 1, 4) for index in range(300)}
    prompts, _ = qa.materialize_prompts(CorpusTokenizer(), specs, "body", "task")
    assert len({tokens[:4] for tokens in prompts.values()}) == 300
    assert all(tokens[4:] == prompts["churn-0"][4:] for tokens in prompts.values())


def test_short_chat_inputs_and_insufficient_mutation_domains_fail():
    with pytest.raises(ValueError, match="complete chat prompt"):
        qa.materialize_prompts(CorpusTokenizer(), {"short": qa.PromptSpec("short", 3)}, "body", "task")
    with pytest.raises(ValueError, match="too few distinct token"):
        qa.materialize_prompts(CorpusTokenizer(), {"branch": qa.PromptSpec("branch", 25, 0, 1000)}, "body", "task")


def test_runtime_import_failure_persists_failed_report_and_does_not_edit_model(tmp_path, monkeypatch):
    config_path = tmp_path / "genai_config.json"
    source = json.dumps(
        {"model": {}, "engine": {"dynamic_batching": {"block_size": 4, "max_batch_size": 2, "num_blocks": 64}}}
    )
    config_path.write_text(source)
    output = tmp_path / "report.json"

    def fail_import(name):
        raise ImportError(f"Missing runtime: {name}")

    monkeypatch.setattr(qa.importlib, "import_module", fail_import)
    with pytest.raises(ImportError, match="Missing runtime"):
        qa.main(["-m", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "ImportError" in report["error"]
    assert config_path.read_text() == source


def test_invalid_model_does_not_leave_a_previous_success_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"status":"passed"}')
    with pytest.raises(FileNotFoundError):
        qa.main(["-m", str(tmp_path / "missing-model"), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["started_at_utc"]


def test_target_only_dspark_overlay_does_not_introduce_conflicting_alias():
    calls = []

    class Config:
        def __init__(self, path):
            pass

        def overlay(self, value):
            calls.append(json.loads(value))

    runtime = SimpleNamespace(Config=Config, Model=lambda config: config)
    args = SimpleNamespace(model_path=Path("model"), execution_provider=None)
    _, overlay = qa.make_model(
        runtime, args, profile(), True, "target-only", {"model": {"dspark": {"filename": "draft.onnx"}}}
    )
    assert calls == [overlay]
    assert "dflash2" not in overlay["model"]
    assert overlay["model"]["dspark"]["filename"] == ""


@pytest.mark.parametrize(
    ("failure", "reference_repeats", "include_prompt_tokens"),
    [(None, 1, False), (None, 2, True), ("cached", 2, False), ("reference", 2, False)],
)
def test_orchestrator_releases_regular_model_and_collects_independent_failures(
    tmp_path, monkeypatch, failure, reference_repeats, include_prompt_tokens
):
    loaded = []

    class Config:
        def __init__(self, path):
            pass

        def overlay(self, value):
            pass

    class Model:
        def __init__(self, config):
            assert all(reference() is None for reference in loaded), "Model weights loaded twice concurrently"
            self.number = len(loaded)
            loaded.append(weakref.ref(self))

    class Coordinator:
        def __init__(self, og, np, model, tokenizer, prompts, args, template_thinks):
            self.model = model
            self.prompts = prompts
            self.reference_runs = 0
            self.engine = SimpleNamespace(get_capabilities=lambda: SimpleNamespace(max_request_length=10000))

        def close(self):
            pass

        def run(self, action):
            rows = []
            for name in action.prompts:
                length = len(self.prompts[name])
                cached = 0 if action.check in ("cold", "evicted") else profile().warm_boundary(length)
                wrong_cached = failure == "cached" and self.model.number == 1 and action.check == "cold"
                wrong_reference = failure == "reference" and self.model.number == 0 and self.reference_runs == 1
                token = 8 if wrong_cached or wrong_reference else 7
                rows.append({**row(cached, length, (token,)), "prompt": name})
            if self.model.number == 0:
                self.reference_runs += 1
            return rows, {"delta": {"draft_tokens_proposed": 0}}

    monkeypatch.setattr(qa, "Driver", Coordinator)
    runtime = SimpleNamespace(Config=Config, Model=Model, Tokenizer=lambda model: CorpusTokenizer())
    args = SimpleNamespace(
        model_path=tmp_path,
        execution_provider=None,
        prompt_file=None,
        user_prompt="task",
        generated_tokens=1,
        reference_repeats=reference_repeats,
        pressure_num_blocks=20,
        output=tmp_path / "report.json",
        max_warm_ttft_ratio=None,
        fail_fast=False,
        include_prompt_tokens=include_prompt_tokens,
    )
    specs, scenarios = qa.make_plan(profile(), [25], ["boundaries", "pressure"], 42, 1, 2, 1, 20)
    report = {"profiles": []}
    qa.run_profile(runtime, None, args, {"model": {}}, profile(), specs, scenarios, "configured", report)
    assert len(loaded) == 3
    assert all(reference() is None for reference in loaded)
    entry = json.loads(args.output.read_text())["profiles"][0]
    assert entry["corpus_sha256"]
    assert entry["prompts"]["churn-0"]["mutation_width"] == 4
    assert entry["selected_scenarios"] == [scenario.name for scenario in scenarios]
    for prompt in entry["prompts"].values():
        assert ("tokens" in prompt) is include_prompt_tokens
        if include_prompt_tokens:
            assert len(prompt["tokens"]) == prompt["length"]
            assert qa.token_digest(prompt["tokens"]) == prompt["sha256"]
    assert len(entry["references"]) == len(entry["prompts"]) * reference_repeats
    if failure == "cached":
        assert entry["scenarios"][0]["status"] == "failed"
        assert len(entry["scenarios"][0]["steps"]) == 3
        assert entry["scenarios"][0]["steps"][0]["rows"][0]["checks"]["parity"] is False
        assert entry["scenarios"][1]["status"] == "passed"
    else:
        assert all(scenario["status"] == "passed" for scenario in entry["scenarios"])
    if failure == "reference":
        assert len(entry["reference_failures"]) == 1
        assert entry["reference_stability"]["length-25"] is False
        assert entry["references"][0]["tokens"] == [7]
        assert entry["references"][1]["tokens"] == [8]
    else:
        assert entry["reference_failures"] == []
        assert all(
            value is (None if reference_repeats == 1 else True) for value in entry["reference_stability"].values()
        )


@pytest.mark.parametrize("failure", ["scenario", "reference", "control"])
def test_cli_returns_failure_after_collecting_validation_findings(tmp_path, monkeypatch, failure):
    (tmp_path / "genai_config.json").write_text(
        json.dumps(
            {"model": {}, "engine": {"dynamic_batching": {"block_size": 4, "max_batch_size": 2, "num_blocks": 64}}}
        )
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(qa.importlib, "import_module", lambda name: SimpleNamespace(__file__="test-runtime"))
    monkeypatch.setattr(qa.importlib.util, "find_spec", lambda name: None)

    def failed_profile(og, np, args, source, profile, specs, scenarios, mode, report):
        report["profiles"].append(
            {
                "reference_failures": ["unstable"] if failure == "reference" else [],
                "scenarios": [{"name": "red-case", "status": "failed" if failure == "scenario" else "passed"}],
                "uncached_controls": [{"name": "batch", "status": "failed" if failure == "control" else "passed"}],
            }
        )

    monkeypatch.setattr(qa, "run_profile", failed_profile)
    message = {
        "reference": r"1 unstable reference",
        "scenario": r"1 scenario.*failed",
        "control": r"1 cache-disabled admission control.*failed",
    }[failure]
    with pytest.raises(qa.ValidationError, match=message):
        qa.main(["-m", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["coverage"] == "selected-suites"


class Flags(IntFlag):
    TOKEN = 1
    TURN_FINISHED = 2
    FAILED = 4
    RETRYABLE = 8
    CAPACITY_BLOCKED = 16


class Reason(Enum):
    NORMAL = 0
    CANCELLED = 1


class FakeRequest:
    def __init__(self):
        self.closed = 0
        self.cancelled = False

    def begin_turn(self, tokens, turn):
        self.tokens = tokens
        return 1

    def close(self):
        self.closed += 1

    def cancel_turn(self, turn_id):
        self.cancelled = True
        return True


class FakeTurn:
    def __init__(self, request):
        pass

    def set_do_sample(self, value):
        pass

    def set_min_generated_tokens(self, value):
        raise AssertionError("A minimum-token floor would suppress speculative verification.")

    def set_max_generated_tokens(self, value):
        pass


class FakeOptions:
    def set_max_session_tokens(self, value):
        pass


class FakeEngine:
    def __init__(self, frames, forever=False):
        self.frames = list(frames)
        self.forever = forever
        self.requests = []
        self.run_calls = 0

    def create_request(self, options):
        request = FakeRequest()
        self.requests.append(request)
        return request

    def get_speculative_stats(self):
        return {**dict.fromkeys(qa.SPECULATIVE_COUNTERS, 0), "acceptance_histogram": [1, 0], "formula_supported": True}

    def create_event_buffer(self, capacity):
        return capacity

    def has_pending_requests(self):
        return self.forever or bool(self.frames)

    def run(self, buffer):
        self.run_calls += 1
        return self.frames.pop(0)(self) if self.frames else []


def event(engine, *, flags=Flags.TOKEN | Flags.TURN_FINISHED, request_index=0, cancelled=False, token=7):
    request = engine.requests[request_index]
    return SimpleNamespace(
        request=request,
        flags=flags,
        turn_id=1,
        token=token,
        error_code="none",
        finish_reason=Reason.CANCELLED if cancelled else Reason.NORMAL,
        usage=SimpleNamespace(
            prompt_tokens=len(request.tokens), generated_tokens=0 if cancelled else 1, cached_prompt_tokens=0
        ),
    )


def driver(engine):
    runtime = SimpleNamespace(
        Engine=lambda model: engine,
        RequestOptions=FakeOptions,
        TurnOptions=FakeTurn,
        EngineEventFlags=Flags,
        FinishReason=Reason,
    )
    tokenizer = SimpleNamespace(create_stream=lambda: SimpleNamespace(decode=lambda token: "We"))
    numpy = SimpleNamespace(asarray=lambda tokens, dtype: tokens, int32="int32")
    args = SimpleNamespace(timeout_seconds=10, max_run_calls=3, generated_tokens=1, event_buffer_size=1)
    return qa.Driver(runtime, numpy, None, tokenizer, {"p": (2, 3, 4, 5, 6), "q": (2, 3)}, args, True)


def test_driver_snapshots_usage_and_releases_completed_requests():
    engine = FakeEngine([lambda current: [event(current)]])
    rows, stats = driver(engine).run(qa.Action(("p",)))
    assert rows[0]["tokens"] == [7]
    assert rows[0]["first_reasoning_s"] is not None
    assert engine.requests[0].closed == 1
    assert stats["before"]["acceptance_histogram"] == [1, 0]
    assert stats["after"]["formula_supported"] is True
    assert stats["delta"]["draft_tokens_proposed"] == 0


def test_driver_routes_concurrent_events_to_the_correct_request():
    engine = FakeEngine([lambda current: [event(current, request_index=1, token=8), event(current)]])
    rows, _ = driver(engine).run(qa.Action(("p", "q")))
    assert {item["prompt"]: item["tokens"] for item in rows} == {"p": [7], "q": [8]}
    assert all(request.closed == 1 for request in engine.requests)


def test_driver_retains_then_explicitly_releases_active_owner():
    engine = FakeEngine([lambda current: [event(current)]])
    runtime = driver(engine)
    runtime.run(qa.Action(("p",), keep_open=True))
    assert engine.requests[0].closed == 0
    runtime.run(qa.Action(("p",), release=True))
    assert engine.requests[0].closed == 1
    runtime.close()
    assert engine.requests[0].closed == 1


def test_driver_cancels_during_prefill_and_drains_terminal_event():
    engine = FakeEngine(
        [lambda current: [], lambda current: [event(current, flags=Flags.TURN_FINISHED, cancelled=True)]]
    )
    rows, _ = driver(engine).run(qa.Action(("p",), cancel_after_runs=1))
    assert engine.requests[0].cancelled
    assert rows[0]["cancelled"]
    assert rows[0]["tokens"] == []
    assert engine.requests[0].closed == 1


@pytest.mark.parametrize("flags", [Flags.FAILED, Flags.RETRYABLE, Flags.CAPACITY_BLOCKED])
def test_driver_errors_do_not_become_successful_empty_output(flags):
    engine = FakeEngine([lambda current: [SimpleNamespace(flags=flags, request=None, error_code="broken")]])
    with pytest.raises(RuntimeError):
        driver(engine).run(qa.Action(("p",)))
    assert engine.requests[0].closed == 1


def test_driver_detects_missing_completion_and_releases_request():
    engine = FakeEngine([lambda current: []])
    with pytest.raises(RuntimeError, match="without terminal"):
        driver(engine).run(qa.Action(("p",)))
    assert engine.requests[0].closed == 1


def test_driver_bounds_stalls_and_releases_request():
    engine = FakeEngine([], forever=True)
    with pytest.raises(TimeoutError, match="budget"):
        driver(engine).run(qa.Action(("p",)))
    assert engine.run_calls == 3
    assert engine.requests[0].closed == 1
