# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Operator-run correctness and reuse checks for the dynamic Engine prefix cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

SUITES = (
    "boundaries",
    "order",
    "branching",
    "alternation",
    "concurrency",
    "leases",
    "cancellation",
    "pressure",
    "random",
)
SPECULATIVE_COUNTERS = (
    "draft_forward_passes",
    "draft_tokens_proposed",
    "draft_tokens_evaluated",
    "draft_tokens_accepted",
    "target_forward_passes",
    "standard_fallback_steps",
    "dflash2_failures",
    "dflash2_disables",
    "dflash2_admission_misses",
    "mtp_failures",
)


class ValidationError(RuntimeError):
    pass


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number.")
    return result


@dataclass(frozen=True)
class Profile:
    block_size: int
    chunk_size: int
    max_batch_size: int
    max_scheduled_tokens: int
    num_blocks: int
    hybrid: bool
    draft_window: int = 0

    def warm_boundary(self, length, adopted=0):
        if not self.hybrid:
            return (length - 1) // self.block_size * self.block_size
        boundary = adopted
        for position in range(adopted + self.chunk_size, length, self.chunk_size):
            if position % self.block_size == 0:
                boundary = position
        return boundary

    def overlay(self, enabled, draft_mode):
        result = {
            "search": {"chunk_size": self.chunk_size},
            "engine": {
                "dynamic_batching": {
                    "prefix_caching": enabled,
                    "max_batch_size": self.max_batch_size,
                    "max_scheduled_tokens": self.max_scheduled_tokens,
                }
            },
        }
        if self.num_blocks:
            result["engine"]["dynamic_batching"]["num_blocks"] = self.num_blocks
        if draft_mode == "target-only":
            result["model"] = {"mtp": {"enabled": False}, "dflash2": {"filename": ""}}
        return result


@dataclass(frozen=True)
class PromptSpec:
    name: str
    length: int
    mutation: int | None = None
    variant: int = 0
    mutation_width: int = 1


@dataclass(frozen=True)
class Action:
    prompts: tuple[str, ...] = ()
    check: str = "safe"
    keep_open: bool = False
    release: bool = False
    cancel_after_runs: int = 0


@dataclass(frozen=True)
class Scenario:
    name: str
    actions: tuple[Action, ...]
    pressure: bool = False


def read_profile(config, chunk_size=None, max_batch_size=None, num_blocks=None):
    batching = config.get("engine", {}).get("dynamic_batching")
    if not batching:
        raise ValueError("The model must declare engine.dynamic_batching.")
    block = batching.get("block_size", 16)
    chunk = chunk_size or config.get("search", {}).get("chunk_size") or 2 * block
    batch = max_batch_size or batching.get("max_batch_size", 16)
    scheduled = batching.get("max_scheduled_tokens", 2048)
    blocks = batching.get("num_blocks") if num_blocks is None else num_blocks
    if blocks is None:
        blocks = 0
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (block, chunk, batch, scheduled, blocks)):
        raise ValueError("Cache geometry must contain integer values.")
    if min(block, chunk, batch, scheduled) <= 0 or blocks < 0:
        raise ValueError("Cache geometry must be positive; num_blocks may be zero for automatic sizing.")
    if scheduled < chunk:
        raise ValueError("max_scheduled_tokens must cover chunk_size for the controlled single-request oracle.")
    groups = config["model"].get("decoder", {}).get("state_groups", [])
    hybrid = any(group["kind"] in ("fixed_conv", "fixed_recurrent") for group in groups)
    draft = config["model"].get("dflash2", config["model"].get("dspark", {}))
    window = draft.get("sliding_window", -1)
    draft_window = window + draft.get("block_size", 0) if draft.get("filename") and window > 0 else 0
    return Profile(block, chunk, batch, scheduled, blocks, hybrid, draft_window)


def pressure_budget(profile, long_length, generated, override=None):
    return override or max(256, 2 * math.ceil((long_length + generated) / profile.block_size) + profile.max_batch_size)


def make_plan(
    profile,
    lengths,
    suites,
    seed,
    random_cases,
    pressure_requests,
    generated=64,
    pressure_blocks=None,
    alternation_split=None,
    alternation_rounds=3,
):
    block, chunk = profile.block_size, profile.chunk_size
    lengths = sorted(
        set(
            lengths
            or (center + delta for center in (block, 2 * block, chunk, 2 * chunk, 3 * chunk) for delta in (-1, 0, 1))
        )
    )
    if not lengths or min(lengths) <= 0:
        raise ValueError("Prompt lengths must be positive.")
    if profile.max_batch_size < 2 and {"concurrency", "leases"} & set(suites):
        raise ValueError(
            "Concurrency/lease suites need max_batch_size >= 2; select other suites or override the profile."
        )
    specs = {f"length-{length}": PromptSpec(f"length-{length}", length) for length in lengths}
    scenarios = []

    def repeat(name):
        return (Action((name,), "cold"), Action((name,), "exact-warm"), Action((name,), "warm"))

    if "boundaries" in suites:
        scenarios.extend(Scenario(f"boundary-{length}", repeat(f"length-{length}")) for length in lengths)
    if "order" in suites:
        for label, order in (("short-first", lengths), ("long-first", list(reversed(lengths)))):
            actions = []
            for index, length in enumerate(order):
                name = f"length-{length}"
                actions.extend((Action((name,), "cold" if index == 0 else "safe"), Action((name,), "warm")))
            scenarios.append(Scenario(label, tuple(actions)))
    long_length = max(*lengths, 4 * chunk + block + 1, 2 * math.lcm(block, chunk) + block + 1)
    specs["long"] = PromptSpec("long", long_length)
    splits = sorted(
        {
            position
            for center in (block, chunk, 2 * chunk)
            for position in (center - 1, center, center + 1)
            if 0 <= position < long_length
        }
    )
    if "branching" in suites:
        for split in splits:
            name = f"branch-{split}"
            specs[name] = PromptSpec(name, long_length, split)
            scenarios.append(
                Scenario(
                    name,
                    (
                        Action(("long",), "cold"),
                        Action((name,)),
                        Action((name,), "warm"),
                        Action(("long",)),
                        Action(("long",), "warm"),
                    ),
                )
            )
    if "alternation" in suites:
        if profile.hybrid and profile.max_batch_size < 2:
            raise ValueError("Hybrid alternation needs at least two fixed-state checkpoints (max_batch_size >= 2).")
        required_blocks = long_length // block + math.ceil((long_length + generated) / block)
        if profile.num_blocks and profile.num_blocks < required_blocks:
            raise ValueError(
                f"Alternation needs a block budget of at least {required_blocks} to retain two independent histories and decode; received {profile.num_blocks}."
            )
        split = 2 * chunk - 1 if alternation_split is None else alternation_split
        if not 0 <= split < long_length or alternation_rounds <= 0:
            raise ValueError("Alternation requires a mutation inside the prompt and a positive round count.")
        name = "alternating-branch"
        specs[name] = PromptSpec(name, long_length, split)
        actions = [
            Action(("long",), "cold"),
            Action(("long",), "exact-warm"),
            Action((name,)),
            Action((name,), "warm"),
        ]
        for _ in range(alternation_rounds):
            actions.extend((Action(("long",), "warm"), Action((name,), "warm")))
        scenarios.append(Scenario("alternating-retained-histories", tuple(actions)))
    seed_name = "short-seed"
    specs[seed_name] = PromptSpec(seed_name, 2 * chunk + 1)
    if "concurrency" in suites:
        scenarios.append(
            Scenario(
                "simultaneous-short-long",
                (Action((seed_name, "long")), Action(("long",)), Action(("long",), "warm")),
            )
        )
    pinned = "pinned-branch"
    specs[pinned] = PromptSpec(pinned, long_length, 2 * chunk - 1)
    if "leases" in suites:
        scenarios.append(
            Scenario(
                "active-owner-defers-replacement",
                (
                    Action(("long",), "cold", keep_open=True),
                    Action((pinned,)),
                    Action((pinned,)),
                    Action(("long",), release=True),
                    Action((pinned,)),
                    Action((pinned,), "warm"),
                ),
            )
        )
    if "cancellation" in suites:
        scenarios.append(
            Scenario(
                "cancel-partial-prefill-and-retry",
                (Action(("long",), "cold", cancel_after_runs=1), Action(("long",)), Action(("long",), "warm")),
            )
        )
    if "pressure" in suites:
        churn = []
        budget = pressure_budget(profile, long_length, generated, pressure_blocks)
        indexed_blocks = profile.warm_boundary(long_length) // block
        pressure_requests = max(pressure_requests, math.ceil(budget / indexed_blocks) + 2)
        for index in range(pressure_requests):
            name = f"churn-{index}"
            specs[name] = PromptSpec(name, long_length, 0, index + 1, mutation_width=block)
            churn.append(Action((name,)))
        scenarios.append(
            Scenario(
                "bounded-pool-churn-and-recovery",
                (Action(("long",), "cold"), *churn, Action(("long",), "evicted"), Action(("long",), "warm")),
                pressure=True,
            )
        )
    if "random" in suites:
        rng = random.Random(seed)
        for index in range(random_cases):
            split = rng.randrange(2 * chunk)
            name = f"random-branch-{index}"
            specs[name] = PromptSpec(name, long_length, split, index + 1)
            scenarios.append(Scenario(name, (Action(("long",), "cold"), Action((name,)), Action((name,), "warm"))))
    used = {name for scenario in scenarios for action in scenario.actions for name in action.prompts}
    return {name: spec for name, spec in specs.items() if name in used}, scenarios


def materialize_prompts(tokenizer, specs, body, user_prompt):
    marker = "PREFIX_CACHE_QA_SYSTEM_BODY"
    messages = json.dumps([{"role": "system", "content": marker}, {"role": "user", "content": user_prompt}])
    template = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
    if template.count(marker) != 1:
        raise ValueError("The chat template must preserve the unique system-body marker.")
    head, tail = template.split(marker)
    head_tokens = list(map(int, tokenizer.encode(head)))
    tail_tokens = list(map(int, tokenizer.encode(tail)))
    corpus = list(map(int, tokenizer.encode(body)))
    alternatives = list(
        dict.fromkeys([*map(int, tokenizer.encode(" alternative independent reference datum")), *corpus])
    )
    if not corpus or len(alternatives) < 2:
        raise ValueError("The tokenizer must produce nonempty corpus tokens and at least two mutation tokens.")
    maximum = max(spec.length for spec in specs.values())
    repetitions = max(1, math.ceil((maximum - len(head_tokens)) / len(corpus)))
    master = head_tokens + corpus * repetitions
    prompts = {}
    for name, spec in specs.items():
        if spec.length <= len(head_tokens) + len(tail_tokens):
            raise ValueError(
                f"{name}: {spec.length} tokens cannot fit a complete chat prompt; specify larger --lengths."
            )
        tokens = master[: spec.length - len(tail_tokens)] + tail_tokens
        if spec.mutation is not None:
            variant = spec.variant
            for position in range(spec.mutation, spec.mutation + spec.mutation_width):
                choices = [token for token in alternatives if token != tokens[position]]
                tokens[position] = choices[variant % len(choices)]
                variant //= len(choices)
                if variant == 0:
                    break
            if variant:
                raise ValueError(
                    "The reference corpus has too few distinct token IDs for the requested mutation count."
                )
        prompts[name] = tuple(tokens)
    return prompts, "<think>" in tail.rsplit("assistant", 1)[-1]


def token_digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode("ascii")).hexdigest()


def check_result(row, reference, profile, rule, previous, generated, warm_ratio=None, cold_ttft=None):
    if rule not in ("safe", "cold", "warm", "exact-warm", "evicted"):
        raise ValueError(f"Unknown cache check: {rule}.")
    name, cached, length = row["prompt"], row["cached_tokens"], row["prompt_length"]
    checks = {
        "safety": 0 <= cached < length and cached % profile.block_size == 0,
        "parity": True,
        "generation_coverage": None if row["cancelled"] else len(row["tokens"]) == generated,
        "reuse": None if rule == "safe" else True,
        "latency": None,
    }
    errors = []
    if not checks["safety"]:
        errors.append(f"{name}: unsafe cached-token boundary {cached} for prompt length {length}.")
    expected = reference[: len(row["tokens"])] if row["cancelled"] else reference
    if row["tokens"] != expected:
        divergence = next(
            (index for index, pair in enumerate(zip(row["tokens"], expected, strict=False)) if pair[0] != pair[1]),
            min(len(row["tokens"]), len(expected)),
        )
        checks["parity"] = False
        errors.append(f"{name}: greedy output differs from cache-disabled reference at token {divergence}.")
    if checks["generation_coverage"] is False:
        errors.append(f"{name}: expected {generated} generated tokens, received {len(row['tokens'])}.")
    if rule == "cold":
        row["expected_cached_tokens"] = 0
        if cached != 0:
            checks["reuse"] = False
            errors.append(f"{name}: a fresh Engine unexpectedly adopted {cached} tokens.")
    elif rule == "evicted":
        if cached >= profile.warm_boundary(length):
            checks["reuse"] = False
            errors.append(f"{name}: pressure scenario did not demonstrate eviction/checkpoint turnover.")
    elif rule in ("warm", "exact-warm"):
        if previous is None or previous["cancelled"]:
            raise ValueError(f"{name}: warm checks require a preceding completed request.")
        expected_cached = profile.warm_boundary(length, previous["cached_tokens"])
        row["expected_cached_tokens"] = expected_cached
        if cached < expected_cached or (rule == "exact-warm" and cached != expected_cached):
            checks["reuse"] = False
            errors.append(
                f"{name}: cache plateau; expected {'exactly' if rule == 'exact-warm' else 'at least'} {expected_cached}, received {cached}."
            )
        if warm_ratio is not None and expected_cached:
            if cold_ttft is None or cold_ttft <= 0:
                raise ValueError("Warm timing checks require a positive cache-disabled reference TTFT.")
            ratio = row["ttft_s"] / cold_ttft
            row["warm_ttft_ratio"] = ratio
            checks["latency"] = ratio <= warm_ratio
            if ratio > warm_ratio:
                errors.append(f"{name}: warm TTFT ratio {ratio:.3f} exceeds {warm_ratio}.")
    row["cache_check"] = rule
    row["checks"] = checks
    if errors:
        raise ValidationError(" ".join(errors))


@dataclass
class Pending:
    name: str
    request: object
    turn_id: int
    started: float
    stream: object
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    ttft: float | None = None
    first_reasoning: float | None = None
    cancel_requested: bool = False


class Driver:
    def __init__(self, og, np, model, tokenizer, prompts, args, template_thinks):
        self.og, self.np = og, np
        self.engine = og.Engine(model)
        self.tokenizer, self.prompts, self.args = tokenizer, prompts, args
        self.template_thinks = template_thinks
        self.retained = {}

    def close(self):
        for request in self.retained.values():
            request.close()
        self.retained.clear()

    def run(self, action):
        if action.release:
            for name in action.prompts:
                self.retained.pop(name).close()
            return [], {}
        pending = {}
        before = dict(self.engine.get_speculative_stats())
        completed = []
        deadline = time.monotonic() + self.args.timeout_seconds
        buffer = self.engine.create_event_buffer(self.args.event_buffer_size)
        try:
            for name in action.prompts:
                tokens = self.prompts[name]
                options = self.og.RequestOptions()
                options.set_max_session_tokens(len(tokens) + self.args.generated_tokens)
                request = self.engine.create_request(options=options)
                state = Pending(name, request, 0, time.perf_counter(), self.tokenizer.create_stream())
                pending[request] = state
                turn = self.og.TurnOptions(request)
                turn.set_do_sample(False)
                turn.set_max_generated_tokens(self.args.generated_tokens)
                state.turn_id = request.begin_turn(self.np.asarray(tokens, dtype=self.np.int32), turn)
            runs = 0
            while self.engine.has_pending_requests():
                if time.monotonic() > deadline or runs >= self.args.max_run_calls:
                    raise TimeoutError("Engine exceeded the phase deadline/run-call budget.")
                for event in self.engine.run(buffer):
                    flags = event.flags
                    if flags & (self.og.EngineEventFlags.FAILED | self.og.EngineEventFlags.RETRYABLE):
                        raise RuntimeError(f"Engine error: flags={flags}, error_code={event.error_code}.")
                    if event.request is None:
                        if flags & self.og.EngineEventFlags.CAPACITY_BLOCKED:
                            raise RuntimeError(
                                "The planned phase is capacity-blocked; its requests cannot currently progress."
                            )
                        raise RuntimeError(f"Invalid request-less event: flags={flags}, error_code={event.error_code}.")
                    if event.request not in pending:
                        raise RuntimeError("Engine returned an unknown or already completed request.")
                    state = pending[event.request]
                    if event.turn_id != state.turn_id:
                        raise RuntimeError("Engine returned an event for a different turn.")
                    elapsed = time.perf_counter() - state.started
                    if flags & self.og.EngineEventFlags.TOKEN:
                        state.tokens.append(int(event.token))
                        state.text += state.stream.decode(int(event.token))
                        if state.ttft is None:
                            state.ttft = elapsed
                        thinking = state.text.split("<think>", 1)[-1].split("</think>", 1)[0]
                        if (
                            state.first_reasoning is None
                            and (self.template_thinks or "<think>" in state.text)
                            and re.search(r"[A-Za-z0-9]", thinking)
                        ):
                            state.first_reasoning = elapsed
                    if flags & self.og.EngineEventFlags.TURN_FINISHED:
                        cancelled = event.finish_reason == self.og.FinishReason.CANCELLED
                        if cancelled != state.cancel_requested:
                            raise RuntimeError(f"{state.name}: unexpected cancellation outcome.")
                        if int(event.usage.generated_tokens) != len(state.tokens):
                            raise RuntimeError(f"{state.name}: usage does not match emitted tokens.")
                        if not cancelled and int(event.usage.prompt_tokens) != len(self.prompts[state.name]):
                            raise RuntimeError(f"{state.name}: usage does not match prompt length.")
                        completed.append(
                            {
                                "prompt": state.name,
                                "prompt_length": len(self.prompts[state.name]),
                                "cached_tokens": int(event.usage.cached_prompt_tokens),
                                "tokens": state.tokens,
                                "cancelled": cancelled,
                                "ttft_s": state.ttft,
                                "first_reasoning_s": state.first_reasoning,
                                "elapsed_s": elapsed,
                                "finish_reason": str(event.finish_reason),
                            }
                        )
                        del pending[event.request]
                        if action.keep_open:
                            self.retained[state.name] = state.request
                        else:
                            state.request.close()
                runs += 1
                if action.cancel_after_runs and runs >= action.cancel_after_runs:
                    for state in pending.values():
                        if not state.cancel_requested:
                            if not state.request.cancel_turn(state.turn_id):
                                raise RuntimeError("Cancellation was refused before the turn finished.")
                            state.cancel_requested = True
            if pending:
                raise RuntimeError("Engine stopped without terminal events for every request.")
            if action.cancel_after_runs and not any(row["cancelled"] for row in completed):
                raise RuntimeError(
                    "Cancellation scenario finished before exercising cancellation; use a longer prompt."
                )
            after = dict(self.engine.get_speculative_stats())
            delta = {name: after[name] - before[name] for name in SPECULATIVE_COUNTERS}
            if delta["dflash2_failures"] or delta["dflash2_disables"] or delta["mtp_failures"]:
                raise RuntimeError(f"Drafter failure/fallback during phase: {delta}.")
            return completed, {"before": before, "after": after, "delta": delta}
        finally:
            for state in pending.values():
                state.request.close()


def make_model(og, args, profile, enabled, draft_mode, source_config):
    config = og.Config(str(args.model_path))
    overlay = profile.overlay(enabled, draft_mode)
    if draft_mode == "target-only" and "dspark" in source_config["model"]:
        overlay["model"]["dspark"] = overlay["model"].pop("dflash2")
    config.overlay(json.dumps(overlay))
    if args.execution_provider:
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)
    return og.Model(config), overlay


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_profile(og, np, args, source_config, profile, specs, scenarios, draft_mode, report):
    model, overlay = make_model(og, args, profile, False, draft_mode, source_config)
    tokenizer = og.Tokenizer(model)
    body = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file
        else "\n".join(
            f"Regression case {index}: nums=[{index}, {index + 3}, {-index}, {index % 17}], target={2 * index + 3}; preserve duplicates and negative values."
            for index in range(256)
        )
    )
    prompts, template_thinks = materialize_prompts(tokenizer, specs, body, args.user_prompt)
    entry = {
        "profile": asdict(profile),
        "draft_mode": draft_mode,
        "reference_overlay": overlay,
        "corpus_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "prompts": {name: {**asdict(specs[name]), "sha256": token_digest(tokens)} for name, tokens in prompts.items()},
        "references": [],
        "reference_stability": {},
        "reference_failures": [],
        "scenarios": [],
    }
    report["profiles"].append(entry)
    references = {}
    driver = Driver(og, np, model, tokenizer, prompts, args, template_thinks)
    try:
        cap = driver.engine.get_capabilities().max_request_length
        entry["reference_max_request_length"] = cap
        if max(map(len, prompts.values())) + args.generated_tokens > cap:
            raise ValueError(
                f"Requested matrix exceeds cache-disabled request capacity {cap}; increase --num-blocks or reduce --lengths."
            )
        for name in prompts:
            entry["reference_stability"][name] = None if args.reference_repeats == 1 else True
            for repeat in range(args.reference_repeats):
                print(
                    f"{draft_mode}/chunk-{profile.chunk_size}/reference-{repeat + 1}: {name} ({len(prompts[name])} tokens)",
                    flush=True,
                )
                rows, stats = driver.run(Action((name,), "cold"))
                row = rows[0]
                row["reference_repeat"] = repeat
                expected = references[name]["tokens"] if name in references else row["tokens"]
                try:
                    check_result(row, expected, profile, "cold", None, args.generated_tokens)
                except ValidationError as error:
                    row["error"] = str(error)
                    entry["references"].append({**row, "speculative_stats": stats})
                    save_report(args.output, report)
                    if args.fail_fast or any(
                        row["checks"][check] is False for check in ("safety", "generation_coverage", "reuse")
                    ):
                        raise
                    entry["reference_stability"][name] = False
                    entry["reference_failures"].append(f"{name}: cache-disabled reference is unstable: {error}")
                    print(f"FAIL: {entry['reference_failures'][-1]}", file=sys.stderr, flush=True)
                else:
                    entry["references"].append({**row, "speculative_stats": stats})
                references.setdefault(name, row)
                save_report(args.output, report)
    finally:
        driver.close()
    del driver, tokenizer, model
    gc.collect()
    for pressure in (False, True):
        selected = [scenario for scenario in scenarios if scenario.pressure == pressure]
        if not selected:
            continue
        cached_profile = profile
        if pressure:
            blocks = pressure_budget(profile, len(prompts["long"]), args.generated_tokens, args.pressure_num_blocks)
            cached_profile = replace(profile, num_blocks=blocks)
            roots = {tokens[: profile.block_size] for name, tokens in prompts.items() if name.startswith("churn-")}
            if len(roots) != sum(name.startswith("churn-") for name in prompts):
                raise ValueError("Pressure prompts must have distinct first KV blocks.")
        model, overlay = make_model(og, args, cached_profile, True, draft_mode, source_config)
        tokenizer = og.Tokenizer(model)
        try:
            for scenario in selected:
                driver = Driver(og, np, model, tokenizer, prompts, args, template_thinks)
                result = {"name": scenario.name, "steps": [], "status": "running"}
                result["overlay"] = overlay
                entry["scenarios"].append(result)
                previous = {}
                try:
                    cap = driver.engine.get_capabilities().max_request_length
                    result["max_request_length"] = cap
                    needed = (
                        max(len(prompts[name]) for action in scenario.actions for name in action.prompts)
                        + args.generated_tokens
                    )
                    if needed > cap:
                        raise ValueError(
                            f"{scenario.name}: needs {needed} session tokens, capacity is {cap}. Increase the corresponding block budget."
                        )
                    for index, action in enumerate(scenario.actions):
                        print(
                            f"{draft_mode}/chunk-{profile.chunk_size}/{scenario.name}/step-{index}: {', '.join(action.prompts)}",
                            flush=True,
                        )
                        step = {"action": asdict(action), "rows": [], "status": "running", "errors": []}
                        result["steps"].append(step)
                        save_report(args.output, report)
                        rows, stats = driver.run(action)
                        step.update(rows=rows, speculative_stats=stats)
                        for row in rows:
                            reference = references[row["prompt"]]
                            try:
                                check_result(
                                    row,
                                    reference["tokens"],
                                    profile,
                                    action.check,
                                    previous.get(row["prompt"]),
                                    args.generated_tokens,
                                    args.max_warm_ttft_ratio,
                                    reference["ttft_s"],
                                )
                            except ValidationError as error:
                                step["errors"].append(str(error))
                                step["status"] = "failed"
                                print(f"FAIL: {scenario.name}: {error}", file=sys.stderr, flush=True)
                                if args.fail_fast or not row["checks"]["safety"]:
                                    raise
                            if (
                                draft_mode == "configured"
                                and profile.draft_window
                                and not row["cancelled"]
                                and row["prompt_length"] - row["cached_tokens"] >= profile.draft_window
                                and args.generated_tokens > 1
                            ):
                                if stats["delta"]["draft_tokens_proposed"] == 0:
                                    error = f"{scenario.name}: complete window rebuilt but drafting did not resume."
                                    step["errors"].append(error)
                                    step["status"] = "failed"
                                    print(f"FAIL: {error}", file=sys.stderr, flush=True)
                                    if args.fail_fast:
                                        raise ValidationError(error)
                            previous[row["prompt"]] = row
                        if step["status"] == "running":
                            step["status"] = "passed"
                        save_report(args.output, report)
                    result["status"] = (
                        "failed" if any(step["status"] == "failed" for step in result["steps"]) else "passed"
                    )
                    save_report(args.output, report)
                finally:
                    driver.close()
                    if result["status"] == "running":
                        result["status"] = "failed"
                    for step in result["steps"]:
                        if step["status"] == "running":
                            step["status"] = "failed"
                del driver
                gc.collect()
        finally:
            del tokenizer, model
            gc.collect()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("-m", "--model-path", type=Path, required=True)
    result.add_argument(
        "-e",
        "--execution-provider",
        choices=("cpu", "cuda", "webgpu"),
        help="Override providers; omitted preserves the release profile, including provider options.",
    )
    result.add_argument("--output", type=Path, default=Path("prefix-cache-qa-results.json"))
    result.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    result.add_argument(
        "--lengths",
        type=positive_int,
        nargs="+",
        help="Exact prompt lengths; default sweeps block/chunk boundaries +/- one token.",
    )
    result.add_argument(
        "--chunk-sizes",
        type=positive_int,
        nargs="+",
        help="Optional chunk-size sweep; otherwise use the model's chunk size.",
    )
    result.add_argument("--max-batch-size", type=positive_int)
    result.add_argument("--num-blocks", type=positive_int)
    result.add_argument("--pressure-num-blocks", type=positive_int)
    result.add_argument("--pressure-requests", type=positive_int, default=10)
    result.add_argument("--random-cases", type=positive_int, default=4)
    result.add_argument(
        "--alternation-split",
        type=int,
        help="Mutation offset for the alternating history; default is the token before the second chunk boundary.",
    )
    result.add_argument("--alternation-rounds", type=positive_int, default=3)
    result.add_argument("--seed", type=int, default=20261002)
    result.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first assertion failure; otherwise collect all healthy-Engine assertion failures.",
    )
    result.add_argument("--generated-tokens", type=positive_int, default=64)
    result.add_argument(
        "--reference-repeats",
        type=positive_int,
        default=2,
        help="Repeat each cache-disabled reference to detect baseline instability; one run leaves stability unchecked.",
    )
    result.add_argument("--event-buffer-size", type=positive_int, default=1)
    result.add_argument("--timeout-seconds", type=positive_float, default=300)
    result.add_argument("--max-run-calls", type=positive_int, default=100000)
    result.add_argument(
        "--max-warm-ttft-ratio",
        type=positive_float,
        help="Optional warm/cold TTFT limit; timing is otherwise diagnostic, not a correctness gate.",
    )
    result.add_argument("--draft-mode", choices=("configured", "target-only", "both"), default="configured")
    result.add_argument("--prompt-file", type=Path, help="Optional UTF-8 system reference corpus; never modified.")
    result.add_argument(
        "--user-prompt",
        default="Implement a solution for the 2 sum problem in Python. Explain the algorithm and complexity.",
    )
    result.add_argument(
        "--device-label",
        default="",
        help="Operator label, e.g. A100-80GB or H100; not automatic hardware verification.",
    )
    result.add_argument("--build-label", default="", help="Runtime build/commit identity supplied by the operator.")
    result.add_argument(
        "--plan-only", action="store_true", help="Print the matrix without importing GenAI or loading a model."
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    arguments = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    started = datetime.now(timezone.utc).isoformat()
    try:
        config_path = args.model_path / "genai_config.json"
        source_bytes = config_path.read_bytes()
        source_config = json.loads(source_bytes)
        profiles = [
            read_profile(source_config, chunk, args.max_batch_size, args.num_blocks)
            for chunk in (args.chunk_sizes or [None])
        ]
        plans = [
            make_plan(
                profile,
                args.lengths,
                args.suites,
                args.seed,
                args.random_cases,
                args.pressure_requests,
                args.generated_tokens,
                args.pressure_num_blocks,
                args.alternation_split,
                args.alternation_rounds,
            )
            for profile in profiles
        ]
        if not args.plan_only and any(profile.num_blocks == 0 for profile in profiles):
            raise ValueError(
                "Controlled GPU QA requires a fixed --num-blocks budget; automatic free-memory sizing is not reproducible across fresh Engines."
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not args.plan_only:
            save_report(
                args.output,
                {
                    "status": "failed",
                    "started_at_utc": started,
                    "arguments": arguments,
                    "error": f"{type(error).__name__}: {error}",
                    "profiles": [],
                },
            )
        raise
    if args.plan_only:
        print(
            json.dumps(
                [
                    {
                        "profile": asdict(profile),
                        "prompts": [asdict(spec) for spec in specs.values()],
                        "scenarios": [asdict(scenario) for scenario in scenarios],
                    }
                    for profile, (specs, scenarios) in zip(profiles, plans, strict=True)
                ],
                indent=2,
            )
        )
        return
    report = {
        "status": "running",
        "started_at_utc": started,
        "arguments": arguments,
        "model_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "profiles": [],
    }
    save_report(args.output, report)
    try:
        og = importlib.import_module("onnxruntime_genai")
        np = importlib.import_module("numpy")
        report["runtime_module"] = og.__file__
        report["runtime_version"] = getattr(og, "__version__", None)
        if importlib.util.find_spec("onnxruntime_ep_cuda") is not None:
            plugin = importlib.import_module("onnxruntime_ep_cuda")
            og.register_execution_provider_library(plugin.get_ep_name(), plugin.get_library_path())
        modes = ("configured", "target-only") if args.draft_mode == "both" else (args.draft_mode,)
        for profile, (specs, scenarios) in zip(profiles, plans, strict=True):
            for mode in modes:
                run_profile(og, np, args, source_config, profile, specs, scenarios, mode, report)
        failures = [
            scenario["name"]
            for entry in report["profiles"]
            for scenario in entry["scenarios"]
            if scenario["status"] != "passed"
        ]
        reference_failures = [failure for entry in report["profiles"] for failure in entry["reference_failures"]]
        if failures or reference_failures:
            raise ValidationError(
                f"{len(failures)} scenario(s) failed: {', '.join(failures)}; "
                f"{len(reference_failures)} unstable reference comparison(s); report={args.output}"
            )
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_report(args.output, report)
    print(
        f"PASS: {sum(len(entry['scenarios']) for entry in report['profiles'])} scenarios; report={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
