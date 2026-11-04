# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Paired off/on benchmark for Engine guidance fast-forward.

Each iteration runs the same guided turns with fast-forward off and on, on one Engine and in an
alternating order, optionally beside unguided peer turns. It reports Run calls, arrival-to-first-
token and end-to-end latency, peer inter-token gaps, cancellation latency, and whether off and on
generated the same tokens. See benchmark/python/README.
"""

import argparse
import contextlib
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import numpy as np
import onnxruntime_genai as og
from benchmark_engine_guidance import percentile, require_request_event

FIXED_WHITESPACE = {"whitespace_flexible": False, "key_separator": ": ", "item_separator": ", "}


def _schema(properties, flexible=False):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        "x-guidance": {"whitespace_flexible": True} if flexible else FIXED_WHITESPACE,
    }


def _json(prompt, schema):
    return {"prompt": prompt, "type": "lark_grammar", "data": f"start: %json {json.dumps(schema)}\n", "schema": schema}


_WEATHER_PROMPT = "Report this observation as JSON: Redmond, 52 F, partly cloudy, 81% humidity."
_WEATHER = {
    "city": {"type": "string"},
    "temperature_f": {"type": "integer"},
    "conditions": {"type": "string"},
    "humidity_percent": {"type": "integer"},
}
_ITEM = {"description": {"type": "string"}, "quantity": {"type": "integer"}, "unit_price": {"type": "number"}}
_FLIGHT = {
    "origin": {"type": "string"},
    "destination": {"type": "string"},
    "date": {"type": "string"},
    "adults": {"type": "integer"},
    "cabin": {"enum": ["economy", "premium_economy", "business"]},
}

# Fixed-whitespace JSON forces the most tokens; flexible whitespace, long free text and a short
# choice force few, so they show where fast-forward gains little.
WORKLOADS = {
    "weather": _json(_WEATHER_PROMPT, _schema(_WEATHER)),
    "weather_flexible": _json(_WEATHER_PROMPT, _schema(_WEATHER, flexible=True)),
    "tool_call": _json(
        "Call the right tool to book a flight from Seattle to Boston on 2026-12-01 for 2 adults.",
        _schema({"name": {"enum": ["get_weather", "search_flights", "book_flight"]}, "arguments": _schema(_FLIGHT)}),
    ),
    "invoice": _json(
        "Extract the invoice: INV-2048 from Contoso Ltd in USD; 3 widgets at 4.50, 1 gadget at 19.99, "
        "2 cables at 7.25; total 48.99.",
        _schema(
            {
                "invoice_number": {"type": "string"},
                "vendor": {"type": "string"},
                "currency": {"enum": ["USD", "EUR", "GBP"]},
                "items": {"type": "array", "items": _schema(_ITEM)},
                "total": {"type": "number"},
            }
        ),
    ),
    "long_text": _json(
        "Summarize the benefits of continuous batching for LLM serving in about 120 words, as JSON.",
        _schema({"title": {"type": "string"}, "summary": {"type": "string"}}),
    ),
    "choice": _json(
        "Classify the sentiment of: 'The update fixed every issue I had.'",
        _schema({"sentiment": {"enum": ["positive", "negative", "neutral"]}}),
    ),
}
PEER_PROMPT = "Write a short paragraph about why reproducible benchmarks matter."
DRAFT_COUNTERS = ("draft_tokens_proposed", "draft_tokens_evaluated", "draft_tokens_accepted")


def summarize(values):
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def encode_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    # A model without a chat template still benchmarks on the plain text.
    with contextlib.suppress(Exception):
        text = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True)
    return np.asarray(tokenizer.encode(text), dtype=np.int32)


@dataclass
class Turn:
    kind: str  # "guided" or "peer"
    workload: str
    request: object
    turn_id: int
    arrived_at: float  # When the turn was due; includes request creation and grammar setup.
    reference: list | None  # Tokens a peer proposes as drafts, from its solo run.
    run_calls_at_arrival: int = 0
    tokens: list = field(default_factory=list)
    token_times: list = field(default_factory=list)
    run_calls_to_first_token: int | None = None
    finish_reason: str | None = None
    finished_at: float | None = None
    cancel_attempted: bool = False
    cancel_requested_at: float | None = None


class Scenario:
    """Drives one Engine through turns that start at scheduled times until all of them finish."""

    def __init__(self, engine, args, max_draft_tokens):
        self.engine = engine
        self.args = args
        self.max_draft_tokens = max_draft_tokens
        self.buffer = engine.create_event_buffer(args.event_buffer)
        self.turns = []
        self.run_calls = 0
        self.model_steps = 0
        self.drafts = {}  # draft_tokens_proposed/evaluated/accepted over this scenario
        # Run hands out a previous step's leftover events before it steps again. Only a Run that
        # filled the buffer can have left some behind; one that returned fewer left none.
        self.maybe_buffered = False
        self.backlog = []  # Closed loop: callbacks that begin a guided turn when one finishes.
        self.guided_expected = 0
        self.replenish_peer = None  # With --replenish_peers: begins a peer when one finishes.

    def begin(self, kind, workload, tokens, max_new_tokens, arrived_at, configure=None, reference=None):
        request_options = og.RequestOptions()
        request_options.set_max_session_tokens(len(tokens) + max_new_tokens)
        request = self.engine.create_request(options=request_options)
        turn_options = og.TurnOptions(request)
        # Greedy, so off and on can be compared token for token.
        turn_options.set_do_sample(False)
        turn_options.set_max_generated_tokens(max_new_tokens)
        if configure:
            configure(turn_options)
        turn_id = request.begin_turn(tokens, turn_options)
        self.turns.append(Turn(kind, workload, request, turn_id, arrived_at, reference, self.run_calls))

    def guided_outstanding(self):
        guided = [turn for turn in self.turns if turn.kind == "guided"]
        return len(guided) < self.guided_expected or any(turn.finished_at is None for turn in guided)

    def _turn_for(self, request):
        return next(turn for turn in self.turns if turn.request is request)

    def _propose_peer_drafts(self):
        """Re-proposes each running peer's drafts from the tokens delivered so far, before every Run.

        Run steps only once every earlier event has been delivered, so the proposal in place when
        a Run steps was made from the complete output and is exact. A proposal made after a full
        buffer may lag and be rejected (the turn already finished, or its session cap is closer than
        the delivered tokens show); such a proposal would only precede a Run that drains, so it is
        cleared instead.
        """
        for turn in self.turns:
            if turn.reference is None or turn.finished_at is not None or not turn.tokens:
                continue
            # Drafts must leave room for the token the verifying step adds.
            width = min(self.max_draft_tokens, self.args.peer_max_new_tokens - len(turn.tokens) - 1)
            proposal = turn.reference[len(turn.tokens) : len(turn.tokens) + max(width, 0)]
            try:
                turn.request.set_draft_tokens(np.asarray(proposal, dtype=np.int32))
            except RuntimeError:
                if not self.maybe_buffered:
                    raise
                turn.request.set_draft_tokens(np.asarray([], dtype=np.int32))

    def _cancel_due_turns(self):
        for turn in self.turns:
            if (
                turn.kind == "guided"
                and turn.finished_at is None
                and not turn.cancel_attempted
                and len(turn.tokens) >= self.args.cancel_after
            ):
                turn.cancel_attempted = True
                requested_at = time.perf_counter()
                # False when the turn already finished and its event is still buffered.
                if turn.request.cancel_turn(turn.turn_id):
                    turn.cancel_requested_at = requested_at

    def run(self, arrivals):
        """`arrivals` holds (seconds after start, callback that begins a turn arriving at a time)."""
        started = time.perf_counter()
        stats_before = self.engine.get_speculative_stats()
        pending = sorted(arrivals, key=lambda arrival: arrival[0])
        while pending or self.maybe_buffered or self.engine.has_pending_requests():
            elapsed = time.perf_counter() - started
            while pending and pending[0][0] <= elapsed:
                # A turn that came due during a Run call still counts from when it was due.
                due, start = pending.pop(0)
                start(started + due)
            if not self.maybe_buffered and not self.engine.has_pending_requests():
                time.sleep(max(0.0, pending[0][0] - elapsed))
                continue
            self._propose_peer_drafts()
            events = self.engine.run(self.buffer)
            self.run_calls += 1
            self.maybe_buffered = len(events) == self.args.event_buffer
            for event in events:
                turn = self._turn_for(require_request_event(event))
                now = time.perf_counter()
                if event.flags & og.EngineEventFlags.TOKEN:
                    turn.tokens.append(event.token)
                    turn.token_times.append(now)
                    if turn.run_calls_to_first_token is None:
                        turn.run_calls_to_first_token = self.run_calls - turn.run_calls_at_arrival
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    turn.finish_reason = event.finish_reason.name
                    turn.finished_at = now
                    # A finished request keeps its cache blocks and batch slot until closed.
                    turn.request.close()
                    if turn.kind == "guided" and self.backlog:
                        self.backlog.pop(0)(now)
                    elif turn.kind == "peer" and self.replenish_peer and self.guided_outstanding():
                        self.replenish_peer(now)
            if self.args.cancel_after:
                self._cancel_due_turns()
        stats = self.engine.get_speculative_stats()
        self.model_steps = stats["target_forward_passes"] - stats_before["target_forward_passes"]
        self.drafts = {key: stats[key] - stats_before[key] for key in DRAFT_COUNTERS}
        return time.perf_counter() - started


def run_scenario(engine, args, max_draft_tokens, prompts, peer_reference, fast_forward):
    scenario = Scenario(engine, args, max_draft_tokens)

    def start_guided(workload, arrived_at):
        spec = WORKLOADS[workload]

        def configure(turn_options):
            turn_options.set_guidance(spec["type"], spec["data"])
            turn_options.set_guidance_fast_forward(fast_forward)

        scenario.begin("guided", workload, prompts[workload], args.max_new_tokens, arrived_at, configure)

    def start_peer(arrived_at):
        scenario.begin("peer", "peer", prompts["peer"], args.peer_max_new_tokens, arrived_at, reference=peer_reference)

    # Peers arriving after --peer_delay_ms show how long a guided turn's forced tokens delay them.
    arrivals = [(args.peer_delay_ms / 1000, start_peer) for _ in range(args.peers)]
    rng = random.Random(args.seed)  # The same arrival times for off and on.
    at = 0.0
    for index in range(args.requests):
        workload = args.workloads[index % len(args.workloads)]
        start = lambda arrived_at, workload=workload: start_guided(workload, arrived_at)  # noqa: E731
        if args.concurrency and index >= args.concurrency:
            scenario.backlog.append(start)
            continue
        arrivals.append((at, start))
        if args.arrival_rate:
            at += rng.expovariate(args.arrival_rate)
    scenario.guided_expected = args.requests
    if args.replenish_peers:
        scenario.replenish_peer = start_peer
    wall_seconds = scenario.run(arrivals)
    if args.peers and peer_reference is not None and scenario.drafts["draft_tokens_evaluated"] == 0:
        # Otherwise the scenario would silently measure ordinary peer decoding.
        raise RuntimeError("A --peer_drafts controlled scenario evaluated no draft tokens.")
    return scenario, wall_seconds


def peer_reference_tokens(engine, args, prompts, max_draft_tokens, controlled, max_passes=8):
    """The peer's solo greedy output, to check peers against and, when controlled, to draft from.

    Without controlled drafts this is one solo run (the model's own drafter, if any, still runs).
    With them, verifying drafts changes the target's numerics, so greedy output with drafts can
    differ from output without them, and drafts taken from the latter stop matching after the first
    difference. Each pass then proposes the previous pass's output until the output stops changing.
    """
    reference = None
    for passes in range(1, max_passes + 1):
        scenario = Scenario(engine, args, max_draft_tokens if reference else 0)
        scenario.begin(
            "peer", "peer", prompts["peer"], args.peer_max_new_tokens, time.perf_counter(), reference=reference
        )
        scenario.run([])
        tokens = scenario.turns[0].tokens
        if not controlled:
            return tokens, {"drafts": "model", "passes": 1, "converged": None}
        if tokens == reference:
            return reference, {"drafts": "controlled", "passes": passes, "converged": True}
        reference = tokens
    return reference, {"drafts": "controlled", "passes": max_passes, "converged": False}


SCHEMA_VALIDATORS = {name: jsonschema.Draft202012Validator(spec["schema"]) for name, spec in WORKLOADS.items()}


def schema_valid(tokenizer, turn):
    """Whether the turn ended by EOS with output valid against its workload's full JSON schema."""
    if turn.finish_reason != "EOS":
        return False
    try:
        parsed = json.loads(tokenizer.decode(np.asarray(turn.tokens, dtype=np.int32)))
    except json.JSONDecodeError:
        return False
    return SCHEMA_VALIDATORS[turn.workload].is_valid(parsed)


def first_difference(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _intersect(a, b):
    both, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            both.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return both


def overlap(guided, peers):
    """Throughput and peer gaps only while guided turns and peers were both present."""
    window = _intersect(
        _merge([(turn.arrived_at, turn.finished_at) for turn in guided]),
        _merge([(turn.arrived_at, turn.finished_at) for turn in peers]),
    )
    seconds = sum(end - start for start, end in window)

    def interval_of(moment):
        return next((index for index, (start, end) in enumerate(window) if start <= moment <= end), None)

    def tokens_per_s(turns):
        inside = sum(interval_of(moment) is not None for turn in turns for moment in turn.token_times)
        return inside / seconds if seconds else None

    return {
        "overlap_ms": seconds * 1000,
        "overlap_guided_tokens_per_s": tokens_per_s(guided),
        "overlap_peer_tokens_per_s": tokens_per_s(peers),
        "overlap_peer_gap_ms": [
            (later - earlier) * 1000
            for turn in peers
            for earlier, later in zip(turn.token_times, turn.token_times[1:], strict=False)
            if interval_of(earlier) is not None and interval_of(earlier) == interval_of(later)
        ],
    }


def measure(scenario, wall_seconds, tokenizer, peer_reference):
    guided = [turn for turn in scenario.turns if turn.kind == "guided"]
    peers = [turn for turn in scenario.turns if turn.kind == "peer"]
    first_token = [(turn.token_times[0] - turn.arrived_at) * 1000 for turn in guided if turn.tokens]
    return {
        "model_steps": scenario.model_steps,
        "drafts": scenario.drafts,
        "run_calls": scenario.run_calls,
        "wall_ms": wall_seconds * 1000,
        **overlap(guided, peers),
        "guided_turns_per_s": len(guided) / wall_seconds,
        "guided_tokens_per_s": sum(len(turn.tokens) for turn in guided) / wall_seconds,
        "peer_tokens_per_s": sum(len(turn.tokens) for turn in peers) / wall_seconds,
        "guided_workloads": [turn.workload for turn in guided],
        "guided_tokens": [turn.tokens for turn in guided],
        "guided_first_token_ms": first_token,
        "guided_end_to_end_ms": [(turn.finished_at - turn.arrived_at) * 1000 for turn in guided],
        "guided_finish_reasons": [turn.finish_reason for turn in guided],
        "guided_schema_valid": [schema_valid(tokenizer, turn) for turn in guided],
        "cancel_latency_ms": [
            (turn.finished_at - turn.cancel_requested_at) * 1000
            for turn in guided
            if turn.cancel_requested_at is not None
        ],
        "peer_gap_ms": [
            (later - earlier) * 1000
            for turn in peers
            for earlier, later in zip(turn.token_times, turn.token_times[1:], strict=False)
        ],
        "peer_run_calls_to_first_token": [turn.run_calls_to_first_token for turn in peers],
        "peer_reference_first_diff": [
            first_difference(turn.tokens, peer_reference) for turn in peers if turn.tokens != peer_reference
        ],
        "peers": len(peers),
    }


def aggregate(runs):
    def flat(name):
        return [value for run in runs for value in run[name]]

    reasons = flat("guided_finish_reasons")
    valid = sum(flat("guided_schema_valid"))
    peers = sum(run["peers"] for run in runs)
    return {
        "model_steps": summarize([run["model_steps"] for run in runs]),
        **{key: summarize([run["drafts"][key] for run in runs]) for key in DRAFT_COUNTERS},
        "run_calls": summarize([run["run_calls"] for run in runs]),
        "wall_ms": summarize([run["wall_ms"] for run in runs]),
        "overlap_ms": summarize([run["overlap_ms"] for run in runs]),
        "overlap_fraction": summarize([run["overlap_ms"] / run["wall_ms"] for run in runs]),
        "overlap_guided_tokens_per_s": summarize(
            [run["overlap_guided_tokens_per_s"] for run in runs if run["overlap_guided_tokens_per_s"] is not None]
        ),
        "overlap_peer_tokens_per_s": summarize(
            [run["overlap_peer_tokens_per_s"] for run in runs if run["overlap_peer_tokens_per_s"] is not None]
        ),
        "overlap_peer_gap_ms": summarize(flat("overlap_peer_gap_ms")),
        "guided_turns_per_s": summarize([run["guided_turns_per_s"] for run in runs]),
        "guided_tokens_per_s": summarize([run["guided_tokens_per_s"] for run in runs]),
        "peer_tokens_per_s": summarize([run["peer_tokens_per_s"] for run in runs]),
        "guided_tokens_per_turn": summarize([len(tokens) for tokens in flat("guided_tokens")]),
        "guided_first_token_ms": summarize(flat("guided_first_token_ms")),
        "guided_end_to_end_ms": summarize(flat("guided_end_to_end_ms")),
        "guided_finish_reasons": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        "guided_schema_valid": valid,
        "guided_turns": len(reasons),
        "guided_schema_valid_rate": valid / len(reasons) if reasons else None,
        "cancel_latency_ms": summarize(flat("cancel_latency_ms")),
        "peer_gap_ms": summarize(flat("peer_gap_ms")),
        "peer_run_calls_to_first_token": summarize(flat("peer_run_calls_to_first_token")),
        "peers": peers,
        "peers_matching_reference": peers - len(flat("peer_reference_first_diff")),
        "peer_reference_first_diff": summarize(flat("peer_reference_first_diff")),
    }


def compare(off_runs, on_runs, tokenizer, max_diffs=20):
    def text(tokens):
        return tokenizer.decode(np.asarray(tokens, dtype=np.int32))

    def pairs(runs_a, runs_b):
        return [
            (iteration, index, workload, a, b)
            for iteration, (run_a, run_b) in enumerate(zip(runs_a, runs_b, strict=True))
            for index, (workload, a, b) in enumerate(
                zip(run_a["guided_workloads"], run_a["guided_tokens"], run_b["guided_tokens"], strict=True)
            )
        ]

    def identical(runs_a, runs_b):
        matched = pairs(runs_a, runs_b)
        return {
            "token_ids": sum(a == b for *_, a, b in matched),
            "text": sum(text(a) == text(b) for *_, a, b in matched),
            "pairs": len(matched),
        }

    def diff(iteration, index, workload, off, on):
        first = first_difference(off, on)
        return {
            "iteration": iteration,
            "turn": index,
            "workload": workload,
            "first_diff": first,
            "lengths": [len(off), len(on)],
            "off_tokens": off[first : first + 16],
            "on_tokens": on[first : first + 16],
            "off_text": text(off),
            "on_text": text(on),
        }

    def ratio(name, numerator, denominator):
        bottom = statistics.median(run[name] for run in denominator)
        return statistics.median(run[name] for run in numerator) / bottom if bottom else None

    return {
        "off_vs_on": identical(off_runs, on_runs),
        # Batched numerics can change greedy output between identical runs; these are the baseline
        # for off_vs_on: each later iteration against the first one of the same mode.
        "off_repeat": identical(off_runs[1:], off_runs[:1] * (len(off_runs) - 1)),
        "on_repeat": identical(on_runs[1:], on_runs[:1] * (len(on_runs) - 1)),
        "off_vs_on_diffs": [diff(*pair) for pair in pairs(off_runs, on_runs) if pair[3] != pair[4]][:max_diffs],
        "model_steps_on_over_off": ratio("model_steps", on_runs, off_runs),
        "run_calls_on_over_off": ratio("run_calls", on_runs, off_runs),
        "guided_turns_per_s_on_over_off": ratio("guided_turns_per_s", on_runs, off_runs),
        "wall_speedup_p50": ratio("wall_ms", off_runs, on_runs),
    }


def run(args):
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    max_draft_tokens = engine.max_draft_tokens_per_proposal()
    if args.peer_drafts == "controlled" and args.peers:
        if max_draft_tokens == 0:
            raise SystemExit("--peer_drafts controlled needs an Engine that verifies draft tokens.")
        if args.peer_max_new_tokens < 3:
            raise SystemExit("--peer_drafts controlled needs --peer_max_new_tokens >= 3 to leave room for a draft.")

    prompts = {name: encode_prompt(tokenizer, WORKLOADS[name]["prompt"]) for name in args.workloads}
    prompts["peer"] = encode_prompt(tokenizer, PEER_PROMPT)
    peer_reference, reference_status = None, None
    if args.peers:
        peer_reference, reference_status = peer_reference_tokens(
            engine, args, prompts, max_draft_tokens, controlled=args.peer_drafts == "controlled"
        )
    # Peers propose drafts only when they are controlled; otherwise the reference is only checked.
    draft_reference = peer_reference if args.peer_drafts == "controlled" else None

    for _ in range(args.warmup):
        for fast_forward in (False, True):
            run_scenario(engine, args, max_draft_tokens, prompts, draft_reference, fast_forward)
    runs = {False: [], True: []}
    for iteration in range(args.iterations):
        # Alternate which mode runs first so warm-cache effects do not favor either.
        for fast_forward in (False, True) if iteration % 2 == 0 else (True, False):
            scenario, wall_seconds = run_scenario(
                engine, args, max_draft_tokens, prompts, draft_reference, fast_forward
            )
            runs[fast_forward].append(measure(scenario, wall_seconds, tokenizer, peer_reference))

    summary = {
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "max_draft_tokens": max_draft_tokens,
        "peer_reference": reference_status,
        "off": aggregate(runs[False]),
        "on": aggregate(runs[True]),
        "comparison": compare(runs[False], runs[True], tokenizer),
    }
    output = json.dumps(summary, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Paired off/on benchmark for Engine guidance fast-forward")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument(
        "-e",
        "--execution_provider",
        default="follow_config",
        choices=["follow_config", "cpu", "cuda", "dml", "webgpu"],
        help="follow_config keeps the providers in genai_config.json",
    )
    parser.add_argument("--workloads", nargs="+", default=["weather", "tool_call", "invoice"], choices=WORKLOADS)
    parser.add_argument("--requests", type=int, default=8, help="Guided turns per scenario")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Closed loop: at most this many guided turns in flight, the next starting when one "
        "finishes; 0 starts them all at once",
    )
    parser.add_argument(
        "--arrival_rate",
        type=float,
        default=0.0,
        help="Guided turns per second as a Poisson process; 0 starts them all at once",
    )
    parser.add_argument("--peers", type=int, default=0, help="Unguided peer turns started with the scenario")
    parser.add_argument(
        "--replenish_peers",
        action="store_true",
        help="Start a new peer whenever one finishes while guided work remains, so peers stay present",
    )
    parser.add_argument(
        "--peer_delay_ms",
        type=float,
        default=0.0,
        help="Start peers this long after the scenario, so they arrive while guided turns force tokens",
    )
    parser.add_argument(
        "--peer_drafts",
        choices=["model", "controlled"],
        default="model",
        help="model: whatever the Engine drafts on its own (for example DFlash); "
        "controlled: peers propose their own solo output as drafts",
    )
    parser.add_argument("--cancel_after", type=int, default=0, help="Cancel each guided turn after this many tokens")
    parser.add_argument("--event_buffer", type=int, default=64, help="Events drained per Run call")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--peer_max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Optional path for the JSON summary")
    args = parser.parse_args(argv)
    if args.concurrency and args.arrival_rate:
        parser.error("--concurrency and --arrival_rate are mutually exclusive")
    return args


if __name__ == "__main__":
    run(parse_args())
