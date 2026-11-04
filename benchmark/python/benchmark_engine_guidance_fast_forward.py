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
    begun_at: float
    reference: list | None  # Tokens a peer proposes as drafts, from its solo run.
    tokens: list = field(default_factory=list)
    token_times: list = field(default_factory=list)
    run_calls_to_first_token: int | None = None
    finish_reason: str | None = None
    finished_at: float | None = None
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

    def begin(self, kind, workload, tokens, max_new_tokens, configure=None, reference=None):
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
        self.turns.append(Turn(kind, workload, request, turn_id, time.perf_counter(), reference))

    def _turn_for(self, request):
        return next(turn for turn in self.turns if turn.request is request)

    def _propose_peer_drafts(self):
        for turn in self.turns:
            if turn.reference is None or turn.finished_at is not None or not turn.tokens:
                continue
            proposal = turn.reference[len(turn.tokens) : len(turn.tokens) + self.max_draft_tokens]
            if proposal:
                turn.request.set_draft_tokens(np.asarray(proposal, dtype=np.int32))

    def _cancel_due_turns(self):
        for turn in self.turns:
            if (
                turn.kind == "guided"
                and turn.finished_at is None
                and turn.cancel_requested_at is None
                and len(turn.tokens) >= self.args.cancel_after
            ):
                turn.cancel_requested_at = time.perf_counter()
                turn.request.cancel_turn(turn.turn_id)

    def run(self, arrivals):
        """`arrivals` holds (seconds after start, callback that begins a turn)."""
        started = time.perf_counter()
        pending = sorted(arrivals, key=lambda arrival: arrival[0])
        while pending or self.engine.has_pending_requests():
            elapsed = time.perf_counter() - started
            while pending and pending[0][0] <= elapsed:
                pending.pop(0)[1]()
            if not self.engine.has_pending_requests():
                time.sleep(max(0.0, pending[0][0] - elapsed))
                continue
            self._propose_peer_drafts()
            events = self.engine.run(self.buffer)
            self.run_calls += 1
            for event in events:
                turn = self._turn_for(require_request_event(event))
                now = time.perf_counter()
                if event.flags & og.EngineEventFlags.TOKEN:
                    turn.tokens.append(event.token)
                    turn.token_times.append(now)
                    if turn.run_calls_to_first_token is None:
                        turn.run_calls_to_first_token = self.run_calls
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    turn.finish_reason = event.finish_reason.name
                    turn.finished_at = now
            if self.args.cancel_after:
                self._cancel_due_turns()
        for turn in self.turns:
            turn.request.close()
        return time.perf_counter() - started


def run_scenario(engine, args, max_draft_tokens, prompts, peer_reference, fast_forward):
    scenario = Scenario(engine, args, max_draft_tokens)

    def start_guided(workload):
        spec = WORKLOADS[workload]

        def configure(turn_options):
            turn_options.set_guidance(spec["type"], spec["data"])
            turn_options.set_guidance_fast_forward(fast_forward)

        scenario.begin("guided", workload, prompts[workload], args.max_new_tokens, configure)

    def start_peer():
        scenario.begin("peer", "peer", prompts["peer"], args.peer_max_new_tokens, reference=peer_reference)

    # Peers start first, so their drafts are in flight when guided turns arrive.
    arrivals = [(0.0, start_peer) for _ in range(args.peers)]
    rng = random.Random(args.seed)  # The same arrival times for off and on.
    at = 0.0
    for index in range(args.requests):
        workload = args.workloads[index % len(args.workloads)]
        arrivals.append((at, lambda workload=workload: start_guided(workload)))
        if args.arrival_rate:
            at += rng.expovariate(args.arrival_rate)
    wall_seconds = scenario.run(arrivals)
    return scenario, wall_seconds


def solo_peer_tokens(engine, args, prompts):
    """The peer's own greedy output, used as perfectly accurate controlled drafts."""
    scenario = Scenario(engine, args, max_draft_tokens=0)
    scenario.begin("peer", "peer", prompts["peer"], args.peer_max_new_tokens)
    scenario.run([])
    return scenario.turns[0].tokens


def valid_json(tokenizer, turn):
    if turn.finish_reason != "EOS":
        return False
    try:
        parsed = json.loads(tokenizer.decode(np.asarray(turn.tokens, dtype=np.int32)))
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and set(WORKLOADS[turn.workload]["schema"]["required"]) <= set(parsed)


def measure(scenario, wall_seconds, tokenizer):
    guided = [turn for turn in scenario.turns if turn.kind == "guided"]
    peers = [turn for turn in scenario.turns if turn.kind == "peer"]
    first_token = [(turn.token_times[0] - turn.begun_at) * 1000 for turn in guided if turn.tokens]
    return {
        "run_calls": scenario.run_calls,
        "wall_ms": wall_seconds * 1000,
        "guided_tokens": [turn.tokens for turn in guided],
        "guided_first_token_ms": first_token,
        "guided_end_to_end_ms": [(turn.finished_at - turn.begun_at) * 1000 for turn in guided],
        "guided_finish_reasons": [turn.finish_reason for turn in guided],
        "guided_valid_json": [valid_json(tokenizer, turn) for turn in guided],
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
    }


def aggregate(runs):
    def flat(name):
        return [value for run in runs for value in run[name]]

    reasons = flat("guided_finish_reasons")
    return {
        "run_calls": summarize([run["run_calls"] for run in runs]),
        "wall_ms": summarize([run["wall_ms"] for run in runs]),
        "guided_tokens_per_turn": summarize([len(tokens) for tokens in flat("guided_tokens")]),
        "guided_first_token_ms": summarize(flat("guided_first_token_ms")),
        "guided_end_to_end_ms": summarize(flat("guided_end_to_end_ms")),
        "guided_finish_reasons": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        "guided_valid_json": sum(flat("guided_valid_json")),
        "guided_turns": len(reasons),
        "cancel_latency_ms": summarize(flat("cancel_latency_ms")),
        "peer_gap_ms": summarize(flat("peer_gap_ms")),
        "peer_run_calls_to_first_token": summarize(flat("peer_run_calls_to_first_token")),
    }


def compare(off_runs, on_runs, tokenizer):
    pairs = [
        (off, on)
        for off_run, on_run in zip(off_runs, on_runs, strict=True)
        for off, on in zip(off_run["guided_tokens"], on_run["guided_tokens"], strict=True)
    ]

    def text(tokens):
        return tokenizer.decode(np.asarray(tokens, dtype=np.int32))

    off_wall = statistics.median(run["wall_ms"] for run in off_runs)
    on_wall = statistics.median(run["wall_ms"] for run in on_runs)
    return {
        "identical_token_ids": sum(off == on for off, on in pairs),
        "identical_text": sum(text(off) == text(on) for off, on in pairs),
        "pairs": len(pairs),
        "run_calls_on_over_off": statistics.median(run["run_calls"] for run in on_runs)
        / statistics.median(run["run_calls"] for run in off_runs),
        "wall_speedup_p50": off_wall / on_wall if on_wall else None,
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
    if args.peer_drafts == "controlled" and args.peers and max_draft_tokens == 0:
        raise SystemExit("--peer_drafts controlled needs an Engine that verifies draft tokens.")

    prompts = {name: encode_prompt(tokenizer, WORKLOADS[name]["prompt"]) for name in args.workloads}
    prompts["peer"] = encode_prompt(tokenizer, PEER_PROMPT)
    peer_reference = solo_peer_tokens(engine, args, prompts) if args.peer_drafts == "controlled" else None

    for _ in range(args.warmup):
        for fast_forward in (False, True):
            run_scenario(engine, args, max_draft_tokens, prompts, peer_reference, fast_forward)
    runs = {False: [], True: []}
    for iteration in range(args.iterations):
        # Alternate which mode runs first so warm-cache effects do not favor either.
        for fast_forward in (False, True) if iteration % 2 == 0 else (True, False):
            scenario, wall_seconds = run_scenario(engine, args, max_draft_tokens, prompts, peer_reference, fast_forward)
            runs[fast_forward].append(measure(scenario, wall_seconds, tokenizer))

    summary = {
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "max_draft_tokens": max_draft_tokens,
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
        "--arrival_rate",
        type=float,
        default=0.0,
        help="Guided turns per second as a Poisson process; 0 starts them all at once",
    )
    parser.add_argument("--peers", type=int, default=0, help="Unguided peer turns running throughout")
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
