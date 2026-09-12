# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json

import numpy as np
import onnxruntime_genai as og

MAX_LENGTH = 1024
SPECULATIVE_COUNT_KEYS = (
    "rounds",
    "draft_tokens_proposed",
    "draft_tokens_evaluated",
    "draft_tokens_accepted",
    "draft_forward_passes",
    "target_forward_passes",
    "standard_fallback_steps",
    "dflash2_failures",
    "dflash2_disables",
    "dflash2_admission_misses",
    "dflash2_context_only_forward_passes",
)


def require_request_event(event: og.EngineEvent) -> og.Request:
    # This single-request workflow cannot continue after any failed turn, even if
    # REQUEST_UNSERVICEABLE leaves the Engine itself healthy.
    if event.flags & og.EngineEventFlags.FAILED:
        raise RuntimeError(f"Generation failed; error_code={event.error_code}")
    if event.request is not None:
        return event.request
    if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
        outcome = "was capacity-blocked"
    elif event.flags & og.EngineEventFlags.RETRYABLE:
        outcome = "reported a retryable failure"
    else:
        outcome = "returned an invalid request-less event"
    raise RuntimeError(f"Engine {outcome}; error_code={event.error_code}")


def speculative_count_delta(before: dict, after: dict) -> dict:
    return {key: after[key] - before[key] for key in SPECULATIVE_COUNT_KEYS}


def prompts(args: argparse.Namespace):
    if args.prompt:
        yield from args.prompt
        return

    while prompt := input("🫵  : "):
        if prompt == "/exit":
            return
        yield prompt


def run(args: argparse.Namespace):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    if args.require_draft_activity and engine.max_draft_tokens_per_proposal() == 0:
        raise RuntimeError("The model does not support speculative draft proposals")

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=MAX_LENGTH,
    )

    session_token_count = 0
    streaming_tokenizer = tokenizer.create_stream()
    request = engine.create_request(params)
    first_turn = True
    try:
        for prompt in prompts(args):
            if args.prompt:
                print(f"🫵  : {prompt}")
            messages = [{"role": "user", "content": prompt}]
            if first_turn:
                messages.insert(0, {"role": "system", "content": ""})
            turn_tokens = tokenizer.encode(
                tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True),
            )

            if session_token_count + len(turn_tokens) >= MAX_LENGTH:
                print("Context exhausted; restart to begin a new conversation.")
                break

            session_token_count += len(turn_tokens)
            input_tokens = np.asarray(turn_tokens, dtype=np.int32)
            first_turn = False
            turn_options = None
            if args.max_new_tokens is not None:
                turn_options = og.TurnOptions(request)
                turn_options.set_max_generated_tokens(args.max_new_tokens)
            stats_before = dict(engine.get_speculative_stats())
            turn_id = request.begin_turn(input_tokens, turn_options)

            print("🤖 :", end="", flush=True)

            event_buffer = engine.create_event_buffer(8)
            while engine.has_pending_requests():
                for event in engine.run(event_buffer):
                    if require_request_event(event) is not request:
                        raise RuntimeError("Engine returned an unknown request")
                    if event.flags & og.EngineEventFlags.TOKEN:
                        token = int(event.token)
                        session_token_count += 1
                        print(
                            streaming_tokenizer.decode(token),
                            end="",
                            flush=True,
                        )
            print()
            stats_after = dict(engine.get_speculative_stats())
            stats_delta = speculative_count_delta(stats_before, stats_after)
            if args.require_draft_activity:
                if stats_delta["draft_tokens_proposed"] == 0 or stats_delta["draft_tokens_evaluated"] == 0:
                    raise RuntimeError(f"Turn {turn_id} produced no speculative draft activity")
                if stats_delta["dflash2_failures"] or stats_delta["dflash2_disables"]:
                    raise RuntimeError(f"Turn {turn_id} reported a DFlash2 failure or disable")
            if args.show_stats or args.require_draft_activity:
                evaluated = stats_delta["draft_tokens_evaluated"]
                stats_delta["acceptance_rate"] = stats_delta["draft_tokens_accepted"] / evaluated if evaluated else 0.0
                print(f"📊 : {json.dumps({'turn_id': turn_id, 'delta': stats_delta})}")
    finally:
        request.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end AI Question/Answer example for gen-ai",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Onnx model folder path (must contain genai_config.json and model.onnx)",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        required=True,
        choices=["cpu", "cuda", "webgpu"],
        help="Execution provider to run ONNX model with",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to run non-interactively. Repeat to exercise continuous decoding across turns.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum generated tokens per turn. By default, generation stops at EOS or max length.",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print the speculative-decoding counter delta after each turn.",
    )
    parser.add_argument(
        "--require-draft-activity",
        action="store_true",
        help="Fail unless every turn proposes and evaluates draft tokens without a DFlash2 failure or disable.",
    )

    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.debug:
        og.set_log_options(
            enabled=True,
            model_input_values=True,
            model_output_values=True,
            model_output_shapes=True,
        )

    run(args)
