# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json

import onnxruntime_genai as og

MAX_LENGTH = 1024


def run(args: argparse.Namespace):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=MAX_LENGTH,
    )

    system_message = json.dumps([{"role": "system", "content": ""}])
    system_tokens = tokenizer.encode(
        tokenizer.apply_chat_template(messages=system_message, add_generation_prompt=False),
    )
    # Temporary DML-only fallback while the low-level Engine continuation API is transitional:
    # replay exact token IDs in a fresh request instead of reconstructing text. Replaying the full
    # session also preserves max_length as a session-total limit. Other advertised providers keep
    # one resident request and use continue_with().
    use_dml_replay = args.execution_provider == "dml"
    logical_token_history = [int(token) for token in system_tokens]

    request = None
    if not use_dml_replay:
        request = og.Request(params)
        request.add_tokens(system_tokens)

    streaming_tokenizer = tokenizer.create_stream()
    request_added = False

    try:
        while prompt := input("🫵  : "):
            if prompt == "/exit":
                break

            user_message = json.dumps([{"role": "user", "content": prompt}])
            turn_tokens = tokenizer.encode(
                tokenizer.apply_chat_template(messages=user_message, add_generation_prompt=True),
            )

            if len(logical_token_history) + len(turn_tokens) >= MAX_LENGTH:
                print("Context exhausted; restart to begin a new conversation.")
                break

            logical_token_history.extend(int(token) for token in turn_tokens)
            if use_dml_replay:
                request = og.Request(params)
                request.add_tokens(logical_token_history)
                engine.add_request(request)
                request_added = True
            elif request_added:
                request.continue_with(turn_tokens)
            else:
                request.add_tokens(turn_tokens)
                engine.add_request(request)
                request_added = True

            print("🤖 :", end="", flush=True)

            while ready_request := engine.step():
                while ready_request.has_unseen_tokens():
                    token = int(ready_request.get_unseen_token())
                    logical_token_history.append(token)
                    print(
                        streaming_tokenizer.decode(token),
                        end="",
                        flush=True,
                    )

            print()
            if use_dml_replay:
                engine.remove_request(request)
                request_added = False
                request = None
    finally:
        if request_added and request is not None:
            engine.remove_request(request)


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
        choices=["cpu", "cuda", "dml", "webgpu"],
        help="Execution provider to run ONNX model with",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    if args.debug:
        og.set_log_options(
            enabled=True,
            model_input_values=True,
            model_output_values=True,
            model_output_shapes=True,
        )

    run(args)
