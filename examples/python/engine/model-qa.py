# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json

import numpy as np
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
    session_token_count = len(system_tokens)
    streaming_tokenizer = tokenizer.create_stream()
    request = engine.create_request(params)
    first_turn = True

    try:
        while prompt := input("🫵  : "):
            if prompt == "/exit":
                break

            user_message = json.dumps([{"role": "user", "content": prompt}])
            turn_tokens = tokenizer.encode(
                tokenizer.apply_chat_template(messages=user_message, add_generation_prompt=True),
            )

            if session_token_count + len(turn_tokens) >= MAX_LENGTH:
                print("Context exhausted; restart to begin a new conversation.")
                break

            session_token_count += len(turn_tokens)
            if first_turn:
                input_tokens = np.concatenate(
                    (
                        np.asarray(system_tokens, dtype=np.int32),
                        np.asarray(turn_tokens, dtype=np.int32),
                    )
                )
                first_turn = False
            else:
                input_tokens = np.asarray(turn_tokens, dtype=np.int32)
            request.begin_turn(input_tokens)

            print("🤖 :", end="", flush=True)

            while engine.has_pending_requests():
                for event in engine.run(8):
                    if event.request is not request:
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

    args = parser.parse_args()
    if args.debug:
        og.set_log_options(
            enabled=True,
            model_input_values=True,
            model_output_values=True,
            model_output_shapes=True,
        )

    run(args)
