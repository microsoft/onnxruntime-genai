# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json

import onnxruntime_genai as og

# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True, model_output_shapes=True)

def run(args: argparse.Namespace):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)

    while prompt:= input("🫵  : "):
        if prompt == "quit()":
            break

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{prompt}"}
        ]
        messages2 = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"What is the capital of France?"}
        ]
        messages = json.dumps(messages)
        messages2 = json.dumps(messages2)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=1024,
        )

        params2 = og.GeneratorParams(model)
        params2.set_search_options(
            do_sample=False,
            max_length=256,
        )
        request = og.Request(tokenizer.encode(tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)), params)
        request2 = og.Request(tokenizer.encode(tokenizer.apply_chat_template(messages=messages2, add_generation_prompt=True)), params2)
        streaming_tokenizer = tokenizer.create_stream()
        streaming_tokenizer2 = tokenizer.create_stream()

        engine.add_request(request)
        # engine.add_request(request2)

        i = 0
        while engine.has_pending_requests():
            i += 1
            if i == 10:
                engine.add_request(request2)  # Add a second request to test multiple requests
            engine.step()

        print(f"🤖 :", end="", flush=True)
        while request.has_unseen_tokens():
            print(streaming_tokenizer.decode(request.get_unseen_token()), end="", flush=True)

        print()
        print(f"🤖 :", end="", flush=True)
        while request2.has_unseen_tokens():
            print(streaming_tokenizer2.decode(request2.get_unseen_token()), end="", flush=True)

        engine.remove_request(request)
        engine.remove_request(request2)

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
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
    args = parser.parse_args()
    run(args)
