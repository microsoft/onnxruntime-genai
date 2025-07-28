# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import random
import threading
import time

import onnxruntime_genai as og
import tqdm
from datasets import load_dataset


def get_random_prompts(num_questions: int, split="validation") -> list[str]:
    dataset = load_dataset("squad_v2", split=split)
    questions = [item["question"] for item in dataset]
    return random.sample(questions, min(num_questions, len(questions)))


class ClientRequest:
    def __init__(
        self, prompt: str, model: og.Model, tokenizer: og.Tokenizer, opaque_data: any
    ):
        self.prompt = prompt
        self.params = og.GeneratorParams(model)
        self.params.set_search_options(
            do_sample=False,
            max_length=256,
        )

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{prompt}"},
        ]
        messages = json.dumps(messages)

        self.request = og.Request(self.params)
        self.request.add_tokens(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    messages=messages, add_generation_prompt=True
                )
            )
        )
        self.request.set_opaque_data(opaque_data)
        self.streaming_tokenizer = tokenizer.create_stream()
        self.token_stream = ""


class RequestPool:
    def __init__(
        self,
        model: og.Model,
        tokenizer: og.Tokenizer,
        engine: og.Engine,
        num_requests: int,
        load_factor: float = 0.2,
        debug: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.num_requests = num_requests
        self.requests: list[ClientRequest] = []
        self.prompts = get_random_prompts(num_requests)
        self.load_factor = load_factor
        self.lock = threading.Lock()
        self.bar = tqdm.tqdm(total=len(self.prompts))
        self.debug = debug

        # Add load_factor * num_requests requests to the engine
        for prompt in self.prompts[: int(num_requests * load_factor)]:
            request = ClientRequest(prompt, model, tokenizer, self)
            self.requests.append(request)
            self.engine.add_request(request.request)

    def fill(self):
        for i, prompt in enumerate(
            self.prompts[int(len(self.prompts) * self.load_factor) :]
        ):
            request = ClientRequest(prompt, self.model, self.tokenizer, self)
            with self.lock:
                self.requests.append(request)
                self.engine.add_request(request.request)
            time.sleep(1)  # Simulate some delay in request generation

    def drain(self, request: og.Request):
        with self.lock:
            client_request = next(
                (r for r in self.requests if r.request == request), None
            )
            while request.has_unseen_tokens():
                token = request.get_unseen_token()
                client_request.token_stream += (
                    client_request.streaming_tokenizer.decode(token)
                )

            if request.is_done():
                assert (
                    client_request is not None
                ), "Client request not found in the pool"

                if self.debug:
                    print(f"🫵  : {client_request.prompt}")
                    print(f"🤖 : {client_request.token_stream}")
                self.engine.remove_request(request)
                self.requests.remove(client_request)
                self.bar.update(1)


class Engine:
    def __init__(self, model_path: str, execution_provider: str, debug: bool):
        self.config = og.Config(model_path)
        self.config.clear_providers()
        if execution_provider != "cpu":
            self.config.append_provider(execution_provider)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)
        self.engine = og.Engine(self.model)
        self.debug = debug
        self.tokens_decoded = 0

    def run(self):
        while request := self.engine.step():
            request_pool = request.get_opaque_data()
            request_pool.drain(request)
            self.tokens_decoded += 1


def run(args: argparse.Namespace):
    engine = Engine(args.model_path, args.execution_provider, args.debug)
    request_pool = RequestPool(
        engine.model,
        engine.tokenizer,
        engine.engine,
        args.num_requests,
        debug=args.debug,
    )

    producer_thread = threading.Thread(target=request_pool.fill)
    producer_thread.start()

    start = time.time()
    engine.run()
    end = time.time()

    request_pool.bar.close()
    print(f"⌛Tokens per second: {engine.tokens_decoded / (end - start):.2f}")


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
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "-n",
        "--num_requests",
        type=int,
        default=1,
        help="Number of requests to process in the pool",
    )
    args = parser.parse_args()

    run(args)
