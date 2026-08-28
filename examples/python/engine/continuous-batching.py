# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import random
import time
from collections import deque

import numpy as np
import onnxruntime_genai as og
import tqdm
from datasets import load_dataset


def get_random_prompts(num_questions: int, split="validation") -> list[str]:
    dataset = load_dataset("squad_v2", split=split)
    questions = [item["question"] for item in dataset]
    return random.sample(questions, min(num_questions, len(questions)))


class ClientRequest:
    def __init__(self, prompt: str, tokenizer: og.Tokenizer):
        self.prompt = prompt
        self.streaming_tokenizer = tokenizer.create_stream()
        self.token_stream = ""


class RequestPool:
    def __init__(
        self,
        model: og.Model,
        tokenizer: og.Tokenizer,
        engine: og.Engine,
        num_requests: int,
        load_factor: float = 1,
        debug: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.num_requests = num_requests
        self.requests: dict[og.Request, ClientRequest] = {}
        self.prompts = get_random_prompts(num_requests)
        self.load_factor = load_factor
        initial_count = int(self.num_requests * self.load_factor)
        self.initial_prompts = self.prompts[:initial_count]
        self.delayed_prompts = deque(self.prompts[initial_count:])
        self.next_admission_time = time.monotonic()
        self.bar = tqdm.tqdm(total=len(self.prompts))
        self.debug = debug

    def admit(self, prompt: str):
        client_request = ClientRequest(prompt, self.tokenizer)
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            do_sample=False,
            max_length=256,
        )
        messages = json.dumps(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ]
        )
        tokens = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
        )
        request = self.engine.create_request(params)
        self.requests[request] = client_request
        turn_params = og.TurnParams(request)
        turn_params.set_max_generated_tokens(128)
        request.begin_turn(
            np.asarray(tokens, dtype=np.int32),
            turn_params,
        )

    def admit_initial_requests(self):
        for prompt in self.initial_prompts:
            self.admit(prompt)

    def admit_due_request(self):
        if not self.delayed_prompts or time.monotonic() < self.next_admission_time:
            return
        self.admit(self.delayed_prompts.popleft())
        self.next_admission_time = time.monotonic() + 1

    def drain(self, event: og.EngineEvent):
        request = event.request
        client_request = self.requests.get(request)
        assert client_request is not None, "Canonical request not found in the pool"
        token_count = 0
        if event.flags & og.EngineEventFlags.TOKEN:
            token = event.token
            client_request.token_stream += client_request.streaming_tokenizer.decode(token)
            token_count += 1

        if event.flags & og.EngineEventFlags.TURN_FINISHED:
            if self.debug:
                print(f"🫵  : {client_request.prompt}")
                print(f"🤖 : {client_request.token_stream}")
            request.close()
            del self.requests[request]
            self.bar.update(1)

        return token_count

    def wait_for_next_admission(self):
        if self.delayed_prompts:
            time.sleep(max(0, self.next_admission_time - time.monotonic()))


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

    def run(self, request_pool: RequestPool):
        request_pool.admit_initial_requests()
        start = time.time()
        try:
            while self.engine.has_pending_requests() or request_pool.delayed_prompts:
                request_pool.admit_due_request()
                if self.engine.has_pending_requests():
                    for event in self.engine.run(8):
                        self.tokens_decoded += request_pool.drain(event)
                    continue
                request_pool.wait_for_next_admission()
        finally:
            for request in list(request_pool.requests):
                request.close()
                del request_pool.requests[request]
        return time.time() - start


def run(args: argparse.Namespace):
    engine = Engine(args.model_path, args.execution_provider, args.debug)
    request_pool = RequestPool(
        engine.model,
        engine.tokenizer,
        engine.engine,
        args.num_requests,
        load_factor=args.load_factor,
        debug=args.debug,
    )

    elapsed = engine.run(request_pool)

    request_pool.bar.close()
    print(f"⌛ Tokens per second: {engine.tokens_decoded / elapsed:.2f}")


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
    parser.add_argument(
        "-l",
        "--load_factor",
        type=float,
        default=1.0,
        help="Load factor to control the number of preloaded in-flight requests (default: 1.0)",
    )
    args = parser.parse_args()

    run(args)
