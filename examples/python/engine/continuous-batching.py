# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import queue
import random
import threading
import time

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
        self.commands = queue.Queue()
        self.stop_producer = threading.Event()
        self.producer_done = False
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
        request.begin_turn(
            np.asarray(tokens, dtype=np.int32),
            max_generated_tokens=128,
        )

    def admit_initial_requests(self):
        for prompt in self.prompts[: int(self.num_requests * self.load_factor)]:
            self.admit(prompt)

    def produce(self):
        try:
            delayed_prompts = self.prompts[int(len(self.prompts) * self.load_factor) :]
            for index, prompt in enumerate(delayed_prompts):
                if index > 0 and self.stop_producer.wait(1):
                    break
                self.commands.put(("admit", prompt))
        finally:
            self.commands.put(("producer-done", None))

    def drain(self, request: og.Request):
        client_request = self.requests.get(request)
        assert client_request is not None, "Canonical request not found in the pool"
        token_count = 0
        while request.has_unseen_tokens():
            token = request.get_unseen_token()
            client_request.token_stream += client_request.streaming_tokenizer.decode(token)
            token_count += 1

        if request.is_turn_complete():
            if self.debug:
                print(f"🫵  : {client_request.prompt}")
                print(f"🤖 : {client_request.token_stream}")
            request.close()
            del self.requests[request]
            self.bar.update(1)

        return token_count

    def handle_command(self, command):
        action, prompt = command
        if action == "admit":
            self.admit(prompt)
        else:
            self.producer_done = True

    def drain_commands(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self.handle_command(command)


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
        producer_thread = threading.Thread(target=request_pool.produce)
        producer_started = False

        try:
            request_pool.admit_initial_requests()
            producer_thread.start()
            producer_started = True
            start = time.time()
            while True:
                request_pool.drain_commands()
                if self.engine.has_pending_requests():
                    request = self.engine.run()
                    if request is None:
                        raise RuntimeError("Engine returned no request while work remained")
                    self.tokens_decoded += request_pool.drain(request)
                    continue
                if request_pool.producer_done:
                    break
                request_pool.handle_command(request_pool.commands.get())
        finally:
            request_pool.stop_producer.set()
            if producer_started:
                producer_thread.join()
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
