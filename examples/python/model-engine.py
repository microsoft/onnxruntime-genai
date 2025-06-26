# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

import onnxruntime_genai as og


# og.set_log_options(enabled=True, model_input_values=True, model_output_values=True, model_output_shapes=True)

class ClientRequest:
    def __init__(self, prompt: str, model: og.Model, tokenizer: og.Tokenizer):
        self.prompt = prompt
        self.request_params = og.GeneratorParams(model)
        self.request_params.set_search_options(
            do_sample=False,
            max_length=256,
        )
        self.params = og.GeneratorParams(model)
        self.params.set_search_options(
            do_sample=False,
            max_length=256,
        )
        
        
        import json
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{prompt}"}
        ]
        messages = json.dumps(messages)

        self.request = og.Request(tokenizer.encode(tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)), self.params)
        self.streaming_tokenizer = tokenizer.create_stream()
        self.token_stream = ""


class RequestPool:
    def __init__(
        self,
        model: og.Model,
        tokenizer: og.Tokenizer,
        engine: og.Engine,
        num_requests: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.num_requests = num_requests
        self.requests: list[ClientRequest] = []

    def fill(self):
        for _ in range(self.num_requests):
            prompt = f"What is 2 + 3?"
            request = ClientRequest(prompt, self.model, self.tokenizer)
            self.requests.append(request)
            self.engine.add_request(request.request)

    def drain(self):
        requests_to_remove = []
        for request in self.requests:
            while request.request.has_unseen_tokens():
                token = request.request.get_unseen_token()
                print("unseen token", token)
                request.token_stream += request.streaming_tokenizer.decode(token)

            if request.request.is_done():
                requests_to_remove.append(request)

        for request in self.requests:
            print(f"🫵: {request.prompt}")
            print(f"🤖: {request.token_stream}")
            self.requests.remove(request)


class Engine:
    def __init__(self, model_path: str, execution_provider: str):
        self.config = og.Config(model_path)
        self.config.clear_providers()
        if execution_provider != "cpu":
            self.config.append_provider(execution_provider)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)
        self.engine = og.Engine(self.model)

    def run(self):
        while self.engine.has_pending_requests():
            self.engine.step()


def run(args: argparse.Namespace):
    engine = Engine(args.model_path, args.execution_provider)
    request_pool = RequestPool(engine.model, engine.tokenizer, engine.engine, 1)
    # TODO: Each of the below should be a separate thread
    request_pool.fill()
    engine.run()
    request_pool.drain()


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
