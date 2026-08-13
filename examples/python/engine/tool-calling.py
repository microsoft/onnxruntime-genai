# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def generate(engine, request, tokenizer):
    stream = tokenizer.create_stream()
    fragments = []
    token_ids = []
    while not request.is_done():
        ready_request = engine.step()
        if ready_request is None:
            raise RuntimeError("Engine stopped before the request completed")
        while ready_request.has_unseen_tokens():
            token_id = ready_request.get_unseen_token()
            token_ids.append(token_id)
            fragments.append(stream.decode(token_id))
    return "".join(fragments), token_ids


def parse_tool_call(output):
    start = output.find(TOOL_CALL_START)
    end = output.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
    if start >= 0 and end >= 0:
        payload = json.loads(output[start + len(TOOL_CALL_START) : end].strip())
    else:
        payload = json.loads(output.strip())
    if isinstance(payload, list):
        if len(payload) != 1:
            raise RuntimeError(f"Expected one tool call, received {len(payload)}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise RuntimeError("Tool call payload must be a JSON object")
    return payload


def contains_token_sequence(tokens, sequence):
    return any(tokens[index : index + len(sequence)] == sequence for index in range(len(tokens) - len(sequence) + 1))


def call_weather_tool(tool_call):
    if tool_call.get("name") != "get_weather":
        raise RuntimeError(f"Expected get_weather, received {tool_call.get('name')!r}")

    arguments = tool_call.get("arguments", tool_call.get("parameters"))
    if not isinstance(arguments, dict) or not isinstance(arguments.get("city"), str):
        raise RuntimeError("get_weather requires a string city argument")

    return {
        "city": arguments["city"],
        "temperature": "52 F",
        "conditions": "Partly cloudy",
    }


def uses_dynamic_batching(model_path):
    with open(Path(model_path) / "genai_config.json", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("engine", {}).get("dynamic_batching") is not None


def run(args):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)

    with open(args.tools_file, encoding="utf-8") as file:
        tools = json.load(file)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages),
        tools=json.dumps(tools),
        add_generation_prompt=True,
    )

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=args.max_length)
    prompt_tokens = list(tokenizer.encode(prompt))
    request = og.Request(params)
    request.add_tokens(np.asarray(prompt_tokens, dtype=np.int32))
    engine.add_request(request)

    tool_call_output, tool_call_tokens = generate(engine, request, tokenizer)
    tool_call_start_tokens = list(tokenizer.encode(TOOL_CALL_START))
    tool_call_end_tokens = list(tokenizer.encode(TOOL_CALL_END))
    if not contains_token_sequence(tool_call_tokens, tool_call_start_tokens) or not contains_token_sequence(
        tool_call_tokens, tool_call_end_tokens
    ):
        raise RuntimeError(f"Assistant did not produce tool-call delimiters: {tool_call_output!r}")
    tool_call = parse_tool_call(tool_call_output)
    tool_result = call_weather_tool(tool_call)
    print(f"Tool call: {tool_call_output}")
    print(f"Tool result: {json.dumps(tool_result)}")

    messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"type": "function", "function": tool_call}],
            },
            {"role": "tool", "content": json.dumps(tool_result)},
        ]
    )
    continued_prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages),
        tools=json.dumps(tools),
        add_generation_prompt=True,
    )
    continued_tokens = list(tokenizer.encode(continued_prompt))
    cached_tokens = prompt_tokens + tool_call_tokens
    if continued_tokens[: len(cached_tokens)] != cached_tokens:
        raise RuntimeError("The continued chat template does not preserve the Engine request token prefix")
    continuation_tokens = continued_tokens[len(cached_tokens) :]
    if not continuation_tokens:
        raise RuntimeError("The continued chat template produced no tool-response tokens")

    if uses_dynamic_batching(args.model_path):
        request.add_tokens(np.asarray(continuation_tokens, dtype=np.int32))
    else:
        engine.remove_request(request)
        request = og.Request(params)
        request.add_tokens(np.asarray(continued_tokens, dtype=np.int32))
        engine.add_request(request)

    final_output, _ = generate(engine, request, tokenizer)
    final_output = final_output.strip()
    engine.remove_request(request)
    if not final_output or TOOL_CALL_START in final_output:
        raise RuntimeError(f"Expected a final assistant answer, received: {final_output!r}")
    print(f"Final answer: {final_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen tool calling with the paged GenAI Engine")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--tools_file", required=True)
    parser.add_argument("--user_prompt", default="What is the weather in Redmond, WA?")
    parser.add_argument("--max_length", type=int, default=256)
    run(parser.parse_args())