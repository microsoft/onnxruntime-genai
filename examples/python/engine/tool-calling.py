# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_lark_grammar, to_tool

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def require_request_event(event: og.EngineEvent) -> og.Request:
    if event.request is not None:
        return event.request
    if event.flags & og.EngineEventFlags.FAILED:
        outcome = "failed"
    elif event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
        outcome = "was capacity-blocked"
    elif event.flags & og.EngineEventFlags.RETRYABLE:
        outcome = "reported a retryable failure"
    else:
        outcome = "returned an invalid request-less event"
    raise RuntimeError(f"Engine {outcome}; error_code={event.error_code}")


def tool_result_fragment(tool_result):
    return (
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
        f"{json.dumps(tool_result)}\n"
        "</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )


def generate(engine, request, tokenizer):
    stream = tokenizer.create_stream()
    fragments = []
    token_ids = []
    while engine.has_pending_requests():
        for event in engine.run(8):
            if require_request_event(event) is not request:
                raise RuntimeError("Engine returned an unknown request")
            if event.flags & og.EngineEventFlags.TOKEN:
                token_id = event.token
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
    if args.guidance:
        grammar = get_lark_grammar(
            tools=to_tool(tools),
            text_output=False,
            tool_output=True,
            tool_call_start=TOOL_CALL_START,
            tool_call_end=TOOL_CALL_END,
        )
        params.set_guidance("lark_grammar", grammar)
    prompt_tokens = list(tokenizer.encode(prompt))
    request = engine.create_request(params)
    try:
        request.begin_turn(np.asarray(prompt_tokens, dtype=np.int32))

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

        continuation_tokens = list(tokenizer.encode(tool_result_fragment(tool_result)))
        if args.guidance:
            request.close()
            final_params = og.GeneratorParams(model)
            final_params.set_search_options(do_sample=False, max_length=args.max_length)
            request = engine.create_request(final_params)
            final_context = prompt_tokens + tool_call_tokens + continuation_tokens
            request.begin_turn(np.asarray(final_context, dtype=np.int32))
        else:
            request.begin_turn(np.asarray(continuation_tokens, dtype=np.int32))

        final_output, _ = generate(engine, request, tokenizer)
        final_output = final_output.strip()
        if not final_output or TOOL_CALL_START in final_output:
            raise RuntimeError(f"Expected a final assistant answer, received: {final_output!r}")
        print(f"Final answer: {final_output}")
    finally:
        request.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen tool calling with the paged GenAI Engine")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--tools_file", required=True)
    parser.add_argument("--user_prompt", default="What is the weather in Redmond, WA?")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="Constrain the tool call with a Lark grammar; requires a guidance-enabled build",
    )
    run(parser.parse_args())
