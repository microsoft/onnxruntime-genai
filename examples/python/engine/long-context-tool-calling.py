# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime_genai as og


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def load_payload(payload_path):
    with open(payload_path, encoding="utf-8") as file:
        payload = json.load(file)
    messages = payload.get("messages")
    tools = payload.get("tools", payload.get("options", {}).get("tools"))
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("The payload must contain a non-empty messages array")
    if not isinstance(tools, list) or not tools:
        raise RuntimeError("The payload must contain tools or options.tools")
    return messages, tools


def tool_function(tool):
    function = tool.get("function", tool)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise RuntimeError(f"Invalid function tool definition: {tool!r}")
    return function


def infer_schema_type(schema):
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return next((value for value in schema_type if value != "null"), "null")
    if schema_type:
        return schema_type
    if "properties" in schema or "required" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "string"


def sample_string(name, schema):
    if schema.get("format") in {"uri", "uri-reference", "url"}:
        return "https://example.com"
    lowered = name.lower()
    if "filepath" in lowered or lowered.endswith("path") or lowered == "path":
        return "sample.txt"
    if "directory" in lowered or lowered.endswith("dir"):
        return "sample-directory"
    if "command" in lowered:
        return "echo long-context-tool-test"
    if "query" in lowered or "prompt" in lowered or "intent" in lowered:
        return "long-context tool-calling validation"
    if "code" in lowered:
        return "print('long-context tool test')"
    if "content" in lowered or "text" in lowered:
        return "synthetic tool input"
    return f"test-{name.replace('_', '-')}"


def sample_value(schema, name="value"):
    if not isinstance(schema, dict):
        return "test-value"
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    alternatives = schema.get("anyOf", schema.get("oneOf"))
    if alternatives:
        alternative = next(
            (value for value in alternatives if isinstance(value, dict) and value.get("type") != "null"),
            alternatives[0],
        )
        return sample_value(alternative, name)

    schema_type = infer_schema_type(schema)
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return {key: sample_value(properties.get(key, {}), key) for key in required}
    if schema_type == "array":
        item_count = max(1, schema.get("minItems", 0))
        return [sample_value(schema.get("items", {}), name) for _ in range(item_count)]
    if schema_type == "integer":
        return max(1, schema.get("minimum", 1))
    if schema_type == "number":
        return max(1.0, schema.get("minimum", 1.0))
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return sample_string(name, schema)


def validate_value(value, schema, location="arguments"):
    if not isinstance(schema, dict) or not schema:
        if value not in ({}, None):
            raise RuntimeError(f"{location} must be empty because the tool declares no parameters")
        return
    alternatives = schema.get("anyOf", schema.get("oneOf"))
    if alternatives:
        errors = []
        for alternative in alternatives:
            try:
                validate_value(value, alternative, location)
                return
            except RuntimeError as error:
                errors.append(str(error))
        raise RuntimeError(f"{location} does not match any allowed schema: {'; '.join(errors)}")
    if "const" in schema and value != schema["const"]:
        raise RuntimeError(f"{location} must equal {schema['const']!r}")
    if schema.get("enum") and value not in schema["enum"]:
        raise RuntimeError(f"{location} must be one of {schema['enum']!r}")

    schema_type = infer_schema_type(schema)
    expected_types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    expected = expected_types.get(schema_type)
    if expected and not isinstance(value, expected):
        raise RuntimeError(f"{location} must be {schema_type}, received {type(value).__name__}")
    if schema_type == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise RuntimeError(f"{location}.{key} is required")
        for key, child in value.items():
            if key in properties:
                validate_value(child, properties[key], f"{location}.{key}")
            elif schema.get("additionalProperties") is False:
                raise RuntimeError(f"{location}.{key} is not allowed")
    if schema_type == "array":
        for index, child in enumerate(value):
            validate_value(child, schema.get("items", {}), f"{location}[{index}]")


def tool_schema(function):
    parameters = function.get("parameters")
    if not parameters:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    return parameters


def synthetic_call(function):
    schema = tool_schema(function)
    arguments = sample_value(schema, "arguments")
    if not isinstance(arguments, dict):
        raise RuntimeError(f"Tool {function['name']!r} does not have an object argument schema")
    validate_value(arguments, schema)
    return {"name": function["name"], "arguments": arguments}


def mock_execute(tool_by_name, tool_call):
    name = tool_call.get("name")
    if name not in tool_by_name:
        raise RuntimeError(f"Unknown tool: {name!r}")
    arguments = tool_call.get("arguments", tool_call.get("parameters", {}))
    validate_value(arguments, tool_schema(tool_by_name[name]))
    return {
        "ok": True,
        "tool": name,
        "message": "Validated by the side-effect-free long-context test executor.",
    }


def parse_tool_calls(output):
    payloads = []
    search_start = 0
    while True:
        start = output.find(TOOL_CALL_START, search_start)
        if start < 0:
            break
        end = output.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
        if end < 0:
            raise RuntimeError(f"Tool call is missing {TOOL_CALL_END!r}: {output!r}")
        payloads.append(output[start + len(TOOL_CALL_START) : end].strip())
        search_start = end + len(TOOL_CALL_END)
    if not payloads:
        payloads = [output.strip()]

    tool_calls = []
    for payload in payloads:
        parsed = json.loads(payload)
        if isinstance(parsed, list):
            tool_calls.extend(parsed)
        else:
            tool_calls.append(parsed)
    if not tool_calls or not all(isinstance(call, dict) and "name" in call for call in tool_calls):
        raise RuntimeError(f"Expected one or more JSON tool-call objects, received: {output!r}")
    return tool_calls


def generate(engine, request, tokenizer):
    stream = tokenizer.create_stream()
    fragments = []
    token_ids = []
    while not request.is_turn_complete():
        ready_request = engine.step()
        if ready_request is None:
            raise RuntimeError("Engine stopped before the request completed")
        while ready_request.has_unseen_tokens():
            token_id = ready_request.get_unseen_token()
            token_ids.append(token_id)
            fragments.append(stream.decode(token_id))
    return "".join(fragments).strip(), token_ids


def model_context_length(model_path):
    with open(Path(model_path) / "genai_config.json", encoding="utf-8") as file:
        config = json.load(file)
    return config["model"]["context_length"]


def new_request(engine, params, tokens):
    request = og.Request(params)
    request.add_tokens(np.asarray(tokens, dtype=np.int32))
    engine.add_request(request)
    return request


def render_prompt(tokenizer, messages, tools):
    return tokenizer.apply_chat_template(
        messages=json.dumps(messages),
        tools=json.dumps(tools),
        add_generation_prompt=True,
    )


def select_live_tools(selection, tool_by_name):
    if not selection:
        return []
    requested = [name.strip() for name in selection.split(",") if name.strip()]
    if requested == ["all"]:
        return list(tool_by_name)
    unknown = [name for name in requested if name not in tool_by_name]
    if unknown:
        raise RuntimeError(f"Unknown --live_tools names: {', '.join(unknown)}")
    return requested


def run_live_tool(args, model, tokenizer, engine, messages, tools, tool_by_name, name):
    expected_call = synthetic_call(tool_by_name[name])
    test_messages = list(messages)
    test_messages.append(
        {
            "role": "user",
            "content": (
                f"Long-context tool test: call {name} exactly once now. "
                f"Use arguments matching this example: {json.dumps(expected_call['arguments'])}. "
                "Do not call any other tool and do not answer with prose."
            ),
        }
    )
    prompt = render_prompt(tokenizer, test_messages, tools)
    prompt_tokens = list(tokenizer.encode(prompt))
    context_length = model_context_length(args.model_path)
    max_length = min(context_length, len(prompt_tokens) + args.max_new_tokens)
    if max_length <= len(prompt_tokens):
        raise RuntimeError(f"Tool {name!r} has no generation headroom in the {context_length}-token context")

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)
    started = time.perf_counter()
    request = new_request(engine, params, prompt_tokens)
    output, generated_tokens = generate(engine, request, tokenizer)
    first_turn_seconds = time.perf_counter() - started
    parsed_calls = parse_tool_calls(output)
    if len(parsed_calls) != 1:
        raise RuntimeError(f"Expected one call to {name!r}, received {len(parsed_calls)} calls: {output!r}")
    if parsed_calls[0].get("name") != name:
        raise RuntimeError(f"Expected tool {name!r}, received {parsed_calls[0].get('name')!r}: {output!r}")
    result = mock_execute(tool_by_name, parsed_calls[0])
    print(
        f"LIVE CALL PASS {name}: prompt={len(prompt_tokens)} generated={len(generated_tokens)} "
        f"seconds={first_turn_seconds:.3f}"
    )

    if args.skip_final_answer:
        engine.remove_request(request)
        return

    continuation_fragment = (
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
        f"{json.dumps(result)}\n"
        "</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    continuation_tokens = list(tokenizer.encode(continuation_fragment))

    started = time.perf_counter()
    request.continue_with(np.asarray(continuation_tokens, dtype=np.int32))
    final_output, final_tokens = generate(engine, request, tokenizer)
    final_seconds = time.perf_counter() - started
    engine.remove_request(request)
    if not final_output or TOOL_CALL_START in final_output:
        raise RuntimeError(f"Expected a final answer after {name!r}, received: {final_output!r}")
    print(
        f"LIVE RESULT PASS {name}: mode=reuse "
        f"appended={len(continuation_tokens)} generated={len(final_tokens)} seconds={final_seconds:.3f}"
    )


def run(args):
    messages, tools = load_payload(args.payload)
    functions = [tool_function(tool) for tool in tools]
    tool_by_name = {function["name"]: function for function in functions}
    if len(tool_by_name) != len(functions):
        raise RuntimeError("Tool names must be unique")

    synthetic_calls = [synthetic_call(function) for function in functions]
    for tool_call in synthetic_calls:
        mock_execute(tool_by_name, tool_call)
    missing_root_types = sum(not function.get("parameters", {}).get("type") for function in functions)
    print(
        f"SCHEMA/MOCK PASS: tools={len(functions)} unique={len(tool_by_name)} "
        f"implicit_or_empty_root_types={missing_root_types}"
    )

    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    prompt = render_prompt(tokenizer, messages, tools)
    prompt_tokens = list(tokenizer.encode(prompt))
    context_length = model_context_length(args.model_path)
    print(
        f"LONG CONTEXT PASS: messages={len(messages)} prompt_chars={len(prompt)} "
        f"prompt_tokens={len(prompt_tokens)} context={context_length} headroom={context_length - len(prompt_tokens)}"
    )
    if len(prompt_tokens) >= context_length:
        raise RuntimeError("The rendered payload does not fit in the model context")

    live_tools = select_live_tools(args.live_tools, tool_by_name)
    for index, name in enumerate(live_tools, 1):
        print(f"LIVE TEST {index}/{len(live_tools)}: {name}")
        run_live_tool(args, model, tokenizer, engine, messages, tools, tool_by_name, name)
    if not live_tools:
        print("No live generation requested; use --live_tools name[,name] or --live_tools all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and exercise a large OpenAI-style tool payload with the GenAI Engine"
    )
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--payload", required=True, help="JSON request containing messages and tools or options.tools")
    parser.add_argument(
        "--live_tools",
        default="",
        help="Comma-separated tool names to generate and mock-execute, or 'all'; empty performs validation only",
    )
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument(
        "--skip_final_answer",
        action="store_true",
        help="Stop after each validated model-generated tool call instead of appending its mock result",
    )
    run(parser.parse_args())