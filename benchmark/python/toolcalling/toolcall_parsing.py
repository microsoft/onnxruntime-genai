# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Parsing and scoring for tool-calling benchmark output.

Models express tool calls in different surface syntaxes, so the parser tries each
known form and keeps whichever one yields calls:

  xml_function   <tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>
  json_call      <tool_call>{"name": NAME, "arguments": {...}}</tool_call>
  tagged_json    <|tool_call|>[{"name": NAME, "arguments": {...}}]
  bare_json      {"name": NAME, "arguments": {...}} with no surrounding tags
"""

import json
import re

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
TAGGED_JSON_BLOCK = re.compile(r"<\|tool_call\|>\s*(\[.*?\]|\{.*?\})", re.DOTALL)
XML_FUNCTION = re.compile(r"<function=([^>]*)>\s*(.*?)\s*(?:</function>|$)", re.DOTALL)
XML_PARAMETER = re.compile(r"<parameter=([^>]*)>\n?(.*?)\n?</parameter>", re.DOTALL)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# A turn ends at the template's end-of-turn token. When that token is not a stop id
# the model keeps going and writes the next turns itself, which decodes as a bare
# role line; with tool calls it then invents the tool's result.
TURN_BREAKS = ("<|im_end|>", "<|im_start|>", "<|end|>", "<|eot_id|>", "\nuser\n", "\nassistant\n", "\ntool\n")

ARGUMENT_KEYS = ("arguments", "parameters", "args")


def turn_text(text):
    """The model's own turn: everything before it starts fabricating the next one."""
    cut = len(text)
    for marker in TURN_BREAKS:
        found = text.find(marker)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def stopped_cleanly(text):
    """True when the model ended its turn instead of writing further turns."""
    return turn_text(text) == text


def strip_reasoning(text):
    """Drop complete reasoning blocks, keeping anything emitted outside them."""
    return THINK_BLOCK.sub("", text)


def _normalize_call(payload):
    """Coerce one decoded JSON tool call into {name, arguments}."""
    if not isinstance(payload, dict):
        return None
    if "function" in payload and isinstance(payload["function"], dict):
        payload = payload["function"]
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    arguments = {}
    for key in ARGUMENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            arguments = value
            break
    return {"name": name.strip(), "arguments": arguments}


def _parse_json_calls(blob):
    try:
        payload = json.loads(blob)
    except ValueError:
        return []
    entries = payload if isinstance(payload, list) else [payload]
    return [call for call in map(_normalize_call, entries) if call]


def _parse_xml_calls(blob):
    calls = []
    for name, body in XML_FUNCTION.findall(blob):
        arguments = {param.strip(): value.strip() for param, value in XML_PARAMETER.findall(body)}
        calls.append({"name": name.strip(), "arguments": arguments})
    return calls


def parse_tool_calls(text):
    """Return (calls, format_name). `format_name` is None when nothing parsed."""
    for blob in TOOL_CALL_BLOCK.findall(text):
        calls = _parse_xml_calls(blob)
        if calls:
            return calls, "xml_function"
        calls = _parse_json_calls(blob)
        if calls:
            return calls, "json_call"

    for blob in TAGGED_JSON_BLOCK.findall(text):
        calls = _parse_json_calls(blob)
        if calls:
            return calls, "tagged_json"

    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        calls = _parse_json_calls(stripped)
        if calls:
            return calls, "bare_json"

    return [], None


def _coerce(value):
    """Best-effort typing so '2', 'true' and '["a"]' compare sensibly."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return json.loads(text)
    except ValueError:
        return text.strip("'\"")


def values_match(expected, actual):
    actual = _coerce(actual)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-6
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(values_match(e, a) for e, a in zip(expected, actual, strict=False))
    return str(expected).strip().lower() == str(actual).strip().lower()


def tool_schemas(tools):
    """Map function name -> allowed/required parameters and enum constraints."""
    schemas = {}
    for tool in tools:
        function = tool["function"] if tool.get("type") == "function" else tool
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        schemas[function["name"]] = {
            "allowed": set(properties),
            "required": set(parameters.get("required") or []),
            "enums": {name: spec["enum"] for name, spec in properties.items() if "enum" in spec},
        }
    return schemas


def score_case(case, text):
    """Grade one generation against the case's pinned expectation.

    `enum_valid` and `no_unknown_params` detect a prompt that advertised a degraded
    schema; `clean_stop` detects an end-of-turn token missing from the stop ids.
    """
    own_turn = turn_text(text)
    calls, call_format = parse_tool_calls(strip_reasoning(own_turn))
    expected = case.get("expected_function")

    result = {
        "name": case["name"],
        "expected_function": expected,
        "call_format": call_format,
        "num_calls": len(calls),
        "called_function": calls[0]["name"] if calls else None,
        "arguments": calls[0]["arguments"] if calls else {},
        "clean_stop": stopped_cleanly(text),
    }

    if expected is None:
        # Success means answering directly instead of inventing a call.
        abstained = not calls
        result.update(
            correct_function=abstained,
            required_present=abstained,
            enum_valid=abstained,
            no_unknown_params=abstained,
            args_exact=abstained,
            correct=abstained,
        )
        return result

    if not calls:
        result.update(
            correct_function=False,
            required_present=False,
            enum_valid=False,
            no_unknown_params=False,
            args_exact=False,
            correct=False,
        )
        return result

    call = calls[0]
    schema = tool_schemas(case["tools"]).get(call["name"])
    result["correct_function"] = call["name"] == expected

    if schema is None:
        # The model invented a function that was never offered.
        result.update(
            required_present=False, enum_valid=False, no_unknown_params=False, args_exact=False, correct=False
        )
        return result

    arguments = call["arguments"]
    result["required_present"] = schema["required"].issubset(arguments)
    result["no_unknown_params"] = set(arguments).issubset(schema["allowed"])
    result["enum_valid"] = all(
        any(values_match(option, arguments[param]) for option in options)
        for param, options in schema["enums"].items()
        if param in arguments
    )
    result["args_exact"] = all(
        param in arguments and values_match(value, arguments[param])
        for param, value in (case.get("expected_arguments") or {}).items()
    )
    result["correct"] = bool(
        result["correct_function"]
        and result["required_present"]
        and result["enum_valid"]
        and result["no_unknown_params"]
        and result["args_exact"]
    )
    return result


def _normalize_for_match(text):
    """Fold away formatting the model is free to choose: case, spacing, markdown.

    A model may render the identifier UA482 as "**UA 482**" and still be quoting the
    tool result exactly, so matching on the raw string would report a false failure.
    """
    return "".join(c for c in text.lower() if c not in " \t\n*_`,")


def score_followup(case, text):
    """Grade the answer produced after the tool result is fed back.

    The expected substrings are values the caller can only know from the tool
    result, so `answer_uses_result` distinguishes a grounded answer from one the
    model invented or recited from memory.
    """
    own_turn = turn_text(text)
    calls, _ = parse_tool_calls(strip_reasoning(own_turn))
    answer = strip_reasoning(own_turn).strip()
    normalized = _normalize_for_match(answer)
    expected = case.get("expected_answer_contains") or []

    missing = [needle for needle in expected if _normalize_for_match(needle) not in normalized]
    return {
        "answer_uses_result": not missing,
        "missing_from_answer": missing,
        "no_repeat_call": not calls,
        "final_clean_stop": stopped_cleanly(text),
        "answer": answer,
    }
