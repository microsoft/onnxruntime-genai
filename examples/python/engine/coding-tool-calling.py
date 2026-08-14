# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import difflib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

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
    return "".join(fragments).strip(), token_ids


def parse_tool_call(output):
    start = output.find(TOOL_CALL_START)
    end = output.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
    payload = output
    if start >= 0 and end >= 0:
        payload = output[start + len(TOOL_CALL_START) : end]
    try:
        tool_call = json.loads(payload.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(tool_call, list):
        if len(tool_call) != 1:
            raise RuntimeError(f"Expected one tool call, received {len(tool_call)}")
        tool_call = tool_call[0]
    if not isinstance(tool_call, dict) or "name" not in tool_call:
        return None
    return tool_call


def workspace_path(workspace, relative_path):
    requested_path = Path(relative_path)
    if requested_path.is_absolute() or requested_path.root:
        requested_path = Path(requested_path.name)
    path = (workspace / requested_path).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise RuntimeError(f"Path escapes the coding workspace: {relative_path!r}") from error
    return path


def execute_tool(workspace, tool_call):
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", tool_call.get("parameters", {}))
    if not isinstance(arguments, dict):
        raise RuntimeError(f"Tool arguments must be an object: {arguments!r}")

    if name == "read_file":
        path = workspace_path(workspace, arguments["path"])
        return {"path": path.relative_to(workspace).as_posix(), "content": path.read_text(encoding="utf-8")}

    if name == "edit_file":
        path = workspace_path(workspace, arguments["path"])
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        content = path.read_text(encoding="utf-8")
        if content.count(old_text) != 1:
            raise RuntimeError("old_text must occur exactly once")
        updated = content.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=arguments["path"],
                tofile=arguments["path"],
            )
        )
        return {"path": path.relative_to(workspace).as_posix(), "diff": diff}

    if name == "run_tests":
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "output": (completed.stdout + completed.stderr)[-4000:],
        }

    raise RuntimeError(f"Unknown tool: {name!r}")


def create_demo_workspace(workspace):
    (workspace / "calculator.py").write_text(
        "def multiply(left, right):\n"
        "    return left + right\n",
        encoding="utf-8",
    )
    (workspace / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import multiply\n\n\n"
        "class CalculatorTests(unittest.TestCase):\n"
        "    def test_multiply(self):\n"
        "        self.assertEqual(multiply(6, 7), 42)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )


def uses_dynamic_batching(model_path):
    with open(Path(model_path) / "genai_config.json", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("engine", {}).get("dynamic_batching") is not None


def new_request(engine, params, tokens):
    request = og.Request(params)
    request.add_tokens(np.asarray(tokens, dtype=np.int32))
    engine.add_request(request)
    return request


def next_action(file_read, file_edited):
    if not file_read:
        return "Call read_file with path calculator.py now."
    if not file_edited:
        return (
            "Call edit_file with path calculator.py, old_text exactly 'return left + right', "
            "and new_text exactly 'return left * right' now."
        )
    return "Call run_tests with no arguments now."


def run_agent(args, workspace):
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
        {
            "role": "system",
            "content": (
                "You are a coding agent. Use tools to inspect the workspace, make the smallest "
                "correct edit, and run the tests. Tool paths must be relative to the workspace; "
                "use calculator.py exactly, never /path/to/calculator.py. Do not claim success "
                "before run_tests returns exit_code 0. If a tool fails, correct the arguments and retry."
            ),
        },
        {"role": "user", "content": args.user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages), tools=json.dumps(tools), add_generation_prompt=True
    )
    request_tokens = list(tokenizer.encode(prompt))
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=args.max_length)
    request = new_request(engine, params, request_tokens)
    dynamic_batching = uses_dynamic_batching(args.model_path)
    file_read = False
    file_edited = False
    tests_passed = False
    tool_rounds = 0

    for step in range(1, args.max_tool_rounds * 2 + 2):
        output, generated_tokens = generate(engine, request, tokenizer)
        tool_call = parse_tool_call(output)
        if tool_call is None:
            if tests_passed:
                engine.remove_request(request)
                print(f"Final answer: {output}")
                return
            checkpoint_output = output[:500]
            if len(output) > len(checkpoint_output):
                checkpoint_output += " [truncated]"
            print(f"Agent output {step}: {checkpoint_output!r}")
            messages.extend(
                [
                    {"role": "assistant", "content": checkpoint_output},
                    {
                        "role": "user",
                        "content": (
                            "The task is not complete because run_tests has not passed. The exact file path "
                            "is calculator.py. Do not ask for clarification. Your entire next response must "
                            f"be a tool call. {next_action(file_read, file_edited)}"
                        ),
                    },
                ]
            )
            print(f"Agent checkpoint {step}: task incomplete; requesting another tool call")
        else:
            tool_rounds += 1
            print(f"Tool call {step}: {json.dumps(tool_call)}")
            try:
                tool_result = execute_tool(workspace, tool_call)
            except Exception as error:
                tool_result = {
                    "error": str(error),
                    "recovery": next_action(file_read, file_edited),
                }
            print(f"Tool result {step}: {json.dumps(tool_result)}")
            if "error" not in tool_result:
                file_read = file_read or tool_call.get("name") == "read_file"
                file_edited = file_edited or tool_call.get("name") == "edit_file"
            tests_passed = tool_call.get("name") == "run_tests" and tool_result.get("exit_code") == 0

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
            messages=json.dumps(messages), tools=json.dumps(tools), add_generation_prompt=True
        )
        continued_tokens = list(tokenizer.encode(continued_prompt))

        if tool_rounds >= args.max_tool_rounds and not tests_passed:
            engine.remove_request(request)
            raise RuntimeError(
                "The model exhausted the tool-call retry budget before completing the coding task. "
                "The Engine tool loop worked, but this model did not produce valid coding-tool arguments; "
                "try a larger tool-trained coding model."
            )
        if dynamic_batching:
            cached_tokens = request_tokens + generated_tokens
            if continued_tokens[: len(cached_tokens)] != cached_tokens:
                raise RuntimeError("The continued chat template does not preserve the Engine request token prefix")
            request.add_tokens(np.asarray(continued_tokens[len(cached_tokens) :], dtype=np.int32))
        else:
            engine.remove_request(request)
            request = new_request(engine, params, continued_tokens)
        request_tokens = continued_tokens

    raise RuntimeError("The coding agent stopped without a final answer")


def run(args):
    if args.workspace:
        workspace = Path(args.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        run_agent(args, workspace)
        return

    with tempfile.TemporaryDirectory(prefix="ortgenai-coding-") as directory:
        workspace = Path(directory).resolve()
        create_demo_workspace(workspace)
        print(f"Demo workspace: {workspace}")
        run_agent(args, workspace)
        print("Final calculator.py:")
        print((workspace / "calculator.py").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandboxed coding tool calls with the GenAI Engine")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--tools_file", default="test/tool-definitions/coding.json")
    parser.add_argument("--workspace", help="Optional existing workspace; tools cannot access paths outside it")
    parser.add_argument(
        "--user_prompt",
        default=(
            "The workspace contains calculator.py and test_calculator.py. Fix the bug in calculator.py "
            "so multiplication is correct. Begin by calling read_file with exactly "
            '{"path": "calculator.py"}. Then edit the exact text you read and run the tests.'
        ),
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_tool_rounds", type=int, default=6)
    run(parser.parse_args())