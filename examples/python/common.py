# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import onnxruntime_genai as og

from dataclasses import dataclass, asdict
from typing import Any

def set_logger(inputs: bool = True, outputs: bool = True) -> None:
    """
    Set log options inside ORT GenAI

    Args:
        inputs (bool): Dump inputs to the model in the console
        outputs (bool): Dump outputs to the model in the console
    Returns:
        None
    """
    og.set_log_options(enabled=True, model_input_values=inputs, model_output_values=outputs)

def register_ep(ep: str, ep_path: str, use_winml: bool) -> None:
    """
    Register execution provider if path is provided or via Windows ML

    Args:
        ep (str): Name of execution provider
        ep_path (str): Path to execution provider to register
        use_winml (bool): Use Windows ML to register execution providers
    Returns:
        None
    """
    if not ep_path:
        return  # No library path specified, skip registration

    print(f"Registering execution provider: {ep}")

    if use_winml:
        # Requies winml.py file
        # Modified from here: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/tutorial?tabs=python#acquiring-the-model-and-preprocessing
        try:
            import winml
            print(winml.register_execution_providers(ort=False, ort_genai=True))
        except ImportError:
            print("WinML not available, using default execution providers")
        except Exception as e:
            print(f"Failed to register WinML execution providers: {e}")
    elif ep == "cuda":
        og.register_execution_provider_library("CUDAExecutionProvider", ep_path)
    elif ep == "NvTensorRtRtx":
        og.register_execution_provider_library("NvTensorRTRTXExecutionProvider", ep_path)
    else:
        print(f"Warning: EP registration not supported for {ep}")
        print("Only 'cuda' and 'NvTensorRtRtx' support plug-in libraries. Use Windows ML via '--use_winml' to register EPs.")
        return

    print(f"Registered {ep} successfully!")

def get_config(path: str, ep: str, ep_options: dict[str, str] = {}, search_options: dict[str, int] = {}) -> og.Config:
    """
    Get og.Config object and set EP-specific and search-specific options inside it

    Args:
        path (str): Path to model folder containing GenAI config
        ep (str): Name of execution provider to set
        ep_options (dict[str, str]): Map of EP-specific option names and their values
        search_options (dict[str, int]): Map of search-specific option names and their values
    Returns:
        og.Config: ORT GenAI config object with all options set
    """
    # Create config with EP
    # - If follow_config, then use the default EP stored inside the GenAI config.
    # - Otherwise, override the stored EP by clearing all providers and appending the desired one.
    config = og.Config(path)
    if ep != "follow_config":
        config.clear_providers()
        if ep != "cpu":
            print(f"Setting model to {ep}")
            config.append_provider(ep)

        # Set any EP-specific options
        for k, v in ep_options.items():
            if k == "enable_cuda_graph" and ep in {"cuda", "NvTensorRtRtx"} and search_options.get("num_beams", 1) > 1:
                # Disable CUDA graph if using beam search (num_beams > 1),
                # num_beams > 1 requires past_present_share_buffer to be false so enable_cuda_graph must be false
                config.set_provider_option(ep, "enable_cuda_graph", "0")
            else:
                config.set_provider_option(ep, k, v)

    if "chunk_size" in search_options and search_options["chunk_size"] == 0:
        # Remove chunk_size of 0
        del search_options["chunk_size"]

    # Set any search-specific options that need to be known before constructing an og.Model object
    # Otherwise they can be set with params.set_search_options(**search_options)
    config.overlay(json.dumps({"search": search_options}))
    return config

def get_search_options(args: argparse.Namespace):
    """
    Get search options for a generator's params during decoding

    Args:
        args (argparse.Namespace): arguments provided by user
    Returns:
        dict[str, Any]: dictionary of key-value pairs to set
    """
    search_options = {}
    names = [
        "batch_size",
        "do_sample",
        "max_length",
        "min_length",
        "num_beams",
        "num_return_sequences",
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
    ]
    for name in names:
        if name in args:
            search_options[name] = getattr(args, name)

    # In case the user doesn't provide the batch size, set it to 1
    search_options["batch_size"] = search_options.get("batch_size", 1)
    return search_options

def apply_chat_template(model_path: str, tokenizer: og.Tokenizer, messages: str, add_generation_prompt: bool, tools: str = "") -> str:
    """
    Apply the chat template with various fallback options

    Args:
        model_path (str): path to folder containing model
        tokenizer (og.Tokenizer): tokenizer object to use
        add_generation_prompt (bool): add tokens to indicate the start of the AI's response
        tools (str): string-encoded list of tools
    Returns:
        str: prompt to encode
    """
    template_str = ""
    jinja_path = os.path.join(model_path, "chat_template.jinja")
    if os.path.exists(jinja_path):
        with open(jinja_path, encoding="utf-8") as f:
            template_str = f.read()

    prompt = tokenizer.apply_chat_template(
        messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, template_str=template_str
    )
    return prompt

def get_user_prompt(prompt: str, non_interactive: bool) -> str:
    """
    Get prompt for 'user' role in chat template

    Args:
        prompt (str): provided prompt
        non_interactive (bool): non-interactive mode (uses either provided prompt or default)
    Returns:
        str: prompt to encode
    """
    text = None

    while True:
        if not non_interactive:
            # If interactive mode is on
            text = input("Prompt (Use quit() to exit): ")
        else:
            # Use provided prompt (whether default or user-provided)
            text = prompt

        if not text:
            print("Error, input cannot be empty")
            continue
        else:
            break

    return text

def get_user_media_paths(media_paths: list[str], non_interactive: bool, media_type: str) -> list[str]:
    """
    Get paths to media for 'user' role in chat template

    Args:
        media_paths (list[str]): user-provided media paths
        non_interactive (bool): non-interactive mode (uses either user-provided media paths or default)
        media_type (str): the media type being obtained
    Returns:
        list[str]: all media filepaths to read and encode
    """
    # Check media type
    media_type = media_type.lower()
    assert media_type in {"audio", "image"}, "Media type must be 'image' or 'audio'"

    paths = []
    if media_paths:
        # If user-provided media paths
        paths = media_paths
    elif not non_interactive:
        # If interactive mode is on
        paths = [
            path.strip()
            for path in input(f"{media_type.capitalize()} Path (comma separated; leave empty if no {media_type}): ").split(",")
        ]

    paths = [path for path in paths if path]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{media_type.capitalize()} file not found: {path}")
        print(f"Using {media_type}: {path}")

    return paths

def get_user_images(image_paths: list[str], non_interactive: bool) -> tuple[og.Images, int]:
    """
    Get images for 'user' role in chat template

    Args:
        image_paths (list[str]): user-provided image paths
        non_interactive (bool): non-interactive mode (uses either user-provided image paths or default)
    Returns:
        (og.Images, int): (all images, number of images) as a tuple
    """
    media_type = "image"
    paths = get_user_media_paths(image_paths, non_interactive, media_type)
    if not paths:
        print(f"No {media_type} provided")
        return None, 0

    images = og.Images.open(*paths)
    return images, len(paths)

def get_user_audios(audio_paths: list[str], non_interactive: bool) -> tuple[og.Audios, int]:
    """
    Get audios for 'user' role in chat template

    Args:
        audio_paths (list[str]): user-provided audio paths
        non_interactive (bool): non-interactive mode (uses either user-provided audio paths or default)
    Returns:
        (og.Audios, int): (all audios, number of audios) as a tuple
    """
    media_type = "audio"
    paths = get_user_media_paths(audio_paths, non_interactive, media_type)
    if not paths:
        print(f"No {media_type} provided")
        return None, 0

    audios = og.Audios.open(*paths)
    return audios, len(paths)

def get_user_content(model_type, num_images, num_audios, prompt) -> str | list[dict[str, str]]:
    """
    Get content for 'user' role in chat template

    Args:
        model_type (str): model type inside ORT GenAI
        num_images (int): number of images
        num_audios (int): number of audios
        prompt (str): user prompt
    Returns:
        str | list[dict[str, str]]: Combined content for 'user' role
    """
    content = None
    # Combine all image tags, audio tags, and text into one user content
    if model_type == "phi3v":
        # Phi-3 vision, Phi-3.5 vision
        image_tags = "".join([f"<|image_{i + 1}|>\n" for i in range(num_images)])
        content = image_tags + prompt
    elif model_type == "phi4mm":
        # Phi-4 multimodal
        image_tags = "".join([f"<|image_{i + 1}|>\n" for i in range(num_images)])
        audio_tags = "".join([f"<|audio_{i + 1}|>\n" for i in range(num_audios)])
        content = image_tags + audio_tags + prompt
    elif model_type in {"qwen2_5_vl", "fara"}:
        # Qwen-2.5 VL, Fara
        image_tags = "".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in range(num_images)])
        content = image_tags + prompt
    else:
        # Gemma-3 style: structured content
        image_tags = [{"type": "image"} for _ in range(num_images)]
        content = image_tags + [{"type": "text", "text": prompt}]
    return content

@dataclass
class ToolSchema:
    """
    A class for defining a tool in a JSON schema compatible way
    """
    description: str
    type: str
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool

@dataclass
class JsonSchema:
    """
    A class for defining a JSON schema for guidance
    """
    x_guidance: dict[str, Any]
    type: str
    items: dict[str, list[ToolSchema]]
    minItems: int

@dataclass
class FunctionDefinition:
    """
    A class for defining a function in an OpenAI-compatible way
    """
    name: str
    description: str
    parameters: dict[str, Any]

@dataclass
class Tool:
    """
    A class for defining a tool in an OpenAI-compatible way
    """
    type: str
    function: FunctionDefinition

def tools_to_schemas(tools: list[Tool]) -> list[ToolSchema]:
    """
    Convert a list of tools to a list of tool schemas

    Args:
        tools (list[Tool]): list of OpenAI-compatible tools
    Returns:
        list[ToolSchema]: list of JSON schema compatible tools
    """
    tool_schemas = []
    for tool in tools:
        properties = {"name": {"const": tool.function.name}}
        tool_parameters_exist = tool.function.parameters != {}

        if tool_parameters_exist:
            parameters = {
                "type": tool.function.parameters.get("type", "object"),
                "properties": tool.function.parameters.get("properties", {}),
                "required": tool.function.parameters.get("required", []),
            }
            properties["parameters"] = parameters

        tool_schema = ToolSchema(
            description=tool.function.description,
            type="object",
            properties=properties,
            required=["name", "parameters"] if tool_parameters_exist else ["name"],
            additionalProperties=False,
        )
        tool_schemas.append(tool_schema)
    return tool_schemas

def get_json_schema(tools: list[Tool], tool_output: bool) -> str:
    """
    Create a JSON schema from a list of tools

    Args:
        tools (list[Tool]): list of OpenAI-compatible tools
        tool_output: output can have a tool call
    Returns:
        str: JSON schema as a JSON-compatible string
    """
    schemas = tools_to_schemas(tools)
    x_guidance = {"whitespace_flexible": False, "key_separator": ": ", "item_separator": ", "}
    json_schema = JsonSchema(x_guidance=x_guidance, type="array", items={"anyOf": schemas}, minItems=int(tool_output))
    d = {k.replace("x_guidance", "x-guidance"): v for k, v in asdict(json_schema).items()}
    return json.dumps(d)

def get_lark_grammar(
    tools: list[Tool],
    text_output: bool,
    tool_output: bool,
    tool_call_start: str,
    tool_call_end: str,
) -> str:
    """
    Create a LARK grammar from a list of tools

    Args:
        tools (list[Tool]): list of OpenAI-compatible tools
        text_output (bool): output can have text
        tool_output (bool): output can have a tool call
        tool_call_start (str): string representation of tool call starting token (e.g. <tool_call>)
        tool_call_end (str): string representation of tool call ending token (e.g. </tool_call>)
    Returns:
        str: LARK grammar as a string
    """
    known_tool_call_ids = tool_call_start != "" and tool_call_end != ""

    rows = []
    if text_output and not tool_output:
        start_row = "start: TEXT"
    elif not text_output and tool_output:
        start_row = f"start: {'toolcall' if known_tool_call_ids else 'functioncall'}"
    elif text_output and tool_output:
        start_row = f"start: TEXT | {'toolcall' if known_tool_call_ids else 'functioncall'}"
    else:
        raise Exception("At least one of 'text_output' and 'tool_output' must be true")
    rows.append(start_row)

    if text_output:
        text_row = "TEXT: /[^{<](.|\\n)*/"
        rows.append(text_row)

    if tool_output:
        schema = get_json_schema(tools=tools, tool_output=tool_output)
        if known_tool_call_ids:
            tool_row = f"toolcall: {tool_call_start} functioncall {tool_call_end}"
            rows.append(tool_row)

        func_row = f"functioncall: %json {schema}"
        rows.append(func_row)

    return "\n".join(rows)

def to_tool(tool_defs: list[dict[str, Any]]) -> list[Tool]:
    """
    Convert a JSON-deserialized object of tools to a list of Tool objects

    Args:
        tool_defs (list[dict[str, Any]]): JSON-deserialized object containing OpenAI-compatible tool definitions
    Returns:
        list[Tool]: list of Tool objects
    """
    tools = []
    for tool_def in tool_defs:
        func = FunctionDefinition(
            name=tool_def["function"]["name"],
            description=tool_def["function"]["description"],
            parameters=tool_def["function"]["parameters"],
        )
        tool = Tool(type="function", function=func)
        tools.append(tool)
    return tools

def get_guidance(
    response_format: str = "",
    filepath: str = "",
    tools_str: str = "",
    tools: list[dict[str, Any] | Tool] = [],
    text_output: bool = True,
    tool_output: bool = False,
    tool_call_start: str = "",
    tool_call_end: str = "",
) -> tuple[str, str, str]:
    """
    Create a grammar to use with LLGuidance

    Args:
        response_format (str): type of format requested
        filepath (str): path to file containing OpenAI-compatible tool definitions
        tools_str (str): JSON-serialized string containing OpenAI-compatible tool definitions
        tools (list[dict[str, Any] | Tool]): list of OpenAI-compatible tools defined in memory
        text_output (bool): output can have text
        tool_output (bool): output can have a tool call
        tool_call_start (str): string representation of tool call starting token (e.g. <tool_call>)
        tool_call_end (str): string representation of tool call ending token (e.g. </tool_call>)
    Returns:
        (str, str, str): (grammar type, grammar data, tools) as a tuple of strings
    """
    guidance_type, guidance_data = "", ""

    # Get list of tools from a range of sources (filepath, JSON-serialized string, in-memory)
    if tool_output:
        if os.path.exists(filepath):
            # If tools are provided as a file
            with open(filepath, 'r') as f:
                tool_defs = json.load(f)
                tools = to_tool(tool_defs)
        elif tools_str != "":
            # If tools are provided as a JSON-serialized string
            try:
                tool_defs = json.loads(tools_str)
                tools = to_tool(tool_defs)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for tools list. Format must be a JSON-serialized string.")
        elif len(tools) > 0:
            if type(tools[0]) != Tool:
                tools = to_tool(tools)
        else:
            raise ValueError("Please provide the list of tools through a file, JSON-serialized string, or a list of tools")

        assert len(tools) > 0, "Could not obtain a list of tools in memory"

    # Create guidance based on user-provided response format
    if response_format in {"text", "lark_grammar"}:
        if response_format == "text":
            assert text_output and not tool_output, "A response format of 'text' requires text_output = True and tool_output = False"

        guidance_type = "lark_grammar"
        guidance_data = get_lark_grammar(
            tools=tools,
            text_output=text_output,
            tool_output=tool_output,
            tool_call_start=tool_call_start,
            tool_call_end=tool_call_end,
        )
    elif response_format in {"json_schema", "json_object"}:
        assert tool_output and not text_output, "A response format of 'json_schema' or 'json_object' requires text_output = False and tool_output = True"

        guidance_type = "json_schema"
        guidance_data = get_json_schema(tools=tools, tool_output=tool_output)
    else:
        raise ValueError("Invalid response format provided")

    return guidance_type, guidance_data, json.dumps([asdict(tool) for tool in tools])

def get_generator_params_args(parser: argparse.ArgumentParser) -> None:
    """
    Add an argument group for the generator params

    Args:
        parser (argparse.ArgumentParser): original parser object with existing arguments
    Returns:
        None
    """
    generator_params = parser.add_argument_group("Generator Params")
    generator_params.add_argument('-c', '--chunk_size', type=int, default=0, help="Chunk size for prefill chunking during context processing (default: 0 = disabled, >0 = enabled)")
    generator_params.add_argument('-s', '--do_sample', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    generator_params.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    generator_params.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    generator_params.add_argument('-b', '--num_beams', type=int, default=1, help='Number of beams to create')
    generator_params.add_argument('-rs', '--num_return_sequences', type=int, default=1, help='Number of return sequences to produce')
    generator_params.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    generator_params.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    generator_params.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    generator_params.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')

def get_guidance_args(parser: argparse.ArgumentParser) -> None:
    """
    Add an argument group for guidance options

    Args:
        parser (argparse.ArgumentParser): original parser object with existing arguments
    Returns:
        None
    """
    guidance = parser.add_argument_group("Guidance Arguments")
    guidance.add_argument('-rf', '--response_format', type=str, default="", choices=["", "text", "json_object", "json_schema", "lark_grammar"], help='Provide response format for the model')
    guidance.add_argument('-tf', '--tools_file', type=str, default="", help='Path to file containing list of OpenAI-compatible tool definitions. Ex: test/test_models/tool-definitions/weather.json')
    guidance.add_argument('-text', '--text_output', action='store_true', default=False, help='Produce a text response in the output')
    guidance.add_argument('-tool', '--tool_output', action='store_true', default=False, help='Produce a tool call in the output')
    guidance.add_argument('-tcs', '--tool_call_start', type=str, default="", help='String representation of tool call start (ex: <|tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work.')
    guidance.add_argument('-tce', '--tool_call_end', type=str, default="", help='String representation of tool call end (ex: <|/tool_call|>). Needs to be marked as special in tokenizer.json for guidance to work.')
