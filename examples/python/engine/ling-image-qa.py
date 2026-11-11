# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Load a packed Ling vision model, then stream responses to image-and-prompt inputs."""

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime_genai as og


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def run_turn(args, tokenizer, engine, template, image, question):
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "\n" + question}],
                }
            ]
        ),
        template_str=template,
        add_generation_prompt=True,
    )
    limits = og.RequestOptions()
    limits.set_max_session_tokens(args.max_session_tokens)
    request = engine.create_request(options=limits)
    try:
        options = og.TurnOptions(request)
        options.set_do_sample(False)
        options.set_max_generated_tokens(args.max_new_tokens)
        opened = time.perf_counter()
        images = og.Images.open(str(image))
        print(f"Image open: {time.perf_counter() - opened:.2f}s", file=sys.stderr, flush=True)
        buffer = engine.create_event_buffer(1)
        stream = tokenizer.create_stream()
        generated = 0
        first_token = last_token = None
        finished = False
        print("\nResponse:", flush=True)
        started = time.perf_counter()
        request.begin_turn(prompt, options, images=images)
        del images
        while engine.has_pending_requests():
            for event in engine.run(buffer):
                if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                    raise RuntimeError(f"Image turn failed or exceeded capacity: {event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    last_token = time.perf_counter()
                    if first_token is None:
                        first_token = last_token
                    generated += 1
                    print(stream.decode(event.token), end="", flush=True)
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finished = True
                    finish_reason = str(event.finish_reason)
                    prompt_tokens = event.usage.prompt_tokens
        elapsed = time.perf_counter() - started
        print(flush=True)
        if not finished or not generated:
            raise RuntimeError("Image turn did not produce a completed, nonempty response")
        decode = last_token - first_token
        rate = (generated - 1) / decode if decode > 0 else 0.0
        print(
            f"\nPrompt: {prompt_tokens} tokens | Generated: {generated} tokens | "
            f"TTFT: {first_token - started:.2f}s | "
            f"Decode: {decode:.2f}s ({rate:.1f} tokens/s after first token) | "
            f"Total: {elapsed:.2f}s | {finish_reason}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Model Builder packed Ling vision package")
    parser.add_argument("--image", type=Path, help="Local PNG/JPEG; omit with --prompt for interactive input")
    parser.add_argument("--prompt", help="Question or instructions; omit with --image for interactive input")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--max-session-tokens", type=positive_int, default=4096, help="Includes expanded image tokens")
    args = parser.parse_args()
    if (args.image is None) != (args.prompt is None):
        parser.error("provide both --image and --prompt, or omit both for interactive input")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    manifest = json.loads((args.model / "genai_config.json").read_text(encoding="utf-8"))
    if manifest["model"]["type"] != "bailing_moe_v3_vl":
        parser.error("--model must be a packed Ling-3.0-flash-VL vision package")
    if args.max_session_tokens > manifest["model"]["context_length"]:
        parser.error("--max-session-tokens exceeds the model context")
    if args.max_new_tokens >= args.max_session_tokens:
        parser.error("--max-new-tokens must be smaller than --max-session-tokens")

    config = og.Config(str(args.model))
    block_size = manifest["engine"]["dynamic_batching"].get("block_size", 256)
    config.overlay(
        json.dumps(
            {
                "model": {"decoder": {"session_options": {"session.use_device_allocator_for_initializers": "1"}}},
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": (args.max_session_tokens + block_size - 1) // block_size,
                        "max_batch_size": 1,
                        "max_scheduled_tokens": 17,
                    }
                },
                "search": {"chunk_size": 17},
            }
        )
    )
    config.set_provider_option("cuda", "enable_cuda_graph", "0")
    config.set_provider_option("cuda", "arena_extend_strategy", "kSameAsRequested")
    print("Loading Ling vision model...", file=sys.stderr, flush=True)
    loading = time.perf_counter()
    model = og.Model(config)
    tokenizer, engine = og.Tokenizer(model), og.Engine(model)
    template = (args.model / "chat_template.jinja").read_text(encoding="utf-8")
    # Use Ling's supported non-thinking template mode without changing the model assets.
    template = "{% set enable_thinking = false %}" + template
    print(f"Model ready. Load time: {time.perf_counter() - loading:.2f}s", file=sys.stderr, flush=True)
    print(
        "TTFT starts at begin_turn and includes preprocessing, encoding and prefill; "
        "model load, image open and user input are excluded.",
        file=sys.stderr,
        flush=True,
    )
    if args.image is not None:
        run_turn(args, tokenizer, engine, template, args.image, args.prompt)
        return
    print("Enter an image path and then a prompt. Blank image path or 'quit' exits.", flush=True)
    while True:
        path = input("\nImage path: ").strip()
        if not path or path.lower() in ("quit", "exit"):
            return
        image = Path(path).expanduser()
        if not image.is_file():
            print(f"Image does not exist: {image}", file=sys.stderr, flush=True)
            continue
        question = ""
        while not question:
            question = input("Prompt: ").strip()
        run_turn(args, tokenizer, engine, template, image, question)


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        print()
    except KeyboardInterrupt:
        sys.exit(130)
