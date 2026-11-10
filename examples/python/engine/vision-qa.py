# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Qwen packed vision Engine example; inputs are formatted turns, not chat messages."""

import argparse
import json

import onnxruntime_genai as og


def run_turn(engine, request, prompt, images, maximum, tokenizer):
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(maximum)
    options.set_do_sample(False)
    request.begin_turn(prompt, options, images=images)
    del images
    buffer = engine.create_event_buffer(32)
    stream = tokenizer.create_stream()
    while engine.has_pending_requests():
        for event in engine.run(buffer):
            if event.flags & og.EngineEventFlags.FAILED:
                raise RuntimeError(f"Engine request failed: {event.error_code}")
            if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
                raise RuntimeError(f"Engine capacity blocked: {event.error_code}")
            if event.flags & og.EngineEventFlags.TOKEN:
                print(stream.decode(event.token), end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Packed Qwen vision package, not a Generator VLM package")
    parser.add_argument("--image", action="append", required=True, help="PNG/JPEG input; repeat in prompt order")
    parser.add_argument("--question", default="Describe these images briefly.")
    parser.add_argument("--follow-up", help="Optional text-only user turn about the same images")
    parser.add_argument("--max-generated-tokens", type=int, default=128)
    args = parser.parse_args()
    model = og.Model(args.model)
    tokenizer, engine = og.Tokenizer(model), og.Engine(model)
    request = engine.create_request()
    content = "<|vision_start|><|image_pad|><|vision_end|>\n" * len(args.image) + args.question
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps([{"role": "user", "content": content}]), add_generation_prompt=True
    )
    run_turn(engine, request, prompt, og.Images.open(*args.image), args.max_generated_tokens, tokenizer)
    if args.follow_up:
        # Explicitly close the preceding assistant even if it ended at a length/stop limit.
        suffix = "<|im_end|>\n<|im_start|>user\n" + args.follow_up + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        run_turn(engine, request, suffix, None, args.max_generated_tokens, tokenizer)
    request.close()


if __name__ == "__main__":
    main()
