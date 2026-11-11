# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Image questions using an Engine vision package and its own chat template."""

import argparse
import json

import onnxruntime_genai as og


def run_turn(engine, tokenizer, messages, images, maximum, session_length):
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(session_length)
    request = engine.create_request(options=request_options)
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(maximum)
    options.set_do_sample(False)
    try:
        prompt = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True)
        try:
            request.begin_turn(prompt, options, images=images)
        except RuntimeError as error:
            print(f"Admission rejected: {error}")
            return None
        buffer = engine.create_event_buffer(32)
        stream = tokenizer.create_stream()
        answer = []
        while engine.has_pending_requests():
            for event in engine.run(buffer):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine request failed: {event.error_code}")
                if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
                    raise RuntimeError(f"Engine capacity blocked: {event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    text = stream.decode(event.token)
                    answer.append(text)
                    print(text, end="", flush=True)
        print()
        return "".join(answer)
    finally:
        request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Engine vision package, not a Generator merge-embedding package")
    parser.add_argument("--image", action="append", required=True, help="PNG/JPEG input; repeat in prompt order")
    parser.add_argument("--question", default="Describe these images briefly.")
    parser.add_argument("--follow-up", help="Optional text-only user turn about the same images")
    parser.add_argument("--max-generated-tokens", type=int, default=128)
    parser.add_argument(
        "--max-session-tokens", type=int, default=8192, help="Includes expanded images and generated tokens"
    )
    args = parser.parse_args()
    if not 0 < args.max_generated_tokens < args.max_session_tokens:
        parser.error("Require 0 < max-generated-tokens < max-session-tokens")
    model = og.Model(args.model)
    tokenizer, engine = og.Tokenizer(model), og.Engine(model)
    content = [{"type": "image", "image": path} for path in args.image]
    content.append({"type": "text", "text": args.question})
    messages = [{"role": "user", "content": content}]
    images = og.Images.open(*args.image)
    answer = run_turn(engine, tokenizer, messages, images, args.max_generated_tokens, args.max_session_tokens)
    if args.follow_up and answer is not None:
        # Replay the full template rather than assuming a model-specific incremental suffix.
        messages.extend([{"role": "assistant", "content": answer}, {"role": "user", "content": args.follow_up}])
        run_turn(engine, tokenizer, messages, images, args.max_generated_tokens, args.max_session_tokens)


if __name__ == "__main__":
    main()
