import argparse
import json
import time

import onnxruntime_genai as og
from datasets import load_dataset


def main(args):
    dataset = load_dataset(path="epfl-dlab/JSONSchemaBench", name="Github_hard", split="test")
    schema = json.loads(dataset[0]["json_schema"])

    system_prompt = "You need to generate a JSON object that matches the schema below."
    user_prompt = json.dumps(schema, indent=2)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    search_options = {
        name: getattr(args, name)
        for name in ["do_sample", "max_length", "min_length", "top_p", "top_k", "temperature", "repetition_penalty"]
        if name in args
    }
    search_options["batch_size"] = 1
    search_options["temperature"] = 0.0

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    schema["x-guidance"] = {"whitespace_flexible": False, "key_separator": ": ", "item_separator": ", "}
    guidance_type = "lark_grammar"
    guidance_input = f"""start: %json {json.dumps(schema)}\n"""
    params.set_search_options(**search_options)
    params.set_guidance(guidance_type, guidance_input, args.enable_ff_tokens)  # set guidance

    generator = og.Generator(model, params)

    final_prompt = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True)
    final_input = tokenizer.encode(final_prompt)
    generator.append_tokens(final_input)

    start_len = len(generator.get_sequence(0))
    prev_len = start_len
    t0 = time.time()
    # for i in range(15):
    full_seq_str = ""
    while not generator.is_done():
        generator.generate_next_token()

        # NOTE: since get_next_tokens returns only the last token, we'll need to use get_sequence instead
        # new_tokens = generator.get_next_tokens()[0]
        # print(tokenizer_stream.decode(new_tokens), end='', flush=True)

        seq = generator.get_sequence(0)
        new_tokens = seq[prev_len:]
        seq_str = ""
        for token in new_tokens:
            seq_str += tokenizer_stream.decode(token)
        print(seq_str, end="", flush=True)
        prev_len = len(seq)
        full_seq_str += seq_str
    latency = time.time() - t0
    print()

    # verify valid json
    _json = json.loads(full_seq_str)
    print(json.dumps(_json, indent=2))

    tps = (len(generator.get_sequence(0)) - start_len) / latency
    print(f"Generation Latency: {latency:.2f} sec")
    print(f"Tokens/sec: {tps:.2f} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai"
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
        required=False,
        default="follow_config",
        choices=["cpu", "cuda", "dml", "follow_config"],
        help="Execution provider to run the ONNX Runtime session with. Defaults to follow_config that uses the execution provider listed in the genai_config.json instead.",
    )
    parser.add_argument(
        "--enable_ff_tokens",
        action="store_true",
        default=False,
        help="Enable feed-forward tokens in the model session if supported (default: False)",
    )
    args = parser.parse_args()
    main(args)
