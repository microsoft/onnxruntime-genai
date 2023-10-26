# Need updated transformers & torch 2.2.0 (currently preview) to run

import argparse
import os
import subprocess
import sys

def check_ort_import():
    try:
        import onnxruntime as ort
    except:
        return False

    return True

def get_models(args: argparse.Namespace):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    subprocess.run([
        sys.executable,
        "-m",
        "onnxruntime.transformers.models.llama.convert_to_onnx",
        "-m",
        "meta-llama/Llama-2-7b-hf",
        "--output",
        "llama2-7b-fp32-cpu",
        "--precision",
        "fp32",
        "--execution_provider",
        "cpu"
    ])
    # subprocess.run(["mv", "gpt2_parity_results.csv", args.output_folder])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default=os.path.join(".", "test_models"),
        required=False,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    assert check_ort_import()
    args = main()
    get_models(args)
