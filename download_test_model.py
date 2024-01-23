import argparse
import os
import subprocess
import sys
from transformers import AutoConfig

def check_ort_import():
    try:
        import onnxruntime as ort
    except Exception as ex:
        return False

    return True

def get_models(args: argparse.Namespace):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    subprocess.run([
        sys.executable,
        "-m",
        "onnxruntime.transformers.models.gpt2.convert_to_onnx",
        "-m",
        args.model_name,
        "--cache_dir",
        args.cache_folder,
        "--output",
        args.output_folder,
        "-o",
        "-p",
        args.precision,
        "-t",
        "1",
    ])
    # subprocess.run(["mv", "gpt2_parity_results.csv", args.output_folder])

    # Save config file
    config = AutoConfig.from_pretrained(args.model_name, use_auth_token=True)
    config.save_pretrained(args.output_folder)

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

    parser.add_argument(
        "-c",
        "--cache_folder",
        default=os.path.join(".", "model_cache"),
        required=False,
    )

    parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        required=False,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    assert check_ort_import()
    args = main()
    get_models(args)
