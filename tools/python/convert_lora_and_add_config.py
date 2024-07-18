# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script helps converting .npz files to .fb files and
# adding and adapter to genai config json file


import argparse

import sys
from util import save_lora_params_to_flatbuffers
from util import add_adapters_to_genai_config

def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_name", type=str, required=True)
    parser.add_argument("--npz_file_path", type=str, required=True)
    parser.add_argument("--fb_file_path", type=str, required=True)
    parser.add_argument("--genai_file_path", type=str, required=True)
    parser.add_argument("--genai_file_path_modified", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    save_lora_params_to_flatbuffers(args.npz_file_path, args.fb_file_path)
    adapters = {args.adapter_name: {"weights": args.fb_file_path}}
    add_adapters_to_genai_config(args.genai_file_path, args.genai_file_path_modified, adapters)
    return 0

if __name__ == "__main__":
    sys.exit(main())
