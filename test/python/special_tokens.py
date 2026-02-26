# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Run this script to mark certain token ids as special

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, type=str, help="Path to tokenizer.json")
parser.add_argument("-s", "--start_tool_call", required=True, type=str, help="String representation of starting tool call token")
parser.add_argument("-e", "--end_tool_call", required=True, type=str, help="String representation of ending tool call token")

args = parser.parse_args()
assert os.path.exists(args.path), "Invalid path to tokenizer.json"
assert os.path.basename(args.path) == "tokenizer.json", "Path is not to a tokenizer.json file"

# Use raw bytes when making comparisons
start_b = args.start_tool_call.encode("ascii", "strict")
end_b = args.end_tool_call.encode("ascii", "strict")
false_b = b'"special": false'
true_b = b'"special": true'

seen = False
temp_path = args.path.replace("tokenizer.json", "temp.json")
with open(args.path, "rb") as in_file, open(temp_path, "wb") as out_file:
    for line in in_file:
        if start_b in line or end_b in line:
            seen = True

        if seen and false_b in line:
            out_file.write(line.replace(false_b, true_b))
            seen = False
        else:
            out_file.write(line)

os.replace(temp_path, args.path)
