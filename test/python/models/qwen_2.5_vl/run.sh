# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

#!/bin/bash
# This script builds and tests either an fp32, bf16 or fp16 Qwen2.5-VL-3B-Instruct model. Append -f to force export.
# Usage: ./run.sh [fp32|bf16|fp16] [-f]

# Exit immediately if a command fails
set -e

# 1. Validate Input
if [ "$1" != "fp32" ] && [ "$1" != "bf16" ] && [ "$1" != "fp16" ]; then
    echo "Error: Invalid precision."
    echo "Usage: $0 fp32|bf16|fp16"
    exit 1
fi

# 2. Define variables based on input
PRECISION=$1
OUTPUT_DIR="./qwen_${PRECISION}"
ONNX_MODEL_PATH="${OUTPUT_DIR}/model.onnx"
CACHE_DIR="./cache"
HF_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# Set the --bf16 or --fp16 flag for the test script
TEST_FLAG=""
if [ "$PRECISION" == "bf16" ]; then
    TEST_FLAG="--bf16"
elif [ "$PRECISION" == "fp16" ]; then
    TEST_FLAG="--fp16"
fi

# 3. Remove output directory only if it exists and -f flag is provided.
if  [ "$2" == "-f" ] && [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing existing directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

# 4. Run the builder script if output directory does not exist.
if ! [ -d "${OUTPUT_DIR}" ]; then
    echo "--- Building ${PRECISION} model ---"
    python -m onnxruntime_genai.models.builder \
        -m ${HF_MODEL} \
        -p ${PRECISION} \
        -o ${OUTPUT_DIR} \
        -e cuda \
        -c ${CACHE_DIR}
fi

# 5. Run the parity test
echo "--- Testing ${PRECISION} model parity ---"
python test_qwen_2.5_vl.py \
    --hf_model ${HF_MODEL} \
    --cache_dir ${CACHE_DIR} \
    --onnx_model ${ONNX_MODEL_PATH} \
    ${TEST_FLAG}

echo "--- ${PRECISION} run complete ---"