#!/bin/bash

# ---
# This script builds and tests either an fp32 or bf16 model.
#
# Usage: ./run_qwen_vl.sh [fp32|bf16]
# ---

# Exit immediately if a command fails
set -e

# 1. Validate Input
if [ "$1" != "fp32" ] && [ "$1" != "bf16" ]; then
    echo "Error: Invalid precision."
    echo "Usage: $0 fp32|bf16"
    exit 1
fi

# 2. Define variables based on input
PRECISION=$1
OUTPUT_DIR="./qwen_${PRECISION}"
ONNX_MODEL_PATH="${OUTPUT_DIR}/model.onnx"
CACHE_DIR="./cache"

# Set the --bf16 flag only if precision is bf16
TEST_FLAG=""
if [ "$PRECISION" == "bf16" ]; then
    TEST_FLAG="--bf16"
fi

# 3. Remove output directory only if it exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Removing existing directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

# 4. Run the builder script
echo "--- Building ${PRECISION} model ---"
python builder.py \
    -m Qwen/Qwen2.5-VL-7B-Instruct \
    -p ${PRECISION} \
    -o ${OUTPUT_DIR} \
    -e cuda \
    -c ${CACHE_DIR}

# 5. Run the parity test
echo "--- Testing ${PRECISION} model parity ---"
python test_qwen_vl_parity.py \
    --cache_dir ${CACHE_DIR} \
    --onnx_model ${ONNX_MODEL_PATH} \
    ${TEST_FLAG}

echo "--- ${PRECISION} run complete ---"