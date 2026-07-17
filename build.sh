#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TELEMETRY_ARG="--use_telemetry"
if [[ "$*" == *"--use_telemetry"* ]]; then
    TELEMETRY_ARG=""
fi

python3 "$SCRIPT_DIR/build.py" $TELEMETRY_ARG "$@"
