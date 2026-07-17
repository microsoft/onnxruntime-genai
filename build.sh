#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Telemetry is enabled by default by the wrapper. Use build.py directly without --use_telemetry to build it out.
# Runtime opt-out remains available via ORT_TELEMETRY_DISABLED=1 / OgaSetTelemetryEnabled(false).
TELEMETRY_ARG="--use_telemetry"
if [[ "$*" == *"--use_telemetry"* ]]; then
    TELEMETRY_ARG=""
fi

python3 "$SCRIPT_DIR/build.py" $TELEMETRY_ARG "$@"
