#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Telemetry is enabled by default when a vcpkg root is available (the 1DS
# cpp-client-telemetry port requires vcpkg) and the target is not mobile. Opt out
# by removing this flag (build via build.py directly), or at runtime via the
# ORT_TELEMETRY_DISABLED=1 env var / OgaSetTelemetryEnabled(false).
TELEMETRY_ARG="--use_telemetry"
if [[ -z "${VCPKG_ROOT}${VCPKG_INSTALLATION_ROOT}" ]] || [[ "$*" == *"--android"* ]] || [[ "$*" == *"--ios"* ]]; then
    TELEMETRY_ARG=""
fi

python3 "$SCRIPT_DIR/build.py" $TELEMETRY_ARG "$@"
