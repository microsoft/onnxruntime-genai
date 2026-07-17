:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
setlocal

rem Requires a Python install to be available in your PATH

rem Telemetry is enabled by default by the wrapper. Use build.py directly without --use_telemetry to build it out.
rem Runtime opt-out remains available via ORT_TELEMETRY_DISABLED=1 / OgaSetTelemetryEnabled(false).
set "TELEMETRY_ARG=--use_telemetry"
echo "%*" | findstr /C:"--use_telemetry" >nul && set "TELEMETRY_ARG="

python "%~dp0\build.py" %TELEMETRY_ARG% %*
