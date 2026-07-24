:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
setlocal

rem Requires a Python install to be available in your PATH

set "TELEMETRY_ARG=--use_telemetry"
echo "%*" | findstr /C:"--use_telemetry" >nul && set "TELEMETRY_ARG="

python "%~dp0\build.py" %TELEMETRY_ARG% %*
