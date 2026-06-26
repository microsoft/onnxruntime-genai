:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
setlocal

rem Requires a Python install to be available in your PATH

rem Telemetry is enabled by default when a vcpkg root is available (the 1DS
rem cpp-client-telemetry port requires vcpkg) and the target is not mobile. Opt out
rem by removing this flag (build via build.py directly), or at runtime via the
rem ORT_TELEMETRY_DISABLED=1 env var / OgaSetTelemetryEnabled(false).
set "TELEMETRY_ARG=--use_telemetry"
if "%VCPKG_ROOT%%VCPKG_INSTALLATION_ROOT%"=="" set "TELEMETRY_ARG="
echo %* | findstr /C:"--android" >nul && set "TELEMETRY_ARG="
echo %* | findstr /C:"--ios" >nul && set "TELEMETRY_ARG="

python "%~dp0\build.py" %TELEMETRY_ARG% %*
