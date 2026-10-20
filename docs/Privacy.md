# Telemetry

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Official Builds

ONNX Runtime GenAI collects telemetry to help improve product quality. Collection is subject to user consent and handled following Microsoft's privacy practices.

Telemetry is turned **ON** by default in official packages.

### Private Builds

The standard `build.sh` and `build.bat` wrappers enable telemetry. For information on how to disable telemetry, see [Disabling Telemetry](#disabling-telemetry) below.

### Collected Information

Telemetry may include model and build metadata, device and platform information, performance measurements, diagnostics, and a pseudonymous device identifier. Local paths and credentials are redacted.

### Disabling Telemetry

- Set `ORT_DISABLE_TELEMETRY=1` before first use to disable native and Python telemetry.
- For Python tools, use the model builder's `--disable_telemetry` flag or call `onnxruntime_genai.telemetry.disable_telemetry()`.
- For private native binaries and Python wheels, pass `--no_telemetry` to the build script or configure CMake with `-DENABLE_TELEMETRY=OFF`.
- Native language bindings provide `disable_telemetry_events`-style controls for non-essential events. These do not disable native process information; use the environment variable for full suppression.

Telemetry is also disabled in CI and native test harnesses.

Python runtime opt-out stops new events and uploads. An upload already admitted may finish, and retained queued events may be sent by a later telemetry-enabled process. Opt out before first use to prevent all uploads.
