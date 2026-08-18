# Telemetry

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Official Builds

ONNX Runtime GenAI collects a small number of trace events with the goal of improving product quality. Official packages on supported platforms include the cross-platform 1DS telemetry SDK. Collection is subject to user consent and handled following Microsoft's privacy practices.

Telemetry is turned **ON** by default in official packages.

### Private Builds

The standard `build.sh` and `build.bat` wrappers enable telemetry. For information on how to disable telemetry, see [Disabling Telemetry](#disabling-telemetry) below.

#### Technical Details

ONNX Runtime GenAI uses the cross-platform 1DS SDK (cpp_client_telemetry) to send ONNX Runtime GenAI trace events to Microsoft's telemetry backend over HTTPS. Based on user consent, this data is handled following GDPR and privacy regulations for anonymity and data access controls.

For ways to disable telemetry, see the [Disabling Telemetry](#disabling-telemetry) section below.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** Telemetry is compiled by default when using `build.py`, `build.bat`, or `build.sh` on supported platforms. Pass `--no_telemetry` to produce a binary that collects no telemetry, or configure CMake directly with `-DENABLE_TELEMETRY=OFF`.
- **Disable native telemetry at runtime.** Set `ORT_DISABLE_TELEMETRY=1` before the native ONNX Runtime GenAI runtime initializes to prevent native events, the native uploader, and native device-ID persistence.
- **Disable Python builder and benchmark details.** For the Python model builder and benchmark telemetry, the same environment variable suppresses detailed events, but one best-effort Heartbeat may still be attempted and may create or use the shared persistent device identifier. CI and automated-test environments suppress all Python telemetry.
- **Disable non-essential events via the API.** The C API (and the C++ wrapper, C#, Python, Java, and Objective-C bindings) can suppress non-essential telemetry. A process information or Heartbeat event may still be emitted.
