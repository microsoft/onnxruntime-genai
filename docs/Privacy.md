# Telemetry

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described below. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Official Builds

ONNX Runtime GenAI collects a small number of trace events with the goal of improving product quality. Official packages on supported platforms include the cross-platform 1DS telemetry SDK. Collection is subject to user consent and handled following Microsoft's privacy practices.

Telemetry is turned **ON** by default in official packages.

The 1DS SDK is obtained one of two ways (see [`cmake/telemetry.cmake`](../cmake/telemetry.cmake)):

- **vcpkg port (preferred for desktop builds):** when a vcpkg root is available (`VCPKG_ROOT` / `VCPKG_INSTALLATION_ROOT`, or `--vcpkg_path`), the `cpp-client-telemetry` port provides the SDK. A static triplet lets the SDK be linked in and dead-stripped for minimum binary footprint.
- **Source build (fallback, and used for mobile cross-builds):** when no vcpkg root is available, or when cross-compiling for Android/iOS, the SDK source (pinned in [`cmake/deps.txt`](../cmake/deps.txt)) is downloaded and built via CMake FetchContent. This path uses the `patch` tool, which ships with Git.

### Private Builds

The standard `build.sh` and `build.bat` wrappers enable telemetry. For information on how to disable telemetry, see [Disabling Telemetry](#disabling-telemetry) below.

#### Technical Details

ONNX Runtime GenAI uses the cross-platform 1DS SDK (cpp_client_telemetry) to send ONNX Runtime GenAI trace events to Microsoft's telemetry backend over HTTPS. Based on user consent, this data is handled following GDPR and privacy regulations for anonymity and data access controls.

For Windows, Linux, and macOS, ONNX Runtime GenAI sends a product-salted hash of a locally generated per-user UUID as its device identifier. For Android and iOS, ONNX Runtime GenAI uses the platform device identifier provided by the 1DS SDK instead of creating a separate GenAI-generated mobile device id.

ProcessInfo is retained at 100%. Non-essential model-session events, including errors, are deterministically sampled as a complete session so correlated events are always retained or dropped together. The current model-session sampling rate is 100%. Retained events carry the 1DS Common Schema `popSample` percentage; an absent value means 100%. To estimate the original event population, sum `100.0 / popSample` for received events, treating an absent or invalid value as 100%.

For ways to disable telemetry, see the [Disabling Telemetry](#disabling-telemetry) section below.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** Telemetry is only compiled when configuring with `--use_telemetry` (`-DENABLE_TELEMETRY=ON`). To produce a binary that collects no data, run the `build.bat` and `build.sh` scripts without `--use_telemetry`.
- **At runtime, via environment variable.** Set `ORT_TELEMETRY_DISABLED=1` before the library initializes to disable non-essential telemetry. The variable also accepts `true` / `yes` / `on` / `y`, case-insensitive. ONNX Runtime GenAI may still send a minimal initialization event.
- **At runtime, via the API.** Call `OgaSetTelemetryEnabled(false)` in the C API, or `Oga::SetTelemetryEnabled(false)` in the C++ wrapper to disable non-essential telemetry. ONNX Runtime GenAI may still send a minimal initialization event.
