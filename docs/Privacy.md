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

GenAI-owned string event properties are limited to **1 KiB (1,024 UTF-8 bytes)**, including error
messages, model metadata, execution-provider lists, CPU names, and input modalities. Values are
bounded before copying or caching and again when constructing events. Truncation never splits a
valid UTF-8 codepoint; malformed UTF-8 bytes are replaced with `?`. Existing tighter limits, such as
the 256-byte Apple CPU-name buffer and fixed-size device identifiers, remain in place.

Error redaction inspects at most 16 KiB. If a longer message could conceal a path anchor beyond that
boundary, its uncertain token/path suffix is conservatively removed before applying the 1 KiB output
limit. Environment inputs are limited to 16 KiB; oversized paths are rejected, never truncated into
different paths. Persistence falls back to another valid per-user location or an in-memory identifier,
and certificate lookup falls back to known system bundles. Unreadable or oversized suppression
variables suppress telemetry. Host-evidence files retain their existing 16 KiB read limit, with a
64 KiB processing limit for combined evidence. These limits govern GenAI's own collection code, not
platform context synthesized internally by the telemetry SDK.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** Telemetry is compiled by default when using `build.py`, `build.bat`, or `build.sh` on supported platforms. Pass `--no_telemetry` to produce a binary that collects no telemetry, or configure CMake directly with `-DENABLE_TELEMETRY=OFF`.
- **Disable all telemetry at runtime.** Set `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime GenAI initializes. This prevents the uploader, events, and persistent device identifier from being created for the process lifetime.
- **Disable non-essential events via the API.** The C API (and the C++ wrapper, C#, Python, Java, and Objective-C bindings) can suppress non-essential telemetry. A process information event may still be emitted when only API suppression is used. For the full process-lifetime opt-out, use `ORT_DISABLE_TELEMETRY` before initialization.
