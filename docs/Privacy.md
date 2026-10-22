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

#### Hardware and Resource Diagnostics

When non-essential telemetry is enabled, a bounded hardware inventory is collected once per process on the first
sampled successful model load. It supplements existing OS, CPU model, processor count and total system RAM
information with available system RAM and CPU/GPU/NPU product names, product vendor/device IDs, driver versions,
core counts, clock rates, memory bus width and memory capacities where supported by the operating system and
drivers. Unsupported measurements are reported as unknown, not zero. Inventory may be incomplete on platforms
without suitable hardware APIs; collection limits are explicitly reported.

Product vendor/device IDs identify hardware models, not individual devices. The inventory does not transmit
adapter UUIDs, serial numbers, MAC addresses, PCI bus addresses, device-instance paths, prompts or generated
content. PCI addresses may be used locally to query allowlisted hardware properties but are not event fields.
Descriptive strings are bounded and filesystem paths are redacted.

Generator resource diagnostics record process resident RAM and cumulative process CPU time at the first prompt
append and at the first observed completion query. Supported accelerator APIs also provide process GPU/NPU memory
snapshots. These are process-wide observations, including concurrent requests and other activity, not isolated
request allocations or peak measurements. Completion snapshots are cached: destroying a generator later does
not take a fresh sample or extend the resource interval. If completion is not observed, end measurements are
unknown and labeled `resourceSampleScope=completion_not_observed`.

Accelerator memory fields distinguish dedicated/shared capacity, device-wide free memory, process memory usage
and the process's driver-managed memory budget. A budget or device-wide free-memory value is not a process
allocation. Measurements include their collection source; unavailable or unreliable process-memory data,
including unsupported container PID mappings, remains unknown. Hardware inventory also covers engine model
loads, but request-resource snapshots currently cover generators, not `OgaEngine` requests or embeddings.

Events are labeled `TelemetryLayer=onnxruntime_genai`. Generator timing and resource intervals differ from
Foundry Local's request timings and must not be summed as independent work. Model/generator identifiers do not
establish a cross-layer request join with Foundry Local correlation IDs.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** Telemetry is compiled by default when using `build.py`, `build.bat`, or `build.sh` on supported platforms. Pass `--no_telemetry` to produce a binary that collects no telemetry, or configure CMake directly with `-DENABLE_TELEMETRY=OFF`.
- **Disable all telemetry at runtime.** Set `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime GenAI initializes. The legacy `ORT_TELEMETRY_DISABLED` variable is also recognized as a full GenAI process opt-out. Either variable prevents the uploader, all events, and the persistent device identifier from being created for the process lifetime; the legacy alias does not merely disable non-essential events.
- **Disable non-essential events via the API.** The C API (and the C++ wrapper, C#, Python, Java, and Objective-C bindings) can suppress non-essential telemetry. A process information event may still be emitted when only API suppression is used. For the full process-lifetime opt-out, use `ORT_DISABLE_TELEMETRY` before initialization.

`ORT_DISABLE_TELEMETRY` accepts `1`, `true`, `yes`, `y`, or `on`, ignoring case and surrounding whitespace.
The legacy alias preserves its permissive truthiness: any nonempty value other than `0`, `false`, `no`, or `off`
disables telemetry. An environment opt-out observed during initialization or a telemetry-control call cannot
be reversed by clearing the variable or re-enabling telemetry through the API. Ordinary API-only suppression
remains reversible. Hardware inventory and resource diagnostics are non-essential and honor these controls.
