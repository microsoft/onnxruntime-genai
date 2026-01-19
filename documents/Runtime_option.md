# Runtime Options

This file will provide details on the usage of SetRuntimeOption API. It will list all the current key value pairs which can be used as an input for this API.

## Set Terminate

Set Terminate is a runtime option to terminate the current session or continue/restart an already terminated session. The current session will throw an exception when the terminate option is enabled and the user will need to handle that scenario, examples/c/src/phi3_terminate.cpp contains an example for this.

To terminate the generation, use this key value pair: ("terminate_session", "1")

To recover from a terminated state, use this key value pair: ("terminate_session", "0")

Key: "terminate_session"

Accepted values: ("0", "1")

## Enable Profiling

Enable Profiling is a runtime option to dynamically enable or disable ONNX Runtime profiling during generation. Once enabled, each subsequent token generation will produce profiling data saved to a separate JSON file. You can stop profiling at any time.

To enable profiling with default file prefix "onnxruntime_run_profile", use this key value pair: ("enable_profiling", "1")

To disable profiling, use this key value pair: ("enable_profiling", "0")

To enable profiling with a custom file prefix, use this key value pair: ("enable_profiling", "<your_custom_prefix>")

Key: "enable_profiling"

Accepted values: ("0", "1", or a custom profile file prefix string)

Note: Difference from SessionOptions `enable_profiling` in genai_config.json

The `enable_profiling` option in `genai_config.json` under `SessionOptions` is a session-level configuration. When enabled, it collects all profiling data from session creation to session end and aggregates them into a single JSON file. This configuration cannot be started or stopped dynamically during inference.

In contrast, ``enable_profiling` in runtime option provides dynamic control:
- Can be enabled or disabled at any point during generation
- Each token generation produces its own profiling file when enabled
- Useful for profiling specific portions of the generation process
