# Runtime Options

This file will provide details on the usage of the SetRuntimeOption API. It will list all the current key value pairs which can be used as an input for this API.

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

In contrast, `enable_profiling` in runtime option provides dynamic control:
- Can be enabled or disabled at any point during generation
- Each token generation produces its own profiling file when enabled
- Useful for profiling specific portions of the generation process

## Language ID (Nemotron Speech)

Language ID is a runtime option for multilingual Nemotron speech models (e.g., `nemotron_speech` with a prompt-conditioned encoder). It sets the language for transcription on a per-generator basis, allowing a single loaded model to serve generators in different languages.

To set the language, use the integer language ID as a string value: `("lang_id", "<integer>")`

Key: `"lang_id"`

Accepted values: A string representation of the integer language ID. The mapping from BCP-47 language codes to integer IDs is:

| Language | Code | ID | Language | Code | ID |
|----------|------|----|----------|------|----|
| English (US) | `en`, `en-US` | 0 | Japanese | `ja` | 14 |
| English (UK) | `en-GB` | 1 | Korean | `ko` | 15 |
| Spanish (Spain) | `es-ES` | 2 | Dutch | `nl` | 16 |
| Spanish (Latin America) | `es`, `es-US` | 3 | Polish | `pl` | 17 |
| Chinese (Simplified) | `zh-CN` | 4 | Turkish | `tr` | 18 |
| Chinese (Traditional) | `zh-TW` | 5 | Ukrainian | `uk` | 19 |
| Hindi | `hi` | 6 | Romanian | `ro` | 20 |
| Arabic | `ar` | 7 | Greek | `el` | 21 |
| French | `fr`, `fr-FR` | 8 | Czech | `cs` | 22 |
| German | `de`, `de-DE` | 9 | Hungarian | `hu` | 23 |
| Italian | `it` | 10 | Swedish | `sv` | 24 |
| Russian | `ru` | 11 | Danish | `da` | 25 |
| Portuguese (Brazil) | `pt-BR` | 12 | Finnish | `fi` | 26 |
| Portuguese (Portugal) | `pt`, `pt-PT` | 13 | Norwegian | `no` | 27 |
| French (Canada) | `fr-CA` | 100 | Slovak | `sk` | 28 |
| Auto-detect | `auto` | 101 | Croatian | `hr` | 29 |
| | | | Bulgarian | `bg` | 30 |

Example (Python):
```python
generator = og.Generator(model, params)
generator.set_runtime_option("lang_id", "9")  # German
```

Example (C#):
```csharp
var generator = new Generator(model, generatorParams);
generator.SetRuntimeOption("lang_id", "9");  // German
```

Note: This option is only supported on models whose encoder graph includes a `lang_id` input (prompt-conditioned multilingual models). Calling it on an English-only model will throw an error.
