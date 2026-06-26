# Telemetry

ONNX Runtime GenAI collects anonymous usage telemetry to help Microsoft
understand which models, hardware, and execution providers are used so the
library can be improved. This page describes what is collected, when, and how to
turn it off.

## How to opt out

Telemetry is on by default. To disable the collection of detailed usage events,
set the environment variable before running:

```bash
# Windows (PowerShell)
$env:ORT_DISABLE_TELEMETRY = "1"

# Linux / macOS
export ORT_DISABLE_TELEMETRY=1
```

When telemetry is disabled this way, no detailed usage events are sent. A
minimal device-id heartbeat (described below) is still sent outside CI/CD
environments so Microsoft can count active devices; it contains no model,
performance, or error data.

In CI/CD environments (detected automatically via `CI`, `TF_BUILD`,
`GITHUB_ACTIONS`, `JENKINS_URL`, and similar variables) nothing is sent at all —
not even the heartbeat. Setting `ORT_DISABLE_TELEMETRY=1` in a CI/CD
environment is therefore fully silent.

## What is collected

### States

| State | Device-id heartbeat | Detailed usage events |
| --- | --- | --- |
| Enabled (default) | Yes | Yes |
| `ORT_DISABLE_TELEMETRY=1` | Yes | No |
| CI/CD environment | No | No |

### Device-id heartbeat

Sent once per process to support active-device counting. It contains:

- A non-reversible hashed device identifier (SHA-256 of a locally generated
  random value; it contains no MAC address, hostname, or user name) and its
  status.
- Operating system name, version, release, and architecture.
- Hardware **model** information only — CPU model, total memory, GPU name, GPU
  driver version, GPU memory and count, and device manufacturer/model. These are
  model/capacity/version strings; no serial numbers, MAC addresses, or unique
  hardware identifiers are collected.
- Python version, ONNX Runtime version, available execution providers, and the
  process name (the base file name of the running program only — no directory
  path or user name).
- The package name, version, and a per-process instance id.

### Detailed usage events (only when enabled)

- **Model build** — model architecture and structure (type, hidden size, layers,
  attention/KV heads, vocabulary size, context length), precision/quantization,
  execution provider, ONNX operator types, and output size.
- **Model load / inference / benchmark** — load time, time-to-first-token,
  generation throughput, token counts, and memory usage.
- **Errors** — the exception type and a message with absolute filesystem paths
  redacted so user names embedded in paths are not collected.

No model weights, prompts, generated text, file contents, or raw file paths are
collected.

## Privacy statement

Your use of the software operates as your consent to these practices. See the
[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704)
for more information.
