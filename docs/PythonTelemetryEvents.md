# Python Telemetry Events

This file tracks the event names and fields emitted by
`src/python/py/telemetry/telemetry.py` and `telemetry_extensions.py`.

Events use Common Schema 4.0 envelopes with `ver`, `name`, `time`, `iKey`, and
`data`. Every `data` object includes:

- `appName` (string)
- `LibraryVersion` (string)
- `AppSessionGuid` (string UUID)

| Event name | Additional `data` fields |
| --- | --- |
| `GenAIHeartbeat` | `sessionId` (integer, always `0`), `deviceId` (hashed string), `deviceIdStatus` (string), `os` (string), `osVersion` (string), `osRelease` (string), `osArchitecture` (string), `processorCount` (integer), `cpuModel` (string), `totalMemoryMB` (integer), `gpuName` (string), `gpuDriverVersion` (string), `gpuMemoryMB` (integer), `gpuCount` (integer), `deviceManufacturer` (string), `deviceModel` (string), `pythonVersion` (string), `ortVersion` (string), `availableProviders` (comma-separated string) |
| `GenAIModelBuild` | `action` (string), `durationMs` (number), `success` (boolean), `modelName` (string), `modelType` (string), `hiddenSize` (integer), `numLayers` (integer), `numAttentionHeads` (integer), `numKeyValueHeads` (integer), `vocabSize` (integer), `contextLength` (integer), `ioDtype` (string), `quantType` (string), `executionProvider` (string), `outputModelSizeBytes` (integer), `numOnnxOperators` (integer), `operatorTypes` (comma-separated string), `hasCustomOps` (boolean), `sourceFormat` (string), `hasAdapter` (boolean), optional `extraOptions` (object) |
| `GenAIBenchmark` | `sessionId` (integer), `modelName` (string), `precision` (string), `backend` (string), `device` (string), `batchSize` (integer), `promptLength` (integer), `tokensGenerated` (integer), `tokenizationLatencyMs` (number), `tokenizationThroughput` (number), `promptProcessingLatencyMs` (number), `promptProcessingThroughput` (number), `tokenGenerationLatencyMs` (number), `tokenGenerationThroughput` (number), `samplingLatencyMs` (number), `samplingThroughput` (number), `wallClockTimeMs` (number), `wallClockThroughput` (number), `timeToFirstTokenMs` (number), `peakGpuMemoryMB` (number), `peakCpuMemoryMB` (number) |
| `GenAIModelLoad` | `sessionId` (integer), `modelName` (string), `modelType` (string), `executionProvider` (string), `totalLoadTimeMs` (number), `numSessions` (integer), `modelFileSizeBytes` (integer) |
| `GenAIInference` | `sessionId` (integer), `modelName` (string), `modelType` (string), `executionProvider` (string), `timeToFirstTokenMs` (number), `totalGenerationTimeMs` (number), `totalTokensGenerated` (integer), `inputTokenCount` (integer), `memoryUsedMB` (number), `gpuMemoryUsedMB` (number) |
| `GenAIAction` | `invokedFrom` (string), `actionName` (string), `durationMs` (number), `success` (boolean), plus optional scrubbed metadata |
| `GenAIError` | `exceptionType` (string), `exceptionMessage` (string), and optional `action` (string), `modelName` (string), `executionProvider` (string), `sessionId` (integer), or scrubbed metadata |

Strings and metadata are scrubbed at final serialization; filesystem paths and
URL-shaped values are replaced with `[path]`.
