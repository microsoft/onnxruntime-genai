// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingProcessor, abstract base class for streaming processing.
#pragma once

#include "model.h"

namespace Generators {

/// Abstract base class for streaming processors.
struct StreamingProcessor : LeakChecked<StreamingProcessor> {
  virtual ~StreamingProcessor() = default;

  /// Feed raw data.
  /// Returns a NamedTensors when a full chunk is ready, or nullptr if more data is needed.
  /// When VAD is enabled, returns nullptr for chunks detected as silence.
  virtual std::unique_ptr<NamedTensors> Process(const float* data, size_t num_samples) = 0;

  /// Flush remaining buffered data.
  /// Returns final NamedTensors, or nullptr if buffer is empty.
  virtual std::unique_ptr<NamedTensors> Flush() = 0;

  /// Set a processor option as a key-value pair.
  /// Supported keys:
  ///   "vad_enabled"             - "true" or "false" (default: "false")
  ///   "vad_threshold"           - float as string, e.g. "0.5"
  ///   "vad_min_silence_chunks"  - int as string, consecutive silence chunks before dropping (default: "5")
  ///   "vad_model_path"          - path to silero_vad.onnx (overrides config filename)
  virtual void SetOption(const char* key, const char* value) = 0;

  /// Get a processor option value by key. Returns the value as a string.
  virtual std::string GetOption(const char* key) const = 0;
};

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model);

}  // namespace Generators
