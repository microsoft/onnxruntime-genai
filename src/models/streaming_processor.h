// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingProcessor, abstract base class for streaming processing.
#pragma once

#include <memory>
#include <string>

#include "model.h"
#include "silero_vad.h"

namespace Generators {

/// Abstract base class for streaming processors.
/// Provides built-in VAD (Voice Activity Detection) support via SetOption/GetOption.
/// Any derived processor (Nemotron, future Whisper streaming, etc.) gets VAD for free
/// by calling ShouldDropChunk() before processing audio.
struct StreamingProcessor : LeakChecked<StreamingProcessor> {
  virtual ~StreamingProcessor() = default;

  /// Feed raw data.
  /// Returns a NamedTensors when a full chunk is ready, or nullptr if more data is needed.
  /// When VAD is enabled, returns nullptr for chunks detected as prolonged silence.
  virtual std::unique_ptr<NamedTensors> Process(const float* data, size_t num_samples) = 0;

  /// Flush remaining buffered data.
  /// Returns final NamedTensors, or nullptr if buffer is empty.
  virtual std::unique_ptr<NamedTensors> Flush() = 0;

  /// Set a processor option as a key-value pair.
  /// Built-in keys (handled by base class):
  ///   "vad_enabled"             - "true" or "false" (default: "false")
  ///   "vad_threshold"           - float as string, e.g. "0.5"
  ///   "vad_min_silence_chunks"  - int as string, consecutive silence chunks before dropping (default: "5")
  ///   "vad_model_path"          - path to silero_vad.onnx (overrides config filename)
  /// Derived classes can override to add model-specific keys, calling base for vad_* keys.
  virtual void SetOption(const char* key, const char* value);

  /// Get a processor option value by key. Returns the value as a string.
  virtual std::string GetOption(const char* key) const;

 protected:
  /// Call from derived Process() to check if a chunk should be dropped.
  /// Returns true if VAD is enabled and the chunk should be skipped (prolonged silence).
  /// Handles consecutive silence tracking internally.
  bool ShouldDropChunk(const float* chunk_data, size_t chunk_size);

  /// Initialize VAD from the model's genai_config.json vad section.
  /// Called by derived constructors if config has vad.enabled = true.
  void InitVadFromConfig(Model& model);

 private:
  Model* model_{nullptr};  // Stored for deferred VAD creation via SetOption
  std::unique_ptr<SileroVad> vad_;
  int consecutive_silence_chunks_{0};
  int min_silence_chunks_{5};

  void EnableVad(const char* vad_model_path, int sample_rate, float threshold);
  void EnableVadFromModel();
};

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model);

}  // namespace Generators
