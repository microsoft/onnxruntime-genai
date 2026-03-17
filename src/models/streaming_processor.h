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
  /// When VAD is enabled, returns nullptr for chunks that contain no speech.
  virtual std::unique_ptr<NamedTensors> Process(const float* data, size_t num_samples) = 0;

  /// Flush remaining buffered data.
  /// Returns final NamedTensors, or nullptr if buffer is empty.
  virtual std::unique_ptr<NamedTensors> Flush() = 0;

  /// Enable Voice Activity Detection. Chunks without speech will be skipped.
  /// @param vad_model_path Path to the silero_vad.onnx model file.
  /// @param threshold Speech probability threshold (default 0.5).
  virtual void EnableVad(const char* vad_model_path, float threshold = 0.5f) = 0;

  /// Disable Voice Activity Detection. All chunks will be processed.
  virtual void DisableVad() = 0;

  /// Set the VAD speech probability threshold.
  virtual void SetVadThreshold(float threshold) = 0;

  /// Returns true if VAD is currently enabled.
  virtual bool IsVadEnabled() const = 0;
};

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model);

}  // namespace Generators
