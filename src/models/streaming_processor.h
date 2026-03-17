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
  virtual std::unique_ptr<NamedTensors> Process(const float* data, size_t num_samples) = 0;

  /// Flush remaining buffered data.
  /// Returns final NamedTensors, or nullptr if buffer is empty.
  virtual std::unique_ptr<NamedTensors> Flush() = 0;
};

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model);

}  // namespace Generators
