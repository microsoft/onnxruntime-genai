// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct QwenImageProcessor : Processor {
  QwenImageProcessor(Config& config, const SessionInfo& session_info);

  QwenImageProcessor() = delete;
  QwenImageProcessor(const QwenImageProcessor&) = delete;
  QwenImageProcessor& operator=(const QwenImageProcessor&) = delete;

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;
  Config& config_;
  ONNXTensorElementDataType pixel_values_type_;
  ONNXTensorElementDataType image_grid_thw_type_;
};

}  // namespace Generators
