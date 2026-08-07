// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "processor.h"

namespace Generators {

struct Mistral3ImageProcessor : Processor {
  Mistral3ImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  int patch_size_;
  int spatial_merge_size_;
};

}  // namespace Generators
