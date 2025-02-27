// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"

namespace Generators {

struct PhiImageProcessor : Processor {
  PhiImageProcessor(Config& config, const SessionInfo& session_info);

  PhiImageProcessor() = delete;
  PhiImageProcessor(const PhiImageProcessor&) = delete;
  PhiImageProcessor& operator=(const PhiImageProcessor&) = delete;

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  std::string input_ids_name_;
  std::string pixel_values_name_;
  ONNXTensorElementDataType pixel_values_type_;
  std::string image_sizes_name_;
};

}  // namespace Generators
