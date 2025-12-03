// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"
#include "processor.h"
#include "ortx_processor.h"

namespace Generators {

struct Qwen2_5VLImageProcessor : Processor {
  Qwen2_5VLImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;
  std::string pixel_values_name_{"pixel_values"};
  std::string image_grid_thw_name_{"image_grid_thw"};
};

}  // namespace Generators
