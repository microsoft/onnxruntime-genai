// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "processor.h"

namespace Generators {

struct NemotronParseProcessor : Processor {
  NemotronParseProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer,
                                        const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;
  ONNXTensorElementDataType pixel_values_type_;
  int64_t target_height_;
  int64_t target_width_;
  int32_t decoder_start_token_id_;
};

}  // namespace Generators
