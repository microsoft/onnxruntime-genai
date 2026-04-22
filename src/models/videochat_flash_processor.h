// Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
// Licensed under the MIT License. See License.txt in the project root for
// license information.

#pragma once

#include "model.h"
#include "processor.h"
#include "ortx_processor.h"

namespace Generators {

struct VideoChatFlashProcessor : Processor {
  VideoChatFlashProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  int64_t num_visual_tokens_;
};

}  // namespace Generators
