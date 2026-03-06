// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "processor.h"
#include "ortx_processor.h"

namespace Generators {

struct Qwen3VLImageProcessor : Processor {
  Qwen3VLImageProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  int64_t spatial_merge_size_;  // 2 for Qwen3-VL
};

}  // namespace Generators
