// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "model.h"
#include "processor.h"
#include "ortx_processor.h"

namespace Generators {

struct QwenImageProcessor : Processor {
  QwenImageProcessor(Config& config, const SessionInfo& session_info);

  virtual std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  int64_t spatial_merge_size_;
  int64_t patch_size_{14};  // Qwen2.5-VL uses 14, Qwen3-VL uses 16
  int64_t qwen4exp_position_grid_size_{};
  int64_t qwen4exp_head_dim_{};
  std::string qwen4exp_pos_embed_indices_name_;
  std::string qwen4exp_pos_embed_weights_name_;
  std::string qwen4exp_vision_cos_name_;
  std::string qwen4exp_vision_sin_name_;
  std::string qwen4exp_vision_attention_bias_name_;
};

}  // namespace Generators
