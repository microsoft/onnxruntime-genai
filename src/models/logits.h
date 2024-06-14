// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "static_buffer.h"

namespace Generators {

struct Logits {
  Logits(const Model& model, State& state);

  void Add();
  RoamingArray<float> Get();

  void Update();

 private:
  void HandleEOSArray(cpu_span<float> logits);

  const Model& model_;
  State& state_;
  size_t output_index_{~0U};

  std::array<int64_t, 3> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<OrtValue> output_to_search_;  // Logits that will be send to Search for token generation
  std::unique_ptr<OrtValue> raw_output_; // Raw logits output from model

  // Used for decoding runs with cuda graphs.
  StaticBuffer* sb_logits32_{};
  StaticBuffer* sb_logits16_{};

#if USE_CUDA
  cuda_unique_ptr<int32_t> cuda_eos_token_ids_ptr_;  // eos_token_ids from params, but in cuda accessible memory
  gpu_span<int32_t> cuda_eos_token_ids_;
#endif

#if USE_DML
  DmlReusedCommandListState logits_cast_command_list_state_{};
  std::unique_ptr<OrtValue> value32_cpu_;
#endif
};

}  // namespace Generators
