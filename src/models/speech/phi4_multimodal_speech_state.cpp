// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/speech/phi4_multimodal_speech_state.h"

#include <numeric>

#include "models/multi_modal.h"

namespace Generators {

int64_t Phi4MultimodalSpeechState::GetNumAudioTokens(const std::vector<ExtraInput>& extra_inputs) const {
  const auto& audio_sizes_name = model_.config_->model.speech.inputs.audio_sizes;
  for (const auto& input : extra_inputs) {
    if (input.name == audio_sizes_name) {
      assert(input.tensor->ort_tensor_);
      const auto info = input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("Unsupported data type " +
                                 std::to_string(static_cast<int64_t>(info->GetElementType())) +
                                 " for audio_sizes tensor. Only int64 is supported.");
      }
      const int64_t* data = input.tensor->ort_tensor_->GetTensorData<int64_t>();
      return std::accumulate(data, data + info->GetElementCount(), 0LL);
    }
  }
  return 0;
}

}  // namespace Generators
