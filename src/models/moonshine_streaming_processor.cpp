// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../generators.h"
#include "moonshine_streaming_processor.h"

namespace Generators {

MoonshineStreamingProcessor::MoonshineStreamingProcessor(Model& model)
    : model_{model} {
  auto* moonshine_model = dynamic_cast<MoonshineStreamingModel*>(&model);
  if (!moonshine_model) {
    throw std::runtime_error(
        "MoonshineStreamingProcessor requires a streaming_enc_dec_asr model type. Got: " +
        model.config_->model.type);
  }
  config_ = moonshine_model->moonshine_config_;
  InitVadFromConfig(model);
}

MoonshineStreamingProcessor::~MoonshineStreamingProcessor() = default;

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::EmitChunk(const float* audio, size_t num,
                                                                     bool is_silent, bool is_final) {
  auto& alloc = model_.allocator_cpu_;
  auto result = std::make_unique<NamedTensors>();

  // audio_chunk: float32 [1, num] (num may be 0 on the Flush tail).
  auto audio_shape = std::array<int64_t, 2>{1, static_cast<int64_t>(num)};
  auto audio_tensor =
      OrtValue::CreateTensor(alloc, audio_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  if (num > 0) {
    std::memcpy(audio_tensor->GetTensorMutableData<float>(), audio, num * sizeof(float));
  }
  result->emplace("audio_chunk", std::make_shared<Tensor>(std::move(audio_tensor)));

  auto flag_shape = std::array<int64_t, 1>{1};

  // is_silent: raw per-chunk VAD verdict (State uses it for segmentation).
  auto silent_tensor =
      OrtValue::CreateTensor(alloc, flag_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *silent_tensor->GetTensorMutableData<int64_t>() = is_silent ? 1 : 0;
  result->emplace("is_silent", std::make_shared<Tensor>(std::move(silent_tensor)));

  // is_final: 1 only on the Flush tail so the State releases held-back
  // lookahead and commits the full segment.
  auto final_tensor =
      OrtValue::CreateTensor(alloc, flag_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  *final_tensor->GetTensorMutableData<int64_t>() = is_final ? 1 : 0;
  result->emplace("is_final", std::make_shared<Tensor>(std::move(final_tensor)));

  return result;
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Process(const float* audio_data,
                                                                   size_t num_samples) {
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);

  const size_t chunk_size = static_cast<size_t>(config_.chunk_samples);
  if (audio_buffer_.size() < chunk_size) {
    return nullptr;  // Not enough audio yet for a streaming chunk.
  }

  // Drain exactly one chunk; leftover stays buffered for the next call.
  std::vector<float> chunk(audio_buffer_.begin(),
                           audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_size));
  audio_buffer_.erase(audio_buffer_.begin(),
                      audio_buffer_.begin() + static_cast<ptrdiff_t>(chunk_size));

  // Raw per-chunk silence verdict (false if VAD is disabled). The State
  // combines this with its own accumulated memory length to segment.
  const bool is_silent = IsChunkSilent(chunk.data(), chunk.size());
  return EmitChunk(chunk.data(), chunk.size(), is_silent, /*is_final=*/false);
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Flush() {
  // Emit the tail (possibly shorter than a chunk, or empty) with is_final=1 so
  // the State releases its held-back lookahead and commits the full segment.
  std::vector<float> tail = std::move(audio_buffer_);
  audio_buffer_.clear();
  const bool is_silent = tail.empty() ? false : IsChunkSilent(tail.data(), tail.size());
  return EmitChunk(tail.data(), tail.size(), is_silent, /*is_final=*/true);
}

}  // namespace Generators
