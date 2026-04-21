// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
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
    throw std::runtime_error("MoonshineStreamingProcessor requires a moonshine_streaming model type. Got: " +
                             model.config_->model.type);
  }
  config_ = moonshine_model->moonshine_config_;

  // Initialize VAD from config (if VAD section exists)
  InitVadFromConfig(model);
}

MoonshineStreamingProcessor::~MoonshineStreamingProcessor() = default;

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Process(const float* audio_data, size_t num_samples) {
  // Just accumulate audio - encoding happens on Flush()
  audio_buffer_.insert(audio_buffer_.end(), audio_data, audio_data + num_samples);
  return nullptr;
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Flush() {
  if (audio_buffer_.empty()) {
    return nullptr;
  }
  return EncodeAllAudio();
}

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::EncodeAllAudio() {
  auto* moonshine_model = dynamic_cast<MoonshineStreamingModel*>(&model_);
  if (!moonshine_model) {
    throw std::runtime_error("Model is not a MoonshineStreamingModel");
  }

  auto& allocator = model_.allocator_cpu_;
  const int chunk_samples = config_.chunk_samples;
  const int overlap_samples = config_.overlap_samples;

  // Pad audio to multiple of 80 (frame length used by Moonshine encoder)
  constexpr int FRAME_LEN = 80;
  size_t audio_len = audio_buffer_.size();
  size_t rem = audio_len % FRAME_LEN;
  if (rem) {
    audio_buffer_.resize(audio_len + FRAME_LEN - rem, 0.0f);
    audio_len = audio_buffer_.size();
  }

  // Encode in overlapping chunks
  std::vector<std::vector<float>> all_enc_data;
  std::vector<int64_t> all_enc_frames;

  int pos = 0;
  int chunk_idx = 0;
  while (pos < static_cast<int>(audio_len)) {
    int start = std::max(0, pos - (chunk_idx > 0 ? overlap_samples : 0));
    int end = std::min(pos + chunk_samples, static_cast<int>(audio_len));

    // Extract and pad chunk
    std::vector<float> chunk(audio_buffer_.begin() + start, audio_buffer_.begin() + end);
    size_t chunk_rem = chunk.size() % FRAME_LEN;
    if (chunk_rem) {
      chunk.resize(chunk.size() + FRAME_LEN - chunk_rem, 0.0f);
    }

    int64_t chunk_len = static_cast<int64_t>(chunk.size());

    // Create input tensors
    auto audio_shape = std::array<int64_t, 2>{1, chunk_len};
    auto audio_tensor = OrtValue::CreateTensor(allocator, audio_shape,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(audio_tensor->GetTensorMutableData<float>(), chunk.data(),
                chunk.size() * sizeof(float));

    auto mask_shape = std::array<int64_t, 2>{1, chunk_len};
    auto mask_tensor = OrtValue::CreateTensor(allocator, mask_shape,
                                              ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto* mask_data = mask_tensor->GetTensorMutableData<int64_t>();
    std::fill_n(mask_data, chunk_len, 1LL);

    // Run encoder
    const char* input_names[] = {"input_values", "attention_mask"};
    OrtValue* input_values[] = {audio_tensor.get(), mask_tensor.get()};
    const char* output_names[] = {"encoder_hidden_states"};
    OrtValue* output_values[] = {nullptr};

    moonshine_model->session_encoder_->Run(
        nullptr,
        input_names, input_values, 2,
        output_names, output_values, 1);

    auto enc_out = std::unique_ptr<OrtValue>(output_values[0]);
    auto enc_shape = enc_out->GetTensorTypeAndShapeInfo()->GetShape();
    // enc_shape: [1, enc_frames, hidden_size]
    int64_t enc_frames = enc_shape[1];
    int64_t hidden_size = enc_shape[2];

    const float* enc_data = enc_out->GetTensorData<float>();

    // Discard overlap frames from encoder output
    int64_t frames_to_skip = 0;
    if (chunk_idx > 0 && start < pos) {
      // overlap_samples in the raw audio space → how many encoder frames?
      int overlap_audio_samples = pos - start;
      // Encoder downsamples: ((audio_length // 80) - 1) // 4 + 1
      // We approximate: overlap_enc_frames = overlap_audio_samples / 320
      frames_to_skip = static_cast<int64_t>(overlap_audio_samples) / 320;
      if (frames_to_skip > enc_frames) frames_to_skip = enc_frames;
    }

    int64_t useful_frames = enc_frames - frames_to_skip;
    if (useful_frames > 0) {
      const float* src = enc_data + frames_to_skip * hidden_size;
      std::vector<float> frame_data(src, src + useful_frames * hidden_size);
      all_enc_data.push_back(std::move(frame_data));
      all_enc_frames.push_back(useful_frames);
    }

    pos += chunk_samples;
    chunk_idx++;
  }

  audio_buffer_.clear();

  // Concatenate all encoder outputs
  int64_t total_frames = 0;
  for (auto f : all_enc_frames) total_frames += f;

  if (total_frames == 0) {
    return nullptr;
  }

  int64_t hidden_size = config_.encoder_hidden_size;
  auto out_shape = std::array<int64_t, 3>{1, total_frames, hidden_size};
  auto full_enc = OrtValue::CreateTensor(allocator, out_shape,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  float* out_ptr = full_enc->GetTensorMutableData<float>();

  for (const auto& chunk_data : all_enc_data) {
    std::memcpy(out_ptr, chunk_data.data(), chunk_data.size() * sizeof(float));
    out_ptr += chunk_data.size();
  }

  // Wrap in NamedTensors
  auto result = std::make_unique<NamedTensors>();
  result->emplace("encoder_hidden_states",
                  std::make_shared<Tensor>(std::move(full_enc)));
  return result;
}

}  // namespace Generators
