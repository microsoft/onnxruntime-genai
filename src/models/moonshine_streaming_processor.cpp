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
  moonshine_model_ = dynamic_cast<MoonshineStreamingModel*>(&model);
  if (!moonshine_model_) {
    throw std::runtime_error("MoonshineStreamingProcessor requires a streaming_enc_dec_asr model type. Got: " +
                             model.config_->model.type);
  }
  config_ = moonshine_model_->moonshine_config_;

  // Initialize VAD from config (if vad.enabled = true).
  InitVadFromConfig(model);
}

MoonshineStreamingProcessor::~MoonshineStreamingProcessor() = default;

std::unique_ptr<NamedTensors> MoonshineStreamingProcessor::Process(const float* audio_data, size_t num_samples) {
  // VAD gating: drop silent chunks before buffering. Saves encoder work on Flush()
  // and keeps the prefix-padding / silence-duration semantics consistent with Nemotron.
  if (ShouldDropChunk(audio_data, num_samples)) {
    return nullptr;
  }
  // Accumulate audio - encoding happens on Flush().
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
  auto& allocator = model_.allocator_cpu_;
  const int chunk_samples = config_.chunk_samples;
  const int overlap_samples = config_.overlap_samples;

  // Moonshine encoder frontend hop = 80 samples (5ms @ 16kHz). Each chunk is
  // padded up to a multiple of FRAME_LEN inside the loop below, which covers
  // both the all-chunks-aligned case and the misaligned tail of the last chunk.
  constexpr int FRAME_LEN = 80;
  const size_t audio_len = audio_buffer_.size();

  // Encode in overlapping chunks. Accumulate useful (non-overlap) encoder frames into
  // a flat float buffer; concatenate into the final OrtValue once the total is known.
  std::vector<float> enc_buffer;
  int64_t total_frames = 0;

  int pos = 0;
  int chunk_idx = 0;
  while (pos < static_cast<int>(audio_len)) {
    int start = std::max(0, pos - (chunk_idx > 0 ? overlap_samples : 0));
    int end = std::min(pos + chunk_samples, static_cast<int>(audio_len));

    // Extract and pad chunk.
    std::vector<float> chunk(audio_buffer_.begin() + start, audio_buffer_.begin() + end);
    size_t chunk_rem = chunk.size() % FRAME_LEN;
    if (chunk_rem) {
      chunk.resize(chunk.size() + FRAME_LEN - chunk_rem, 0.0f);
    }

    int64_t chunk_len = static_cast<int64_t>(chunk.size());

    // Create input tensors.
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

    // Run encoder.
    const char* input_names[] = {"input_values", "attention_mask"};
    OrtValue* input_values[] = {audio_tensor.get(), mask_tensor.get()};
    const char* output_names[] = {"encoder_hidden_states"};
    OrtValue* output_values[] = {nullptr};

    moonshine_model_->session_encoder_->Run(
        nullptr,
        input_names, input_values, 2,
        output_names, output_values, 1);

    auto enc_out = std::unique_ptr<OrtValue>(output_values[0]);
    auto enc_shape = enc_out->GetTensorTypeAndShapeInfo()->GetShape();
    // enc_shape: [1, enc_frames, hidden_size]
    int64_t enc_frames = enc_shape[1];
    int64_t hidden_size = enc_shape[2];

    const float* enc_data = enc_out->GetTensorData<float>();

    // Discard overlap frames from this chunk's encoder output.
    int64_t frames_to_skip = 0;
    if (chunk_idx > 0 && start < pos) {
      // Encoder downsample factor ~= 320 (hop 80 × stride 4).
      int overlap_audio_samples = pos - start;
      frames_to_skip = static_cast<int64_t>(overlap_audio_samples) / 320;
      if (frames_to_skip > enc_frames) frames_to_skip = enc_frames;
    }

    int64_t useful_frames = enc_frames - frames_to_skip;
    if (useful_frames > 0) {
      const float* src = enc_data + frames_to_skip * hidden_size;
      enc_buffer.insert(enc_buffer.end(), src, src + useful_frames * hidden_size);
      total_frames += useful_frames;
    }

    pos += chunk_samples;
    chunk_idx++;
  }

  audio_buffer_.clear();

  if (total_frames == 0) {
    return nullptr;
  }

  // Concatenate accumulated frames into the final encoder_hidden_states tensor.
  const int64_t hidden_size = config_.encoder_hidden_size;
  auto out_shape = std::array<int64_t, 3>{1, total_frames, hidden_size};
  auto full_enc = OrtValue::CreateTensor(allocator, out_shape,
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::memcpy(full_enc->GetTensorMutableData<float>(),
              enc_buffer.data(),
              enc_buffer.size() * sizeof(float));

  auto result = std::make_unique<NamedTensors>();
  result->emplace("encoder_hidden_states",
                  std::make_shared<Tensor>(std::move(full_enc)));
  return result;
}

}  // namespace Generators
