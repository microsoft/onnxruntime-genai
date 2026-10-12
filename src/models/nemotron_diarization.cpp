// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_diarization.h"
#include "nemotron_speech.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Generators {
NemotronDiarizationState::NemotronDiarizationState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model, model.diarization_device_}, model_{model} {
  input_names_ = {"input_features", "num_lookahead_frames", "past_cache_embeds", "past_cache_probs",
                  "past_fifo", "past_num_cache_frames", "past_num_fifo_frames", "past_is_compressed"};
  output_names_ = {"speaker_probs", "present_cache_embeds", "present_cache_probs", "present_fifo",
                   "present_num_cache_frames", "present_num_fifo_frames", "present_is_compressed"};
  inputs_.resize(input_names_.size());
  outputs_.resize(output_names_.size());
  lookahead_ = OrtValue::CreateTensor<int64_t>(model.allocator_cpu_, std::span<const int64_t>{});
  if (model.config_->model.diarization.run_options)
    SetRunOptions(*model.config_->model.diarization.run_options);
  Reset();
}

void NemotronDiarizationState::Reset() {
  auto& allocator = model_.allocator_cpu_;
  const std::array<int64_t, 3> embeds{1, 264, 512}, probs{1, 264, 8};
  cache_[0] = OrtValue::CreateTensor<float>(allocator, embeds);
  cache_[1] = OrtValue::CreateTensor<float>(allocator, probs);
  cache_[2] = OrtValue::CreateTensor<float>(allocator, embeds);
  for (size_t i = 0; i < 3; ++i)
    std::fill_n(cache_[i]->GetTensorMutableData<float>(), cache_[i]->GetTensorTypeAndShapeInfo()->GetElementCount(), 0.0f);
  for (size_t i = 3; i < 5; ++i) {
    cache_[i] = OrtValue::CreateTensor<int64_t>(allocator, std::span<const int64_t>{});
    *cache_[i]->GetTensorMutableData<int64_t>() = 0;
  }
  cache_[5] = OrtValue::CreateTensor<bool>(allocator, std::span<const int64_t>{});
  *cache_[5]->GetTensorMutableData<bool>() = false;
  for (size_t i = 0; i < cache_.size(); ++i) inputs_[i + 2] = cache_[i].get();
  total_ = {};
  previous_frame_ = {};
  chunk_previous_ = {};
  boundaries_.clear();
}

DeviceSpan<float> NemotronDiarizationState::Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) {
  State::Run(*model_.session_diarization_);
  return {};
}

void NemotronDiarizationState::Process(OrtValue& features, int64_t lookahead) {
  const auto shape = features.GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != 128 || shape[2] <= lookahead * 8 || lookahead < 0)
    throw std::runtime_error("Invalid diarization feature window or look-ahead");
  inputs_[0] = &features;
  *lookahead_->GetTensorMutableData<int64_t>() = lookahead;
  inputs_[1] = lookahead_.get();
  for (size_t i = 0; i < cache_.size(); ++i) inputs_[i + 2] = cache_[i].get();
  DeviceSpan<int32_t> tokens;
  Run(0, tokens);
  std::array<std::unique_ptr<OrtValue>, 7> owned;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    owned[i].reset(outputs_[i]);
    outputs_[i] = nullptr;
  }
  const auto info = owned[0]->GetTensorTypeAndShapeInfo();
  const auto out_shape = info->GetShape();
  const int64_t central_frames = shape[2] - lookahead * 8;
  if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out_shape != std::vector<int64_t>{1, central_frames, 8})
    throw std::runtime_error("Diarization output must be [1, central_mel_frames, 8] float probabilities");
  auto cpu = OrtValue::CreateTensor<float>(model_.allocator_cpu_, out_shape);
  auto& cpu_device = *GetDeviceInterface(DeviceType::CPU);
  auto& source_device = owned[0]->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU
                            ? cpu_device
                            : *p_session_device_;
  ByteWrapTensor(cpu_device, *cpu).CopyFrom(ByteWrapTensor(source_device, *owned[0]));
  const auto* values = cpu->GetTensorData<float>();
  for (size_t i = 0; i < info->GetElementCount(); ++i)
    if (!std::isfinite(values[i]) || values[i] < 0 || values[i] > 1)
      throw std::runtime_error("Diarization returned an invalid speaker probability");
  chunk_previous_ = previous_frame_;
  boundaries_.clear();
  boundaries_.reserve(static_cast<size_t>(central_frames) + 1);
  boundaries_.push_back(total_);
  for (int64_t frame = 0; frame < central_frames; ++frame) {
    if (frame == central_frames - std::min<int64_t>(8, central_frames)) previous_frame_ = total_;
    for (size_t speaker = 0; speaker < 8; ++speaker) {
      const float probability = values[frame * 8 + speaker];
      total_.sums[speaker] += probability;
    }
    ++total_.count;
    boundaries_.push_back(total_);
  }
  for (size_t i = 0; i < cache_.size(); ++i) {
    cache_[i] = std::move(owned[i + 1]);
    inputs_[i + 2] = cache_[i].get();
  }
}

OgaTokenMetadataSpeakerEvidence NemotronDiarizationState::Evidence(int64_t local_frame) const {
  if (boundaries_.empty() || local_frame < 0 ||
      static_cast<uint64_t>(local_frame) >= (boundaries_.size() - 1 + 7) / 8)
    throw std::runtime_error("ASR token frame lies outside evaluated diarization audio");
  const size_t start = static_cast<size_t>(std::max<int64_t>(0, local_frame - 1) * 8);
  const size_t stop = std::min(static_cast<size_t>((local_frame + 1) * 8), boundaries_.size() - 1);
  const auto& left = local_frame == 0 ? chunk_previous_ : boundaries_[start];
  const auto& right = boundaries_[stop];
  OgaTokenMetadataSpeakerEvidence result{};
  std::copy(left.sums.begin(), left.sums.end(), result.start_probability_sums);
  std::copy(right.sums.begin(), right.sums.end(), result.stop_probability_sums);
  result.start_frame_count = left.count;
  result.stop_frame_count = right.count;
  return result;
}
}  // namespace Generators
