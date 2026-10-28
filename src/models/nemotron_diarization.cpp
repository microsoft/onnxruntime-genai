// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_diarization.h"
#include "nemotron_speech.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace Generators {

void ValidateNemotronDiarizationGraph(const Config::Model& config, const SessionInfo& info) {
  const auto require_tensor = [&](const char* name, ONNXTensorElementDataType type, size_t rank, bool output = false) {
    const auto shape = output ? info.GetOutputShape(name) : info.GetInputShape(name);
    const auto actual_type = output ? info.GetOutputDataType(name) : info.GetInputDataType(name);
    if (actual_type != type || shape.size() != rank)
      throw std::runtime_error(std::string{"Invalid diarization tensor type or rank: "} + name);
    return shape;
  };
  const auto single_batch = [](const std::vector<int64_t>& shape) {
    return shape[0] == -1 || shape[0] == 1;
  };

  const auto features = require_tensor("input_features", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
  if (!single_batch(features) || features[1] != config.diarization.num_mels || features[2] != -1)
    throw std::runtime_error("Diarization input_features must match declared num_mels and support dynamic time");

  const auto embeds = require_tensor("past_cache_embeds", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
  const auto probs = require_tensor("past_cache_probs", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
  const auto fifo = require_tensor("past_fifo", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 3);
  if (!single_batch(embeds) || !single_batch(probs) || !single_batch(fifo) ||
      embeds[1] <= 0 || embeds[2] <= 0 || fifo[1] < 0 || fifo[2] != embeds[2] ||
      probs[1] != embeds[1] || probs[2] != 8)
    throw std::runtime_error("Diarization requires fixed cache/FIFO capacities, positive speaker-cache capacity, matching embeddings, and eight speakers");

  const auto check_output = [&](const char* name, const std::vector<int64_t>& expected) {
    const auto actual = require_tensor(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, expected.size(), true);
    for (size_t i = 0; i < actual.size(); ++i)
      if (actual[i] != -1 && expected[i] != -1 && actual[i] != expected[i])
        throw std::runtime_error(std::string{"Diarization output shape does not match its input contract: "} + name);
    if (!single_batch(actual))
      throw std::runtime_error(std::string{"Diarization output must support batch one: "} + name);
  };
  check_output("present_cache_embeds", embeds);
  check_output("present_cache_probs", probs);
  check_output("present_fifo", fifo);
  check_output("speaker_probs", {1, -1, 8});

  for (const char* name : {"num_lookahead_frames", "past_num_cache_frames", "past_num_fifo_frames"})
    require_tensor(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 0);
  require_tensor("past_is_compressed", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 0);
  for (const char* name : {"present_num_cache_frames", "present_num_fifo_frames"})
    require_tensor(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 0, true);
  require_tensor("present_is_compressed", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 0, true);
}

NemotronDiarizationState::NemotronDiarizationState(const NemotronSpeechModel& model, const GeneratorParams& params)
    : State{params, model, model.diarization_device_}, model_{model} {
  ValidateTimestampConfiguration(model.config_->model);
  ValidateNemotronDiarizationGraph(model.config_->model, model.diarization_session_info_);
  const auto& config = model.config_->model;
  feature_frames_per_asr_frame_ = static_cast<int64_t>(config.hop_length) * config.subsampling_factor /
                                  config.diarization.hop_length;
  size_t cache_index = 0;
  for (const char* name : {"past_cache_embeds", "past_cache_probs", "past_fifo"}) {
    cache_shapes_[cache_index] = model.diarization_session_info_.GetInputShape(name);
    cache_shapes_[cache_index++][0] = 1;
  }

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
  for (size_t i = 0; i < cache_shapes_.size(); ++i) {
    cache_[i] = OrtValue::CreateTensor<float>(allocator, cache_shapes_[i]);
    std::fill_n(cache_[i]->GetTensorMutableData<float>(), cache_[i]->GetTensorTypeAndShapeInfo()->GetElementCount(), 0.0f);
  }
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
  central_feature_frames_ = 0;
  inputs_[0] = nullptr;
  inputs_[1] = lookahead_.get();
}

void NemotronDiarizationState::SetFeaturesInput(OrtValue& features, int64_t lookahead) {
  const auto& config = model_.config_->model.diarization;
  const auto shape = features.GetTensorTypeAndShapeInfo()->GetShape();
  if (lookahead < 0 || lookahead > config.right_context)
    throw std::runtime_error("Diarization look-ahead must be within the configured encoder-frame limit");
  const int64_t right_frames = lookahead * config.subsampling_factor;
  if (features.GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      shape.size() != 3 || shape[0] != 1 || shape[1] != config.num_mels || shape[2] <= right_frames)
    throw std::runtime_error("Invalid diarization feature window or look-ahead");

  inputs_[0] = &features;
  *lookahead_->GetTensorMutableData<int64_t>() = lookahead;
  inputs_[1] = lookahead_.get();
  central_feature_frames_ = shape[2] - right_frames;
}

DeviceSpan<float> NemotronDiarizationState::Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) {
  if (!inputs_[0])
    throw std::runtime_error("Diarization features must be set before Run");

  State::Run(*model_.session_diarization_);
  std::array<std::unique_ptr<OrtValue>, 7> owned;
  for (size_t i = 0; i < outputs_.size(); ++i) {
    owned[i].reset(outputs_[i]);
    outputs_[i] = nullptr;
  }

  const auto info = owned[0]->GetTensorTypeAndShapeInfo();
  const auto out_shape = info->GetShape();
  const int64_t central_frames = central_feature_frames_;
  if (info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out_shape != std::vector<int64_t>{1, central_frames, 8})
    throw std::runtime_error("Diarization output must be [1, central_mel_frames, 8] float probabilities");
  for (size_t i = 0; i < cache_shapes_.size(); ++i) {
    const auto cache_info = owned[i + 1]->GetTensorTypeAndShapeInfo();
    if (cache_info->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        cache_info->GetShape() != cache_shapes_[i])
      throw std::runtime_error("Diarization returned a cache inconsistent with the graph input shape");
  }

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
    if (frame == central_frames - std::min(feature_frames_per_asr_frame_, central_frames)) previous_frame_ = total_;
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
  inputs_[0] = nullptr;
  return {};
}

TokenSpeakerEvidence NemotronDiarizationState::GetTokenSpeakerEvidence(int64_t local_asr_frame) const {
  const size_t stride = static_cast<size_t>(feature_frames_per_asr_frame_);
  if (boundaries_.empty() || local_asr_frame < 0 ||
      static_cast<uint64_t>(local_asr_frame) >= (boundaries_.size() - 1 + stride - 1) / stride)
    throw std::runtime_error("ASR token frame lies outside evaluated diarization audio");
  const size_t start = static_cast<size_t>(std::max<int64_t>(0, local_asr_frame - 1)) * stride;
  const size_t stop = std::min(static_cast<size_t>(local_asr_frame + 1) * stride, boundaries_.size() - 1);
  const auto& left = local_asr_frame == 0 ? chunk_previous_ : boundaries_[start];
  const auto& right = boundaries_[stop];
  TokenSpeakerEvidence result{};
  std::copy(left.sums.begin(), left.sums.end(), result.start_probability_sums);
  std::copy(right.sums.begin(), right.sums.end(), result.stop_probability_sums);
  result.start_frame_count = left.count;
  result.stop_frame_count = right.count;
  return result;
}
}  // namespace Generators
