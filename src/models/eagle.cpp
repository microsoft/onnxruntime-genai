// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "eagle.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>

#include "utils.h"

namespace Generators {
namespace {

constexpr int kTreeTokens = 60;
constexpr int kTreeDepth = 7;
constexpr int kTreeTopK = 10;

std::unique_ptr<Config> CloneConfigForTarget(const Config& source) {
  return std::make_unique<Config>(source);
}

bool IsDynamicOr(int64_t actual, int64_t expected) {
  return actual < 0 || actual == expected;
}

void ValidateTensor(const SessionInfo& session_info, const std::string& name,
                    bool input, ONNXTensorElementDataType expected_type,
                    size_t expected_rank,
                    std::span<const std::pair<size_t, int64_t>> dimensions) {
  const bool exists = input ? session_info.HasInput(name) : session_info.HasOutput(name);
  const char* role = input ? "input" : "output";
  if (!exists)
    throw std::runtime_error("EAGLE " + std::string(role) + " '" + name +
                             "' was not found in the ONNX model.");

  const auto type = input ? session_info.GetInputDataType(name)
                          : session_info.GetOutputDataType(name);
  if (type != expected_type)
    throw std::runtime_error("EAGLE " + std::string(role) + " '" + name +
                             "' has an unsupported data type. Expected " +
                             TypeToString(expected_type) + ", got " + TypeToString(type) + ".");

  const auto shape = input ? session_info.GetInputShape(name)
                           : session_info.GetOutputShape(name);
  if (shape.size() != expected_rank)
    throw std::runtime_error("EAGLE " + std::string(role) + " '" + name +
                             "' must have rank " + std::to_string(expected_rank) +
                             ", got rank " + std::to_string(shape.size()) + ".");
  for (const auto& [axis, expected] : dimensions) {
    if (!IsDynamicOr(shape[axis], expected))
      throw std::runtime_error("EAGLE " + std::string(role) + " '" + name +
                               "' has dimension " + std::to_string(shape[axis]) +
                               " at axis " + std::to_string(axis) + ", expected " +
                               std::to_string(expected) + ".");
  }
}

ONNXTensorElementDataType ValidateEagleContract(const EagleModel& model) {
  const auto& config = *model.config_->model.eagle;
  const auto& target_config = model.config_->model.decoder;
  const auto& target_info = model.target_model().session_info_;
  if (config.target_hidden_state_names.size() != 3)
    throw std::runtime_error(
        "model.eagle.target_hidden_state_names must contain exactly three ordered target outputs.");
  if (config.hidden_size <= 0 || config.draft_vocab_size <= 0 ||
      config.num_key_value_heads <= 0 || config.head_size <= 0)
    throw std::runtime_error(
        "model.eagle hidden_size, draft_vocab_size, num_key_value_heads, and head_size must be positive.");
  if (config.total_tokens != kTreeTokens || config.depth != kTreeDepth ||
      config.top_k != kTreeTopK)
    throw std::runtime_error(
        "EAGLE v0 requires model.eagle total_tokens=60, depth=7, and top_k=10.");
  if (target_config.inputs.attention_bias.empty())
    throw std::runtime_error(
        "EAGLE tree decoding requires model.decoder.inputs.attention_bias.");

  ONNXTensorElementDataType data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (const auto& name : config.target_hidden_state_names) {
    if (!target_info.HasOutput(name))
      throw std::runtime_error("EAGLE target hidden-state output '" + name +
                               "' was not found in the target ONNX model.");
    const auto type = target_info.GetOutputDataType(name);
    if (type != Ort::TypeToTensorType<float> &&
        type != Ort::TypeToTensorType<Ort::BFloat16_t>)
      throw std::runtime_error("EAGLE target hidden-state output '" + name +
                               "' must have data type float32 or bfloat16.");
    if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
      data_type = type;
    else if (type != data_type)
      throw std::runtime_error("EAGLE target hidden-state outputs must use one data type.");
    const auto shape = target_info.GetOutputShape(name);
    if (shape.size() != 3 || !IsDynamicOr(shape[0], 1) ||
        !IsDynamicOr(shape[2], config.hidden_size))
      throw std::runtime_error("EAGLE target hidden-state output '" + name +
                               "' must have shape [1, sequence_length, hidden_size].");
  }

  const std::array<std::pair<size_t, int64_t>, 1> batch_one{
      std::pair<size_t, int64_t>{0, 1}};
  const std::array<std::pair<size_t, int64_t>, 2> bias_dims{
      std::pair<size_t, int64_t>{0, 1}, std::pair<size_t, int64_t>{1, 1}};
  ValidateTensor(target_info, target_config.inputs.attention_mask, true,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(target_info, target_config.inputs.position_ids, true,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(target_info, target_config.inputs.attention_bias, true,
                 data_type, 4, bias_dims);

  const auto& i = config.inputs;
  const auto& o = config.outputs;
  const auto& info = model.eagle_session_info_;
  const int64_t fused_width =
      static_cast<int64_t>(config.hidden_size) *
      static_cast<int64_t>(config.target_hidden_state_names.size());
  const std::array<std::pair<size_t, int64_t>, 2> target_dims{
      std::pair<size_t, int64_t>{0, 1}, std::pair<size_t, int64_t>{2, fused_width}};
  const std::array<std::pair<size_t, int64_t>, 2> hidden_dims{
      std::pair<size_t, int64_t>{0, 1},
      std::pair<size_t, int64_t>{2, config.hidden_size}};
  const std::array<std::pair<size_t, int64_t>, 3> cache_dims{
      std::pair<size_t, int64_t>{0, 1},
      std::pair<size_t, int64_t>{1, config.num_key_value_heads},
      std::pair<size_t, int64_t>{3, config.head_size}};
  const std::array<std::pair<size_t, int64_t>, 2> logits_dims{
      std::pair<size_t, int64_t>{0, 1},
      std::pair<size_t, int64_t>{2, config.draft_vocab_size}};
  const std::array<std::pair<size_t, int64_t>, 2> topk_dims{
      std::pair<size_t, int64_t>{0, 1},
      std::pair<size_t, int64_t>{2, config.top_k}};

  ValidateTensor(info, i.input_ids, true, Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(info, i.target_hidden_states, true, data_type, 3, target_dims);
  ValidateTensor(info, i.recurrent_hidden_states, true, data_type, 3, hidden_dims);
  ValidateTensor(info, i.use_target_hidden_states, true,
                 Ort::TypeToTensorType<bool>, 0, {});
  ValidateTensor(info, i.attention_mask, true,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(info, i.attention_bias, true,
                 Ort::TypeToTensorType<float>, 4, bias_dims);
  ValidateTensor(info, i.position_ids, true,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(info, i.past_key, true, data_type, 4, cache_dims);
  ValidateTensor(info, i.past_value, true, data_type, 4, cache_dims);

  ValidateTensor(info, o.draft_hidden_states, false, data_type, 3, hidden_dims);
  ValidateTensor(info, o.draft_logits, false, data_type, 3, logits_dims);
  ValidateTensor(info, o.draft_token_id, false,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(info, o.mapped_token_id, false,
                 Ort::TypeToTensorType<int64_t>, 2, batch_one);
  ValidateTensor(info, o.draft_topk_ids, false,
                 Ort::TypeToTensorType<int64_t>, 3, topk_dims);
  ValidateTensor(info, o.draft_topk_log_scores, false, data_type, 3, topk_dims);
  ValidateTensor(info, o.mapped_topk_ids, false,
                 Ort::TypeToTensorType<int64_t>, 3, topk_dims);
  ValidateTensor(info, o.present_key, false, data_type, 4, cache_dims);
  ValidateTensor(info, o.present_value, false, data_type, 4, cache_dims);

  const DeviceType device_type = model.p_device_->GetType();
  if (device_type == DeviceType::CPU && data_type != Ort::TypeToTensorType<float>)
    throw std::runtime_error("EAGLE CPU execution requires FP32 target and EAGLE graphs.");
  if (device_type == DeviceType::CUDA &&
      data_type != Ort::TypeToTensorType<Ort::BFloat16_t>)
    throw std::runtime_error("EAGLE CUDA execution requires BF16 target and EAGLE graphs.");
  if (device_type != DeviceType::CPU && device_type != DeviceType::CUDA)
    throw std::runtime_error("EAGLE v0 supports CPU/FP32 or CUDA/BF16 execution only.");
  return data_type;
}

std::unique_ptr<OrtValue> CreateTensor(OrtAllocator& allocator,
                                       std::span<const int64_t> shape,
                                       ONNXTensorElementDataType type) {
  return OrtValue::CreateTensor(allocator, shape, type);
}

template <typename T>
std::unique_ptr<OrtValue> MakeTensor(OrtAllocator& allocator,
                                     std::span<const int64_t> shape,
                                     std::span<const T> values) {
  auto tensor = OrtValue::CreateTensor<T>(allocator, shape);
  if (!values.empty())
    std::copy(values.begin(), values.end(), tensor->GetTensorMutableData<T>());
  return tensor;
}

std::unique_ptr<OrtValue> MakeZeroTensor(OrtAllocator& allocator,
                                         std::span<const int64_t> shape,
                                         ONNXTensorElementDataType type) {
  auto tensor = CreateTensor(allocator, shape, type);
  const size_t bytes =
      tensor->GetTensorTypeAndShapeInfo()->GetElementCount() * Ort::SizeOf(type);
  if (bytes != 0)
    std::memset(tensor->GetTensorMutableRawData(), 0, bytes);
  return tensor;
}

std::vector<int64_t> ReadInt64Tensor(const OrtValue& value,
                                     size_t expected_count,
                                     const char* name) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info->GetElementType() != Ort::TypeToTensorType<int64_t> ||
      info->GetElementCount() != expected_count)
    throw std::runtime_error(std::string("EAGLE ") + name +
                             " output has an invalid runtime shape or data type.");
  const int64_t* data = value.GetTensorData<int64_t>();
  return {data, data + expected_count};
}

std::vector<float> ReadScoreTensor(const OrtValue& value,
                                   size_t expected_count,
                                   ONNXTensorElementDataType type) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info->GetElementType() != type || info->GetElementCount() != expected_count)
    throw std::runtime_error(
        "EAGLE top-k score output has an invalid runtime shape or data type.");
  std::vector<float> result(expected_count);
  if (type == Ort::TypeToTensorType<float>) {
    std::copy_n(value.GetTensorData<float>(), expected_count, result.begin());
  } else {
    const auto* data = static_cast<const uint16_t*>(value.GetTensorRawData());
    for (size_t i = 0; i < expected_count; ++i)
      result[i] = BFloat16ToFloat32(data[i]);
  }
  return result;
}

float AccumulateTreeScore(float parent_score, float child_score,
                          ONNXTensorElementDataType type) {
  const float score = parent_score + child_score;
  if (type != Ort::TypeToTensorType<Ort::BFloat16_t>)
    return score;

  // AngelSlim accumulates in the graph dtype, so reproduce BF16 round-to-nearest-even
  // at every tree level rather than retaining an FP32 sum.
  uint32_t bits;
  std::memcpy(&bits, &score, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return BFloat16ToFloat32(static_cast<uint16_t>(bits >> 16));
}

std::unique_ptr<OrtValue> GatherRows(OrtAllocator& allocator,
                                     const OrtValue& source,
                                     std::span<const size_t> rows,
                                     int width,
                                     ONNXTensorElementDataType type) {
  const auto shape = source.GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[2] != width)
    throw std::runtime_error("EAGLE hidden-state tensor has an invalid runtime shape.");
  for (size_t row : rows) {
    if (row >= static_cast<size_t>(shape[1]))
      throw std::runtime_error("EAGLE hidden-state gather index is out of range.");
  }
  const std::array<int64_t, 3> result_shape{
      1, static_cast<int64_t>(rows.size()), width};
  auto result = CreateTensor(allocator, result_shape, type);
  const size_t row_bytes = static_cast<size_t>(width) * Ort::SizeOf(type);
  const auto* source_bytes = static_cast<const uint8_t*>(source.GetTensorRawData());
  auto* result_bytes = static_cast<uint8_t*>(result->GetTensorMutableRawData());
  for (size_t index = 0; index < rows.size(); ++index) {
    std::memcpy(result_bytes + index * row_bytes,
                source_bytes + rows[index] * row_bytes, row_bytes);
  }
  return result;
}

std::vector<size_t> SelectTopKIndices(std::span<const float> scores, size_t count) {
  if (count > scores.size())
    throw std::runtime_error("EAGLE top-k selection exceeds the score count.");
  std::vector<size_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), size_t{0});
  const auto better = [&](size_t left, size_t right) {
    return (std::isnan(scores[left]) && !std::isnan(scores[right])) ||
           scores[left] > scores[right];
  };
  // Mirror PyTorch's CPU TopKImpl so BF16 score ties reproduce the pinned oracle.
  if (count * 64 <= scores.size()) {
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), better);
  } else {
    std::nth_element(indices.begin(), indices.begin() + count - 1, indices.end(), better);
    if (count > 1)
      std::sort(indices.begin(), indices.begin() + count - 1, better);
  }
  indices.resize(count);
  return indices;
}

int32_t CheckedTargetToken(int64_t token, int vocab_size) {
  if (token < 0 || token >= vocab_size)
    throw std::runtime_error("EAGLE mapped token is outside the target vocabulary range.");
  return static_cast<int32_t>(token);
}

std::unique_ptr<OrtValue> CopyCachePrefix(OrtAllocator& allocator,
                                          const OrtValue& source,
                                          size_t length,
                                          ONNXTensorElementDataType type) {
  const auto shape = source.GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 4 || shape[0] != 1 || length > static_cast<size_t>(shape[2]))
    throw std::runtime_error("EAGLE cache rewind received an invalid length.");
  const std::array<int64_t, 4> result_shape{
      shape[0], shape[1], static_cast<int64_t>(length), shape[3]};
  auto result = CreateTensor(allocator, result_shape, type);
  const size_t row_bytes = static_cast<size_t>(shape[3]) * Ort::SizeOf(type);
  const size_t source_head_bytes = static_cast<size_t>(shape[2]) * row_bytes;
  const size_t result_head_bytes = length * row_bytes;
  const auto* source_bytes = static_cast<const uint8_t*>(source.GetTensorRawData());
  auto* result_bytes = static_cast<uint8_t*>(result->GetTensorMutableRawData());
  for (int64_t head = 0; head < shape[1]; ++head) {
    if (result_head_bytes != 0) {
      std::memcpy(result_bytes + static_cast<size_t>(head) * result_head_bytes,
                  source_bytes + static_cast<size_t>(head) * source_head_bytes,
                  result_head_bytes);
    }
  }
  return result;
}

}  // namespace

EagleTargetState::EagleTargetState(const EagleModel& model,
                                   DeviceSpan<int32_t> sequence_lengths,
                                   const GeneratorParams& params)
    : State{params, model.target_model()},
      eagle_model_{model},
      inner_{std::make_unique<DecoderOnly_State>(
          model.target_model(), sequence_lengths, params)} {}

std::unique_ptr<OrtValue> EagleTargetState::CaptureTargetFeatures() const {
  const auto& config = *eagle_model_.config_->model.eagle;
  const size_t feature_count = config.target_hidden_state_names.size();
  const size_t element_size = Ort::SizeOf(eagle_model_.data_type());
  size_t sequence_length = 0;
  std::vector<std::vector<uint8_t>> cpu_features;
  cpu_features.reserve(feature_count);

  for (const auto& name : config.target_hidden_state_names) {
    OrtValue* output = inner_->GetOutput(name.c_str());
    if (!output)
      throw std::runtime_error("Target state has no EAGLE feature output named '" + name + "'.");
    auto info = output->GetTensorTypeAndShapeInfo();
    const auto shape = info->GetShape();
    if (info->GetElementType() != eagle_model_.data_type() || shape.size() != 3 ||
        shape[0] != 1 || shape[2] != config.hidden_size || shape[1] < 0)
      throw std::runtime_error("Target EAGLE feature output '" + name +
                               "' has an invalid runtime shape or data type.");
    if (sequence_length == 0)
      sequence_length = static_cast<size_t>(shape[1]);
    else if (sequence_length != static_cast<size_t>(shape[1]))
      throw std::runtime_error(
          "Target EAGLE feature outputs have inconsistent sequence lengths.");

    auto source =
        ByteWrapTensor(*eagle_model_.target_model().p_device_inputs_, *output);
    auto cpu = source.CopyDeviceToCpu();
    cpu_features.emplace_back(cpu.begin(), cpu.end());
  }

  const size_t fused_width =
      static_cast<size_t>(config.hidden_size) * feature_count;
  const std::array<int64_t, 3> fused_shape{
      1, static_cast<int64_t>(sequence_length), static_cast<int64_t>(fused_width)};
  auto fused =
      CreateTensor(eagle_model_.allocator_cpu_, fused_shape, eagle_model_.data_type());
  auto* destination = static_cast<uint8_t*>(fused->GetTensorMutableRawData());
  const size_t source_row_bytes =
      static_cast<size_t>(config.hidden_size) * element_size;
  const size_t destination_row_bytes = fused_width * element_size;
  for (size_t row = 0; row < sequence_length; ++row) {
    for (size_t feature = 0; feature < feature_count; ++feature) {
      std::memcpy(destination + row * destination_row_bytes +
                      feature * source_row_bytes,
                  cpu_features[feature].data() + row * source_row_bytes,
                  source_row_bytes);
    }
  }
  return fused;
}

void EagleTargetState::RecordFeatures(
    size_t start, std::unique_ptr<OrtValue> features) {
  const auto shape = features->GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 3 || shape[1] < 0)
    throw std::runtime_error("EAGLE target features have an invalid shape.");
  const size_t count = static_cast<size_t>(shape[1]);

  auto first_replaced = feature_segments_.end();
  for (auto it = feature_segments_.begin(); it != feature_segments_.end(); ++it) {
    const size_t end = it->start + it->count;
    if (it->start >= start) {
      first_replaced = it;
      break;
    }
    if (end > start) {
      it->count = start - it->start;
      first_replaced = std::next(it);
      break;
    }
  }
  feature_segments_.erase(first_replaced, feature_segments_.end());
  feature_segments_.push_back(
      FeatureSegment{start, count, 0, std::move(features)});
}

DeviceSpan<float> EagleTargetState::Run(int total_length,
                                        DeviceSpan<int32_t>& next_tokens,
                                        DeviceSpan<int32_t> next_indices) {
  if (next_tokens.empty())
    throw std::runtime_error("EAGLE target execution requires at least one token.");
  if (total_length < static_cast<int>(next_tokens.size()))
    throw std::runtime_error("EAGLE target execution received an invalid total length.");
  auto logits = inner_->Run(total_length, next_tokens, next_indices);
  const size_t feature_start =
      static_cast<size_t>(total_length) - next_tokens.size();
  RecordFeatures(feature_start, CaptureTargetFeatures());
  stable_length_ = static_cast<size_t>(total_length);
  tree_features_.reset();
  return logits;
}

DeviceSpan<float> EagleTargetState::RunTree(
    DeviceSpan<int32_t>& tree_tokens,
    std::span<const int64_t> position_ids,
    std::span<const uint8_t> tree_mask) {
  auto logits = inner_->RunTree(
      static_cast<int>(stable_length_), tree_tokens, position_ids, tree_mask);
  tree_features_ = CaptureTargetFeatures();
  const auto shape = tree_features_->GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 3 || shape[1] != static_cast<int64_t>(tree_tokens.size()))
    throw std::runtime_error("EAGLE target tree features have an invalid sequence length.");
  return logits;
}

void EagleTargetState::CompactTree(std::span<const size_t> tree_indices) {
  if (!tree_features_)
    throw std::runtime_error("EAGLE target tree features are unavailable for compaction.");
  const auto& config = *eagle_model_.config_->model.eagle;
  const auto shape = tree_features_->GetTensorTypeAndShapeInfo()->GetShape();
  for (size_t index : tree_indices) {
    if (index >= static_cast<size_t>(shape[1]))
      throw std::runtime_error("EAGLE target feature compaction index is out of range.");
  }

  const size_t element_size = Ort::SizeOf(eagle_model_.data_type());
  const size_t fused_width =
      static_cast<size_t>(config.hidden_size) *
      config.target_hidden_state_names.size();
  const size_t row_bytes = fused_width * element_size;
  const std::array<int64_t, 3> selected_shape{
      1, static_cast<int64_t>(tree_indices.size()),
      static_cast<int64_t>(fused_width)};
  auto selected =
      CreateTensor(eagle_model_.allocator_cpu_, selected_shape, eagle_model_.data_type());
  const auto* source =
      static_cast<const uint8_t*>(tree_features_->GetTensorRawData());
  auto* destination = static_cast<uint8_t*>(selected->GetTensorMutableRawData());
  for (size_t row = 0; row < tree_indices.size(); ++row) {
    std::memcpy(destination + row * row_bytes,
                source + tree_indices[row] * row_bytes, row_bytes);
  }

  inner_->CompactTreeCache(stable_length_, tree_indices);
  RecordFeatures(stable_length_, std::move(selected));
  stable_length_ += tree_indices.size();
  tree_features_.reset();
}

std::unique_ptr<OrtValue> EagleTargetState::CopyFeatures(
    size_t start, size_t count) const {
  const auto& config = *eagle_model_.config_->model.eagle;
  const size_t fused_width =
      static_cast<size_t>(config.hidden_size) *
      config.target_hidden_state_names.size();
  const size_t row_bytes = fused_width * Ort::SizeOf(eagle_model_.data_type());
  const std::array<int64_t, 3> shape{
      1, static_cast<int64_t>(count), static_cast<int64_t>(fused_width)};
  auto result =
      CreateTensor(eagle_model_.allocator_cpu_, shape, eagle_model_.data_type());

  size_t cursor = start;
  auto* destination = static_cast<uint8_t*>(result->GetTensorMutableRawData());
  while (cursor < start + count) {
    const auto segment = std::find_if(
        feature_segments_.begin(), feature_segments_.end(),
        [cursor](const FeatureSegment& candidate) {
          return candidate.start <= cursor &&
                 cursor < candidate.start + candidate.count;
        });
    if (segment == feature_segments_.end())
      throw std::runtime_error(
          "EAGLE target features do not cover the requested stable-cache range.");

    const size_t available =
        segment->start + segment->count - cursor;
    const size_t rows = std::min(available, start + count - cursor);
    const size_t source_row =
        segment->value_offset + cursor - segment->start;
    std::memcpy(destination + (cursor - start) * row_bytes,
                static_cast<const uint8_t*>(segment->value->GetTensorRawData()) +
                    source_row * row_bytes,
                rows * row_bytes);
    cursor += rows;
  }
  return result;
}

void EagleTargetState::DiscardFeaturesBefore(size_t index) {
  while (!feature_segments_.empty() &&
         feature_segments_.front().start + feature_segments_.front().count <= index)
    feature_segments_.erase(feature_segments_.begin());

  if (!feature_segments_.empty() && feature_segments_.front().start < index) {
    auto& segment = feature_segments_.front();
    const size_t discarded = index - segment.start;
    segment.start = index;
    segment.count -= discarded;
    segment.value_offset += discarded;
  }
}

void EagleTargetState::RewindTo(size_t index) {
  inner_->RewindTo(index);
  stable_length_ = index;
  tree_features_.reset();
  auto first_discarded = feature_segments_.end();
  for (auto it = feature_segments_.begin(); it != feature_segments_.end(); ++it) {
    if (it->start >= index) {
      first_discarded = it;
      break;
    }
    if (it->start + it->count > index) {
      it->count = index - it->start;
      first_discarded = std::next(it);
      break;
    }
  }
  feature_segments_.erase(first_discarded, feature_segments_.end());
}

OrtValue* EagleTargetState::GetInput(const char* name) {
  return inner_->GetInput(name);
}
OrtValue* EagleTargetState::GetOutput(const char* name) {
  return inner_->GetOutput(name);
}
void EagleTargetState::SetActiveAdapter(Adapters* adapters,
                                        const std::string& adapter_name) {
  inner_->SetActiveAdapter(adapters, adapter_name);
}
void EagleTargetState::SetRunOption(const char* key, const char* value) {
  inner_->SetRunOption(key, value);
}
void EagleTargetState::SetExtraInputs(
    const std::vector<ExtraInput>& extra_inputs) {
  inner_->SetExtraInputs(extra_inputs);
}

EagleDraftState::EagleDraftState(const EagleModel& model)
    : model_{model},
      data_type_{model.data_type()},
      run_options_{OrtRunOptions::Create()} {
  const auto& config = *model_.config_->model.eagle;
  if (config.run_options) {
    for (const auto& [key, value] : *config.run_options)
      run_options_->AddConfigEntry(key.c_str(), value.c_str());
  }
  Reset();
}

EagleDraftState::RunResult EagleDraftState::Run(
    std::span<const int32_t> input_ids,
    std::unique_ptr<OrtValue> target_hidden_states,
    std::unique_ptr<OrtValue> recurrent_hidden_states,
    bool use_target_hidden_states,
    const OrtValue& past_key,
    const OrtValue& past_value,
    std::span<const int64_t> position_ids,
    std::span<const uint8_t> tree_mask,
    size_t stable_length) const {
  if (session_terminated_)
    throw std::runtime_error("EAGLE session was terminated.");
  const auto& config = *model_.config_->model.eagle;
  const size_t sequence_length = input_ids.size();
  if (sequence_length == 0 || position_ids.size() != sequence_length)
    throw std::runtime_error("EAGLE runtime received invalid token or position inputs.");

  const auto target_shape =
      target_hidden_states->GetTensorTypeAndShapeInfo()->GetShape();
  const auto recurrent_shape =
      recurrent_hidden_states->GetTensorTypeAndShapeInfo()->GetShape();
  const auto past_shape = past_key.GetTensorTypeAndShapeInfo()->GetShape();
  if (target_shape != std::vector<int64_t>({
                          1, static_cast<int64_t>(sequence_length),
                          static_cast<int64_t>(config.hidden_size) * 3}) ||
      recurrent_shape != std::vector<int64_t>({
                             1, static_cast<int64_t>(sequence_length),
                             config.hidden_size}) ||
      past_shape.size() != 4 || past_shape[0] != 1 ||
      past_shape[1] != config.num_key_value_heads ||
      past_shape[3] != config.head_size ||
      past_value.GetTensorTypeAndShapeInfo()->GetShape() != past_shape)
    throw std::runtime_error("EAGLE runtime tensor shapes are inconsistent.");
  const size_t past_length = static_cast<size_t>(past_shape[2]);
  const size_t total_length = past_length + sequence_length;
  if (stable_length > total_length)
    throw std::runtime_error("EAGLE runtime stable length exceeds the attention width.");
  const size_t branch_width = total_length - stable_length;
  if (!tree_mask.empty() && tree_mask.size() != sequence_length * branch_width)
    throw std::runtime_error("EAGLE runtime tree mask has an invalid shape.");

  std::vector<int64_t> input_ids_i64(input_ids.begin(), input_ids.end());
  std::vector<int64_t> attention_mask(total_length, 1);
  const std::array<int64_t, 2> token_shape{
      1, static_cast<int64_t>(sequence_length)};
  const std::array<int64_t, 2> mask_shape{1, static_cast<int64_t>(total_length)};
  const std::array<int64_t, 4> bias_shape{
      1, 1, static_cast<int64_t>(sequence_length),
      static_cast<int64_t>(total_length)};
  const std::array<bool, 1> selector{use_target_hidden_states};

  auto input_ids_value =
      MakeTensor<int64_t>(model_.allocator_cpu_, token_shape, input_ids_i64);
  auto selector_value =
      MakeTensor<bool>(model_.allocator_cpu_, {}, selector);
  auto mask_value =
      MakeTensor<int64_t>(model_.allocator_cpu_, mask_shape, attention_mask);
  auto bias_value =
      MakeZeroTensor(model_.allocator_cpu_, bias_shape, Ort::TypeToTensorType<float>);
  auto* bias = bias_value->GetTensorMutableData<float>();
  for (size_t query = 0; query < sequence_length && !tree_mask.empty(); ++query) {
    for (size_t key = 0; key < branch_width; ++key) {
      if (!tree_mask[query * branch_width + key])
        bias[query * total_length + stable_length + key] =
            std::numeric_limits<float>::lowest();
    }
  }
  auto position_value =
      MakeTensor<int64_t>(model_.allocator_cpu_, token_shape, position_ids);

  const auto& names = config.inputs;
  const std::array<const char*, 9> input_names{
      names.input_ids.c_str(), names.target_hidden_states.c_str(),
      names.recurrent_hidden_states.c_str(),
      names.use_target_hidden_states.c_str(), names.attention_mask.c_str(),
      names.attention_bias.c_str(), names.position_ids.c_str(),
      names.past_key.c_str(), names.past_value.c_str()};
  const std::array<OrtValue*, 9> inputs{
      input_ids_value.get(), target_hidden_states.get(),
      recurrent_hidden_states.get(), selector_value.get(), mask_value.get(),
      bias_value.get(), position_value.get(), const_cast<OrtValue*>(&past_key),
      const_cast<OrtValue*>(&past_value)};

  const std::array<int64_t, 3> hidden_shape{
      1, static_cast<int64_t>(sequence_length), config.hidden_size};
  const std::array<int64_t, 3> topk_shape{
      1, static_cast<int64_t>(sequence_length), config.top_k};
  const std::array<int64_t, 4> present_shape{
      1, config.num_key_value_heads, static_cast<int64_t>(total_length),
      config.head_size};
  RunResult result;
  result.hidden_states =
      CreateTensor(model_.allocator_cpu_, hidden_shape, data_type_);
  result.topk_log_scores =
      CreateTensor(model_.allocator_cpu_, topk_shape, data_type_);
  result.mapped_topk_ids =
      CreateTensor(model_.allocator_cpu_, topk_shape,
                   Ort::TypeToTensorType<int64_t>);
  result.key = CreateTensor(model_.allocator_cpu_, present_shape, data_type_);
  result.value = CreateTensor(model_.allocator_cpu_, present_shape, data_type_);

  const auto& output_config = config.outputs;
  const std::array<const char*, 5> output_names{
      output_config.draft_hidden_states.c_str(),
      output_config.draft_topk_log_scores.c_str(),
      output_config.mapped_topk_ids.c_str(),
      output_config.present_key.c_str(), output_config.present_value.c_str()};
  std::array<OrtValue*, 5> outputs{
      result.hidden_states.get(), result.topk_log_scores.get(),
      result.mapped_topk_ids.get(), result.key.get(), result.value.get()};
  model_.eagle_session_->Run(
      run_options_.get(), input_names.data(), inputs.data(), inputs.size(),
      output_names.data(), outputs.data(), outputs.size());
  return result;
}

void EagleDraftState::Prepare(
    std::unique_ptr<OrtValue> target_hidden_states,
    std::span<const int32_t> shifted_input_ids) {
  const auto& config = *model_.config_->model.eagle;
  if (shifted_input_ids.empty())
    throw std::runtime_error("EAGLE stable update requires at least one feature row.");
  const size_t sequence_length = shifted_input_ids.size();
  const auto target_shape =
      target_hidden_states->GetTensorTypeAndShapeInfo()->GetShape();
  if (target_hidden_states->GetTensorTypeAndShapeInfo()->GetElementType() != data_type_ ||
      target_shape != std::vector<int64_t>({
                          1, static_cast<int64_t>(sequence_length),
                          static_cast<int64_t>(config.hidden_size) * 3}))
    throw std::runtime_error("EAGLE stable target features have an invalid shape or data type.");

  const std::array<int64_t, 3> recurrent_shape{
      1, static_cast<int64_t>(sequence_length), config.hidden_size};
  auto recurrent =
      MakeZeroTensor(model_.allocator_cpu_, recurrent_shape, data_type_);
  std::vector<int64_t> positions(sequence_length);
  std::iota(positions.begin(), positions.end(),
            static_cast<int64_t>(cache_length_));
  RunResult result = Run(
      shifted_input_ids, std::move(target_hidden_states), std::move(recurrent),
      true, *key_, *value_, positions, {}, cache_length_);

  const std::array<size_t, 1> last_row{sequence_length - 1};
  last_hidden_state_ = GatherRows(
      model_.allocator_cpu_, *result.hidden_states, last_row,
      config.hidden_size, data_type_);
  const size_t topk_count = sequence_length * static_cast<size_t>(config.top_k);
  auto all_scores =
      ReadScoreTensor(*result.topk_log_scores, topk_count, data_type_);
  auto all_ids =
      ReadInt64Tensor(*result.mapped_topk_ids, topk_count, "mapped top-k");
  initial_scores_.assign(
      all_scores.end() - config.top_k, all_scores.end());
  initial_mapped_ids_.assign(
      all_ids.end() - config.top_k, all_ids.end());
  key_ = std::move(result.key);
  value_ = std::move(result.value);
  cache_length_ += sequence_length;
  conditioning_token_ = shifted_input_ids.back();
  initialized_ = true;
}

EagleTree EagleDraftState::BuildTree() const {
  if (!initialized_ || !last_hidden_state_ ||
      initial_scores_.size() != kTreeTopK ||
      initial_mapped_ids_.size() != kTreeTopK)
    throw std::runtime_error("EAGLE stable state is not initialized for tree drafting.");
  const auto& config = *model_.config_->model.eagle;

  std::vector<float> all_scores(initial_scores_.begin(), initial_scores_.end());
  std::vector<int64_t> all_tokens(initial_mapped_ids_.begin(),
                                  initial_mapped_ids_.end());
  std::vector<int64_t> all_parents{0};

  std::vector<float> active_scores(initial_scores_.begin(), initial_scores_.end());
  std::vector<size_t> selected_indices(kTreeTopK);
  std::iota(selected_indices.begin(), selected_indices.end(), size_t{0});
  std::vector<int32_t> input_ids(kTreeTopK);
  for (int index = 0; index < kTreeTopK; ++index)
    input_ids[index] =
        CheckedTargetToken(initial_mapped_ids_[index],
                           model_.config_->model.vocab_size);

  std::vector<size_t> repeated_rows(kTreeTopK, 0);
  auto input_hidden = GatherRows(
      model_.allocator_cpu_, *last_hidden_state_, repeated_rows,
      config.hidden_size, data_type_);
  std::vector<uint8_t> active_tree_mask(kTreeTopK * kTreeTopK);
  for (int index = 0; index < kTreeTopK; ++index)
    active_tree_mask[index * kTreeTopK + index] = 1;
  size_t active_tree_width = kTreeTopK;

  const OrtValue* past_key = key_.get();
  const OrtValue* past_value = value_.get();
  std::unique_ptr<OrtValue> transient_key;
  std::unique_ptr<OrtValue> transient_value;

  for (int level = 0; level < kTreeDepth; ++level) {
    const std::array<int64_t, 3> target_shape{
        1, kTreeTopK, static_cast<int64_t>(config.hidden_size) * 3};
    auto target =
        MakeZeroTensor(model_.allocator_cpu_, target_shape, data_type_);
    std::vector<int64_t> positions(kTreeTopK,
                                   static_cast<int64_t>(cache_length_ + level));
    RunResult result = Run(
        input_ids, std::move(target), std::move(input_hidden), false,
        *past_key, *past_value, positions, active_tree_mask, cache_length_);

    const size_t output_count = kTreeTopK * kTreeTopK;
    const auto mapped_ids =
        ReadInt64Tensor(*result.mapped_topk_ids, output_count, "mapped top-k");
    const auto topk_scores =
        ReadScoreTensor(*result.topk_log_scores, output_count, data_type_);
    std::vector<float> candidate_scores(output_count);
    for (int parent = 0; parent < kTreeTopK; ++parent) {
      for (int child = 0; child < kTreeTopK; ++child) {
        candidate_scores[parent * kTreeTopK + child] =
            AccumulateTreeScore(active_scores[parent], topk_scores[child],
                                data_type_);
      }
    }

    const int64_t parent_bias =
        1 + (level > 0 ? kTreeTopK : 0) +
        kTreeTopK * kTreeTopK * std::max(0, level - 1);
    for (size_t selected_index : selected_indices)
      all_parents.push_back(static_cast<int64_t>(selected_index) + parent_bias);
    all_scores.insert(all_scores.end(),
                      candidate_scores.begin(), candidate_scores.end());
    all_tokens.insert(all_tokens.end(), mapped_ids.begin(), mapped_ids.end());

    auto next_selected = SelectTopKIndices(candidate_scores, kTreeTopK);
    std::vector<size_t> parent_rows(kTreeTopK);
    std::vector<float> next_scores(kTreeTopK);
    std::vector<int32_t> next_ids(kTreeTopK);
    for (int index = 0; index < kTreeTopK; ++index) {
      parent_rows[index] = next_selected[index] / kTreeTopK;
      next_scores[index] = candidate_scores[next_selected[index]];
      next_ids[index] =
          CheckedTargetToken(mapped_ids[next_selected[index]],
                             model_.config_->model.vocab_size);
    }
    auto next_hidden = GatherRows(
        model_.allocator_cpu_, *result.hidden_states, parent_rows,
        config.hidden_size, data_type_);

    const size_t next_width = active_tree_width + kTreeTopK;
    std::vector<uint8_t> next_tree_mask(kTreeTopK * next_width);
    for (int row = 0; row < kTreeTopK; ++row) {
      const size_t parent = parent_rows[row];
      std::copy_n(active_tree_mask.data() + parent * active_tree_width,
                  active_tree_width,
                  next_tree_mask.data() + static_cast<size_t>(row) * next_width);
      next_tree_mask[static_cast<size_t>(row) * next_width +
                     active_tree_width + row] = 1;
    }

    input_hidden = std::move(next_hidden);
    input_ids = std::move(next_ids);
    active_scores = std::move(next_scores);
    selected_indices = std::move(next_selected);
    active_tree_mask = std::move(next_tree_mask);
    active_tree_width = next_width;
    transient_key = std::move(result.key);
    transient_value = std::move(result.value);
    past_key = transient_key.get();
    past_value = transient_value.get();
  }

  const size_t selected_draft_count = static_cast<size_t>(kTreeTokens - 1);
  auto top_indices = SelectTopKIndices(all_scores, selected_draft_count);
  std::sort(top_indices.begin(), top_indices.end());

  EagleTree tree;
  tree.tokens.reserve(kTreeTokens);
  tree.tokens.push_back(conditioning_token_);
  for (size_t index : top_indices)
    tree.tokens.push_back(
        CheckedTargetToken(all_tokens[index], model_.config_->model.vocab_size));
  tree.selected_candidate_indices = top_indices;

  std::vector<size_t> parent_rows(selected_draft_count);
  for (size_t index = 0; index < selected_draft_count; ++index) {
    const int64_t parent = all_parents[top_indices[index] / kTreeTopK];
    if (parent == 0) {
      parent_rows[index] = 0;
      continue;
    }
    const size_t parent_candidate = static_cast<size_t>(parent - 1);
    auto found = std::lower_bound(top_indices.begin(), top_indices.end(),
                                  parent_candidate);
    if (found == top_indices.end() || *found != parent_candidate)
      throw std::runtime_error(
          "EAGLE tree pruning selected a node without its parent.");
    parent_rows[index] =
        static_cast<size_t>(found - top_indices.begin()) + 1;
  }

  tree.attention_mask.assign(kTreeTokens * kTreeTokens, uint8_t{});
  std::vector<int64_t> depths(kTreeTokens);
  for (int row = 0; row < kTreeTokens; ++row) {
    tree.attention_mask[row * kTreeTokens + row] = 1;
    tree.attention_mask[row * kTreeTokens] = 1;
  }
  for (size_t index = 0; index < selected_draft_count; ++index) {
    uint8_t* destination =
        tree.attention_mask.data() + (index + 1) * kTreeTokens;
    const uint8_t* parent =
        tree.attention_mask.data() + parent_rows[index] * kTreeTokens;
    for (int column = 0; column < kTreeTokens; ++column)
      destination[column] = static_cast<uint8_t>(
          destination[column] || parent[column]);
  }
  for (int row = 0; row < kTreeTokens; ++row) {
    depths[row] = std::accumulate(
                      tree.attention_mask.begin() +
                          static_cast<ptrdiff_t>(row * kTreeTokens),
                      tree.attention_mask.begin() +
                          static_cast<ptrdiff_t>((row + 1) * kTreeTokens),
                      int64_t{0}) -
                  1;
  }
  tree.position_ids = depths;

  std::set<size_t> nonleaf(parent_rows.begin(), parent_rows.end());
  for (size_t node = 0; node < static_cast<size_t>(kTreeTokens); ++node) {
    if (nonleaf.contains(node))
      continue;
    std::vector<size_t> path(static_cast<size_t>(depths[node]) + 1);
    size_t current = node;
    for (size_t position = path.size(); position-- > 0;) {
      path[position] = current;
      if (current != 0)
        current = parent_rows[current - 1];
    }
    tree.retrieve_indices.push_back(std::move(path));
  }
  if (tree.tokens.size() != kTreeTokens || tree.retrieve_indices.empty())
    throw std::runtime_error("EAGLE tree construction produced an invalid topology.");
  return tree;
}

void EagleDraftState::RewindTo(size_t index) {
  if (index > cache_length_)
    throw std::runtime_error("EAGLE cache cannot rewind forward.");
  key_ = CopyCachePrefix(
      model_.allocator_cpu_, *key_, index, data_type_);
  value_ = CopyCachePrefix(
      model_.allocator_cpu_, *value_, index, data_type_);
  cache_length_ = index;
  last_hidden_state_.reset();
  initial_scores_.clear();
  initial_mapped_ids_.clear();
  conditioning_token_ = 0;
  initialized_ = false;
}

void EagleDraftState::Reset() {
  const auto& config = *model_.config_->model.eagle;
  const std::array<int64_t, 4> empty_shape{
      1, config.num_key_value_heads, 0, config.head_size};
  key_ = CreateTensor(model_.allocator_cpu_, empty_shape, data_type_);
  value_ = CreateTensor(model_.allocator_cpu_, empty_shape, data_type_);
  last_hidden_state_.reset();
  initial_scores_.clear();
  initial_mapped_ids_.clear();
  cache_length_ = 0;
  conditioning_token_ = 0;
  initialized_ = false;
}

void EagleDraftState::SetRunOption(const char* key, const char* value) {
  if (!key || !value)
    throw std::runtime_error(
        "EAGLE runtime option key and value must not be null.");
  if (std::strcmp(key, "terminate_session") == 0) {
    if (std::strcmp(value, "0") == 0) {
      session_terminated_ = false;
      run_options_->UnsetTerminate();
    } else if (std::strcmp(value, "1") == 0) {
      session_terminated_ = true;
      run_options_->SetTerminate();
    } else {
      throw std::runtime_error(
          std::string("terminate_session key value unexpected: ") + value);
    }
    return;
  }
  run_options_->AddConfigEntry(key, value);
}

EagleModel::EagleModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  if (!config_->model.eagle || config_->model.eagle->filename.empty())
    throw std::runtime_error(
        "model.eagle.filename is not set in genai_config.json.");
  if (config_->model.draft)
    throw std::runtime_error(
        "model.draft and model.eagle cannot both be configured.");

  const auto& eagle_config = *config_->model.eagle;
  if (!ProviderConfigurationMatches(
          config_->model.decoder.session_options,
          eagle_config.session_options))
    throw std::runtime_error(
        "Target and EAGLE must use the same execution provider. "
        "Cross-EP speculative decoding is not supported in this release.");
  ValidateSpeculativeModelCompatibility(*this, nullptr);

  target_model_ =
      std::make_shared<DecoderOnly_Model>(
          CloneConfigForTarget(*config_), ort_env);
  eagle_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(
      eagle_config.session_options, *eagle_session_options_, false, true);
  eagle_session_ = CreateSession(
      ort_env, eagle_config.filename, eagle_session_options_.get());
  eagle_session_info_.Add(*eagle_session_);
  data_type_ = ValidateEagleContract(*this);
  session_info_.Add(*target_model_->session_decoder_);
}

std::unique_ptr<State> EagleModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params) const {
  return std::make_unique<EagleState>(*this, sequence_lengths, params);
}

EagleState::EagleState(const EagleModel& model,
                       DeviceSpan<int32_t> sequence_lengths,
                       const GeneratorParams& params)
    : State{params, model},
      model_{model},
      target_state_{std::make_unique<EagleTargetState>(
          model, sequence_lengths, params)},
      draft_state_{model} {
  ValidateSpeculativeGeneratorParams(params);
  if (params.search.do_sample)
    throw std::runtime_error("EAGLE v0 supports greedy decoding only.");
  if (params.search.repetition_penalty != 1.0f ||
      params.search.min_length != 0 ||
      params.search.no_repeat_ngram_size != 0)
    throw std::runtime_error(
        "EAGLE v0 does not support repetition_penalty, min_length, or no_repeat_ngram_size.");
  const auto& eagle_config = *model_.config_->model.eagle;
  if (params.speculative.max_draft_tokens != eagle_config.depth + 1)
    throw std::runtime_error(
        "EAGLE v0 requires speculative.max_draft_tokens=8.");
  if (params.speculative.ngram_size != 0 ||
      params.speculative.adaptive_k_bool != 0 ||
      params.speculative.cooldown_bool != 0)
    throw std::runtime_error(
        "EAGLE v0 does not support n-gram lookup, adaptive K, or cooldown.");
  if (params.search.past_present_share_buffer || params.use_graph_capture)
    throw std::runtime_error(
        "EAGLE tree decoding does not support shared KV buffers or graph capture.");
  if (params.search.chunk_size.has_value() &&
      params.search.chunk_size.value() > 0)
    throw std::runtime_error(
        "EAGLE tree decoding does not support chunked target prefill.");
  if (model_.config_->model.decoder.sliding_window.has_value())
    throw std::runtime_error(
        "EAGLE tree decoding does not support sliding-window target caches.");
}

DeviceSpan<float> EagleState::Run(
    int total_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  return target_state_->Run(total_length, next_tokens, next_indices);
}

void EagleState::RewindTo(size_t index) {
  target_state_->RewindTo(index);
  if (index == 0) {
    draft_state_.Reset();
    return;
  }

  const size_t draft_length = std::min(index - 1, draft_state_.cache_length());
  if (draft_length < draft_state_.cache_length())
    draft_state_.RewindTo(draft_length);
}

OrtValue* EagleState::GetInput(const char* name) {
  return target_state_->GetInput(name);
}
OrtValue* EagleState::GetOutput(const char* name) {
  return target_state_->GetOutput(name);
}
void EagleState::SetActiveAdapter(
    Adapters* adapters, const std::string& adapter_name) {
  target_state_->SetActiveAdapter(adapters, adapter_name);
}
void EagleState::SetRunOption(const char* key, const char* value) {
  target_state_->SetRunOption(key, value);
  draft_state_.SetRunOption(key, value);
}
void EagleState::SetExtraInputs(
    const std::vector<ExtraInput>& extra_inputs) {
  target_state_->SetExtraInputs(extra_inputs);
}

}  // namespace Generators
