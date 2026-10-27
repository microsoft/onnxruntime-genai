// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <utility>

#include "models/io/token_type_ids.h"

namespace Generators {

TokenTypeIds::TokenTypeIds(State& state)
    : state_{state} {
  type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.token_type_ids);
  shape_ = {state_.params_->BatchBeamSize(), 0};
}

namespace {
// Zero marks every decoder position as text.
void ZeroTokenTypeIds(OrtValue& tensor) {
  auto info = tensor.GetTensorTypeAndShapeInfo();
  const size_t count = info->GetElementCount();
  if (info->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    std::fill_n(tensor.GetTensorMutableData<int64_t>(), count, int64_t{0});
  else
    std::fill_n(tensor.GetTensorMutableData<int32_t>(), count, int32_t{0});
}
}  // namespace

void TokenTypeIds::Add() {
  value_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape_, type_);
  ZeroTokenTypeIds(*value_);
  input_index_ = state_.inputs_.size();
  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(model_.config_->model.decoder.inputs.token_type_ids.c_str());
}

void TokenTypeIds::SetPending(std::span<const int32_t> markers) {
  pending_.assign(markers.begin(), markers.end());
}

void TokenTypeIds::Update(size_t sequence_length) {
  // Generated tokens are text; prompt updates reinstate the cached image markers once.
  if (shape_[1] != static_cast<int64_t>(sequence_length)) {
    shape_[1] = static_cast<int64_t>(sequence_length);
    value_ = OrtValue::CreateTensor(model_.allocator_cpu_, shape_, type_);
    state_.inputs_[input_index_] = value_.get();
  }
  UpdatePendingTokenTypeIds(*value_, pending_);
}

std::vector<int32_t> ReadTokenTypeIds(const OrtValue& token_type_ids) {
  const auto info = token_type_ids.GetTensorTypeAndShapeInfo();
  const size_t count = info->GetElementCount();
  if (count == 0) return {};

  switch (info->GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* values = token_type_ids.GetTensorData<int32_t>();
      return std::vector<int32_t>(values, values + count);
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* values = token_type_ids.GetTensorData<int64_t>();
      std::vector<int32_t> converted;
      converted.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        if (values[i] < std::numeric_limits<int32_t>::min() ||
            values[i] > std::numeric_limits<int32_t>::max()) {
          throw std::runtime_error("token_type_ids values must fit in int32");
        }
        converted.push_back(static_cast<int32_t>(values[i]));
      }
      return converted;
    }
    default:
      throw std::runtime_error("token_type_ids must have int32 or int64 elements");
  }
}

void UpdatePendingTokenTypeIds(OrtValue& token_type_ids, std::vector<int32_t>& pending_token_type_ids) {
  std::vector<int32_t> markers = std::move(pending_token_type_ids);
  pending_token_type_ids.clear();

  auto info = token_type_ids.GetTensorTypeAndShapeInfo();
  const size_t count = info->GetElementCount();
  const auto type = info->GetElementType();
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("token_type_ids input must be int32 or int64");
  }
  ZeroTokenTypeIds(token_type_ids);
  if (markers.empty()) return;
  if (markers.size() != count) {
    throw std::runtime_error("pending token_type_ids has " + std::to_string(markers.size()) +
                             " values but the decoder input has " + std::to_string(count) + " tokens");
  }

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto* dst = token_type_ids.GetTensorMutableData<int64_t>();
    std::copy(markers.begin(), markers.end(), dst);
  } else {
    auto* dst = token_type_ids.GetTensorMutableData<int32_t>();
    std::copy(markers.begin(), markers.end(), dst);
  }
}

}  // namespace Generators
