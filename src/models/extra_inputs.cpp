#include "../generators.h"
#include "model.h"
#include "extra_inputs.h"

namespace Generators {

PresetExtraInputs::PresetExtraInputs(State& state)
    : state_(state),
      registry_{
          {"num_logits_to_keep", [&state = state_]() -> std::unique_ptr<OrtValue> {
             std::vector<int64_t> shape{1};
             auto num_logits_to_keep = OrtValue::CreateTensor<int64_t>(state.model_.allocator_cpu_, shape);
             *num_logits_to_keep->GetTensorMutableData<int64_t>() = 0;
             return num_logits_to_keep;
           }}} {}

void PresetExtraInputs::Add() {
  const auto input_names_vector = state_.model_.session_info_->GetInputNames();
  const std::unordered_set<std::string> input_names(state_.input_names_.begin(), state_.input_names_.end());
  std::vector<std::string> unclaimed_input_names;
  // Add any model input for which we don't have a corresponding input in the state to the unclaimed_input_names
  for (const auto& input_name : input_names_vector) {
    if (input_names.find(input_name) == input_names.end()) {
      unclaimed_input_names.push_back(input_name);
    }
  }

  // Try to claim the unclaimed inputs from the registry
  for (const auto& input_name : unclaimed_input_names) {
    auto it = registry_.find(input_name);
    if (it != registry_.end()) {
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(it->second());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    } else if (input_name.rfind("onnx::Neg_", 0) == 0) {
      // The unclaimed input has a prefix of onnx::Neg_, which is a special case
      // We treat this as an alias to num_logits_to_keep
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(registry_.at("num_logits_to_keep")());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    }
  }
}

ExtraInputs::ExtraInputs(State& state)
    : state_{state} {
  extra_inputs_.reserve(state_.params_->extra_inputs.size());

  if (state_.GetCapturedGraphInfo()) {
    owned_extra_inputs_.reserve(state_.params_->extra_inputs.size());

    for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
      auto type_and_shape_info = state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      const auto& input_name = state_.params_->extra_inputs[i].name;

      sb_extra_inputs_.emplace(input_name, state_.GetCapturedGraphInfo()->sb_extra_inputs_.at(input_name).get());
      owned_extra_inputs_.push_back(sb_extra_inputs_.at(input_name)->CreateTensorOnStaticBuffer(type_and_shape_info->GetShape(), type_and_shape_info->GetElementType()));
      extra_inputs_.push_back(owned_extra_inputs_.back().get());
    }
  } else {
    // We don't use graph capture, so simply use the existing pointers
    for (auto& extra_input : state_.params_->extra_inputs) {
      extra_inputs_.push_back(extra_input.tensor->ort_tensor_.get());
    }
  }
}

void ExtraInputs::Add(const std::vector<std::string>& required_input_names) {
  std::unordered_set<std::string> required_input_names_set(required_input_names.begin(), required_input_names.end());
  // Add extra user inputs
  for (int i = 0; i < state_.params_->extra_inputs.size(); ++i) {
    if (required_input_names_set.empty() || required_input_names_set.count(state_.params_->extra_inputs[i].name)) {
      state_.input_names_.push_back(state_.params_->extra_inputs[i].name.c_str());
      state_.inputs_.push_back(extra_inputs_[i]);
    }
  }

  // Copy the data from the CPU-backed ORT value to the static buffers
  for (int i = 0; i < sb_extra_inputs_.size(); ++i) {
    auto tensor = ByteWrapTensor(*model_.p_device_, *extra_inputs_[i]);
    auto source = std::span{state_.params_->extra_inputs[i].tensor->ort_tensor_->GetTensorData<uint8_t>(), tensor.size()};
    copy(source, tensor.CpuSpan());
    tensor.CopyCpuToDevice();
  }

  registrar_.Add();
}

}  // namespace Generators
