#pragma once

#include "static_buffer.h"
#include "../tensor.h"

namespace Generators {

struct ExtraInputs {
  ExtraInputs(const Model& model, State& state);
  void Add();

 private:
  const Model& model_;
  State& state_;
  std::vector<const char*> lora_input_names_;
  std::vector<std::shared_ptr<Tensor>> lora_tensors_;
  // The actual ordered vector of extra inputs that may either
  // come from lora_tensors_ or from owned_extra_inputs
  std::vector<OrtValue*> extra_inputs_;
  // device_buffers and owned_extra_inputs_
  // are only utilized for CUDA and DML devices
  std::vector<StaticBuffer> device_buffers_;
  std::vector<std::unique_ptr<OrtValue>> owned_extra_inputs_;
};

}  // namespace Generators
