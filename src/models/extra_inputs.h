// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_api.h"

#include <memory>
#include <string>
#include <vector>

namespace Generators {

struct Model;
struct State;

struct ExtraInputs {
  ExtraInputs(const Model& model, State& state);
  ExtraInputs(const ExtraInputs&) = delete;
  ExtraInputs& operator=(const ExtraInputs&) = delete;

  void Add();

 private:
  const Model& model_;
  State& state_;
  std::vector<std::string> extra_input_names_;
  std::vector<std::shared_ptr<OrtValue>> extra_inputs_;
};

}  // namespace Generators
