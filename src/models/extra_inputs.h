// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct Model;
struct State;

struct ExtraInputs {
  ExtraInputs(const Model& model, State& state);
  void Add();

 private:
  const Model& model_;
  State& state_;
  std::vector<std::string> extra_input_names_;
  std::vector<std::shared_ptr<OrtValue>> extra_inputs_;
};

}  // namespace Generators
