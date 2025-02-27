#pragma once

#include "static_buffer.h"

namespace Generators {

struct PresetExtraInputs {
  PresetExtraInputs(State& state);
  void Add();

 private:
  using FuncType = std::function<std::unique_ptr<OrtValue>()>;
  State& state_;
  std::unordered_map<std::string, FuncType> registry_;
  std::vector<std::unique_ptr<OrtValue>> extra_inputs_;
  std::vector<std::string> extra_input_names_;
};

struct ExtraInputs {
  ExtraInputs(State& state);
  void Add(const std::vector<std::string>& required_input_names = {});

 private:
  State& state_;
  const Model& model_{state_.model_};
  std::vector<OrtValue*> extra_inputs_;
  std::vector<std::unique_ptr<OrtValue>> owned_extra_inputs_;
  std::unordered_map<std::string, StaticBuffer*> sb_extra_inputs_;
  PresetExtraInputs registrar_{state_};
};

}  // namespace Generators
