#pragma once

#include "static_buffer.h"

namespace Generators {

struct ExtraInputs {
  ExtraInputs(State& state);
  void Add();

 private:
  State& state_;
  const Model& model_{state_.model_};
  std::vector<OrtValue*> extra_inputs_;
  std::vector<std::unique_ptr<OrtValue>> owned_extra_inputs_;
  std::unordered_map<std::string, StaticBuffer*> sb_extra_inputs_;
};

}  // namespace Generators
