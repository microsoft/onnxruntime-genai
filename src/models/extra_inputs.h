#pragma once

#include "static_buffer.h"

namespace Generators {

struct ExtraInputs {
  ExtraInputs(const Model& model, State& state);
  void Add();

 private:
  const Model& model_;
  State& state_;
  std::vector<OrtValue*> extra_inputs_;
  std::vector<std::unique_ptr<OrtValue>> owned_extra_inputs_;
  std::unordered_map<std::string, StaticBuffer*> sb_extra_inputs_;
};

}  // namespace Generators
