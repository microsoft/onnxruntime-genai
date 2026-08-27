// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct ExtraOutputs {
 public:
  ExtraOutputs(State& state);
  void Add(const std::vector<std::string>& all_output_names);
  void Update();
  void RegisterOutputs();

 private:
  State& state_;
  // manage output ortvalues not specified in output_names_
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> output_ortvalues_;
  std::vector<std::string> all_output_names_;  // keep output strings in scope
  size_t extra_outputs_start_{std::numeric_limits<size_t>::max()};
};

}  // namespace Generators