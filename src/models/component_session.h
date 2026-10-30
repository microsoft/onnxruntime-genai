// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../filesystem.h"
#include "../generator/generators.h"
#include "onnxruntime_api.h"

namespace Generators {

struct Config;

class ComponentPackage {
 public:
  explicit ComponentPackage(const fs::path& package_path);
  bool Has(std::string_view component) const;
  const fs::path& Path(std::string_view component) const;

 private:
  std::unordered_map<std::string, fs::path> components_;
};

struct ComponentInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<std::string> symbolic_dimensions;
  ONNXTensorElementDataType type{};
};

struct ComponentSession {
  ComponentSession(const Config& config, const ComponentPackage& package,
                   std::string component);
  ~ComponentSession();
  NamedTensors Run(const NamedTensors& inputs,
                   const std::vector<std::string>& outputs = {});
  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::vector<ComponentInfo>& Inputs() const { return inputs_; }
  bool UsesCuda() const { return uses_cuda_; }

 private:
  std::unique_ptr<OrtSession> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<ComponentInfo> inputs_;
  bool uses_cuda_{};
  std::mutex mutex_;
};

}  // namespace Generators
