// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../filesystem.h"
#include "../ort_genai.h"
#include "onnxruntime_api.h"

namespace Generators {

struct ComponentSession {
  ComponentSession(const fs::path& package_path, std::string component,
                   const std::vector<std::string>& providers);
  std::vector<OgaComponentTensor> Run(const std::vector<OgaComponentInput>& inputs,
                                      const std::vector<std::string>& outputs);
  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::vector<OgaComponentInfo>& Inputs() const { return inputs_; }

 private:
  std::unique_ptr<OrtSession> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OgaComponentInfo> inputs_;
  std::mutex mutex_;
};

}  // namespace Generators
