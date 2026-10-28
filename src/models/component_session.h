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
#include "onnxruntime_api.h"
#include "preprocessing/genai_tokenizer.h"

namespace Generators {

bool PackageHasComponent(const fs::path& package_path,
                         std::string_view component);

struct ComponentInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<std::string> symbolic_dimensions;
  ONNXTensorElementDataType type{};
};

struct ComponentSession {
  ComponentSession(const fs::path& package_path, std::string component,
                   const std::vector<std::string>& providers);
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

class PackageTokenizer {
 public:
  explicit PackageTokenizer(const fs::path& package_path);
  std::vector<int32_t> Encode(const std::string& text) const;
  int32_t PadTokenId() const { return pad_token_id_; }
  size_t ContextLength() const { return context_length_; }

 private:
  Config config_;
  Tokenizer tokenizer_;
  int32_t pad_token_id_{};
  size_t context_length_{8192};
};

}  // namespace Generators
