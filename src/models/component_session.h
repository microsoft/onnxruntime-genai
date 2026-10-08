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

struct ComponentCudaGraphState;
struct ComponentPackageResources;

struct ComponentPackageTokenizer {
  explicit ComponentPackageTokenizer(const fs::path& package_path);
  ~ComponentPackageTokenizer();
  ComponentPackageTokenizer(ComponentPackageTokenizer&&) noexcept;
  ComponentPackageTokenizer& operator=(ComponentPackageTokenizer&&) noexcept;
  ComponentPackageTokenizer(const ComponentPackageTokenizer&) = delete;
  ComponentPackageTokenizer& operator=(const ComponentPackageTokenizer&) = delete;

  std::vector<int32_t> Encode(const std::string& text) const;
  std::vector<std::vector<int32_t>> EncodeBatch(
      const std::vector<std::string>& texts) const;
  int32_t PadTokenId() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ComponentSession {
  ComponentSession(const fs::path& package_path, std::string component,
                   const std::vector<std::string>& providers);
  ~ComponentSession();
  std::vector<OgaComponentTensor> Run(const std::vector<OgaComponentInput>& inputs,
                                      const std::vector<std::string>& outputs);
  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::vector<OgaComponentInfo>& Inputs() const { return inputs_; }

 private:
  std::shared_ptr<ComponentPackageResources> package_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<ComponentCudaGraphState> cuda_graph_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OgaComponentInfo> inputs_;
  std::mutex mutex_;
};

}  // namespace Generators
