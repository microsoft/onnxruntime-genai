// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../config.h"

#include "onnxruntime_c_api.h"

#include <set>

namespace Generators {

struct ModelStateMetadata {
  virtual ~ModelStateMetadata() = default;

  virtual bool HasInput(const std::string& name) const = 0;
  virtual bool HasOutput(const std::string& name) const = 0;
  virtual ONNXTensorElementDataType GetInputDataType(const std::string& name) const = 0;
  virtual ONNXTensorElementDataType GetOutputDataType(const std::string& name) const = 0;
  virtual std::vector<int64_t> GetInputShape(const std::string& name) const = 0;
  virtual std::vector<int64_t> GetOutputShape(const std::string& name) const = 0;
};

class ModelStateManifest {
 public:
  explicit ModelStateManifest(const Config::Model::Decoder& decoder);

  static void ValidateConfig(const Config::Model::Decoder& decoder);
  static void ValidateDynamicEngineCompatibility(const Config::Model::Decoder& decoder);
  static void ValidatePagedCacheGeometry(
      const Config::Model::Decoder& decoder,
      const Config::Model::Decoder::StateGroup& paged_group,
      const ModelStateMetadata& metadata,
      size_t full_block_count,
      size_t window_block_count,
      const std::set<int>& windowed_layers,
      size_t block_size);
  void ValidateSession(const ModelStateMetadata& metadata) const;

  // Read-only access to the validated decoder state groups, in declaration order. Owners of a
  // specific kind (for example the fixed-state pool) select the groups they manage from this view.
  const std::vector<Config::Model::Decoder::StateGroup>& StateGroups() const { return state_groups_; }
  bool HasStateGroupKind(Config::Model::Decoder::StateGroupKind kind) const;
  bool HasFixedStateGroups() const;

 private:
  std::vector<Config::Model::Decoder::StateGroup> state_groups_;
  Config::Model::Decoder::Inputs inputs_;
  Config::Model::Decoder::Outputs outputs_;
};

}  // namespace Generators
