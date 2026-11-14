// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "engine.h"

namespace Generators {

struct NemotronBoltModel;
struct NemotronBoltExecutionState {
  virtual ~NemotronBoltExecutionState() = default;
  virtual bool PrefixCachingEnabled() const = 0;
  virtual size_t Prefill(std::span<const int32_t> tokens, int max_length, PrefixCacheMetrics& events) = 0;
  virtual int32_t NextToken(size_t public_length) = 0;
  virtual void Reset() = 0;
  virtual SpeculativeStats Stats() const = 0;
};

EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<NemotronBoltModel> model);
EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<Model> model,
                                                  std::unique_ptr<NemotronBoltExecutionState> state);

}  // namespace Generators
