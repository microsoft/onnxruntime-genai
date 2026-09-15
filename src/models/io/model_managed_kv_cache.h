// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "kv_cache.h"

namespace Generators {

// A (mostly) NO-OP KeyValueCache variant that is used for stateful models
// i.e. Models that manage KV Cache internally to the session.
struct ModelManagedKeyValueCache : KeyValueCache {
  ModelManagedKeyValueCache(State& state);

  virtual void Add() override;
  virtual void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  virtual void RewindTo(size_t index) override;
  virtual bool IsModelManaged() const override { return true; }

 private:
  State& state_;
  const Model& model_{state_.model_};
};

}  // namespace Generators
