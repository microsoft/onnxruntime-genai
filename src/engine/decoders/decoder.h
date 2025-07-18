// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../scheduled_requests.h"
#include "../../models/model.h"
#include "../cache_manager.h"

namespace Generators {

struct ModelIO : State {
  ModelIO(const GeneratorParams& params, const Model& model) : State(params, model) {}

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override {
    return {};
  }

  void RewindTo(size_t index) override { (void)index; };

  OrtValue* GetOutput(const char* name) override { return nullptr; };
};

struct DecoderIO : ModelIO {
  DecoderIO(std::shared_ptr<Model> model,
            ScheduledRequests& scheduled_requests,
            std::shared_ptr<CacheManager> cache_manager)
      : ModelIO(*scheduled_requests.Params(), *model),
        scheduled_requests_{scheduled_requests},
        cache_manager_{cache_manager} {}

  virtual std::vector<DeviceSpan<float>> ProcessLogits() = 0;

 protected:
  ScheduledRequests& scheduled_requests_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::unique_ptr<Tensor> logits_;
};

struct Decoder {
  Decoder() = default;

  virtual void Decode(ScheduledRequests& scheduled_requests) = 0;

  virtual ~Decoder() = default;
};

}  // namespace Generators
