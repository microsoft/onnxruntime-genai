// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../scheduled_requests.h"
#include "../../models/model.h"
#include "../cache_manager.h"

namespace Generators {

/*
 * ModelIO is a base class for managing model inputs and outputs.
 * It inherits from State since its purpose is to maintain the model's state.
 * This class can be extended for specific model types that require input/output management.
 */
struct ModelIO : State {
  ModelIO(const GeneratorParams& params, const Model& model) : State(params, model) {}

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t> next_indices = {}) override {
    throw std::runtime_error("Unexpected call to ModelIO::Run, this function is not implemented for ModelIO.");
  }

  void RewindTo(size_t index) override {
    throw std::runtime_error("Unexpected call to ModelIO::RewindTo, this function is not implemented for ModelIO.");
  };

  OrtValue* GetOutput(const char* name) override {
    throw std::runtime_error("Unexpected call to ModelIO::GetOutput, this function is not implemented for ModelIO.");
  };
};

/*
 * DecoderIO is a base class for managing inputs and outputs specific to decoder models.
 * It extends ModelIO and adds functionality for handling scheduled requests and cache management.
 * This class can be further extended for different types of decoder models.
 * Classes inheriting from DecoderIO must implement the ProcessLogits method to return
 * the logits output from the model for each request in the scheduled batch.
 */
struct DecoderIO : ModelIO {
  DecoderIO(std::shared_ptr<Model> model,
            ScheduledRequests& scheduled_requests,
            std::shared_ptr<CacheManager> cache_manager)
      : ModelIO(*scheduled_requests.Params(), *model),
        scheduled_requests_{scheduled_requests},
        cache_manager_{cache_manager} {}

  /*
   * ProcessLogits processes the logits output from the model and returns a vector of DeviceSpan<float>,
   * where each DeviceSpan corresponds to the logits for a specific request in the scheduled batch.
   * The implementation of this method will depend on the specific decoder model and how it outputs logits.
   * This is a pure virtual function, so any class inheriting from DecoderIO must provide an implementation.
   */
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
