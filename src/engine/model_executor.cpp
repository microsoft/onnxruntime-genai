// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_executor.h"
#include "decoders/simple_decoder.h"

#include <string_view>
#include <typeinfo>

namespace Generators {

namespace {

ExecutionFailureKind ClassifyOrtExecutionFailure(std::string_view message) {
  if (message.find("Failed to allocate memory for requested buffer") !=
      std::string_view::npos) {
    return ExecutionFailureKind::CapacityExceeded;
  }
  return ExecutionFailureKind::Unknown;
}

std::unique_ptr<Decoder> CreateDecoder(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager) {
  if (auto decoder_only_model = std::dynamic_pointer_cast<DecoderOnly_Model>(model)) {
    return std::make_unique<SimpleDecoder>(decoder_only_model, cache_manager);
  }

  throw std::runtime_error("The model type is not supported for decoding. Expected a decoder-only model.");
}

}  // namespace

std::unique_ptr<ModelExecutor> ModelExecutor::Create(std::shared_ptr<Model> model,
                                                     std::shared_ptr<CacheManager> cache_manager) {
  return std::make_unique<DecoderModelExecutor>(model, cache_manager);
}

DecoderModelExecutor::DecoderModelExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : model_{model},
      cache_manager_{cache_manager},
      decoder_{CreateDecoder(model, cache_manager)} {}

void DecoderModelExecutor::Decode(ScheduledRequests& scheduled_requests,
                                  ExecutionContext& context) {
  try {
    cache_manager_->PrepareStep(scheduled_requests.Requests(), context);
    context.block_table_columns = cache_manager_->BlockTableColumns();
    decoder_->Decode(scheduled_requests, context);
  } catch (const Ort::Exception& error) {
    const auto failure_kind = ClassifyOrtExecutionFailure(error.what());
    if (failure_kind == ExecutionFailureKind::CapacityExceeded) {
      throw ModelExecutionError{
          failure_kind,
          std::string{"Model execution exceeded available memory. Cause: "} +
              error.what(),
      };
    }
    throw;
  }
}

}  // namespace Generators
