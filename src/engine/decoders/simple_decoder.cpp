// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"
#include "static_batch_decoder_io.h"
#include "varlen_decoder_io.h"

namespace Generators {

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model,
                             std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  cache_manager_->Step();
  std::unique_ptr<DecoderIO> decoder_state =
      cache_manager_->SupportsDynamicBatching()
          ? static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<VarlenDecoderIO>(model_, scheduled_requests, cache_manager_))
          : static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<StaticBatchDecoderIO>(model_, scheduled_requests, cache_manager_));

  auto run_options = scheduled_requests.RunOptions();
  decoder_state->DumpInputs();
  model_->session_decoder_->Run(run_options.get(),
                                decoder_state->input_names_.data(),
                                decoder_state->inputs_.data(),
                                decoder_state->input_names_.size(),
                                decoder_state->output_names_.data(),
                                decoder_state->outputs_.data(),
                                decoder_state->output_names_.size());
  decoder_state->DumpOutputs();

  scheduled_requests.AddDecoderState(std::move(decoder_state));
}

}  // namespace Generators
