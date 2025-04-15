// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_executor.h"

#include <typeinfo>

namespace Generators {

SimpleDecoderIO::SimpleDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                                 ScheduledRequests& scheduled_requests,
                                 std::shared_ptr<CacheManager> cache_manager)
    : State(scheduled_requests.Params(), *model) {
  input_names_ = {model_->config_->model.decoder.inputs.input_ids.c_str(),
                  model_->config_->model.decoder.inputs.position_ids.c_str(),
                  model_->config_->model.decoder.inputs.cumulative_sequence_lengths.c_str(),
                  model_->config_->model.decoder.inputs.sequence_lengths.c_str(),
                  model_->config_->model.decoder.inputs.max_query_length.c_str(),
                  model_->config_->model.decoder.inputs.max_sequence_length.c_str(),
                  model_->config_->model.decoder.inputs.block_table.c_str(),
                  model_->config_->model.decoder.inputs.slot_mapping.c_str()};
  output_names_ = {model_->config_->model.decoder.outputs.logits.c_str()};
}

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model)
    : model_{model} {}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  cache_manager_->Step();
  SimpleDecoderIO decoder_state(model_, scheduled_requests, cache_manager_);
  model_->session_decoder_->Run(decoder_state.run_options_.get(),
                                decoder_state.input_names_.data(),
                                decoder_state.inputs_.data(),
                                decoder_state.input_names_.size(),
                                decoder_state.output_names_.data(),
                                decoder_state.outputs_.data(),
                                decoder_state.output_names_.size());
}

}  // namespace Generators
