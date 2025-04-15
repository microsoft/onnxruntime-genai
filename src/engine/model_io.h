// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../models/decoder_only.h"
#include "scheduled_requests.h"

namespace Generators {

struct ModelIO {
  Decoder() = default;

  std::vector<const char*> input_names, output_names;
  std::vector<OrtValue*> inputs, outputs;
};

struct DecoderWithPagedAttention : ModelIO {
  DecoderWithPagedAttention(std::shared_ptr<DecoderOnly_Model> model,
                            ScheduledRequests& scheduled_requests)
      : model_{model},
        scheduled_requests_{scheduled_requests} {
    input_names = {
        model_->config_->model.decoder.inputs.input_ids.c_str(),
        model_->config_->model.decoder.inputs.position_ids.c_str(),
        model_->config_->model.decoder.inputs.cumulative_sequence_lengths.c_str(),
        model_->config_->model.decoder.inputs.sequence_lengths.c_str(),
        model_->config_->model.decoder.inputs.max_query_length.c_str(),
        model_->config_->model.decoder.inputs.max_sequence_length.c_str(),
        model_->config_->model.decoder.inputs.block_table.c_str(),
        model_->config_->model.decoder.inputs.slot_mapping.c_str()};
    output_names = {model_->config_->model.decoder.outputs.logits.c_str()};
  }

  std::shared_ptr<DecoderOnly_Model> model_;
};

}  // namespace Generators
