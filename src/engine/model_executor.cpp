// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_executor.h"

#include <typeinfo>

namespace Generators {

namespace {

std::unique_ptr<Decoder> CreateDecoder(std::shared_ptr<Model> model) {
  if (auto decoder_only_model = std::dynamic_pointer_cast<DecoderOnly_Model>(model)) {
    return std::make_unique<SimpleDecoder>(decoder_only_model);
  } else {
    throw std::runtime_error("Continuous batching does not support the requested model type.");
  }
  return nullptr;
}

}  // namespace

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model)
    : model_{model} {}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  std::vector<const char*>
      input_names{
          model_->config_->model.decoder.inputs.input_ids.c_str(),
          model_->config_->model.decoder.inputs.position_ids.c_str(),
          model_->config_->model.decoder.inputs.cumulative_sequence_lengths.c_str(),
          model_->config_->model.decoder.inputs.sequence_lengths.c_str(),
          model_->config_->model.decoder.inputs.max_query_length.c_str(),
          model_->config_->model.decoder.inputs.max_sequence_length.c_str(),
          model_->config_->model.decoder.inputs.block_table.c_str(),
          model_->config_->model.decoder.inputs.slot_mapping.c_str()},
      output_names{
          model_->config_->model.decoder.outputs.logits.c_str(),
      };
  std::vector<OrtValue*>
      inputs{
          scheduled_requests.InputIds(),
          scheduled_requests.PositionIds(),
          scheduled_requests.CumulativeSequenceLengths(),
          scheduled_requests.SequenceLengths(),
          scheduled_requests.MaxQueryLength(),
          scheduled_requests.MaxSequenceLength(),
          scheduled_requests.BlockTable(),
          scheduled_requests.SlotMapping()},
      outputs{
          scheduled_requests.Logits()};

  std::vector<OrtValue*> key_caches = scheduled_requests.KeyCaches();
  std::vector<OrtValue*> value_caches = scheduled_requests.ValueCaches();
  std::vector<std::string> key_caches_names, value_caches_names;

  for (size_t i = 0; i < key_caches.size(); ++i) {
    key_caches_names.push_back(
        ComposeKeyValueName(model_->config_->model.decoder.inputs.past_key_names, i));
    value_caches_names.push_back(
        ComposeKeyValueName(model_->config_->model.decoder.inputs.past_value_names, i));

    inputs.push_back(key_caches[i]);
    inputs.push_back(value_caches[i]);

    input_names.push_back(key_caches_names.back().c_str());
    input_names.push_back(value_caches_names.back().c_str());
  }

  auto run_options = scheduled_requests.RunOptions();
  model_->session_decoder_->Run(run_options.get(),
                                input_names.data(),
                                inputs.data(),
                                input_names.size(),
                                output_names.data(),
                                outputs.data(),
                                output_names.size());
}

ModelExecutor::ModelExecutor(std::shared_ptr<Model> model)
    : model_{model},
      decoder_{CreateDecoder(model)} {}

void ModelExecutor::Decode(ScheduledRequests& scheduled_requests) {
  decoder_->Decode(scheduled_requests);
}

}  // namespace Generators
