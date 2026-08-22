// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"
#include "hybrid_decoder_io.h"
#include "static_batch_decoder_io.h"
#include "varlen_decoder_io.h"
#include "../../models/model_state_manifest.h"

namespace Generators {

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model,
                             std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {
  const ModelStateManifest manifest{model_->config_->model.decoder};
  has_fixed_state_groups_ = manifest.HasFixedStateGroups();
  const bool has_position_ids = model_->session_info_.HasInput(
      model_->config_->model.decoder.inputs.position_ids);
  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      cache_manager_->SupportsDynamicBatching() &&
      !has_fixed_state_groups_ &&
      !has_position_ids) {
    graph_buffers_ = std::make_unique<VarlenGraphBuffers>(*model_);
  }
}

namespace {

// A step can be captured only if it looks like every other step the graph will serve: one new token
// per sequence. Prefill and chunked-prefill steps carry a variable number of tokens and are shaped
// differently from run to run, so they always execute eagerly.
bool IsPureDecodeStep(ScheduledRequests& scheduled_requests) {
  if (scheduled_requests.size() == 0) {
    return false;
  }
  for (auto& request : scheduled_requests) {
    if (request->IsPrefill() || request->ScheduledTokenCount() != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests,
                           ExecutionContext& context) {
  const bool capture = graph_buffers_ != nullptr &&
                       graph_buffers_->Fits(scheduled_requests.size()) &&
                       (context.plan ? context.plan->graph_capture_eligible
                                     : IsPureDecodeStep(scheduled_requests));

  std::unique_ptr<DecoderIO> decoder_state;
  if (!cache_manager_->SupportsDynamicBatching()) {
    decoder_state = std::make_unique<StaticBatchDecoderIO>(
        model_, scheduled_requests, cache_manager_);
  } else if (has_fixed_state_groups_) {
    decoder_state = std::make_unique<HybridDecoderIO>(
        model_, scheduled_requests, cache_manager_, context);
  } else {
    decoder_state = std::make_unique<VarlenDecoderIO>(
        model_, scheduled_requests, cache_manager_, &context,
        capture ? graph_buffers_.get() : nullptr);
  }

  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      graph_buffers_ == nullptr) {
    // Inputs without persistent graph buffers have per-step addresses and must stay eager.
    context.run_options->AddConfigEntry("gpu_graph_id", "-1");
  } else if (graph_buffers_ != nullptr) {
    // -1 tells the CUDA EP to run eagerly. Otherwise every distinct decode shape gets its own
    // annotation id: the EP runs that id once eagerly, captures it on the next occurrence, and
    // replays from then on.
    const std::string graph_id =
        capture ? std::to_string(VarlenGraphBuffers::GraphId(scheduled_requests.size(),
                                                             context.block_table_columns))
                : std::string("-1");
    context.run_options->AddConfigEntry("gpu_graph_id", graph_id.c_str());
  }

  decoder_state->DumpInputs();
  model_->session_decoder_->Run(context.run_options.get(),
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
