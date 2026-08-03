// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"
#include "static_batch_decoder_io.h"
#include "varlen_decoder_io.h"

namespace Generators {

SimpleDecoder::SimpleDecoder(std::shared_ptr<DecoderOnly_Model> model,
                             std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {
  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      cache_manager_->SupportsDynamicBatching()) {
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
    if (request->IsPrefill() || request->UnprocessedTokens().size() != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests) {
  cache_manager_->Step();

  const bool capture = graph_buffers_ != nullptr &&
                       graph_buffers_->Fits(scheduled_requests.size()) &&
                       IsPureDecodeStep(scheduled_requests);

  std::unique_ptr<DecoderIO> decoder_state =
      cache_manager_->SupportsDynamicBatching()
          ? static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<VarlenDecoderIO>(
                model_, scheduled_requests, cache_manager_, capture ? graph_buffers_.get() : nullptr))
          : static_cast<std::unique_ptr<DecoderIO>>(std::make_unique<StaticBatchDecoderIO>(model_, scheduled_requests, cache_manager_));

  auto run_options = scheduled_requests.RunOptions();
  if (graph_buffers_ != nullptr) {
    // -1 tells the CUDA EP to run eagerly. Otherwise every distinct decode shape gets its own
    // annotation id: the EP runs that id once eagerly, captures it on the next occurrence, and
    // replays from then on.
    const std::string graph_id =
        capture ? std::to_string(VarlenGraphBuffers::GraphId(scheduled_requests.size(),
                                                             cache_manager_->BlockTableColumns()))
                : std::string("-1");
    run_options->AddConfigEntry("gpu_graph_id", graph_id.c_str());
  }

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
