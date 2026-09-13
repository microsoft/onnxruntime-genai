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
  if (cache_manager_->SupportsDynamicBatching() && has_position_ids) {
    const auto& position_ids =
        model_->config_->model.decoder.inputs.position_ids;
    const auto position_shape =
        model_->session_info_.GetInputShape(position_ids);
    ValidatePackedPositionIdsInput(
        model_->session_info_.GetInputDataType(position_ids),
        position_shape,
        model_->session_info_.GetInputSymbolicShape(position_ids));
    if (position_shape.size() == 2 &&
        (!model_->config_->model.vision.filename.empty() ||
         !model_->config_->model.vision.pipeline.empty() ||
         !model_->config_->model.speech.filename.empty())) {
      throw std::runtime_error(
          "Packed [3, num_tokens] position_ids support text-only models; "
          "multimodal coordinates are not supported by the Engine.");
    }
    position_planes_ = position_shape.size() == 2 ? 3 : 1;
  }
  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      cache_manager_->SupportsDynamicBatching()) {
    graph_buffers_ = std::make_unique<VarlenGraphBuffers>(*model_, position_planes_);
  }
}

namespace {

// A step can be captured only if it looks like every other step the graph will serve: the same
// number of new tokens for every sequence. Prefill and chunked-prefill steps carry a variable number
// of tokens and are shaped differently from run to run, so they always execute eagerly. Returns the
// shared per-request token count, or zero when the step cannot be captured.
size_t UniformDecodeTokenCount(ScheduledRequests& scheduled_requests) {
  size_t tokens_per_request = 0;
  for (auto& request : scheduled_requests) {
    const size_t token_count = request->ScheduledTokenCount();
    if (request->IsPrefill() || token_count == 0) {
      return 0;
    }
    if (tokens_per_request == 0) {
      tokens_per_request = token_count;
    } else if (tokens_per_request != token_count) {
      return 0;
    }
  }
  return tokens_per_request;
}

// The scheduler already proved a capture-eligible plan is uniform, so any row carries the count.
size_t PlanDecodeTokenCount(const StepPlan& plan) {
  if (!plan.graph_capture_eligible || plan.requests.empty()) {
    return 0;
  }
  return plan.requests.front().unprocessed_token_count;
}

}  // namespace

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests,
                           ExecutionContext& context) {
  const size_t tokens_per_request = context.plan
                                        ? PlanDecodeTokenCount(*context.plan)
                                        : UniformDecodeTokenCount(scheduled_requests);
  // -1 tells the CUDA EP to run eagerly. Otherwise every distinct decode shape gets its own
  // annotation id: the EP runs that id once eagerly, captures it on the next occurrence, and
  // replays from then on. The id also covers the fixed-state binding layout, whose device
  // addresses move when the persistent bank flips.
  int annotation_id = -1;
  if (graph_buffers_ != nullptr && tokens_per_request != 0 &&
      graph_buffers_->Fits(scheduled_requests.size(), tokens_per_request)) {
    annotation_id = graph_buffers_->GraphId(scheduled_requests.size(), tokens_per_request,
                                            context.block_table_columns,
                                            context.fixed_state_binding_key);
  }
  const bool capture = annotation_id > 0;

  std::unique_ptr<DecoderIO> decoder_state;
  if (!cache_manager_->SupportsDynamicBatching()) {
    decoder_state = std::make_unique<StaticBatchDecoderIO>(
        model_, scheduled_requests, cache_manager_);
  } else if (has_fixed_state_groups_) {
    decoder_state = std::make_unique<HybridDecoderIO>(
        model_, scheduled_requests, cache_manager_, context,
        capture ? graph_buffers_.get() : nullptr, position_planes_);
  } else {
    decoder_state = std::make_unique<VarlenDecoderIO>(
        model_, scheduled_requests, cache_manager_, &context,
        capture ? graph_buffers_.get() : nullptr, position_planes_);
  }

  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      cache_manager_->SupportsDynamicBatching()) {
    context.run_options->AddConfigEntry("gpu_graph_id", std::to_string(annotation_id).c_str());
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
