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
    position_planes_ = PackedPositionIdPlanes(*model_);
    if (position_planes_ == 3 &&
        (!model_->config_->model.vision.filename.empty() ||
         !model_->config_->model.vision.pipeline.empty() ||
         !model_->config_->model.speech.filename.empty())) {
      throw std::runtime_error(
          "Packed [3, num_tokens] position_ids support text-only models; "
          "multimodal coordinates are not supported by the Engine.");
    }
  }
  if (IsGraphCaptureEnabled(model_->config_->model.decoder.session_options) &&
      cache_manager_->SupportsDynamicBatching()) {
    // The scheduler clamps every request's draft width to what the cache can roll back, and a
    // verify step runs one token plus those drafts. A per-request speculative option can raise the
    // width above the config default, so the cache's ceiling is what the buffers have to cover.
    graph_buffers_ = std::make_unique<VarlenGraphBuffers>(
        *model_, position_planes_, cache_manager_->MaxDraftTokensPerStep() + 1);
  }
}

SimpleDecoder::~SimpleDecoder() {
#if ORT_API_VERSION >= 27
  // The session outlives this decoder and keeps every graph it captured, so the graphs have to go
  // before the buffers whose addresses they recorded. Destructors must not throw.
  if (graph_buffers_ && model_->session_decoder_) {
    for (const int annotation_id : graph_buffers_->graph_ids.AssignedIds()) {
      try {
        model_->session_decoder_->ReleaseCapturedGraph(annotation_id);
      } catch (...) {
        if (g_log.enabled && g_log.graph_capture) {
          Log("graph_capture") << "ReleaseCapturedGraph(id=" << annotation_id << ") failed"
                               << std::endl;
        }
      }
    }
  }
#endif
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

void SimpleDecoder::LogGraphDecision(size_t batch_size, size_t tokens_per_request,
                                     const ExecutionContext& context, int annotation_id,
                                     GraphFallback fallback, size_t known_shapes) {
  const size_t shapes = graph_buffers_ ? graph_buffers_->graph_ids.size() : 0;
  const bool new_shape = shapes != known_shapes;
  const bool reason_changed = fallback != last_fallback_;
  last_fallback_ = fallback;
  if (!g_log.enabled || !g_log.graph_capture || (!new_shape && !reason_changed)) {
    return;
  }
  auto& stream = Log("graph_capture");
  stream << "decode batch=" << batch_size << " tokens_per_request=" << tokens_per_request
         << " block_table_columns=" << context.block_table_columns
         << " state_binding=" << context.fixed_state_binding_key << " -> ";
  switch (fallback) {
    case GraphFallback::kCaptured:
      stream << "graph id " << annotation_id;
      break;
    case GraphFallback::kDisabled:
      stream << "eager (graph capture not enabled for this decoder)";
      break;
    case GraphFallback::kNonUniformStep:
      stream << "eager (prefill, or requests contribute different token counts)";
      break;
    case GraphFallback::kStepTooWide:
      stream << "eager (step exceeds the persistent buffers: max_batch_size="
             << graph_buffers_->max_batch_size
             << " max_query_tokens=" << graph_buffers_->max_query_tokens
             << " max_token_rows=" << graph_buffers_->max_token_rows << ")";
      break;
    case GraphFallback::kUncapturableShape:
      stream << "eager ("
             << (shapes >= GraphAnnotationIds::kMaxCapturedShapes ? "shape budget exhausted"
                                                                  : "shape has no annotation key")
             << ")";
      break;
  }
  stream << " [" << shapes << "/" << GraphAnnotationIds::kMaxCapturedShapes << " shapes assigned]"
         << std::endl;
}

void SimpleDecoder::Decode(ScheduledRequests& scheduled_requests,
                           ExecutionContext& context) {
  const size_t tokens_per_request = context.plan
                                        ? PlanDecodeTokenCount(*context.plan)
                                        : UniformDecodeTokenCount(scheduled_requests);
  // -1 tells the CUDA EP to run eagerly. Otherwise every distinct decode shape gets its own
  // annotation id: the EP runs that id once eagerly, captures it on the next occurrence, and
  // replays from then on. The id also covers the fixed-state binding layout, whose device
  // addresses move when the persistent bank flips.
  const size_t known_shapes = graph_buffers_ ? graph_buffers_->graph_ids.size() : 0;
  int annotation_id = -1;
  GraphFallback fallback = GraphFallback::kCaptured;
  if (graph_buffers_ == nullptr) {
    fallback = GraphFallback::kDisabled;
  } else if (tokens_per_request == 0) {
    fallback = GraphFallback::kNonUniformStep;
  } else if (!graph_buffers_->Fits(scheduled_requests.size(), tokens_per_request)) {
    fallback = GraphFallback::kStepTooWide;
  } else {
    annotation_id = graph_buffers_->GraphId(scheduled_requests.size(), tokens_per_request,
                                            context.block_table_columns,
                                            context.fixed_state_binding_key);
    if (annotation_id <= 0) {
      fallback = GraphFallback::kUncapturableShape;
    }
  }
  const bool capture = annotation_id > 0;
  LogGraphDecision(scheduled_requests.size(), tokens_per_request, context, annotation_id, fallback,
                   known_shapes);

  std::unique_ptr<DecoderIO> decoder_state;
  if (!cache_manager_->SupportsDynamicBatching()) {
    decoder_state = std::make_unique<StaticBatchDecoderIO>(
        model_, scheduled_requests, cache_manager_);
  } else if (has_fixed_state_groups_) {
    decoder_state = std::make_unique<HybridDecoderIO>(
        model_, scheduled_requests, cache_manager_, context,
        capture ? graph_buffers_.get() : nullptr, position_planes_, &embedding_workspace_);
  } else {
    decoder_state = std::make_unique<VarlenDecoderIO>(
        model_, scheduled_requests, cache_manager_, &context,
        capture ? graph_buffers_.get() : nullptr, position_planes_, &embedding_workspace_);
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
