// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_decoder.h"
#include "hybrid_decoder_io.h"
#include "static_batch_decoder_io.h"
#include "varlen_decoder_io.h"
#include "../../models/env_utils.h"
#include "../../models/model_state_manifest.h"

#include <array>
#include <fstream>

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

void DumpDiagnosticRows(Tensor& output, const StepPlan& plan, const std::string& path) {
  auto device_bytes = output.GetByteSpan();
  auto bytes = device_bytes.CopyDeviceToCpu();
  if (plan.token_count == 0 || bytes.size() % plan.token_count != 0) {
    throw std::runtime_error("Diagnostic output size does not match the step token count.");
  }
  const size_t row_bytes = bytes.size() / plan.token_count;
  std::ofstream stream{path, std::ios::app};
  if (!stream) {
    throw std::runtime_error("Failed to open diagnostic output file: " + path);
  }
  for (size_t request_index = 0; request_index < plan.requests.size(); ++request_index) {
    const auto& entry = plan.requests[request_index];
    for (size_t local_row = 0; local_row < entry.unprocessed_token_count; ++local_row) {
      const size_t packed_row = entry.packed_token_offset + local_row;
      uint64_t hash = 1469598103934665603ull;
      for (uint8_t value : bytes.subspan(packed_row * row_bytes, row_bytes)) {
        hash = (hash ^ value) * 1099511628211ull;
      }
      stream << plan.transaction_id << ' ' << request_index << ' ' << entry.sequence_length_before << ' '
             << entry.unprocessed_token_count << ' ' << entry.draft_token_count << ' ' << local_row << ' '
             << packed_row << ' ' << hash << '\n';
    }
  }
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

  const std::string diagnostic_output_name = GetEnv("ORTGENAI_DIAGNOSTIC_OUTPUT_NAME");
  const std::string diagnostic_output_path = GetEnv("ORTGENAI_DIAGNOSTIC_OUTPUT_PATH");
  std::unique_ptr<Tensor> diagnostic_output;
  if (!diagnostic_output_name.empty() && !diagnostic_output_path.empty() && context.plan != nullptr &&
      context.hidden_states_input == nullptr && model_->session_info_.HasOutput(diagnostic_output_name)) {
    diagnostic_output = std::make_unique<Tensor>(
        model_->p_device_inputs_, model_->session_info_.GetOutputDataType(diagnostic_output_name));
    const std::array<int64_t, 2> diagnostic_shape{
      static_cast<int64_t>(context.plan->token_count),
      static_cast<int64_t>(model_->config_->model.decoder.hidden_size)};
    diagnostic_output->CreateTensor(diagnostic_shape);
    decoder_state->output_names_.push_back(diagnostic_output_name.c_str());
    decoder_state->outputs_.push_back(diagnostic_output->GetOrtTensor());
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

  if (diagnostic_output != nullptr) {
    DumpDiagnosticRows(*diagnostic_output, *context.plan, diagnostic_output_path);
  }

  scheduled_requests.AddDecoderState(std::move(decoder_state));
}

}  // namespace Generators
