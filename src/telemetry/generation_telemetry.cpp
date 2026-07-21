// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generation_telemetry.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include <atomic>
#include <chrono>
#include "telemetry.h"

namespace Generators {

namespace {
std::atomic<uint32_t> g_next_generator_id{1};
}

GenerationTelemetry::AppendTrackingSuppression::AppendTrackingSuppression(
    GenerationTelemetry& telemetry)
    : telemetry_{telemetry},
      previous_value_{telemetry.append_tracking_suppressed_} {
  telemetry_.append_tracking_suppressed_ = true;
}

GenerationTelemetry::AppendTrackingSuppression::~AppendTrackingSuppression() {
  telemetry_.append_tracking_suppressed_ = previous_value_;
}

GenerationTelemetry::GenerationTelemetry(uint32_t session_id, bool is_transducer)
    : session_id_{session_id},
      generator_id_{g_next_generator_id.fetch_add(1)},
      is_transducer_{is_transducer},
      input_modality_{is_transducer ? "audio" : "text"},
      start_time_{std::chrono::steady_clock::now()} {}

GenerationTelemetry::~GenerationTelemetry() {
  Finish();
}

void GenerationTelemetry::LogGeneratorCreate(int batch_size, int num_beams, int max_length,
                                             int top_k, float top_p, float temperature,
                                             bool do_sample, bool use_graph_capture, bool has_guidance) {
  GenAiTelemetry::Instance().LogGeneratorCreate(
      session_id_, generator_id_, batch_size, num_beams, max_length,
      top_k, top_p, temperature, do_sample, use_graph_capture, has_guidance);
}

void GenerationTelemetry::LogAdapterActivated() {
  GenAiTelemetry::Instance().LogAdapterActivated(session_id_, generator_id_);
}

bool GenerationTelemetry::IsEnabled() const {
  return !GenAiTelemetry::IsDestroyed() && GenAiTelemetry::Instance().IsEnabled();
}

bool GenerationTelemetry::BeginAppend() {
  append_tracking_ = false;
  if (append_tracking_suppressed_) return false;

  CloseRequestBoundary();
  append_tracking_ = IsEnabled();
  if (append_tracking_) append_start_time_ = std::chrono::steady_clock::now();
  return append_tracking_;
}

void GenerationTelemetry::CompleteAppend(size_t input_token_count, int num_beams,
                                         std::string_view input_modality) {
  if (!append_tracking_) return;
  append_tracking_ = false;

  if (prompt_tokens_ == 0 && !first_token_logged_) start_time_ = append_start_time_;
  prompt_tokens_ += static_cast<int>(input_token_count) * num_beams;
  input_modality_ = input_modality;
}

void GenerationTelemetry::OnTokenGenerated(int active_token_count) {
  const bool track_telemetry = IsEnabled();
  if (track_telemetry && !generation_abandoned_) {
    const auto now = std::chrono::steady_clock::now();
    if (prompt_tokens_ == 0 && generated_tokens_ == 0) start_time_ = now;
    generated_tokens_ += active_token_count;
    last_token_time_ = now;
    if (!first_token_logged_) {
      first_token_time_ = now;
      first_token_logged_ = true;
      generate_start_logged_ = GenAiTelemetry::Instance().LogGenerateStart(
          session_id_, generator_id_, prompt_tokens_, input_modality_);
    }
  } else if (!track_telemetry && !generation_abandoned_) {
    generation_abandoned_ = true;
    Reset();
  }
}

void GenerationTelemetry::OnRewind() {
  if (append_tracking_suppressed_) return;
  CloseRequestBoundary();
}

GenerationTelemetry::AppendTrackingSuppression GenerationTelemetry::SuppressAppendTracking() {
  return AppendTrackingSuppression{*this};
}

void GenerationTelemetry::CloseRequestBoundary() {
  if (generated_tokens_ > 0) {
    Finish();
  } else if (generation_abandoned_) {
    Reset();
  }
  generation_abandoned_ = false;
}

void GenerationTelemetry::Finish() {
  if (generated_tokens_ > 0 && generate_start_logged_ && !GenAiTelemetry::IsDestroyed()) {
    const auto total_time_ms =
        std::chrono::duration<double, std::milli>(last_token_time_ - start_time_).count();
    double time_to_first_token_ms = 0.0;
    if (first_token_logged_) {
      time_to_first_token_ms =
          std::chrono::duration<double, std::milli>(first_token_time_ - start_time_).count();
    }
    const double tokens_per_second =
        total_time_ms > 0 ? generated_tokens_ * 1000.0 / total_time_ms : 0.0;

    GenAiTelemetry::Instance().LogGenerateEnd(
        session_id_, generator_id_, prompt_tokens_ + generated_tokens_,
        time_to_first_token_ms, total_time_ms, tokens_per_second);
  }
  Reset();
}

void GenerationTelemetry::Reset() {
  prompt_tokens_ = 0;
  generated_tokens_ = 0;
  first_token_logged_ = false;
  generate_start_logged_ = false;
  start_time_ = std::chrono::steady_clock::now();
  first_token_time_ = {};
  last_token_time_ = {};
  input_modality_ = is_transducer_ ? "audio" : "text";
  append_tracking_ = false;
}

}  // namespace Generators

#endif
