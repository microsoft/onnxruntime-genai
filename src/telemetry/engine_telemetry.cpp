// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine_telemetry.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include <algorithm>
#include <atomic>
#include <string_view>

#include "telemetry.h"

namespace Generators {

namespace {
std::atomic<uint32_t> g_next_engine_id{1};
std::atomic<uint32_t> g_next_engine_request_id{1};
std::atomic<EngineRequestTelemetry::FinalizeCallback> g_finalize_callback{nullptr};
}  // namespace

uint32_t AllocateEngineTelemetryId() {
  return g_next_engine_id.fetch_add(1);
}

void EngineRequestTelemetry::SetFinalizeCallbackForTest(FinalizeCallback callback) {
  g_finalize_callback.store(callback);
}

EngineRequestTelemetry::~EngineRequestTelemetry() {
  if (!active_) return;
  Finish(completed_ ? Outcome::Completed : Outcome::Abandoned,
         completed_ ? completed_time_ : std::chrono::steady_clock::now());
}

void EngineRequestTelemetry::Begin(uint32_t session_id, uint32_t engine_id,
                                   bool dynamic_batching, int64_t prompt_tokens) {
  if (active_) {
    Finish(completed_ ? Outcome::Completed : Outcome::Abandoned,
           completed_ ? completed_time_ : std::chrono::steady_clock::now());
  }

  if (!IsEnabled()) return;

  session_id_ = session_id;
  engine_id_ = engine_id;
  request_id_ = g_next_engine_request_id.fetch_add(1);
  prompt_tokens_ = prompt_tokens;
  dynamic_batching_ = dynamic_batching;
  assigned_time_ = std::chrono::steady_clock::now();
  steady_clock_anchor_ = assigned_time_;
  system_clock_anchor_ = std::chrono::system_clock::now();
  active_ = true;
}

void EngineRequestTelemetry::OnScheduled() {
  if (!active_ || scheduled_) return;
  if (!IsEnabled()) {
    Reset();
    return;
  }
  scheduled_time_ = std::chrono::steady_clock::now();
  scheduled_ = true;
}

void EngineRequestTelemetry::OnStep(int64_t sequence_length, size_t batch_size,
                                    bool completed) {
  if (!active_) return;
  if (!IsEnabled()) {
    Reset();
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (!first_token_recorded_) {
    first_token_time_ = now;
    first_token_recorded_ = true;
  }
  generated_tokens_ = std::max<int64_t>(0, sequence_length - prompt_tokens_);
  ++step_count_;
  batch_size_total_ += static_cast<int64_t>(batch_size);
  max_batch_size_ = std::max(max_batch_size_, static_cast<int64_t>(batch_size));
  if (completed && !completed_) {
    completed_time_ = now;
    completed_ = true;
  }
}

void EngineRequestTelemetry::OnRemoved(bool completed) {
  if (!active_) return;
  const auto now = std::chrono::steady_clock::now();
  Finish(completed || completed_ ? Outcome::Completed : Outcome::Removed,
         completed_ ? completed_time_ : now);
}

bool EngineRequestTelemetry::IsEnabled() const {
  return g_finalize_callback.load() != nullptr ||
         (!GenAiTelemetry::IsDestroyed() && GenAiTelemetry::Instance().IsEnabled());
}

void EngineRequestTelemetry::Finish(Outcome outcome,
                                    std::chrono::steady_clock::time_point end_time) {
  if (!active_) return;
  const auto finalize_callback = g_finalize_callback.load();
  if (!finalize_callback && GenAiTelemetry::IsDestroyed()) {
    Reset();
    return;
  }

  EngineRequestInfo info{};
  info.dynamic_batching = dynamic_batching_;
  info.prompt_tokens = prompt_tokens_;
  info.generated_tokens = generated_tokens_;
  info.step_count = step_count_;
  info.max_batch_size = max_batch_size_;
  info.average_batch_size =
      step_count_ > 0 ? static_cast<double>(batch_size_total_) / step_count_ : 0.0;
  info.queue_time_ms =
      scheduled_
          ? std::chrono::duration<double, std::milli>(scheduled_time_ - assigned_time_).count()
          : 0.0;
  info.time_to_first_token_ms =
      first_token_recorded_
          ? std::chrono::duration<double, std::milli>(first_token_time_ - assigned_time_).count()
          : 0.0;
  info.decode_time_ms =
      first_token_recorded_
          ? std::chrono::duration<double, std::milli>(end_time - first_token_time_).count()
          : 0.0;
  info.total_time_ms =
      std::chrono::duration<double, std::milli>(end_time - assigned_time_).count();
  switch (outcome) {
    case Outcome::Completed:
      info.outcome = "completed";
      break;
    case Outcome::Removed:
      info.outcome = "removed";
      break;
    case Outcome::Abandoned:
      info.outcome = "abandoned";
      break;
  }

  const auto end_wall_time =
      system_clock_anchor_ +
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          end_time - steady_clock_anchor_);
  const auto end_timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          end_wall_time.time_since_epoch())
          .count();
  if (finalize_callback) {
    finalize_callback(session_id_, engine_id_, request_id_, info, end_timestamp_ms);
  }
  if (!GenAiTelemetry::IsDestroyed()) {
    GenAiTelemetry::Instance().LogEngineRequest(
        session_id_, engine_id_, request_id_, info, end_timestamp_ms);
  }
  Reset();
}

void EngineRequestTelemetry::Reset() {
  session_id_ = 0;
  engine_id_ = 0;
  request_id_ = 0;
  prompt_tokens_ = 0;
  generated_tokens_ = 0;
  step_count_ = 0;
  batch_size_total_ = 0;
  max_batch_size_ = 0;
  dynamic_batching_ = false;
  active_ = false;
  scheduled_ = false;
  first_token_recorded_ = false;
  completed_ = false;
  assigned_time_ = {};
  scheduled_time_ = {};
  first_token_time_ = {};
  completed_time_ = {};
  steady_clock_anchor_ = {};
  system_clock_anchor_ = {};
}

}  // namespace Generators

#else

namespace Generators {

uint32_t AllocateEngineTelemetryId() {
  return 0;
}

}  // namespace Generators

#endif
