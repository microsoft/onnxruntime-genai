// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry.h"
#include "device_info.h"
#include "../models/env_utils.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)
#include "LogManager.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

// Version string defined by the build system
#ifndef ORTGENAI_VERSION
#define ORTGENAI_VERSION "unknown"
#endif

#if defined(ORTGENAI_ENABLE_TELEMETRY)
// LOGMANAGER_INSTANCE declares a unique 1DS LogManager. It expands to the
// MAT (::Microsoft::Applications::Events) namespace, so it MUST be at global
// scope — not inside namespace Generators.
LOGMANAGER_INSTANCE
#endif

namespace Generators {

namespace {

// Decode a base64-encoded string at runtime.
// The instrumentation key is stored encoded to avoid plain-text exposure in binaries.
std::string DecodeBase64(const std::string& encoded) {
  auto DecodeChar = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };

  std::string result;
  result.reserve(encoded.size() * 3 / 4);
  uint32_t accum = 0;
  int bits = 0;
  for (char c : encoded) {
    int val = DecodeChar(c);
    if (val < 0) continue;
    accum = (accum << 6) | static_cast<uint32_t>(val);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      result.push_back(static_cast<char>((accum >> bits) & 0xFF));
    }
  }
  return result;
}

// Base64-encoded instrumentation key.
// To update: encode your key with  echo -n "YOUR-KEY" | base64
constexpr const char* kInstrumentationKeyB64 = "OWQ1ZGRhZWM2MWUyNDU2N2I3ODhhMjBhZWEzMjQ2MzEtMzE2NTQxNmEtNGJmZi00ZDQwLTgzMDUtZDlhODdlMDM4NDY5LTcyMzk=";

}  // namespace

#if defined(ORTGENAI_ENABLE_TELEMETRY)

struct GenAiTelemetry::Impl {
  MAT::ILogger* logger{nullptr};
};

#else

struct GenAiTelemetry::Impl {};

#endif  // ORTGENAI_ENABLE_TELEMETRY

// Guard against access after static destruction
static std::atomic<bool> s_instance_destroyed{false};

GenAiTelemetry& GenAiTelemetry::Instance() {
  static GenAiTelemetry instance;
  return instance;
}

bool GenAiTelemetry::IsDestroyed() {
  return s_instance_destroyed.load();
}

void GenAiTelemetry::Initialize() {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_.load()) return;

  // Check environment variable opt-out
  std::string env_val = GetEnv("ORTGENAI_TELEMETRY_ENABLED");
  if (env_val == "0" || env_val == "false" || env_val == "FALSE") {
    enabled_.store(false);
  }

  impl_ = std::make_unique<Impl>();

  auto& config = MAT::LogManager::GetLogConfiguration();
  // Allow env var override, otherwise use the embedded (base64-obfuscated) key
  std::string ikey = GetEnv("ORTGENAI_TELEMETRY_IKEY");
  if (ikey.empty()) {
    ikey = DecodeBase64(kInstrumentationKeyB64);
  }
  if (ikey.empty()) {
    // No valid instrumentation key — telemetry will not be sent
    enabled_.store(false);
    return;
  }
  config[MAT::CFG_STR_PRIMARY_TOKEN] = ikey;
  config["name"] = "Microsoft.OnnxRuntimeGenAI";
  config["version"] = ORTGENAI_VERSION;
  config["config"]["host"] = "OnnxRuntimeGenAI";

  // Scrub the client IP at the collector for privacy (direct-upload mode).
  // Uses the literal config key (ILogConfiguration::CFG_BOOL_ENABLE_IP_SCRUBBING,
  // "enableIpScrubbing") so it stays forward-compatible: it is a harmless no-op
  // on SDK releases that predate the IP-scrubbing feature and becomes effective
  // (and explicit) once the cpp-client-telemetry port is bumped to a release
  // that includes it.
  config["enableIpScrubbing"] = true;

  impl_->logger = MAT::LogManager::Initialize();
  initialized_.store(true);
#endif
}

void GenAiTelemetry::Shutdown() {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (!initialized_.load()) return;

  MAT::LogManager::FlushAndTeardown();
  impl_.reset();
  initialized_.store(false);
#endif
}

GenAiTelemetry::~GenAiTelemetry() {
  Shutdown();
  s_instance_destroyed.store(true);
}

bool GenAiTelemetry::IsEnabled() const {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  return enabled_.load() && initialized_.load();
#else
  return false;
#endif
}

void GenAiTelemetry::SetEnabled(bool enabled) {
  enabled_.store(enabled);
}

uint32_t GenAiTelemetry::AllocateSessionId() {
  return next_session_id_.fetch_add(1);
}

void GenAiTelemetry::LogProcessInfo() {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  // Only log once per process
  bool expected = false;
  if (!process_info_logged_.compare_exchange_strong(expected, true)) return;

  // Re-check enabled and impl validity after CAS
  if (!IsEnabled() || !impl_ || !impl_->logger) {
    process_info_logged_.store(false);  // Allow retry
    return;
  }

  const auto& device = GetDeviceInfo();

  MAT::EventProperties event("OnnxRuntimeGenAI.ProcessInfo");
  event.SetProperty("libraryVersion", ORTGENAI_VERSION);
  event.SetProperty("deviceId", device.device_id);
  event.SetProperty("os", device.os);
  event.SetProperty("osVersion", device.os_version);
  event.SetProperty("osArchitecture", device.os_architecture);
  event.SetProperty("processorCount", static_cast<int64_t>(device.processor_count));
  event.SetProperty("totalMemoryMB", static_cast<int64_t>(device.total_memory_mb));
  event.SetProperty("cpuModel", device.cpu_model);
  event.SetProperty("userLocale", device.user_locale);
  event.SetProperty("userTimezone", device.user_timezone);

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogModelLoadStart(uint32_t session_id) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoadStart");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogModelLoad(uint32_t session_id,
                                   const std::string& model_type,
                                   const std::string& execution_providers,
                                   const std::string& selected_device,
                                   int vocab_size,
                                   int context_length,
                                   bool is_in_memory) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoad");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("modelType", model_type);
  event.SetProperty("executionProviders", execution_providers);
  event.SetProperty("selectedDevice", selected_device);
  event.SetProperty("vocabSize", static_cast<int64_t>(vocab_size));
  event.SetProperty("contextLength", static_cast<int64_t>(context_length));
  event.SetProperty("isInMemory", is_in_memory);

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogModelLoadEnd(uint32_t session_id, bool is_success,
                                      double load_time_ms,
                                      const std::string& error_message) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoadEnd");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("isSuccess", is_success);
  event.SetProperty("loadTimeMs", load_time_ms);
  if (!error_message.empty()) {
    // Truncate error message to 256 chars for privacy
    auto truncated = error_message.substr(0, 256);
    event.SetProperty("errorMessage", truncated);
  }

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogGeneratorCreate(uint32_t session_id,
                                         int batch_size, int num_beams, int max_length,
                                         float top_k, float top_p, float temperature,
                                         bool use_graph_capture, bool has_guidance) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.GeneratorCreate");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("batchSize", static_cast<int64_t>(batch_size));
  event.SetProperty("numBeams", static_cast<int64_t>(num_beams));
  event.SetProperty("maxLength", static_cast<int64_t>(max_length));
  event.SetProperty("topK", static_cast<double>(top_k));
  event.SetProperty("topP", static_cast<double>(top_p));
  event.SetProperty("temperature", static_cast<double>(temperature));
  event.SetProperty("useGraphCapture", use_graph_capture);
  event.SetProperty("hasGuidance", has_guidance);

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogGenerateStart(uint32_t session_id, uint32_t generator_id,
                                       int prompt_tokens) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.GenerateStart");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("generatorId", static_cast<int64_t>(generator_id));
  event.SetProperty("promptTokens", static_cast<int64_t>(prompt_tokens));

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogGenerateEnd(uint32_t session_id, uint32_t generator_id,
                                     int total_tokens, int prompt_tokens,
                                     double time_to_first_token_ms, double total_time_ms,
                                     double tokens_per_second,
                                     const std::string& execution_provider) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.GenerateEnd");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("generatorId", static_cast<int64_t>(generator_id));
  event.SetProperty("totalTokens", static_cast<int64_t>(total_tokens));
  event.SetProperty("promptTokens", static_cast<int64_t>(prompt_tokens));
  event.SetProperty("timeToFirstTokenMs", time_to_first_token_ms);
  event.SetProperty("totalTimeMs", total_time_ms);
  event.SetProperty("tokensPerSecond", tokens_per_second);
  event.SetProperty("executionProvider", execution_provider);

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogRuntimeError(uint32_t session_id,
                                      const std::string& error_type,
                                      const std::string& error_message,
                                      const std::string& context) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.RuntimeError");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("errorType", error_type);
  // Truncate error message for privacy
  auto truncated = error_message.substr(0, 256);
  event.SetProperty("errorMessage", truncated);
  event.SetProperty("context", context);

  impl_->logger->LogEvent(event);
#endif
}

void GenAiTelemetry::LogRuntimePerf(uint32_t session_id, int total_runs,
                                     double total_run_duration_ms) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  if (!IsEnabled()) return;

  MAT::EventProperties event("OnnxRuntimeGenAI.RuntimePerf");
  event.SetProperty("sessionId", static_cast<int64_t>(session_id));
  event.SetProperty("totalRuns", static_cast<int64_t>(total_runs));
  event.SetProperty("totalRunDurationMs", total_run_duration_ms);

  impl_->logger->LogEvent(event);
#endif
}

}  // namespace Generators
