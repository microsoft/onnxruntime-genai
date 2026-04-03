// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry.h"
#include "device_info.h"
#include "../models/env_utils.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)
#include "LogManager.hpp"
#include "LogManagerProvider.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Version string defined by the build system
#ifndef ORTGENAI_VERSION
#define ORTGENAI_VERSION "unknown"
#endif

namespace Generators {

namespace {

// Decode a base64-encoded string at runtime.
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

// Obfuscated 1DS ingestion key: the key bytes are XOR'd with a fixed repeating
// key, then base64-encoded, so it does not appear in a plain `strings | base64 -d`
// scan. This is light obfuscation, NOT encryption — the key is a write-only
// OneCollector tenant token (like an Application Insights ikey), not a secret: it
// only permits sending events, never reading them. Overridable via the
// ORTGENAI_TELEMETRY_IKEY env var or the -DORTGENAI_TELEMETRY_TOKEN build option.
// To regenerate: XOR the key bytes with the repeating ASCII key below, then base64.
constexpr const char* kInstrumentationKeyObfuscated =
    "LA1WHDYXWhILD1FwUQhyLXdfX0g3RFoRX19Vf1IPcnFiCFcaZUAIEgxAUH5cX2x9eQpYVTNHC0REVVclVFx4fC0LDU9hWFlHXlo=";

// Recover the ingestion key: base64-decode, then XOR with the repeating key.
std::string DeobfuscateKey() {
  static constexpr char kXorKey[] = "OnnxRuntimeGenAI";
  constexpr size_t klen = sizeof(kXorKey) - 1;
  std::string decoded = DecodeBase64(kInstrumentationKeyObfuscated);
  for (size_t i = 0; i < decoded.size(); ++i)
    decoded[i] = static_cast<char>(decoded[i] ^ kXorKey[i % klen]);
  return decoded;
}

// Case-insensitive check for common "disabled" values used by the env opt-out.
bool IsEnvOptOut(const std::string& value) {
  std::string v;
  v.reserve(value.size());
  for (char c : value) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return v == "0" || v == "false" || v == "off" || v == "no" || v == "disabled" || v == "n";
}

// Generate a random v4 UUID as a hex string (e.g. "f81d4fae-7dec-41d0-8f12-00a0c91e6bf6").
// Used for the process-wide AppSessionGuid (Tier 1 identity).
std::string GenerateGuidV4() {
  // Draw the 128 bits directly from a CSPRNG-backed std::random_device rather
  // than seeding a PRNG, so the full entropy is preserved and the value is
  // non-predictable.
  std::random_device rd;
  uint64_t hi = (static_cast<uint64_t>(rd()) << 32) | rd();
  uint64_t lo = (static_cast<uint64_t>(rd()) << 32) | rd();
  // Set version (4) and variant (10xx) bits.
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

  char buf[37];
  std::snprintf(buf, sizeof(buf),
                "%08x-%04x-%04x-%04x-%012llx",
                static_cast<uint32_t>(hi >> 32),
                static_cast<uint32_t>((hi >> 16) & 0xFFFF),
                static_cast<uint32_t>(hi & 0xFFFF),
                static_cast<uint32_t>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return std::string(buf);
}

// Returns the index of the first filesystem-path anchor in s, or npos. Anchors:
// a drive prefix (C:\ or C:/), a UNC prefix (\\), a home prefix (~/ or ~\), or an
// absolute POSIX path with >= 2 segments (/a/b...). Single-slash tokens such as
// "n/a" or "read/write" are not anchors.
size_t FindPathAnchor(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\\' && i + 1 < s.size() && s[i + 1] == '\\') return i;  // UNC
    if (c == '~' && i + 1 < s.size() && (s[i + 1] == '/' || s[i + 1] == '\\')) return i;
    if (std::isalpha(static_cast<unsigned char>(c)) && i + 2 < s.size() &&
        s[i + 1] == ':' && (s[i + 2] == '\\' || s[i + 2] == '/'))
      return i;  // drive prefix
    if (c == '/') {
      // Absolute POSIX path with >= 2 '/'-delimited non-empty segments. Segments
      // may contain spaces (e.g. "/home/John Smith"), which is why the whole run
      // is treated as a path.
      size_t segments = 0;
      size_t j = i;
      while (j < s.size() && s[j] == '/') {
        size_t seg_start = ++j;
        // A path segment for detection is non-empty and space-free; this avoids
        // treating unrelated slashes (e.g. "n/a ... read/write") as a path while
        // still detecting real paths like "/home/jdoe" or "/Users/Jane Doe"
        // (whose spaced tail is removed by the to-end-of-message redaction).
        while (j < s.size() && s[j] != '/' && s[j] != '\r' && s[j] != '\n' &&
               s[j] != ' ' && s[j] != '\t')
          ++j;
        if (j > seg_start)
          ++segments;
        else
          break;
      }
      if (segments >= 2) return i;
    }
  }
  return std::string::npos;
}

// Redact filesystem paths from free-text error strings before transmission.
// Load and runtime exceptions routinely embed the user's config/model path
// (e.g. C:\Users\<name>\... or /home/<name>/...), which would expose the username
// and directory layout. Paths frequently contain spaces, so a per-token classifier
// is bypassable; instead, everything from the first path anchor to the end of the
// message is replaced with "[path]".
std::string ScrubErrorMessage(const std::string& msg) {
  size_t anchor = FindPathAnchor(msg);
  if (anchor == std::string::npos) return msg;
  return msg.substr(0, anchor) + "[path]";
}

}  // namespace

#if defined(ORTGENAI_ENABLE_TELEMETRY)

struct GenAiTelemetry::Impl {
  MAT::ILogConfiguration config;  // must outlive log_manager (SDK holds a reference)
  MAT::ILogManager* log_manager{nullptr};
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

  // Environment opt-out: do not initialize the SDK or write any identifier.
  if (IsEnvOptOut(GetEnv("ORTGENAI_TELEMETRY_ENABLED"))) {
    enabled_.store(false);
    return;
  }

  // Telemetry must never affect host control flow: any SDK failure here is
  // swallowed and simply leaves telemetry uninitialized (a later call retries).
  // enabled_ is not cleared on failure paths, so a successful retry still emits.
  try {
    // Allow env var override, otherwise use the embedded ingestion key.
    std::string ikey = GetEnv("ORTGENAI_TELEMETRY_IKEY");
    if (ikey.empty()) {
#if defined(ORTGENAI_TELEMETRY_TENANT_TOKEN)
      ikey = ORTGENAI_TELEMETRY_TENANT_TOKEN;  // build-time override (CI secret)
#else
      ikey = DeobfuscateKey();  // embedded in-repo default
#endif
    }
    if (ikey.empty()) return;  // no key -> cannot send; stay uninitialized

    impl_ = std::make_unique<Impl>();

    // Configuration is owned by impl_ so it outlives the log manager (the SDK holds
    // a reference to it). enableIpScrubbing requests collector-side client-IP
    // obfuscation on the direct-upload path.
    auto& config = impl_->config;
    config[MAT::CFG_STR_PRIMARY_TOKEN] = ikey;
    config["version"] = ORTGENAI_VERSION;
    config["enableIpScrubbing"] = true;

    // Do not block process teardown to upload: persisted events are sent on the next
    // run. 0 keeps Shutdown non-blocking and avoids adding exit latency to host apps.
    config[MAT::CFG_INT_MAX_TEARDOWN_TIME] = 0;

    // Create a dedicated log manager via LogManagerProvider (recommended over the
    // LOGMANAGER_INSTANCE singleton for library use). wantController=true makes this
    // instance the host (name == host), so Flush/FlushAndTeardown act on it.
    MAT::status_t status = MAT::STATUS_SUCCESS;
    impl_->log_manager =
        MAT::LogManagerProvider::CreateLogManager("OnnxRuntimeGenAI", true, config, status);
    if (status != MAT::STATUS_SUCCESS || impl_->log_manager == nullptr) {
      impl_.reset();
      return;
    }
    impl_->logger = impl_->log_manager->GetLogger(ikey);

    // Process-wide AppSessionGuid stamped on every event as logger context (not a
    // per-event property). An explicit, SDK-independent correlator that answers
    // "which events came from the same process run?" and makes the lightweight
    // per-event sessionId / generatorId counters globally unique.
    app_session_guid_ = GenerateGuidV4();
    impl_->logger->SetContext("AppSessionGuid", app_session_guid_);

    // Override the 1DS SDK's device id with our privacy-preserving value so the
    // SDK's automatic (potentially hardware-derived) device id is never sent. This
    // flows as ext.device.localId on every event. The value is a stable hash of a
    // locally-generated UUID, not a hardware identifier.
    impl_->logger->GetSemanticContext()->SetDeviceId(GetDeviceInfo().device_id);

    initialized_.store(true);
  } catch (...) {
    // Never propagate a telemetry-init failure to the host.
    impl_.reset();
  }
#endif
}

void GenAiTelemetry::Shutdown() {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (!initialized_.load()) return;

  // Mark uninitialized first so concurrent Log* calls bail in IsEnabled() before
  // teardown begins. Log* and Shutdown must not run concurrently (OgaShutdown API
  // contract); this only narrows the window.
  initialized_.store(false);

  // 1DS-recommended shutdown sequence: Flush() persists in-memory events to disk,
  // FlushAndTeardown() quiesces background activity (PauseActivity/WaitPause), and
  // LogManagerProvider::Release() disposes the instance. Wrapped so a teardown
  // failure can never propagate (Shutdown runs from ~GenAiTelemetry and the
  // extern-C OgaShutdown, where an escaping exception would call std::terminate).
  try {
    if (impl_ && impl_->log_manager) {
      impl_->log_manager->Flush();
      impl_->log_manager->FlushAndTeardown();
      MAT::LogManagerProvider::Release(impl_->config);
      impl_->log_manager = nullptr;
      impl_->logger = nullptr;
    }
  } catch (...) {
  }
  impl_.reset();
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

#if defined(ORTGENAI_ENABLE_TELEMETRY)
std::unique_lock<std::mutex> GenAiTelemetry::LockForLogging() {
  std::unique_lock<std::mutex> lock(init_mutex_);
  if (!enabled_.load() || !initialized_.load() || !impl_ || !impl_->logger)
    return {};  // empty (unlocked) -> caller bails
  return lock;  // locked; ownership moves to caller
}

void GenAiTelemetry::RunLocked(const std::function<void()>& fn) {
  auto lock = LockForLogging();
  if (!lock) return;
  try {
    fn();
  } catch (...) {
    // Telemetry must never affect host control flow.
  }
}
#endif

void GenAiTelemetry::LogProcessInfo() {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  // Only log once per process
  bool expected = false;
  if (!process_info_logged_.compare_exchange_strong(expected, true)) return;

  bool emitted = false;
  RunLocked([&] {
    const auto& device = GetDeviceInfo();

    MAT::EventProperties event("OnnxRuntimeGenAI.ProcessInfo");
    // sessionId 0 = process scope (model sessions are numbered from 1); ProcessInfo
    // correlates with model/generate events via the AppSessionGuid logger context.
    event.SetProperty("sessionId", static_cast<int64_t>(0));
    event.SetProperty("libraryVersion", ORTGENAI_VERSION);
    // Device id is sent as ext.device.localId (set via SetDeviceId in Initialize),
    // so it is not duplicated as a custom property here.
    event.SetProperty("os", device.os);
    event.SetProperty("osVersion", device.os_version);
    event.SetProperty("osArchitecture", device.os_architecture);
    event.SetProperty("processorCount", static_cast<int64_t>(device.processor_count));
    event.SetProperty("totalMemoryMB", static_cast<int64_t>(device.total_memory_mb));
    event.SetProperty("cpuModel", device.cpu_model);

    impl_->logger->LogEvent(event);
    emitted = true;
  });
  // Allow a retry if telemetry was not live (or threw before emitting).
  if (!emitted) process_info_logged_.store(false);
#endif
}

void GenAiTelemetry::LogModelLoadStart(uint32_t session_id) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoadStart");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogModelLoad(uint32_t session_id, const ModelLoadInfo& info) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoad");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("modelType", info.model_type);
    event.SetProperty("modelFamily", info.model_family);
    event.SetProperty("modality", info.modality);
    event.SetProperty("attentionType", info.attention_type);
    event.SetProperty("transcriptionMode", info.transcription_mode);
    event.SetProperty("executionProviders", info.execution_providers);
    event.SetProperty("selectedDevice", info.selected_device);
    event.SetProperty("vocabSize", static_cast<int64_t>(info.vocab_size));
    event.SetProperty("contextLength", static_cast<int64_t>(info.context_length));
    event.SetProperty("numHiddenLayers", static_cast<int64_t>(info.num_hidden_layers));
    event.SetProperty("hiddenSize", static_cast<int64_t>(info.hidden_size));
    event.SetProperty("numAttentionHeads", static_cast<int64_t>(info.num_attention_heads));
    event.SetProperty("numKeyValueHeads", static_cast<int64_t>(info.num_key_value_heads));
    event.SetProperty("intraOpNumThreads", static_cast<int64_t>(info.intra_op_num_threads));
    event.SetProperty("graphOptimizationLevel", static_cast<int64_t>(info.graph_optimization_level));
    event.SetProperty("gpuMemoryMB", static_cast<int64_t>(info.gpu_memory_mb));
    event.SetProperty("isInMemory", info.is_in_memory);

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogModelLoadEnd(uint32_t session_id, bool is_success,
                                     double load_time_ms,
                                     const std::string& error_message) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.ModelLoadEnd");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("isSuccess", is_success);
    event.SetProperty("loadTimeMs", load_time_ms);
    if (!error_message.empty()) {
      // Scrub filesystem paths (privacy), then cap length (size guard).
      auto scrubbed = ScrubErrorMessage(error_message);
      if (scrubbed.size() > 256) scrubbed = scrubbed.substr(0, 256);
      event.SetProperty("errorMessage", scrubbed);
    }

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogGeneratorCreate(uint32_t session_id, uint32_t generator_id,
                                        int batch_size, int num_beams, int max_length,
                                        float top_k, float top_p, float temperature,
                                        bool do_sample, bool use_graph_capture, bool has_guidance) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.GeneratorCreate");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("generatorId", static_cast<int64_t>(generator_id));
    event.SetProperty("batchSize", static_cast<int64_t>(batch_size));
    event.SetProperty("numBeams", static_cast<int64_t>(num_beams));
    event.SetProperty("maxLength", static_cast<int64_t>(max_length));
    event.SetProperty("topK", static_cast<double>(top_k));
    event.SetProperty("topP", static_cast<double>(top_p));
    event.SetProperty("temperature", static_cast<double>(temperature));
    event.SetProperty("doSample", do_sample);
    event.SetProperty("useGraphCapture", use_graph_capture);
    event.SetProperty("hasGuidance", has_guidance);

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogGenerateStart(uint32_t session_id, uint32_t generator_id,
                                      int prompt_tokens, const std::string& input_modality) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.GenerateStart");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("generatorId", static_cast<int64_t>(generator_id));
    event.SetProperty("promptTokens", static_cast<int64_t>(prompt_tokens));
    event.SetProperty("inputModality", input_modality);

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogGenerateEnd(uint32_t session_id, uint32_t generator_id,
                                    int total_tokens,
                                    double time_to_first_token_ms, double total_time_ms,
                                    double tokens_per_second) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.GenerateEnd");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("generatorId", static_cast<int64_t>(generator_id));
    event.SetProperty("totalTokens", static_cast<int64_t>(total_tokens));
    event.SetProperty("timeToFirstTokenMs", time_to_first_token_ms);
    event.SetProperty("totalTimeMs", total_time_ms);
    event.SetProperty("tokensPerSecond", tokens_per_second);

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogAdapterActivated(uint32_t session_id, uint32_t generator_id) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.AdapterActivated");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("generatorId", static_cast<int64_t>(generator_id));

    impl_->logger->LogEvent(event);
  });
#endif
}

void GenAiTelemetry::LogRuntimeError(uint32_t session_id,
                                     const std::string& error_type,
                                     const std::string& error_message,
                                     const std::string& context) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  RunLocked([&] {
    MAT::EventProperties event("OnnxRuntimeGenAI.RuntimeError");
    event.SetProperty("sessionId", static_cast<int64_t>(session_id));
    event.SetProperty("errorType", error_type);
    // Scrub filesystem paths (privacy), then cap length (size guard).
    auto scrubbed = ScrubErrorMessage(error_message);
    if (scrubbed.size() > 256) scrubbed = scrubbed.substr(0, 256);
    event.SetProperty("errorMessage", scrubbed);
    event.SetProperty("context", context);

    impl_->logger->LogEvent(event);
  });
#endif
}

}  // namespace Generators
