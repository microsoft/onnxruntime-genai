// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_telemetry.h"

#include "telemetry.h"
#include "../models/model.h"
#include "../models/model_type.h"
#include "../smartptrs.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>

namespace Generators {

#if defined(ORTGENAI_ENABLE_TELEMETRY)

namespace {

std::string DeriveModelFamily(const std::string& model_type) {
  struct Prefix {
    const char* prefix;
    const char* family;
  };
  static constexpr Prefix kPrefixes[] = {
      {"qwen", "qwen"},
      {"phi", "phi"},
      {"gemma", "gemma"},
      {"llama", "llama"},
      {"mistral", "mistral"},
      {"nemotron", "nemotron"},
      {"parakeet", "parakeet"},
      {"whisper", "whisper"},
      {"granite", "granite"},
      {"olmo", "olmo"},
      {"chatglm", "chatglm"},
      {"ernie", "ernie"},
      {"internlm", "internlm"},
      {"smollm", "smollm"},
      {"gptoss", "gptoss"},
      {"gpt2", "gpt2"},
      {"lfm2", "lfm2"},
      {"hunyuan", "hunyuan"},
  };
  for (const auto& prefix : kPrefixes) {
    if (model_type.rfind(prefix.prefix, 0) == 0) {
      return prefix.family;
    }
  }
  return model_type;
}

std::string DeriveAttentionType(const Config::Model::Decoder& decoder) {
  const bool has_conv = std::any_of(
      decoder.layer_types.begin(), decoder.layer_types.end(),
      [](const std::string& layer_type) { return layer_type == "conv"; });
  if (has_conv) {
    return "hybrid";
  }
  if (decoder.sliding_window.has_value()) {
    return "sliding_window";
  }
  if (decoder.num_key_value_heads > 0 && decoder.num_attention_heads > 0 &&
      decoder.num_key_value_heads < decoder.num_attention_heads) {
    return "gqa";
  }
  return "full";
}

ModelLoadInfo BuildModelLoadInfo(const Model& model) {
  const auto& config = *model.config_;
  const auto& decoder = config.model.decoder;

  ModelLoadInfo info;
  info.model_type = config.model.type;
  info.model_family = DeriveModelFamily(config.model.type);
  info.selected_device = to_string(model.p_device_->GetType());
  info.attention_type = DeriveAttentionType(decoder);
  info.vocab_size = config.model.vocab_size;
  info.context_length = config.model.context_length;
  info.num_hidden_layers = decoder.num_hidden_layers;
  info.hidden_size = decoder.hidden_size;
  info.num_attention_heads = decoder.num_attention_heads;
  info.num_key_value_heads = decoder.num_key_value_heads;
  info.is_in_memory = !config.model_data_spans_.empty();

  for (const auto& provider : decoder.session_options.providers) {
    if (!info.execution_providers.empty()) {
      info.execution_providers += ",";
    }
    info.execution_providers += provider;
  }

  if (decoder.session_options.intra_op_num_threads.has_value()) {
    info.intra_op_num_threads = *decoder.session_options.intra_op_num_threads;
  }
  if (decoder.session_options.graph_optimization_level.has_value()) {
    info.graph_optimization_level =
        static_cast<int>(*decoder.session_options.graph_optimization_level);
  }

  const auto& model_type = config.model.type;
  if (ModelType::IsMMM(model_type)) {
    info.modality = "multimodal";
  } else if (ModelType::IsVLM(model_type)) {
    info.modality = "vision";
  } else if (ModelType::IsALM(model_type) || ModelType::IsTransducer(model_type)) {
    info.modality = "audio";
  } else {
    info.modality = "text";
  }

  if (ModelType::IsTransducer(model_type)) {
    info.transcription_mode = "streaming";
  } else if (ModelType::IsALM(model_type)) {
    info.transcription_mode = "batch";
  }

  if (model.p_device_->GetType() == DeviceType::CUDA) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    model.p_device_->GetAvailableMemory(free_bytes, total_bytes);
    info.gpu_memory_mb = static_cast<int>(total_bytes / (1024 * 1024));
  }

  return info;
}

}  // namespace

#endif

std::shared_ptr<Model> CreateModelWithTelemetry(
    const std::function<std::shared_ptr<Model>()>& create_model) {
#if defined(ORTGENAI_ENABLE_TELEMETRY)
  auto& telemetry = GenAiTelemetry::Instance();
  telemetry.Initialize();
  telemetry.LogProcessInfo();
  const uint32_t session_id = telemetry.AllocateSessionId();

  if (telemetry.IsEnabled()) {
    const bool model_load_start_logged = telemetry.LogModelLoadStart(session_id);
    const auto start = std::chrono::steady_clock::now();
    try {
      auto model = create_model();
      if (model_load_start_logged) {
        telemetry.LogModelLoad(session_id, BuildModelLoadInfo(*model));
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        telemetry.LogModelLoadEnd(session_id, true, elapsed_ms);
      }
      model->telemetry_session_id_ = session_id;
      return model;
    } catch (const std::exception& error) {
      if (model_load_start_logged) {
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        telemetry.LogModelLoadEnd(session_id, false, elapsed_ms, error.what());
      }
      telemetry.LogRuntimeError(
          session_id, "std::exception", error.what(), "model_load");
      throw;
    }
  }

  auto model = create_model();
  model->telemetry_session_id_ = session_id;
  return model;
#else
  return create_model();
#endif
}

}  // namespace Generators
