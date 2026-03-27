// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <stdexcept>
#include <string>

#include "../generators.h"
#include "streaming_processor.h"

namespace Generators {

void StreamingProcessor::InitVadFromConfig(Model& model) {
  model_ = model.shared_from_this();
  auto& vad_config = model.config_->model.vad;
  // VAD is enabled if the "vad" section exists in genai_config.json (detected by non-empty filename)
  if (!vad_config.filename.empty()) {
    EnableVadFromModel();

    // Convert ms-based config to chunk counts
    int sample_rate = model.config_->model.sample_rate;
    int chunk_samples = model.config_->model.chunk_samples;
    float chunk_duration_ms = (static_cast<float>(chunk_samples) / sample_rate) * 1000.0f;

    silence_duration_chunks_ = std::max(1, static_cast<int>(vad_config.silence_duration_ms / chunk_duration_ms));
    prefix_padding_chunks_ = std::max(1, static_cast<int>(vad_config.prefix_padding_ms / chunk_duration_ms));
  }
}

void StreamingProcessor::EnableVadFromModel() {
  if (!model_) {
    throw std::runtime_error("Cannot enable VAD: no model reference. Call InitVadFromConfig first.");
  }
  vad_ = CreateSileroVad(*model_);
  consecutive_silence_chunks_ = 0;
}

bool StreamingProcessor::ShouldDropChunk(const float* chunk_data, size_t chunk_size) {
  if (!vad_) {
    return false;
  }

  bool has_speech = vad_->ContainsSpeech(chunk_data, chunk_size);
  if (has_speech) {
    consecutive_silence_chunks_ = 0;
    return false;
  }

  consecutive_silence_chunks_++;

  // Keep at least prefix_padding_chunks_ of silence for context before potential speech.
  // After silence_duration_chunks_ of continuous silence, start dropping.
  int effective_threshold = std::max(prefix_padding_chunks_, silence_duration_chunks_);
  if (consecutive_silence_chunks_ >= effective_threshold) {
    return true;
  }

  return false;  // Within tolerance — keep to preserve context
}

void StreamingProcessor::SetOption(const char* key, const char* value) {
  std::string_view k{key};

  if (k == "vad_enabled") {
    std::string_view v{value};
    if (v == "true" || v == "1") {
      if (!vad_) {
        EnableVadFromModel();
      }
    } else if (v == "false" || v == "0") {
      vad_.reset();
      consecutive_silence_chunks_ = 0;
    } else {
      throw std::runtime_error(
          "Invalid value for vad_enabled: '" + std::string(value) + "'. Expected 'true' or 'false'.");
    }
  } else if (k == "vad_threshold") {
    float threshold = std::stof(value);
    if (vad_) {
      vad_->SetThreshold(threshold);
    }
  } else if (k == "silence_duration_ms") {
    int ms = std::stoi(value);
    if (model_) {
      float chunk_duration_ms = (static_cast<float>(model_->config_->model.chunk_samples) /
                                 model_->config_->model.sample_rate) * 1000.0f;
      silence_duration_chunks_ = std::max(1, static_cast<int>(ms / chunk_duration_ms));
    }
  } else if (k == "prefix_padding_ms") {
    int ms = std::stoi(value);
    if (model_) {
      float chunk_duration_ms = (static_cast<float>(model_->config_->model.chunk_samples) /
                                 model_->config_->model.sample_rate) * 1000.0f;
      prefix_padding_chunks_ = std::max(1, static_cast<int>(ms / chunk_duration_ms));
    }
  } else {
    throw std::runtime_error("Unknown StreamingProcessor option: '" + std::string(key) + "'");
  }
}

std::string StreamingProcessor::GetOption(const char* key) const {
  std::string_view k{key};

  if (k == "vad_enabled") {
    return vad_ ? "true" : "false";
  } else if (k == "vad_threshold") {
    return std::to_string(vad_ ? vad_->GetThreshold() : 0.5f);
  } else if (k == "silence_duration_ms") {
    if (model_) {
      float chunk_duration_ms = (static_cast<float>(model_->config_->model.chunk_samples) /
                                 model_->config_->model.sample_rate) * 1000.0f;
      return std::to_string(static_cast<int>(silence_duration_chunks_ * chunk_duration_ms));
    }
    return "500";
  } else if (k == "prefix_padding_ms") {
    if (model_) {
      float chunk_duration_ms = (static_cast<float>(model_->config_->model.chunk_samples) /
                                 model_->config_->model.sample_rate) * 1000.0f;
      return std::to_string(static_cast<int>(prefix_padding_chunks_ * chunk_duration_ms));
    }
    return "300";
  } else {
    throw std::runtime_error("Unknown StreamingProcessor option: '" + std::string(key) + "'");
  }
}

}  // namespace Generators
