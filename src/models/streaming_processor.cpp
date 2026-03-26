// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <string>

#include "../generators.h"
#include "streaming_processor.h"

namespace Generators {

void StreamingProcessor::InitVadFromConfig(Model& model) {
  model_ = model.shared_from_this();  // Store for deferred creation via SetOption("vad_enabled", "true")
  auto& vad_config = model.config_->model.vad;
  if (vad_config.enabled) {
    EnableVadFromModel();
    min_silence_chunks_ = vad_config.min_silence_chunks;
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
  if (consecutive_silence_chunks_ > min_silence_chunks_) {
    return true;  // Enough consecutive silence — drop
  }

  return false;  // Not enough consecutive silence yet — keep to preserve context
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
  } else if (k == "vad_min_silence_chunks") {
    min_silence_chunks_ = std::stoi(value);
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
  } else if (k == "vad_min_silence_chunks") {
    return std::to_string(min_silence_chunks_);
  } else {
    throw std::runtime_error("Unknown StreamingProcessor option: '" + std::string(key) + "'");
  }
}

}  // namespace Generators
