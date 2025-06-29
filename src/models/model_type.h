// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace Generators {

struct ModelType {
  // Large-language model (LLM)
  static constexpr std::array<std::string_view, 17> LLM = {"chatglm", "decoder", "gemma", "gemma2", "gemma3_text", "gpt2", "granite", "llama", "mistral", "nemotron", "olmo", "phi", "phimoe", "phi3", "phi3small", "qwen2", "qwen3"};

  // Vision-language model (VLM)
  static constexpr std::array<std::string_view, 2> VLM = {"gemma3", "phi3v"};

  // Audio-language model (ALM)
  static constexpr std::array<std::string_view, 1> ALM = {"whisper"};

  // Multi-modal model (MMM)
  static constexpr std::array<std::string_view, 1> MMM = {"phi4mm"};

  // Pipeline (Pipe)
  static constexpr std::array<std::string_view, 1> Pipe = {"decoder-pipeline"};

  inline static bool IsLLM(const std::string& model_type) {
    return std::find(LLM.begin(), LLM.end(), model_type) != LLM.end();
  }

  inline static bool IsVLM(const std::string& model_type) {
    return std::find(VLM.begin(), VLM.end(), model_type) != VLM.end();
  }

  inline static bool IsALM(const std::string& model_type) {
    return std::find(ALM.begin(), ALM.end(), model_type) != ALM.end();
  }

  inline static bool IsMMM(const std::string& model_type) {
    return std::find(MMM.begin(), MMM.end(), model_type) != MMM.end();
  }

  inline static bool IsPipe(const std::string& model_type) {
    return std::find(Pipe.begin(), Pipe.end(), model_type) != Pipe.end();
  }
};

}  // namespace Generators