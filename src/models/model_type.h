// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// --------------------------------------------------------------------------
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// Portions of this file consist of AI generated content.
// --------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace Generators {

// model.type predicate table. As of issue #2114 PR5 these are NO LONGER the runtime model-CLASS
// dispatch: production routing is structural (see Generators::ClassifyStructuralRoute / CreatePipeline
// in model.cpp). They are retained because every one still has a non-dispatch caller:
//   * TranslateV1ToPipeline (config.cpp) uses Is{QwenVLFamily,Pixtral,VLM,MMM,ALM,Pipe} to SYNTHESIZE
//     the structural pipeline signals from a legacy v1 config -- the one legitimate type->structure map;
//   * the transducer bypass uses Is{RNNT,TDT,Transducer} (generators.cpp, ClassifyStructuralRoute);
//   * config.cpp's context_length guard uses IsRNNT;
//   * kv_cache.cpp's conv-state cache uses IsLFM2 (also the ClassifyStructuralRoute lfm2 residual);
//   * ClassifyLegacyRoute (model.cpp) keeps the full table as the ground-truth oracle the PR5
//     zero-regression test (PipelineDispatchTests) compares the structural router against.
// None became unused, so none were deleted (issue §5's "delete model_type.h" reduces to "keep").
struct ModelType {
  inline static bool IsLLM(const std::string& model_type) {
    // Large-language model (LLM)
    static constexpr std::array<std::string_view, 26> LLM = {"chatglm", "decoder", "ernie4_5", "gemma", "gemma2", "gemma3_text", "gemma4_text", "gpt2", "gptoss", "granite", "hunyuandensev1", "internlm2", "lfm2", "llama", "mistral", "nemotron", "olmo", "phi", "phimoe", "phi3", "phi3small", "qwen2", "qwen3", "qwen3_5_moe_text", "qwen3_5_text", "smollm3"};
    return std::find(LLM.begin(), LLM.end(), model_type) != LLM.end();
  }

  inline static bool IsVLM(const std::string& model_type) {
    // Vision-language model (VLM)
    static constexpr std::array<std::string_view, 9> VLM = {"fara", "gemma3", "mistral3", "phi3v", "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "videochat_flash_qwen"};
    return std::find(VLM.begin(), VLM.end(), model_type) != VLM.end();
  }

  inline static bool IsQwenVLFamily(const std::string& model_type) {
    // Qwen-VL family: models requiring 3D mRoPE position IDs
    static constexpr std::array<std::string_view, 5> QwenVL = {"fara", "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe"};
    return std::find(QwenVL.begin(), QwenVL.end(), model_type) != QwenVL.end();
  }

  inline static bool IsPixtralFamily(const std::string& model_type) {
    // Pixtral family: per-image vision loop with variable resolution
    return model_type == "mistral3";
  }

  inline static bool IsALM(const std::string& model_type) {
    // Audio-language model (ALM)
    static constexpr std::array<std::string_view, 1> ALM = {"whisper"};
    return std::find(ALM.begin(), ALM.end(), model_type) != ALM.end();
  }

  inline static bool IsRNNT(const std::string& model_type) {
    // RNNT models bypass the search/logits pipeline entirely.
    static constexpr std::array<std::string_view, 1> rnnt_types = {"nemotron_speech"};
    return std::find(rnnt_types.begin(), rnnt_types.end(), model_type) != rnnt_types.end();
  }

  inline static bool IsTDT(const std::string& model_type) {
    static constexpr std::array<std::string_view, 1> TDT = {"parakeet_tdt"};
    return std::find(TDT.begin(), TDT.end(), model_type) != TDT.end();
  }

  // Transducer models (RNNT, TDT) bypass the standard search/logits pipeline
  // and drive a custom encoder/decoder/joiner loop via TransducerState.
  inline static bool IsTransducer(const std::string& model_type) {
    return IsRNNT(model_type) || IsTDT(model_type);
  }

  inline static bool IsMMM(const std::string& model_type) {
    // Multi-modal model (MMM)
    static constexpr std::array<std::string_view, 2> MMM = {"gemma4", "phi4mm"};
    return std::find(MMM.begin(), MMM.end(), model_type) != MMM.end();
  }

  inline static bool IsPipe(const std::string& model_type) {
    // Pipeline (Pipe)
    static constexpr std::array<std::string_view, 1> Pipe = {"decoder-pipeline"};
    return std::find(Pipe.begin(), Pipe.end(), model_type) != Pipe.end();
  }

  inline static bool IsLFM2(const std::string& model_type) {
    // Liquid Foundation Model 2: hybrid attention/conv architecture with conv state cache
    return model_type == "lfm2";
  }
};

}  // namespace Generators
