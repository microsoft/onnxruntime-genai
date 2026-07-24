// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_tag_utils.h"
#include "model.h"

namespace Generators {

std::optional<int32_t> ResolveFallbackTokenId(const std::string& model_type,
                                              const std::string& tag_name,
                                              const Tokenizer& /*tokenizer*/) {
  // Hardcoded fallback token IDs for models whose genai_config.json doesn't yet include
  // bot/eot/bor/eor fields. Provides backward compatibility for Foundry Local when
  // consuming older model packages that predate these config fields.
  //
  // Model type  | Tag | Token string    | Token ID
  // ------------|-----|-----------------|--------
  // qwen2/qwen3| bot | <tool_call>     | 151657
  // qwen2/qwen3| eot | </tool_call>    | 151658
  // qwen2/qwen3| bor | <think>         | 151667
  // qwen2/qwen3| eor | </think>        | 151668
  // phi3        | bot | <|tool_call|>   | 200025
  // phi3        | eot | <|/tool_call|>  | 200026
  // clang-format off
  static const std::unordered_map<std::string, std::unordered_map<std::string, int32_t>> fallback_map = {
      {"qwen2", {{"bot", 151657}, {"eot", 151658}, {"bor", 151667}, {"eor", 151668}}},
      {"qwen3", {{"bot", 151657}, {"eot", 151658}, {"bor", 151667}, {"eor", 151668}}},
      {"phi3",  {{"bot", 200025}, {"eot", 200026}}},
  };
  // clang-format on

  auto type_it = fallback_map.find(model_type);
  if (type_it == fallback_map.end()) return std::nullopt;
  auto tag_it = type_it->second.find(tag_name);
  if (tag_it == type_it->second.end()) return std::nullopt;

  return tag_it->second;
}

}  // namespace Generators
