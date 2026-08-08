// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_tag_utils.h"
#include "model.h"
#include "../config.h"

namespace Generators {

std::optional<int32_t> ResolveFallbackTokenId(const std::string& model_type,
                                              const std::string& tag_name,
                                              const Tokenizer& /*tokenizer*/) {
  // Hardcoded fallback token IDs for models whose genai_config.json doesn't yet include
  // bot/eot/bor/eor fields. Provides backward compatibility for Foundry Local when
  // consuming older model packages that predate these config fields.
  //
  // Model type  | Tag            | Token string    | Token ID
  // ------------|----------------|-----------------|--------
  // qwen2/qwen3| bot_token_id   | <tool_call>     | 151657
  // qwen2/qwen3| eot_token_id   | </tool_call>    | 151658
  // qwen3      | bor_token_id   | <think>         | 151667
  // qwen3      | eor_token_id   | </think>        | 151668
  // phi3        | bot_token_id   | <|tool_call|>   | 200025
  // phi3        | eot_token_id   | <|/tool_call|>  | 200026

  using D = Config::Defaults;
  // clang-format off
  static const std::unordered_map<std::string, std::unordered_map<std::string_view, int32_t>> fallback_map = {
      {"qwen2", {{D::BotTokenIdName, 151657}, {D::EotTokenIdName, 151658}}},
      {"qwen3", {{D::BotTokenIdName, 151657}, {D::EotTokenIdName, 151658}, {D::BorTokenIdName, 151667}, {D::EorTokenIdName, 151668}}},
      {"phi3",  {{D::BotTokenIdName, 200025}, {D::EotTokenIdName, 200026}}},
  };
  // clang-format on

  auto type_it = fallback_map.find(model_type);
  if (type_it == fallback_map.end()) return std::nullopt;
  auto tag_it = type_it->second.find(tag_name);
  if (tag_it == type_it->second.end()) return std::nullopt;

  return tag_it->second;
}

}  // namespace Generators
