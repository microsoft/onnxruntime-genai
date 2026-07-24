// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_tag_utils.h"
#include "model.h"

namespace Generators {

std::optional<int32_t> ResolveFallbackTokenId(const std::string& model_type,
                                              const std::string& tag_name,
                                              const Tokenizer& tokenizer) {
  static const std::unordered_map<std::string, std::unordered_map<std::string, std::string>> fallback_map = {
      {"qwen2", {{"bot", "<tool_call>"}, {"eot", "</tool_call>"}, {"bor", "<think>"}, {"eor", "</think>"}}},
      {"qwen3", {{"bot", "<tool_call>"}, {"eot", "</tool_call>"}, {"bor", "<think>"}, {"eor", "</think>"}}},
      {"phi3", {{"bot", "<|tool_call|>"}, {"eot", "<|/tool_call|>"}}},
      {"gptoss", {{"bot", "<|start|>"}, {"eot", "<|call|>"}}},
  };

  auto type_it = fallback_map.find(model_type);
  if (type_it == fallback_map.end()) return std::nullopt;
  auto tag_it = type_it->second.find(tag_name);
  if (tag_it == type_it->second.end()) return std::nullopt;

  int32_t resolved = tokenizer.TokenToTokenId(tag_it->second.c_str());
  if (resolved >= 0) return resolved;
  return std::nullopt;
}

}  // namespace Generators
