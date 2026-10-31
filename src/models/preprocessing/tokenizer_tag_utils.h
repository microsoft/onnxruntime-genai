// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace Generators {

struct Tokenizer;

// Resolves a fallback token ID for models whose genai_config.json doesn't yet include
// bot/eot/bor/eor token IDs in the model section. This exists specifically for
// Foundry Local backward compatibility with older model packages that predate
// these config fields.
//
// Returns the resolved token ID if found in the fallback map and tokenizer vocabulary,
// or std::nullopt if the model type/tag name is not in the map or the token string
// doesn't resolve in the vocabulary.
std::optional<int32_t> ResolveFallbackTokenId(const std::string& model_type,
                                              const std::string& tag_name,
                                              const Tokenizer& tokenizer);

}  // namespace Generators
