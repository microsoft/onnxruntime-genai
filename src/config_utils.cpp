// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "config_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace Generators {

namespace {
constexpr std::string_view kKvCacheQuantizationBitsDisabled = "0";
constexpr std::string_view kKvCacheQuantizationBits4 = "4";
constexpr std::string_view kKvCacheQuantizationBits8 = "8";
constexpr std::string_view kKvCacheQuantizationBitsOptionName = "kvCacheQuantizationBits";
}  // namespace

int GetKvCacheQuantizationBits(const Config::SessionOptions& session_options,
                               std::string_view provider_name) {
  const auto normalized_provider = NormalizeProviderName(provider_name);
  for (auto provider_options_it = session_options.provider_options.rbegin();
       provider_options_it != session_options.provider_options.rend();
       ++provider_options_it) {
    if (NormalizeProviderName(provider_options_it->name) != normalized_provider) {
      continue;
    }

    const auto option_it = std::find_if(provider_options_it->options.begin(),
                                        provider_options_it->options.end(),
                                        [](const Config::NamedString& option) {
                                          return option.first == kKvCacheQuantizationBitsOptionName;
                                        });
    if (option_it == provider_options_it->options.end() || option_it->second == kKvCacheQuantizationBitsDisabled) {
      return 0;
    }

    const auto& quantization_bits = option_it->second;

    if (quantization_bits == kKvCacheQuantizationBits4) {
      return 4;
    }

    if (quantization_bits == kKvCacheQuantizationBits8) {
      return 8;
    }

    throw std::runtime_error("Unsupported kvCacheQuantizationBits value: " + quantization_bits +
                             ". Only " + std::string(kKvCacheQuantizationBitsDisabled) +
                             " (disabled), " + std::string(kKvCacheQuantizationBits4) +
                             ", and " + std::string(kKvCacheQuantizationBits8) +
                             " are supported.");
  }

  return 0;
}

}  // namespace Generators
