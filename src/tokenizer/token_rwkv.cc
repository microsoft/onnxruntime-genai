// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "token_rwkv.h"

using namespace tfm;

static std::string MakeString(const char* s1, std::string_view s2) {
  return std::string(s1) + std::string(s2);
}

TfmStatus RwkvTokenizer::Onload() {
  auto text_tokens = GetDataDir();
  std::istringstream file(text_tokens);
  std::string line;

  while (std::getline(file, line)) {
    auto l_ws = line.find(' ');
    auto r_ws = line.rfind(' ');
    if (l_ws == std::string::npos || r_ws == std::string::npos || l_ws == r_ws) {
      return {kTfmErrorInvalidFile, MakeString("[TrieTokenizer] vocab line: ", line)};
    }

    int idx = 0;
    std::from_chars(line.data(), line.data() + line.size(), idx);
    if (idx == 0) {
      return {kTfmErrorInvalidFile, MakeString("[TrieTokenizer] bad index in vocab line: ", line)};
    }

    std::string raw = line.substr(line.find(' ') + 1, line.rfind(' ') - line.find(' ') - 1);
    std::string x;
    int key_length = 0;
    if (UnquoteString(raw, x)) {
      std::from_chars(line.data() + r_ws + 1, line.data() + line.size(), key_length);
    }
    if (x.length() != key_length) {
      return {kTfmErrorInvalidFile, MakeString("[TrieTokenizer] bad len in vocab line: ", line)};
    }

    idx2token_[idx] = x;
  }

  for (const auto& kv : idx2token_) {
    root_.add(kv.second, 0, kv.first);
  }

  return {};
}

TfmStatus RwkvTokenizer::Encode(std::string_view src, std::vector<tfmTokenId_t>& ids) const {
  size_t idx = 0;
  std::vector<tfmTokenId_t>& tokens = ids;
  while (idx < src.length()) {
    auto result = root_.find_longest(std::string(src), idx);
    tokens.push_back(result);
  }
  return {};
}

TfmStatus RwkvTokenizer::Decode(const span<tfmTokenId_t const>& ids, std::string& text) const {
  std::string result;
  for (auto i : ids) {
    result += idx2token_.at(i);
  }
  text = result;
  return {};
}
