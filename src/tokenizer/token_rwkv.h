// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <charconv>
#include <optional>

#include "utils/unescape.h"
#include "trietree.hpp"
#include "tokenizer.h"

namespace tfm {
// This Trie Tree is C++ implementation of
// https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/rwkv_tokenizer.py
// Perf optimized by leveraging C++ features, but the algorithm is the same.
class RWKVTrieTree : public tfm::TrieTree<char> {
 public:
  static constexpr int kMaxTokenLength_ = 128;

  explicit RWKVTrieTree(char ch = 0) : TrieTree(ch) {}

  // keep the same function for source code understanding.
  void add(const std::string& key, int idx = 0,
           std::optional<int> value = std::optional<int>()) {
    Add(key, idx, value);
  }

  int find_longest(const std::string& key, size_t& idx) const {
    return FindLongest(key, idx);
  }
};

class RwkvTokenizer : public TokenizerImpl {
 public:
  RwkvTokenizer() = default;

  TfmStatus Onload() override;
  TfmStatus Encode(std::string_view input, std::vector<tfmTokenId_t>& ids) const override;

  TfmStatus Decode(const span<tfmTokenId_t const>& ids, std::string& text) const override;

 private:
  std::map<tfmTokenId_t, std::string> idx2token_;
  RWKVTrieTree root_;
};

}  // namespace tfm
