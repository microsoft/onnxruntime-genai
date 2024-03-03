// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <unordered_map>
#include <iostream>
#include <utility>
#include <charconv>
#include <limits>

#include "simdjson.h"
#include "gsl/narrow"
#include "utils/unescape.h"
#include "utils/unicode.h"
#include "trietree.hpp"
#include "bpe_utils.hpp"

namespace tfm {

inline bool IsSpmByteWord(std::string_view word) {
  return word.size() == 6 && word[0] == '<' && word[1] == '0' && word[2] == 'x' && word[5] == '>';
}

class BpeEncoder {
 public:
  BpeEncoder() = default;

  TfmStatus Load(const simdjson::dom::element& tok_json, const TokenConfig& config) {
    auto model_node = tok_json.at_key("model");
    if (model_node.is_null()) {
      return TfmStatus{kTfmErrorInvalidFile, "Cannot find the model key in the the tokenizer.json"};
    }

    auto unk_token_obj = model_node.at_key("unk_token");
    std::string unk_token = config.unk_token_.content_;
    if (!unk_token_obj.is_null()) {
      unk_token = model_node.at_key("unk_token").get_c_str().value();
      if (config.unk_token_.content_ != unk_token) {
        return TfmStatus{kTfmErrorInvalidFile,
                         "The unk_token in the tokenizer.json is not the same as the one in the config"};
      }
    }
    if (model_node.at_key("vocab").is_null() || model_node.at_key("merges").is_null()) {
      return TfmStatus{kTfmErrorInvalidFile, "Cannot find the vocab/merges key in the tokenizer.json"};
    }

    auto ewsuffix = model_node.at_key("end_of_word_suffix");
    if (ewsuffix.is_string()) {
      end_of_word_suffix_ = model_node.at_key("end_of_word_suffix").get_c_str().value();
    }

    simdjson::dom::object vocab_obj;
    auto error = model_node.at_key("vocab").get(vocab_obj);
    if (error) {
      return TfmStatus{kTfmErrorInvalidFile, "Cannot find the vocab key in the the tokenizer.json"};
    }

    std::string spm_byte;
    for (auto [_key, _value] : vocab_obj) {
      uint32_t id = gsl::narrow_cast<uint32_t>(_value.get_uint64());
      if (IsSpmByteWord(_key)) {
        char buf[3] = {_key[3], _key[4], 0};  // something like <0x20>
        spm_byte = {static_cast<char>(strtol(buf, NULL, 16))};
        _key = spm_byte;
      }

      vocab_map_[std::string(_key)] = id;
      if (id > max_token_id_) {
        max_token_id_ = id;
      }
    }

    auto it = vocab_map_.find(unk_token);
    if (it != vocab_map_.end()) {
      unk_id_ = it->second;
    } else {
      auto id = gsl::narrow<uint32_t>(vocab_map_.size());
      vocab_map_[unk_token] = id;
      unk_id_ = id;
    }

    simdjson::dom::element merges_node = model_node.at_key("merges");
    if (!merges_node.is_array()) {
      return TfmStatus{kTfmErrorInvalidFile, "Cannot find the merges key in the the tokenizer.json"};
    }

    return LoadMerges(merges_node);
  }

  TfmStatus LoadMerges(simdjson::dom::element merges_node) {
    auto lines = merges_node.get_array();
    for (auto each_line : lines) {
      std::string_view line = each_line.get_string();
      if (line.empty()) continue;
      auto pos = line.find(' ');
      if (pos == std::string::npos) {
        return {kTfmErrorInvalidFile, "Cannot know how to parse line: " + std::string(line)};
      }
      std::string w1(line.substr(0, pos));
      std::string w2(line.substr(pos + 1));
      int token_length = gsl::narrow<int>(w1.length() + w2.length());
      if (w2.find("</w>") != std::string::npos || w1.find("</w>") != std::string::npos) {
        token_length -= 4;
      }
      auto iw1 = GetTokenId(w1);
      auto iw2 = GetTokenId(w2);
      auto iww = GetTokenId(w1 + w2);
      BpeNode value{iww, gsl::narrow<uint32_t>(bpe_rank_.size()), token_length};
      bpe_rank_[GetRankKey(iw1, iw2)] = value;
    }

    return {};
  }

  void PerformBPE(std::list<std::pair<uint32_t, uint32_t>>& vals) const {
    while (vals.size() >= 2) {
      auto pos_it = vals.end();
      uint32_t min_val = std::numeric_limits<uint32_t>::max();
      uint32_t ori_id1 = 0, ori_id2 = 0;
      uint32_t aim_id = 0;
      int token_length = 0;
      for (auto it = vals.begin(); it != vals.end(); ++it) {
        auto it2 = it;
        ++it2;
        if (it2 == vals.end()) {
          break;
        }

        auto map_it = bpe_rank_.find(GetRankKey(it->first, it2->first));
        if (map_it == bpe_rank_.end()) {
          continue;
        }

        if (min_val > map_it->second.value) {
          ori_id1 = it->first;
          ori_id2 = it2->first;
          min_val = map_it->second.value;
          pos_it = it;
          aim_id = map_it->second.id;
        }
      }

      if (pos_it == vals.end()) {
        break;
      }

      token_length = pos_it->second;
      pos_it = vals.erase(pos_it);
      pos_it->first = aim_id;
      pos_it->second += token_length;
      for (++pos_it; pos_it != vals.end(); ++pos_it) {
        if (pos_it->first != ori_id1) continue;
        auto it2 = pos_it;
        ++it2;
        if (it2 == vals.end()) break;
        if (it2->first != ori_id2) continue;
        token_length = pos_it->second;
        pos_it = vals.erase(pos_it);
        pos_it->first = aim_id;
        pos_it->second += token_length;
      }
    }
  }

  uint32_t GetTokenId(const std::string& key) const {
    auto it = vocab_map_.find(key);
    if (it != end(vocab_map_)) {
      return it->second;
    } else {
      return unk_id_;
    }
  }

  uint32_t GetTokenId(std::string_view key, tfmTokenId_t default_id = bpe::kInvalidTokenId) const {
    auto it = vocab_map_.find(std::string(key));
    return it != end(vocab_map_) ? it->second : default_id;
  }

  std::vector<std::string_view> BuildDecoder() const {
    std::vector<std::string_view> decoder;
    decoder.resize(static_cast<size_t>(max_token_id_) + 1);
    for (const auto& [str, id] : vocab_map_) {
      assert(id <= max_token_id_);
      decoder[id] = str.c_str();
    }
    return decoder;
  }

  const std::string& GetEndWordSuffix() const { return end_of_word_suffix_; }

 private:
  struct BpeNode {
    uint32_t id;
    uint32_t value;
    int length;
  };

  static uint64_t GetRankKey(uint32_t i0, uint32_t i1) {
    return (static_cast<uint64_t>(i1) << 32) | (i0 & 0xFFFFFFFFLL);
  }

 private:
  std::map<uint64_t, BpeNode> bpe_rank_;
  std::unordered_map<std::string, uint32_t> vocab_map_;

  uint32_t unk_id_ = std::numeric_limits<uint32_t>::max();
  uint32_t max_token_id_ = 0;
  std::string end_of_word_suffix_;
};

}  // namespace tfm
