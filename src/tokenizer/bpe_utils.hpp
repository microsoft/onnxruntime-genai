// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <list>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "utils/unicode.h"


namespace tfm::bpe {

using TokenPairs = std::vector<std::pair<std::u32string_view, int>>;
using u32string_view = std::u32string_view;

constexpr int kInvalidTokenId = -1;

class SpecialTokenMap {
 public:
  void Add(std::u32string p_str, int p_id) {
    auto it = token_map_.find(p_str);
    if (it != token_map_.end()) {
      assert(it->second == p_id && "Duplicate special tokens.");
    } else {
      token_map_[p_str] = p_id;
      token_list_.emplace_back(std::move(p_str), p_id);
    }
  }

  [[nodiscard]] TokenPairs SplitBySpecialTokens(const std::u32string_view& input) const {
    TokenPairs res;
    res.emplace_back(input, kInvalidTokenId);
    for (const auto& st : token_list_) {
      TokenPairs new_split_res;
      for (auto& str : res) {
        if (str.second != kInvalidTokenId) {
          new_split_res.emplace_back(str);
          continue;
        }

        auto it = str.first.begin();
        size_t search_pos = 0;
        while (it != str.first.end()) {
// works fine for all clang-based platform: macOS, Android, WebAssembly
#if defined(__clang__)
          auto search_it = std::search(it, str.first.end(), st.str.begin(), st.str.end());
#else
          auto search_it = std::search(it, str.first.end(),
                                       std::boyer_moore_searcher(st.str.begin(), st.str.end()));
#endif
          if (search_it == str.first.end()) {
            new_split_res.emplace_back(u32string_view(
                                           str.first.data() + search_pos, str.first.size() - search_pos),
                                       kInvalidTokenId);
            break;
          }

          auto prefixLen = search_it - it;
          if (prefixLen != 0) {
            new_split_res.emplace_back(u32string_view(str.first.data() + search_pos, prefixLen), kInvalidTokenId);
            search_pos += prefixLen;
          }

          new_split_res.emplace_back(u32string_view(str.first.data() + search_pos, st.str.size()), st.id);
          it = search_it + st.str.size();
          search_pos += st.str.size();
        }
      }

      std::swap(new_split_res, res);
    }

    return res;
  }

 private:
  struct SpecialTokenInfo {
    std::u32string str;
    int id;

    SpecialTokenInfo(std::u32string p_str, int p_id)
        : str(std::move(p_str)), id(p_id) {
    }
  };

  std::list<SpecialTokenInfo> token_list_;
  std::unordered_map<std::u32string, int> token_map_;
};

class TokenWithRegularExp {
 public:
  void Set(std::u32string_view val) {
    m_text = val;
  }

  std::pair<bool, std::u32string_view> GetNextToken() {
    while (!m_text.empty()) {
      auto res = TryMatch();
      if (res.empty()) {
        m_text = m_text.substr(1);
        continue;
      }
      return {true, res};
    }
    return {false, {}};
  }

 private:
  std::u32string_view TryMatch() {
    // python pattern:
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

    // 's|'t|'re|'ve|'m|'ll|'d|
    // Note: the sequential of the following
    // if statement should not be switched, which follows the python regex's syntax
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') ||
          (m_text[1] == U'm') || (m_text[1] == U'd')) {
        std::u32string_view res = m_text.substr(0, 2);
        m_text = m_text.substr(2);
        return res;
      }

      if (m_text.size() > 2) {
        if (((m_text[1] == U'r') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'v') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'l') && (m_text[2] == U'l'))) {
          std::u32string_view res = m_text.substr(0, 3);
          m_text = m_text.substr(3);
          return res;
        }
      }
    }

    // ?\p{L}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::L)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::L) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?\p{N}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::N)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::N) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?[^\s\p{L}\p{N}]+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (NotLNZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // \s+(?!\S)|\s+
    if ((!m_text.empty()) && (IsZ(m_text[0]))) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i])) break;
      }
      if ((i > 1) && (i != m_text.size())) {  //\s+(?!\S)
        i--;
        std::u32string_view res = m_text.substr(0, i);
        m_text = m_text.substr(i);
        return res;
      }
      // \s+
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return std::u32string_view{};
  }

  static bool IsZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::Z) != 0;
  }

  static bool NotLNZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    if (category & ufal::unilib::unicode::L) return false;
    if (category & ufal::unilib::unicode::N) return false;
    if (category & ufal::unilib::unicode::Z) return false;
    return true;
  }

 private:
  std::u32string_view m_text;
};

} // namespace tfm::bpe
