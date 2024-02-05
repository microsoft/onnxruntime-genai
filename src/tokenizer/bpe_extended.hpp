// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "utils/unescape.h"
#include "trietree.hpp"
#include "bpe_utils.hpp"

namespace tfm::bpe {

class ExtendedToken {
 public:
  TfmStatus LoadAddedTokens(const std::string_view added_tokens[], size_t tok_num) {
    // int id = bpe::kInvalidTokenId;
    // for (size_t n = 0; n < tok_num; ++n) {
    //   std::string token(added_tokens[n]);  // Convert std::string_view to std::string
    //   id = GetTokenId(token);
    //   added_tokens_.Add(FromUTF8(added_tokens[n]), 0, std::make_optional(id));
    // }

    return {};
  }

  TfmStatus LoadAddedTokens(const simdjson::dom::element& added_tokens, std::map<int64_t, std::string>& dict) {
    for (simdjson::dom::object tok: added_tokens) {
        int id = bpe::kInvalidTokenId;
        std::string_view content;
        bool special = false;
        for (auto field: tok) {
          if (field.key == "id") {
            id = gsl::narrow_cast<int>(field.value.get_int64());
          } else if (field.key == "content") {
            content = field.value;
          } else if (field.key == "special") {
            special = field.value;
          }
        }
        if (id == bpe::kInvalidTokenId || content.empty() ){         // skip the token if id is not specified
          continue;
        }
        // Need to find the case where added_token cannot cover the special_token
        // if (special) {
        //   special_tokens_.Add(content, id);
        // }
        dict.emplace(id, std::string(content));
        added_tokens_.Add(FromUTF8(content), 0, std::make_optional(id));
    }
    return {};
  }


  // REF:
  //   https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/tokenization_utils.py#L52
  bpe::TokenPairs Split(const std::u32string& input) const {
    // split by added tokens
    bpe::TokenPairs added_result;
    bpe::TokenPairs final_result;
    added_tokens_.Split(input, added_result);
    for (const auto& [token, id] : added_result) {
      if (id != bpe::kInvalidTokenId) {
        final_result.emplace_back(token, id);
      } else {
        auto special_result = special_tokens_.SplitBySpecialTokens(token);
        for (const auto& [_token, _id] : special_result) {
          final_result.emplace_back(_token, _id);
        }
      }
    }

    return final_result;
  }

 private:
  SpecialTokenMap special_tokens_;
  TrieTree<char32_t> added_tokens_;
};

}  // namespace tfm::bpe
