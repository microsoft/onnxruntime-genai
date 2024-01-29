// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "simdjson.h"
#include "tfmtok.h"

namespace tfm {

template <typename T>
inline simdjson::error_code TryToGetJson(const simdjson::dom::element& element,
                                         const std::string_view& key,
                                         T& value) {
  auto object = element.at_key(key);
  if (!object.is_null()) {
    auto err = object.get(value);
    if (err == simdjson::NO_SUCH_FIELD) {
      return simdjson::SUCCESS;
    }
    return err;
  }

  return simdjson::SUCCESS;
}

template <>
simdjson::error_code inline TryToGetJson<std::string>(
    const simdjson::dom::element& element, const std::string_view& key,
    std::string& value) {
  std::string_view raw_value;
  auto err = TryToGetJson(element, key, raw_value);
  if (err == simdjson::SUCCESS) {
    value = raw_value;
  }

  return err;
}

struct AddedToken final {
  std::string token_type_;
  std::string content_;
  bool lstrip_{};
  bool normalized_{};
  bool rstrip_{};
  bool single_word_{};

  bool FromJson(const simdjson::dom::element& element);
};

class TokenConfig final {
 public:
  TokenConfig(/* args */);
  ~TokenConfig();

 public:
  TfmStatus LoadJson(const std::string& json_path);

 private:
  bool FromJson(const simdjson::dom::element& element);

 public:
  bool add_bos_token_{};
  bool add_eos_token_{};
  bool clean_up_tokenization_spaces_{};
  double model_max_length_{};

  std::string pad_token_;
  std::string tokenizer_class_;

  AddedToken bos_token_;
  AddedToken eos_token_;
  AddedToken unk_token_;
};

}  // namespace tfm
