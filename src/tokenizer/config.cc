// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <fstream>
#include <streambuf>
#include <filesystem>

#include "config.h"

namespace tfm {

bool AddedToken::FromJson(const simdjson::dom::element& element) {
  if (element.is_string()) {
    content_ = element.get_c_str();
    return true;
  }
  if (TryToGetJson(element, "__type", token_type_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "content", content_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "lstrip", lstrip_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "normalized", normalized_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "rstrip", rstrip_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "single_word", single_word_) != simdjson::SUCCESS)
    return false;
  return true;
}

TokenConfig::TokenConfig() = default;

TokenConfig::~TokenConfig() = default;

static std::string PatchJsonText(const std::string& json_path) {
  std::string json_text;
  std::ifstream t(json_path);

  t.seekg(0, std::ios::end);
  json_text.reserve(t.tellg());
  t.seekg(0, std::ios::beg);

  json_text.assign((std::istreambuf_iterator<char>(t)),
                   std::istreambuf_iterator<char>());

  for (size_t n = 0; n < json_text.size(); ++n) {
    size_t num_len = 0;
    while (std::isdigit(json_text[n])) {
      num_len++;
      n++;
      if (n >= json_text.size()) {
        break;
      }
    }

    // if some number is too long, simpjson will fail to parse it
    if (num_len > 20) {
      json_text[n - 2] = '.';  // convert it to a floating number
    }
  }

  return json_text;
}

TfmStatus TokenConfig::LoadJson(const std::string& json_path) {
  simdjson::dom::parser parser;
  simdjson::dom::element root;

  if (!std::filesystem::exists(
          std::filesystem::path(json_path).lexically_normal())) {
    return {kTfmErrorInvalidFile, std::string(json_path) + " not found"};
  }
  std::string json_text = PatchJsonText(json_path);
  simdjson::error_code error = parser.parse(json_text).get(root);
  if (error) {
    std::string error_msg = simdjson::error_message(error);
    return {kTfmErrorInvalidFile, error_msg};
  }

  if (!FromJson(root)) {
    return {kTfmErrorInvalidFile, "failed to parse json config file"};
  }

  return {};
}

bool TokenConfig::FromJson(const simdjson::dom::element& element) {
  if (TryToGetJson(element, "add_bos_token", add_bos_token_) !=
      simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "add_eos_token", add_eos_token_) !=
      simdjson::SUCCESS)
    return false;
  if (!bos_token_.FromJson(element["bos_token"]))
    return false;
  if (TryToGetJson(element, "clean_up_tokenization_spaces",
                   clean_up_tokenization_spaces_) != simdjson::SUCCESS)
    return false;
  if (!eos_token_.FromJson(element["eos_token"]))
    return false;
  if (TryToGetJson(element, "model_max_length", model_max_length_) !=
      simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "pad_token", pad_token_) != simdjson::SUCCESS)
    return false;
  if (TryToGetJson(element, "tokenizer_class", tokenizer_class_) !=
      simdjson::SUCCESS)
    return false;
  if (!unk_token_.FromJson(element["unk_token"]))
    return false;
  return true;
}

}  // namespace tfm
