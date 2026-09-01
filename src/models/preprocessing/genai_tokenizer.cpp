// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "genai_tokenizer.h"

#include "models/model.h"
#include "models/preprocessing/tokenizer_tag_utils.h"
#include "tensor.h"

#include <algorithm>

namespace Generators {

std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id) {
  bool pad_right_{true};

  size_t max_length = 0;
  for (auto& sequence : sequences)
    max_length = std::max(max_length, sequence.size());

  std::vector<int32_t> result(max_length * sequences.size());
  std::span<int32_t> result_span(result);

  // Copy and pad the sequences with pad_token_id
  for (size_t i = 0; i < sequences.size(); i++) {
    auto output_span = result_span.subspan(i * max_length, max_length);
    auto input_span = sequences[i];

    auto pad_count = max_length - input_span.size();
    if (pad_right_) {
      std::copy(input_span.begin(), input_span.end(), output_span.begin());
      std::fill(output_span.end() - pad_count, output_span.end(), pad_token_id);
    } else {
      std::fill(output_span.begin(), output_span.begin() + pad_count, pad_token_id);
      std::copy(input_span.begin(), input_span.end(), output_span.begin() + pad_count);
    }
  }

  return result;
}

TokenizerStream::TokenizerStream(const Tokenizer& tokenizer)
    : tokenizer_{tokenizer.shared_from_this()} {
  CheckResult(OrtxCreate(kOrtxKindDetokenizerCache, cache_.Address()));
}

const std::string& TokenizerStream::Decode(int32_t token) {
  const char* string;
  CheckResult(OrtxDetokenizeCached(tokenizer_->tokenizer_, cache_, token, &string));
  chunk_ = string;
  return chunk_;
}

Tokenizer::Tokenizer(const Config& config) : bos_token_id_{config.model.bos_token_id},
                                             eos_token_id_{config.model.eos_token_id},
                                             pad_token_id_{config.model.pad_token_id},
                                             bot_token_id_{config.model.bot_token_id},
                                             eot_token_id_{config.model.eot_token_id},
                                             bor_token_id_{config.model.bor_token_id},
                                             eor_token_id_{config.model.eor_token_id} {
  // Default tokenizer options
  const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
  const char* values[] = {"false", "true"};

  // Resolve tokenizer_dir (may be empty, relative, absolute, or a "sha256:" shared-asset reference).
  const fs::path tokenizer_dir = config.ResolvePath(config.model.tokenizer_dir);
  CheckResult(OrtxCreateTokenizerWithOptions(tokenizer_.Address(), tokenizer_dir.string().c_str(), keys, values, 2));

  // Resolve any unset bot/eot/bor/eor IDs via model-type fallback strings.
  // Resolve any unset bot/eot/bor/eor IDs via model-type fallback.
  if (!bot_token_id_) bot_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::BotTokenIdName), *this);
  if (!eot_token_id_) eot_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::EotTokenIdName), *this);
  if (!bor_token_id_) bor_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::BorTokenIdName), *this);
  if (!eor_token_id_) eor_token_id_ = ResolveFallbackTokenId(config.model.type, std::string(Config::Defaults::EorTokenIdName), *this);
}

int32_t Tokenizer::GetBotTokenId() const {
  if (!bot_token_id_) throw std::runtime_error("bot_token_id is not defined for this model");
  return *bot_token_id_;
}

int32_t Tokenizer::GetEotTokenId() const {
  if (!eot_token_id_) throw std::runtime_error("eot_token_id is not defined for this model");
  return *eot_token_id_;
}

int32_t Tokenizer::GetBorTokenId() const {
  if (!bor_token_id_) throw std::runtime_error("bor_token_id is not defined for this model");
  return *bor_token_id_;
}

int32_t Tokenizer::GetEorTokenId() const {
  if (!eor_token_id_) throw std::runtime_error("eor_token_id is not defined for this model");
  return *eor_token_id_;
}

std::unique_ptr<TokenizerStream> Tokenizer::CreateStream() const {
  return std::make_unique<TokenizerStream>(*this);
}

void Tokenizer::UpdateOptions(const char* const* keys, const char* const* values, size_t num_options) {
  // Tap into ORT Extensions API
  CheckResult(OrtxUpdateTokenizerOptions(tokenizer_, const_cast<const char**>(keys), const_cast<const char**>(values), num_options));
}

std::vector<int32_t> Tokenizer::Encode(const char* text) const {
  OrtxPtr<OrtxTokenId2DArray> ids;
  CheckResult(OrtxTokenize(tokenizer_, &text, 1, ids.Address()));

  const extTokenId_t* tokens;
  size_t count;
  CheckResult(OrtxTokenId2DArrayGetItem(ids, 0, &tokens, &count));
  return {tokens, tokens + count};
}

std::string Tokenizer::Decode(std::span<const int32_t> tokens) const {
  OrtxPtr<OrtxStringArray> ortx_string_array;
  CheckResult(OrtxDetokenize1D(tokenizer_, reinterpret_cast<const uint32_t*>(tokens.data()), tokens.size(), ortx_string_array.Address()));

  const char* string;
  CheckResult(OrtxStringArrayGetItem(ortx_string_array, 0, &string));
  return string;
}

std::string Tokenizer::ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const {
  OrtxPtr<OrtxTensorResult> templated_text;
  CheckResult(OrtxApplyChatTemplate(tokenizer_, template_str, messages, tools, templated_text.Address(), add_generation_prompt, false /*tokenize*/));

  OrtxPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(templated_text, 0, tensor.Address()));

  const char* text_ptr{};
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr));

  return text_ptr;
}

std::string Tokenizer::ApplyChatTemplateWithOptions(const char* template_str, const char* messages, const char* tools,
                                                    const char* template_kwargs, bool add_generation_prompt) const {
  OrtxPtr<OrtxTensorResult> templated_text;
  CheckResult(OrtxApplyChatTemplateWithOptions(tokenizer_, template_str, messages, tools, template_kwargs,
                                               templated_text.Address(), add_generation_prompt,
                                               false /*tokenize*/));

  OrtxPtr<OrtxTensor> tensor;
  CheckResult(OrtxTensorResultGetAt(templated_text, 0, tensor.Address()));

  const char* text_ptr{};
  CheckResult(OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr));

  return text_ptr;
}

std::vector<int32_t> Tokenizer::EncodeBatch(std::span<const std::string> strings) const {
  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i].c_str()));
    span_sequences.emplace_back(sequences.back());
  }

  return PadInputs(span_sequences, pad_token_id_);
}

std::shared_ptr<Tensor> Tokenizer::EncodeBatch(std::span<const char*> strings) const {
  if (strings.empty()) {
    throw std::runtime_error("EncodeBatch: input strings must not be empty");
  }
  for (size_t i = 0; i < strings.size(); i++) {
    if (strings[i] == nullptr) {
      throw std::runtime_error("EncodeBatch: input string at index " + std::to_string(i) + " must not be null");
    }
  }

  std::vector<std::vector<int32_t>> sequences;
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < strings.size(); i++) {
    sequences.emplace_back(Encode(strings[i]));
    span_sequences.emplace_back(sequences.back());
  }

  auto encoded = PadInputs(span_sequences, pad_token_id_);  // TODO: Pad directly into tensor vs copying?

  auto shape = std::array<int64_t, 2>{static_cast<int64_t>(strings.size()), static_cast<int64_t>(encoded.size() / strings.size())};
  auto ort_tensor_ = OrtValue::CreateTensor<int32_t>(Ort::Allocator::GetWithDefaultOptions(), shape);
  auto tensor = std::make_shared<Tensor>(std::move(ort_tensor_));
  std::copy(encoded.begin(), encoded.end(), tensor->GetMutableData<int32_t>());

  return tensor;
}

std::vector<std::string> Tokenizer::DecodeBatch(std::span<const int32_t> sequences, size_t count) const {
  if (sequences.size() % count != 0)
    throw std::runtime_error("DecodeBatch: sequences must be evenly divisible by the count");
  size_t sequence_length = sequences.size() / count;
  std::vector<std::string> strings;
  for (size_t i = 0; i < count; i++)
    strings.emplace_back(Decode(sequences.subspan(sequence_length * i, sequence_length)));
  return strings;
}

int32_t Tokenizer::TokenToTokenId(const char* token) const {
  extTokenId_t token_id;
  CheckResult(OrtxConvertTokenToId(tokenizer_, token, &token_id));
  return token_id;
}

std::shared_ptr<Tokenizer> Model::CreateTokenizer() const {
  return std::make_shared<Tokenizer>(*config_);
}

}  // namespace Generators
