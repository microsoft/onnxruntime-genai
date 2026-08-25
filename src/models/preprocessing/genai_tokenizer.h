// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "generator/generators.h"
#include "models/utils.h"
#include "ortx_tokenizer.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Generators {

struct Config;
struct Tensor;

struct Tokenizer;

struct TokenizerStream : LeakChecked<TokenizerStream> {
  TokenizerStream(const Tokenizer& tokenizer);

  const std::string& Decode(int32_t token);

 private:
  std::shared_ptr<const Tokenizer> tokenizer_;
  OrtxPtr<OrtxObject> cache_;
  std::string chunk_;
};

// Turn an array of ragged token sequences into a 2D input suitable for batching. Handles padding for the model.
std::vector<int32_t> PadInputs(std::span<std::span<const int32_t>> sequences, int32_t pad_token_id);

struct Tokenizer : std::enable_shared_from_this<Tokenizer>, LeakChecked<Tokenizer>, ExternalRefCounted<Tokenizer> {
  Tokenizer(Config& config);

  std::unique_ptr<TokenizerStream> CreateStream() const;

  void UpdateOptions(const char* const* keys, const char* const* values, size_t num_options);
  std::vector<int32_t> Encode(const char* text) const;
  std::string Decode(std::span<const int32_t> tokens) const;
  std::string ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const;
  std::string ApplyChatTemplateWithOptions(const char* template_str, const char* messages, const char* tools,
                                           const char* template_kwargs, bool add_generation_prompt) const;

  std::vector<int32_t> EncodeBatch(std::span<const std::string> strings) const;
  std::shared_ptr<Tensor> EncodeBatch(std::span<const char*> strings) const;
  std::vector<std::string> DecodeBatch(std::span<const int32_t> sequences, size_t count) const;

  int32_t TokenToTokenId(const char* token) const;
  int32_t GetBosTokenId() const { return bos_token_id_; }
  const std::vector<int32_t>& GetEosTokenIds() const { return eos_token_id_; }
  int32_t GetPadTokenId() const { return pad_token_id_; }

  int32_t GetBotTokenId() const;
  int32_t GetEotTokenId() const;
  int32_t GetBorTokenId() const;
  int32_t GetEorTokenId() const;

  OrtxPtr<OrtxTokenizer> tokenizer_;

 private:
  int32_t bos_token_id_;
  std::vector<int32_t> eos_token_id_;
  int32_t pad_token_id_;
  std::optional<int32_t> bot_token_id_;
  std::optional<int32_t> eot_token_id_;
  std::optional<int32_t> bor_token_id_;
  std::optional<int32_t> eor_token_id_;
};

}  // namespace Generators
