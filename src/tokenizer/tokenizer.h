// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "config.h"

namespace tfm {

class TokenizerImpl : public Tokenizer {
 public:
  TokenizerImpl() = default;
  virtual ~TokenizerImpl() = default;

  const TokenConfig* GetConfig() const { return token_cfg_.get(); }
  void BindConfig(std::unique_ptr<TokenConfig> token_cfg) { token_cfg_ = std::move(token_cfg); }
  const std::string& GetDataDir() const { return tokenizer_dir_; }
  void SetDataDir(const std::string& dir) { tokenizer_dir_ = dir; }

 public:
  TfmStatus OnLoad() override;

  TfmStatus BatchEncode(const std::vector<std::string_view>& input, std::vector<std::vector<tfmTokenId_t>>& t_ids) const override;

  TfmStatus BatchDecode(const std::vector<span<tfmTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const override;

  TfmStatus Id2Token(tfmTokenId_t /* id */, std::string& /* token */, DecoderState** /* state */) const override;

 public:
  virtual TfmStatus Encode(std::string_view /* input */, std::vector<tfmTokenId_t>& /* ids */) const {
    return TfmStatus::OK();
  }

  virtual TfmStatus Decode(const span<tfmTokenId_t const>& /* ids */, std::string& /* text */) const {
    return TfmStatus::OK();
  }

 private:
  std::string tokenizer_dir_;
  std::unique_ptr<TokenConfig> token_cfg_;
};

}  // namespace tfm
