// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <set>
#include <string>
#include "tokenizer.h"

#include "bpe_encoder.hpp"

namespace tfm {

class BPETokenizer : public TokenizerImpl {
 public:
  BPETokenizer();
  ~BPETokenizer() override;

 public:
  TfmStatus Onload() override;
  TfmStatus Encode(std::string_view input, std::vector<tfmTokenId_t>& ids) const override;
  TfmStatus Decode(const span<tfmTokenId_t const>& ids, std::string& text) const override;
  std::string_view ModelName() const;

 private:
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<tfmTokenId_t> Encode(std::string_view sv_input,
                                   int64_t max_length,
                                   bool compute_offset_mapping,
                                   std::list<OffsetMappingType>& offset_map) const;
  void CreateByteEncoder();
  void LoadPredefinedTokens(const TokenConfig& config);

 protected:
  std::string_view model_name_{"GPT2"};

  std::string bos_token_{"<|endoftext|>"};
  std::string eos_token_{"<|endoftext|>"};
  std::string unk_token_{"<|endoftext|>"};
  std::string pad_token_{};  // no padding by default

  bool en_normalization_{};
  bool skip_special_tokens_{true};
  bool whitespace_token_{};
  std::vector<std::string_view> arr_vocab_;
  std::map<int64_t, std::string> added_tokens_;
  std::set<int64_t> all_special_ids_;

  uint32_t byte_encoder_[256] = {};
  std::unordered_map<char32_t, unsigned char> byte_decoder_;
  std::unique_ptr<BpeEncoder> bbpe_encoder_;

  int64_t padding_length_ = -1;
  uint32_t unk_token_id_{};
  uint32_t bos_token_id_{};
  uint32_t eos_token_id_{};
  uint32_t pad_token_id_{};
};

}  // namespace tfm
