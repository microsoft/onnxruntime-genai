// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tokenizer.h"

#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"

namespace tfm {

class SpmTokenizer : public TokenizerImpl {
 public:
  SpmTokenizer(/* args */);
  ~SpmTokenizer() override;

 public:
  TfmStatus Onload() override;
  TfmStatus Encode(std::string_view input, std::vector<tfmTokenId_t>& ids) const override;
  TfmStatus Decode(const span<tfmTokenId_t const>& ids, std::string& text) const override;
  TfmStatus Id2Token(tfmTokenId_t id, std::string& token, DecoderState** state) const override;

 private:
  class SpmDeocerState : public DecoderState {
   public:
    SpmDeocerState() = default;
    ~SpmDeocerState() override = default;
    bool last_control_char_ = false;
  };

  std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_processor_;
};

}  // namespace tfm
