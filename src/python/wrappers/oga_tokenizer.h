// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_borrowed_view.h"
#include "oga_utils.h"

namespace OgaPy {

struct OgaTokenizer : OgaObject {
  explicit OgaTokenizer(::OgaTokenizer* p) : ptr_(p) {}
  ~OgaTokenizer() override { if (ptr_) OgaDestroyTokenizer(ptr_); }
  ::OgaTokenizer* get() const { return ptr_; }
  
  // Update tokenizer options
  void UpdateOptions(const char** option_keys, const char** option_values, size_t num_options) {
    OgaCheckResult(OgaUpdateTokenizerOptions(ptr_, option_keys, option_values, num_options));
  }
  
  // Get BOS token ID
  int32_t GetBosTokenId() const {
    int32_t token_id = 0;
    OgaCheckResult(OgaTokenizerGetBosTokenId(ptr_, &token_id));
    return token_id;
  }
  
  // Get EOS token IDs as a borrowed view (automatically handles reference counting)
  EosTokenIdsView* GetEosTokenIds() const {
    const int32_t* eos_token_ids = nullptr;
    size_t token_count = 0;
    OgaCheckResult(OgaTokenizerGetEosTokenIds(ptr_, &eos_token_ids, &token_count));
    return new EosTokenIdsView(const_cast<OgaTokenizer*>(this), eos_token_ids, token_count);
  }
  
  // Get PAD token ID
  int32_t GetPadTokenId() const {
    int32_t token_id = 0;
    OgaCheckResult(OgaTokenizerGetPadTokenId(ptr_, &token_id));
    return token_id;
  }
  
  // Encode a single string and add to sequences
  void Encode(const char* str, OgaSequences* sequences) const {
    OgaCheckResult(OgaTokenizerEncode(ptr_, str, sequences->get()));
  }
  
  // Encode a batch of strings
  OgaTensor* EncodeBatch(const char** strings, size_t count) const {
    OgaTensor* out = nullptr;
    OgaCheckResult(OgaTokenizerEncodeBatch(ptr_, strings, count, &out));
    return out;
  }
  
  // Decode a batch of token sequences
  OgaStringArray* DecodeBatch(const OgaTensor* tensor) const {
    ::OgaStringArray* out = nullptr;
    OgaCheckResult(OgaTokenizerDecodeBatch(ptr_, tensor, &out));
    return new OgaStringArray(out);
  }
  
  // Decode tokens to a string
  const char* Decode(const int32_t* tokens, size_t token_count) const {
    const char* out_string = nullptr;
    OgaCheckResult(OgaTokenizerDecode(ptr_, tokens, token_count, &out_string));
    return out_string;
  }
  
private:
  ::OgaTokenizer* ptr_;
};

struct OgaTokenizerStream : OgaObject {
  explicit OgaTokenizerStream(::OgaTokenizerStream* p) : ptr_(p) {}
  ~OgaTokenizerStream() override { if (ptr_) OgaDestroyTokenizerStream(ptr_); }
  ::OgaTokenizerStream* get() const { return ptr_; }
  
  // Decode a token in streaming mode
  const char* Decode(int32_t token) {
    const char* out = nullptr;
    OgaCheckResult(OgaTokenizerStreamDecode(ptr_, token, &out));
    return out;
  }
  
private:
  ::OgaTokenizerStream* ptr_;
};

} // namespace OgaPy
