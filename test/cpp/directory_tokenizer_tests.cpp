// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"
#include "models/component_session.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

TEST(DirectoryTokenizerTest, EncodesAndIsMovable) {
  DirectoryTokenizer tokenizer{
      std::string(MODEL_PATH) + "multimodal-decoder-no-input-ids"};
  auto tokens = tokenizer.Encode("hello");
  EXPECT_FALSE(tokens.empty());
  EXPECT_TRUE(tokenizer.Encode("").empty());

  const auto pad_token_id = tokenizer.PadTokenId();
  EXPECT_EQ(pad_token_id, 1);
  DirectoryTokenizer moved{std::move(tokenizer)};
  EXPECT_EQ(moved.PadTokenId(), pad_token_id);
  EXPECT_EQ(moved.Encode("hello"), tokens);
}

TEST(DirectoryTokenizerTest, InternalBatchRowsMatchIndividualEncoding) {
  Generators::ComponentPackageTokenizer tokenizer{
      std::string(MODEL_PATH) + "multimodal-decoder-no-input-ids"};
  const std::vector<std::string> texts{
      "hello", "a longer input with punctuation!", ""};

  const auto rows = tokenizer.EncodeBatch(texts);
  ASSERT_EQ(rows.size(), texts.size());
  for (size_t index = 0; index < texts.size(); ++index)
    EXPECT_EQ(rows[index], tokenizer.Encode(texts[index]));
}
