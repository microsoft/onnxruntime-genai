// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

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
