// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/model_type.h"

TEST(ModelTypeTests, NemotronHybridTextType) {
  EXPECT_TRUE(Generators::ModelType::IsLLM("nemotron_h"));
  EXPECT_TRUE(Generators::ModelType::IsLLM("nemotron"));
  EXPECT_FALSE(Generators::ModelType::IsLFM2("nemotron_h"));
  EXPECT_FALSE(Generators::ModelType::IsTransducer("nemotron_h"));
  EXPECT_FALSE(Generators::ModelType::IsLLM("nemotron_speech"));
}
