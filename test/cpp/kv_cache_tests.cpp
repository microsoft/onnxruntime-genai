// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/io/static_kv_cache.h"

namespace {

TEST(KvCacheTests, InfersShapeForSingleSessionDecoder) {
  Generators::Config::Model::Decoder decoder;
  EXPECT_TRUE(Generators::ShouldInferKeyValueCacheShape(decoder));
}

TEST(KvCacheTests, DoesNotInferShapeForPipelineDecoder) {
  Generators::Config::Model::Decoder decoder;
  decoder.pipeline.emplace_back();
  EXPECT_FALSE(Generators::ShouldInferKeyValueCacheShape(decoder));
}

}  // namespace
