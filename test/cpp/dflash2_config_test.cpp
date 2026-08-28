// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "dflash2_drafter.h"

namespace Generators::test {
namespace {

Config MakeDflash2Config() {
  Config config;
  auto& decoder = config.model.decoder;
  decoder.filename = "target.onnx";
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::Fixed});

  auto& dflash2 = config.model.dflash2;
  dflash2.filename = "dflash2.onnx";
  dflash2.num_hidden_layers = 3;
  dflash2.num_key_value_heads = 2;
  dflash2.head_size = 8;
  dflash2.block_size = 4;
  dflash2.num_draft_tokens = 3;
  dflash2.selector_top_k = 2;
  dflash2.sliding_window = 17;
  return config;
}

}  // namespace

TEST(Dflash2ConfigTest, RequiresDrafterFilename) {
  auto config = MakeDflash2Config();
  config.model.dflash2.filename.clear();
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresCompleteGeometry) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_key_value_heads = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresOneDraftPerNonAnchorBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = 2;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, ProjectsDrafterWithoutTargetState) {
  const auto projected = CreateDflash2Config(MakeDflash2Config());
  const auto& decoder = projected->model.decoder;
  EXPECT_EQ(decoder.filename, "dflash2.onnx");
  EXPECT_EQ(decoder.num_hidden_layers, 3);
  EXPECT_EQ(decoder.num_key_value_heads, 2);
  EXPECT_EQ(decoder.head_size, 8);
  EXPECT_FALSE(decoder.sliding_window.has_value());
  EXPECT_FALSE(decoder.state_groups.has_value());
}

TEST(Dflash2ConfigTest, AccountsForWindowedPagedCache) {
  const auto config = MakeDflash2Config();
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(config, 16), 3072u);
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 15u);
}

TEST(Dflash2ConfigTest, RejectsUnboundedCachePool) {
  auto config = MakeDflash2Config();
  config.model.dflash2.sliding_window = 0;
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 8, 3), std::runtime_error);
}

}  // namespace Generators::test
