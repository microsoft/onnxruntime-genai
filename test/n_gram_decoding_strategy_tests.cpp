// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "n_gram_decoding_strategy.h"

#include <stdexcept>

#include <gtest/gtest.h>

namespace Generators::test {

TEST(NGramDecodingCapabilitiesTest, AcceptsSupportedConfiguration) {
  EXPECT_NO_THROW(ValidateNGramDecodingCapabilities(Config::Search{}, {}));
}

TEST(NGramDecodingCapabilitiesTest, RejectsDraftModelCombination) {
  NGramDecodingCapabilities capabilities;
  capabilities.uses_draft_model = true;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsBatchSizeGreaterThanOne) {
  Config::Search search;
  search.batch_size = 2;
  EXPECT_THROW(ValidateNGramDecodingCapabilities(search, {}), std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsBeamSearch) {
  Config::Search search;
  search.num_beams = 2;
  EXPECT_THROW(ValidateNGramDecodingCapabilities(search, {}), std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsMultipleReturnSequences) {
  Config::Search search;
  search.num_return_sequences = 2;
  EXPECT_THROW(ValidateNGramDecodingCapabilities(search, {}), std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, AcceptsGuidance) {
  NGramDecodingCapabilities capabilities;
  capabilities.uses_guidance = true;
  EXPECT_NO_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities));
}

TEST(NGramDecodingCapabilitiesTest, RejectsNonDecoderOnlyTextModel) {
  NGramDecodingCapabilities capabilities;
  capabilities.is_plain_decoder_only_text = false;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsSlidingKvCache) {
  NGramDecodingCapabilities capabilities;
  capabilities.uses_sliding_kv_cache = true;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsHybridState) {
  NGramDecodingCapabilities capabilities;
  capabilities.uses_hybrid_state = true;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsModelManagedState) {
  NGramDecodingCapabilities capabilities;
  capabilities.uses_model_managed_state = true;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

TEST(NGramDecodingCapabilitiesTest, RejectsPrunedLogits) {
  NGramDecodingCapabilities capabilities;
  capabilities.has_pruned_logits = true;
  EXPECT_THROW(
      ValidateNGramDecodingCapabilities(Config::Search{}, capabilities),
      std::runtime_error);
}

}  // namespace Generators::test
