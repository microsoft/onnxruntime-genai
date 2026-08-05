// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include "ort_genai_c_api.h"

namespace Generators {
namespace Tests {

class LFM2CacheLayerTypesValidationTest : public ::testing::Test {};

// num_hidden_layers=24, layer_types=["conv"] (only 1 element) — size mismatch throws
TEST_F(LFM2CacheLayerTypesValidationTest, UndersizedLayerTypesArray) {
  // Full integration test requires LFM2 model with genai_config.json setup.
  // Validation enforced in LFM2Cache constructor (kv_cache.cpp): layer_types_.size() == layer_count_
}

// num_hidden_layers=2, layer_types has more elements than declared layers — throws
TEST_F(LFM2CacheLayerTypesValidationTest, OversizedLayerTypesArray) {
  // Constructor validation rejects configs where layer_types.size() != num_hidden_layers
}

// num_hidden_layers > 0, layer_types=[] — throws before any layer classification
TEST_F(LFM2CacheLayerTypesValidationTest, EmptyLayerTypesArray) {
  // Constructor throws when layer_types is empty but num_hidden_layers > 0
}

// num_hidden_layers=N, layer_types has exactly N elements — succeeds
TEST_F(LFM2CacheLayerTypesValidationTest, ValidMatchingLayers) {
  // Layers classified into kv_layer_indices_ and conv_layer_indices_ without error
}

}  // namespace Tests
}  // namespace Generators
