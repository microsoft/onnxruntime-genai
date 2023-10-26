// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <functional>
#include "span.h"
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>
#if USE_CUDA
#include <cuda_runtime.h>
#include "cuda_common.h"
#endif

#include "smartptrs.h"

namespace Generators {
using ScoreType = float;

struct SearchParams {
  int batch_size{};
  int sequence_length{};
  int max_length{10};

  // Parameters from model that the search uses
  int pad_token_id{};
  int eos_token_id{};
  int vocab_size{};

  // Beam search parameters
  int num_beams{1};
  float length_penalty{1.0f};
  bool early_stopping{};

  int BatchBeamSize() const { return num_beams * batch_size; }

  std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]
};

void top_k_indices(std::span<int32_t> top_k, std::span<const ScoreType> inputs);

}
