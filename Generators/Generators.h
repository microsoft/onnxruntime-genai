// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <assert.h>
#include <functional>
#include <span>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_set>
#include <vector>
#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include "SafeInt.hpp"
#include "onnxruntime_cxx_api_2.h"
#include "debugging.h"
#include "smartptrs.h"

namespace Generators {
using ScoreType = float;

struct SearchParams {
  int num_beams{1};
  int batch_size{};
  int sequence_length{};
  int max_length{10};
  int pad_token_id{98};
  int eos_token_id{98};
  int vocab_size{};

  float length_penalty{1.0f};
  bool early_stopping{false};

  int BatchBeamSize() const { return num_beams * batch_size; }

  std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]
};

}
