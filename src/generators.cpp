// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"

namespace Generators {

// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float Float16ToFloat32(uint16_t v) {
  // Extract sign, exponent, and fraction from numpy.float16
  int sign = (v & 0x8000) >> 15;
  int exponent = (v & 0x7C00) >> 10;
  int fraction = v & 0x03FF;

  // Handle special cases
  if (exponent == 0) {
    if (fraction == 0) {
      // Zero
      return sign ? -0.0f : 0.0f;
    } else {
      // Subnormal number
      return std::ldexp((sign ? -1.0f : 1.0f) * fraction / 1024.0f, -14);
    }
  } else if (exponent == 31) {
    if (fraction == 0) {
      // Infinity
      return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    } else {
      // NaN
      return std::numeric_limits<float>::quiet_NaN();
    }
  }

  // Normalized number
  return std::ldexp((sign ? -1.0f : 1.0f) * (1.0f + fraction / 1024.0f), exponent - 15);
}

Config::Config(const std::filesystem::path& path) : config_path{path} {
  ParseConfig(path / "config.json", *this);
}

SearchParams::SearchParams(const Model& model) :
  pad_token_id{ model.config_.pad_token_id },
  eos_token_id{ model.config_.eos_token_id },
  vocab_size{ model.config_.vocab_size },
  max_length{ model.config_.max_length },
  length_penalty{ model.config_.length_penalty },
  early_stopping{ model.config_.early_stopping }
#if USE_CUDA
  ,cuda_stream{model.cuda_stream_}
#endif
 {
}

ProviderOptions GetDefaultProviderOptions(DeviceType device_type) {
  ProviderOptions options;

  if (device_type == DeviceType::CUDA) {
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);

    Generators::ProviderOptions provider_options;
    auto& cuda_options = provider_options.emplace<OrtCUDAProviderOptions>();
    cuda_options.has_user_compute_stream = true;
    cuda_options.user_compute_stream = cuda_stream;
  }

  return options;
  }


}