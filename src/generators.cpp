// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"
#include "search.h"
#if USE_CUDA
#include "search_cuda.h"
#endif

namespace Generators {

// IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
float Float16ToFloat32(uint16_t v) {
  // Extract sign, exponent, and fraction from numpy.float16
  int const sign = (v & 0x8000) >> 15;
  int const exponent = (v & 0x7C00) >> 10;
  int const fraction = v & 0x03FF;

  // Handle special cases
  if (exponent == 0) {
    if (fraction == 0) {
      // Zero
      return sign != 0 ? -0.0f : 0.0f;
    }  // Subnormal number
    return std::ldexp((sign != 0 ? -1.0f : 1.0f) * static_cast<float>(fraction) / 1024.0f, -14);
  }
  if (exponent == 31) {
    if (fraction == 0) {
      // Infinity
      return sign != 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }  // NaN
    return std::numeric_limits<float>::quiet_NaN();
  }

  // Normalized number
  return std::ldexp((sign != 0 ? -1.0f : 1.0f) * (1.0f + static_cast<float>(fraction) / 1024.0f), exponent - 15);
}

SearchParams::SearchParams(const Model& model)
    : pad_token_id{model.config_->pad_token_id},
      eos_token_id{model.config_->eos_token_id},
      vocab_size{model.config_->model.vocab_size},
      max_length{model.config_->max_length},
      length_penalty{model.config_->length_penalty},
      early_stopping{model.config_->early_stopping},
      num_beams{model.config_->num_beams},
      device_type{model.device_type_},
      cuda_stream{model.cuda_stream_} {
}

ProviderOptions GetDefaultProviderOptions([[maybe_unused]] DeviceType device_type) {
  ProviderOptions options;
#if USE_CUDA
  if (device_type == DeviceType::CUDA) {
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);

    auto& cuda_options = options.emplace<OrtCUDAProviderOptions>();
    cuda_options.has_user_compute_stream = true;
    cuda_options.user_compute_stream = cuda_stream;
  }
#endif

  return options;
}

std::unique_ptr<Search> SearchParams::CreateSearch() const {
#if USE_CUDA
  if (device_type == DeviceType::CUDA) {
    if (num_beams > 1)
      return std::make_unique<BeamSearch_Cuda>(*this);
    return std::make_unique<GreedySearch_Cuda>(*this);
  }
#endif

  if (num_beams > 1) {
    return std::make_unique<BeamSearch_Cpu>(*this);
  }
  return std::make_unique<GreedySearch_Cpu>(*this);
}

std::unique_ptr<Generator> CreateGenerator(Model& model, const SearchParams& search_params) {
  return std::make_unique<Generator>(model, search_params);
}

Generator::Generator(Model& model, const SearchParams& search_params) : model_{model} {
  search_ = search_params.CreateSearch();
  state_ = model.CreateState(search_->GetSequenceLengths(), search_params);
}

void Generator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling AppendNextToken* first");

  search_->SetLogits(state_->Run(search_->GetSequenceLength(), search_->GetNextTokens(), search_->GetNextIndices()));
  computed_logits_ = true;
}

bool Generator::IsDone() const {
  if (computed_logits_)
    throw std::runtime_error("IsDone() can't be called in the middle of processing logits");

  return search_->IsDone();
}

void Generator::AppendNextToken_TopK_TopP(int top_k, float top_p, float temperature) {
  if (search_->params_.num_beams != 1)
    throw std::runtime_error("TopK and TopP cannot be used with a beam search");

  if (!computed_logits_)
    throw std::runtime_error("Must call ComputeLogits before AppendNextToken*");
  computed_logits_ = false;

  // TODO: Do TopK if top_k >1 then do TopP on the results
  if (top_p < 1.0f) {
    search_->SampleTopP(top_p, temperature);
  } else if (top_k > 1) {
    search_->SampleTopK(top_k, temperature);
  } else {
    search_->SelectTop();
  }
}

void Generator::AppendNextToken() {
  if (search_->params_.num_beams > 1) {
    if (!computed_logits_)
      throw std::runtime_error("Must call ComputeLogits before AppendNextToken*");
    computed_logits_ = false;
    search_->SelectTop();
    return;
  }

  auto& config = *model_.config_;
  AppendNextToken_TopK_TopP(config.top_k, config.top_p, config.temperature);
}

}  // namespace Generators
