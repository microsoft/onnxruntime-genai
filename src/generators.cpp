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

GeneratorParams::GeneratorParams(const Model& model)
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

void GeneratorParams::SetInputSequences(const TokenSequences& sequences) {
  const bool pad_right = true; // FUTURE: Pull from model config, but default to padding on the right

  size_t max_count = 0;
  for (auto& sequence : sequences)
    max_count = std::max(max_count, sequence.size());

  const size_t input_ids_count = max_count * sequences.size();

  input_ids_owner_ = std::make_unique<int32_t[]>(input_ids_count);
  auto new_input_ids = std::span<int32_t>(input_ids_owner_.get(), input_ids_count);

  input_ids = new_input_ids;
  sequence_length = static_cast<int>(max_count);
  batch_size = static_cast<int>(sequences.size());

  // Copy and pad the input sequences with pad_token_id
  for (size_t sequence_index = 0; sequence_index < sequences.size(); sequence_index++) {
    auto output_span = new_input_ids.subspan(sequence_index * max_count, max_count);
    auto input_span = sequences[sequence_index];

    auto pad_count = max_count - input_span.size();
    if (pad_right) {
      std::copy(input_span.begin(), input_span.end(), output_span.begin());
      std::fill(output_span.end() - pad_count, output_span.end(), pad_token_id);
    } else {
      std::fill(output_span.begin(), output_span.begin() + pad_count, pad_token_id);
      std::copy(input_span.begin(), input_span.end(), output_span.begin() + pad_count);
    }
  }
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

std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& search_params) {
  return std::make_unique<Generator>(model, search_params);
}

std::unique_ptr<Search> CreateSearch(const GeneratorParams& params) {
#if USE_CUDA
  if (params.device_type == DeviceType::CUDA) {
    if (params.num_beams > 1)
      return std::make_unique<BeamSearch_Cuda>(params);
    return std::make_unique<GreedySearch_Cuda>(params);
  }
#endif

  if (params.num_beams > 1) {
    return std::make_unique<BeamSearch_Cpu>(params);
  }
  return std::make_unique<GreedySearch_Cpu>(params);
}

Generator::Generator(const Model& model, const GeneratorParams& search_params) : model_{model} {
  search_ = CreateSearch(search_params);
  state_ = model.CreateState(search_->GetSequenceLengths(), search_params);
}

void Generator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling GenerateNextToken* first");

  search_->SetLogits(state_->Run(search_->GetSequenceLength(), search_->GetNextTokens(), search_->GetNextIndices()));
  computed_logits_ = true;
}

bool Generator::IsDone() const {
  if (computed_logits_)
    throw std::runtime_error("IsDone() can't be called in the middle of processing logits");

  return search_->IsDone();
}

void Generator::GenerateNextToken_TopK_TopP(int top_k, float top_p, float temperature) {
  if (search_->params_.num_beams != 1)
    throw std::runtime_error("TopK and TopP cannot be used with a beam search");

  if (!computed_logits_)
    throw std::runtime_error("Must call ComputeLogits before GenerateNextToken*");
  computed_logits_ = false;

  if (top_p < 1.0f && top_k > 1) {
    search_->SampleTopPAndK(top_p, top_k, temperature);
  } else if (top_p < 1.0f) {
    search_->SampleTopP(top_p, temperature);
  } else if (top_k > 1) {
    search_->SampleTopK(top_k, temperature);
  } else {
    search_->SelectTop();
  }
}

void Generator::GenerateNextToken() {
  if (search_->params_.num_beams > 1) {
    if (!computed_logits_)
      throw std::runtime_error("Must call ComputeLogits before GenerateNextToken*");
    computed_logits_ = false;
    search_->SelectTop();
    return;
  }

  auto& config = *model_.config_;
  GenerateNextToken_TopK_TopP(config.top_k, config.top_p, config.temperature);
}

RoamingArray<int32_t> Generator::GetSequence(int index) const {
  return search_->GetSequence(index);
}

TokenSequences Generate(const Model& model, const GeneratorParams& params) {
  auto generator = CreateGenerator(model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  TokenSequences result;

  for (int i = 0; i < params.batch_size; i++) {
    auto sequence = generator->search_->GetSequence(i);
    auto sequence_cpu = sequence.GetCPU();

    auto& v = result.emplace_back();
    v.assign(sequence_cpu.begin(), sequence_cpu.end());
  }
  return result;
}

}  // namespace Generators
