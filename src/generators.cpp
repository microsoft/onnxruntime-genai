﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"
#include "search.h"
#if USE_CUDA
#include "search_cuda.h"
#endif

namespace Generators {

static bool _ = (Ort::InitApi(), false);

OrtGlobals::OrtGlobals() : env_{OrtEnv::Create(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)} {}

std::unique_ptr<OrtGlobals>& GetOrtGlobals() {
  static auto globals = std::make_unique<OrtGlobals>();
  return globals;
}

void Shutdown() {
  GetOrtGlobals().reset();
}

OrtEnv& GetOrtEnv() {
  return *GetOrtGlobals()->env_;
}

GeneratorParams::GeneratorParams(const Model& model)
    : search{model.config_->search},
      pad_token_id{model.config_->model.pad_token_id},
      eos_token_id{model.config_->model.eos_token_id},
      vocab_size{model.config_->model.vocab_size},
      hidden_size{model.config_->model.decoder.hidden_size},
      device_type{model.device_type_},
      cuda_stream{model.cuda_stream_},
      is_cuda_graph_enabled_{IsCudaGraphEnabled(model.config_->model.decoder.session_options)},
      config_{model.config_.get()} {
  use_cuda_graph = is_cuda_graph_enabled_;
  if (use_cuda_graph) {
    max_batch_size = 1;  // set it to 1 by default
  }
}

void GeneratorParams::TryGraphCapture(int max_bs) {
  if (!is_cuda_graph_enabled_ || device_type == DeviceType::CPU) {
    // no-op
    return;
  }

  if (DeviceType::CUDA == device_type || DeviceType::DML == device_type) {
    if (max_bs == 0) {
      throw std::runtime_error("Graph capture is enabled, but max_batch_size is not set.");
    }
    use_cuda_graph = true;
    max_batch_size = max_bs;
  } else {
    throw std::runtime_error("CUDA graph is not supported on this device");
  }
}

void GeneratorParams::SetInputs(const NamedTensors& named_tensors) {
  for (const auto& [name, tensor] : named_tensors) {
    if (name == Config::Defaults::InputIdsName) {
      input_ids = std::span<const int32_t>(tensor->ort_tensor_->GetTensorMutableData<int32_t>(),
                                           tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
      batch_size = static_cast<int>(tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape()[0]);
      sequence_length = static_cast<int>(input_ids.size()) / batch_size;
    } else {
      // If the nominal name is found in the map, use the graph name.
      // Else, use the nominal name as the graph name.
      [[maybe_unused]] const auto [graph_name, found] = config_->GetGraphName(name);
      extra_inputs.push_back({graph_name, tensor});
    }
  }
}

std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params) {
  return std::make_unique<Generator>(model, params);
}

std::unique_ptr<Search> CreateSearch(const GeneratorParams& params) {
#if USE_CUDA
  if (params.device_type == DeviceType::CUDA) {
    if (params.search.num_beams > 1)
      return std::make_unique<BeamSearch_Cuda>(params);
    return std::make_unique<GreedySearch_Cuda>(params);
  }
#endif

  if (params.search.num_beams > 1) {
    return std::make_unique<BeamSearch_Cpu>(params);
  }
  return std::make_unique<GreedySearch_Cpu>(params);
}

Generator::Generator(const Model& model, const GeneratorParams& params) : model_{model.shared_from_this()} {
  if (params.search.max_length == 0)
    throw std::runtime_error("search max_length is 0");
  if (params.search.max_length > model.config_->model.context_length)
    throw std::runtime_error("max_length (" + std::to_string(params.search.max_length) + ") cannot be greater than model context_length (" + std::to_string(model.config_->model.context_length) + ")");
  if (params.batch_size < 1)
    throw std::runtime_error("batch_size must be 1 or greater, is " + std::to_string(params.batch_size));
  if (params.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.vocab_size));
  if (params.sequence_length >= params.search.max_length)
    throw std::runtime_error("input sequence_length (" + std::to_string(params.sequence_length) + ") is >= max_length (" + std::to_string(params.search.max_length) + ")");

  search_ = CreateSearch(params);
  state_ = model.CreateState(search_->GetSequenceLengths(), params);
}

void Generator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling GenerateNextToken first");

  auto logits = state_->Run(search_->GetSequenceLength(), search_->GetNextTokens(), search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpSpan(stream, logits.GetCPU());
    stream << std::endl;
  }
  search_->SetLogits(logits);
  computed_logits_ = true;

  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);
}

bool Generator::IsDone() const {
  if (computed_logits_)
    throw std::runtime_error("IsDone() can't be called in the middle of processing logits");

  return search_->IsDone();
}

void Generator::GenerateNextToken() {
  if (!computed_logits_)
    throw std::runtime_error("Must call ComputeLogits before GenerateNextToken");
  computed_logits_ = false;
  auto& search = search_->params_->search;

  if (g_log.enabled && g_log.generate_next_token) {
    auto& stream = Log("generate_next_token");
    stream << SGR::Fg_Green << "do_sample: " << SGR::Reset << search.do_sample << ' '
           << SGR::Fg_Green << "top_k: " << SGR::Reset << search.top_k << ' '
           << SGR::Fg_Green << "top_p: " << SGR::Reset << search.top_p << ' '
           << SGR::Fg_Green << "temperature: " << SGR::Reset << search.temperature << ' '
           << SGR::Fg_Cyan << "sequence length: " << SGR::Reset << search_->GetSequenceLength()
           << std::endl;
  }

  if (!search.do_sample || search.top_k == 1) {
    search_->SelectTop();
    return;
  }

  // The user explicitly called TopK_TopP on a beam search
  if (search.num_beams != 1)
    throw std::runtime_error("TopK and TopP cannot be used with a beam search");

  // Sanity checks
  if (search.top_p < 0.0f || search.top_p > 1.0f)
    throw std::runtime_error("top_p must be between 0.0 and 1.0");
  if (search.top_k < 0)
    throw std::runtime_error("top_k must be 0 or greater");

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1) {
    search_->SampleTopKTopP(search.top_k, search.top_p, search.temperature);
  } else if (search.top_k > 1) {
    search_->SampleTopK(search.top_k, search.temperature);
  } else {
    assert(search.top_k == 0);
    search_->SampleTopP(search.top_p, search.temperature);
  }
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
