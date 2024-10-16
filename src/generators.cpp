// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"
#include "models/decoder_only.h"
#include "search.h"
#if USE_CUDA
#include "search_cuda.h"
#endif

namespace Generators {

static bool _ = (Ort::InitApi(), false);

OrtGlobals::OrtGlobals()
    : env_{OrtEnv::Create(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)} {
  auto arena_config = OrtArenaCfg::Create(0, -1, -1, -1);
  Ort::Allocator& allocator_cpu{Ort::Allocator::GetWithDefaultOptions()};
  env_->CreateAndRegisterAllocator(allocator_cpu.GetInfo(), *arena_config);
}

// Ensure Shutdown() has been called before process exit
struct ValidateShutdown {
  ~ValidateShutdown() {
    if (GetOrtGlobals()) {
      std::cerr << "OGA Error: Shutdown must be called before process exit, please check the documentation for the proper API to call to ensure clean shutdown." << std::endl;
      std::abort();
    }
  }
};

std::unique_ptr<OrtGlobals>&
GetOrtGlobals() {
  static auto globals = std::make_unique<OrtGlobals>();
  static auto validate = std::make_unique<ValidateShutdown>();  // Must be after the above line so the destructor runs before the above destructor
  return globals;
}

// Used by Shutdown() to display the counts and types of any leaked objects
template <typename... Types>
bool LeakTypeList<Types...>::Dump() {
  ((LeakChecked<Types>::Count() != 0 ? std::cerr << "OGA Error: " << LeakChecked<Types>::Count() << " instances of " << typeid(Types).name() << " were leaked." << std::endl : std::cerr), ...);
  return ((LeakChecked<Types>::Count() != 0) || ...);
}

void Shutdown() {
  if (LeakTypes::Dump()) {
    std::cerr << "    Please see the documentation for the API being used to ensure proper cleanup." << std::endl;
    std::abort();
  }

  GetOrtGlobals().reset();  // Delete now because on process exit is too late
}

OrtEnv& GetOrtEnv() {
  return *GetOrtGlobals()->env_;
}

std::string to_string(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::DML:
      return "DirectML";
  }
  throw std::runtime_error("Unknown device type");
}

GeneratorParams::GeneratorParams(const Config& config) : config{config} {
}

GeneratorParams::GeneratorParams(const Model& model)
    : config{*model.config_.get()},
      device_type{model.device_type_},
      cuda_stream{model.cuda_stream_},
      is_cuda_graph_enabled_{IsCudaGraphEnabled(model.config_->model.decoder.session_options)} {
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

// TODO(aciddelgado): Does this work?
void GeneratorParams::SetInputs(const NamedTensors& named_tensors) {
  for (const auto& [name, tensor] : named_tensors) {
    if (name == Config::Defaults::InputIdsName) {
      aux_input_ids = cpu_span<int32_t>(tensor->ort_tensor_->GetTensorMutableData<int32_t>(),
                                        tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
    } else {
      // If the nominal name is found in the map, use the graph name.
      // Else, use the nominal name as the graph name.
      [[maybe_unused]] const auto [graph_name, found] = config.GetGraphName(name);
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
  if (params.search.batch_size < 1)
    throw std::runtime_error("batch_size must be 1 or greater, is " + std::to_string(params.search.batch_size));
  if (params.config.model.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.config.model.vocab_size));

  search_ = CreateSearch(params);
  state_ = model.CreateState(search_->GetSequenceLengths(), params);  // Search sequence lengths set when creating state

  // Temporary solution for multimodal and whisper models
  if (!params.aux_input_ids.empty() && params.aux_input_ids.data() != nullptr) {
    AddTokens(params.aux_input_ids);
  }
}

void Generator::AddTokens(const cpu_span<int32_t>& input_ids) {
  // TODO(aciddelgado): check for batch_size > 1 requires full rewind
  search_->SetUserTokens(input_ids);

  computed_logits_ = false;
  ComputeLogits(input_ids);
}

void Generator::ComputeLogits(const RoamingArray<int32_t>& next_tokens) {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling AddTokens or GenerateNextToken first");

  auto logits = state_->Run(search_->GetSequenceLength(), next_tokens, search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpSpan(stream, logits.GetCPU());
    stream << std::endl;
  }
  search_->SetLogits(logits);
  computed_logits_ = true;
}

bool Generator::IsDone() const {
  // TODO(aciddelgado): Is this the correct approach to handling computed_logits_ now?
  if (computed_logits_) {
    return false;
  }

  bool is_done = search_->IsDone();
  if (is_done) {
    state_->Finalize();
  }

  return is_done;
}

void Generator::GenerateNextToken() {
  // TODO(aciddelgado): check that AddTokens has been called at least once
  if (!computed_logits_) {
    ComputeLogits(search_->GetNextTokens());
  }
  computed_logits_ = false;
  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);

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

void Generator::RewindToLength(size_t new_length) {
  if (new_length > search_->GetSequenceLength())
    throw std::runtime_error("Cannot rewind to a length greater than the current sequence length");
  if (new_length == search_->GetSequenceLength())
    return;
  size_t batch_size = search_->params_->search.batch_size;
  if (batch_size > 1 && new_length != 0)
    throw std::runtime_error("RewindToLength must be called with new_length=0 when batch_size > 1");
  search_->RewindTo(new_length);
  state_->RewindTo(new_length);
  computed_logits_ = false;
}

RoamingArray<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

}  // namespace Generators
