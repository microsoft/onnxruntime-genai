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

OrtGlobals::OrtGlobals() : env_{OrtEnv::Create(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)} {}

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

std::unique_ptr<AssistantGenerator> CreateAssistantGenerator(const Model& model, const GeneratorParams& params) {
  return std::make_unique<AssistantGenerator>(model, params);
}

std::unique_ptr<SpeculativeDecodingGenerator> CreateSpeculativeDecodingGenerator(const Model& model, const Model& assistant_model, const GeneratorParams& params) {
  return std::make_unique<SpeculativeDecodingGenerator>(model, assistant_model, params);
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

std::unique_ptr<Search> CreateSpeculativeSearch(const GeneratorParams& params) {
#if USE_CUDA
  throw std::runtime_error("Speculative decoding is not supported on CUDA");
#endif
  if (params.search.num_beams > 1) {
    throw std::runtime_error("Speculative decoding is not supported with beam search");
  }
  return std::make_unique<SpeculativeGreedySearch_Cpu>(params);
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
  if (params.input_ids.empty() || params.input_ids.data() == nullptr)
    throw std::runtime_error("input_ids not set in GeneratorParams");

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

RoamingArray<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

AssistantGenerator::AssistantGenerator(const Model& model, const GeneratorParams& params)
    : Generator(model, params) {
  if (params.search.num_beams != 1)
    throw std::runtime_error("AssistantGenerator only supports num_beams=1, got " + std::to_string(params.search.num_beams));
  if (params.batch_size != 1)
    throw std::runtime_error("AssistantGenerator only supports batch_size=1, got " + std::to_string(params.batch_size));
  if (params.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.vocab_size));
  if (params.sequence_length >= params.search.max_length)
    throw std::runtime_error("input sequence_length (" + std::to_string(params.sequence_length) + ") is >= max_length (" + std::to_string(params.search.max_length) + ")");

  state_ = std::make_unique<SpeculativeDecodingDecoderOnly_State>(
      *std::dynamic_pointer_cast<const DecoderOnly_Model>(model_), search_->GetSequenceLengths(), params);
}

void AssistantGenerator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling GenerateNextToken first");

  auto sequence_length = search_->GetSequenceLength();
  auto next_token_length = first_run_in_assist_ ? 2 : 1;
  auto past_length = sequence_length - next_token_length;
  auto logits = state_->Run(search_->GetSequence(0), next_token_length, past_length, 1);
  if (g_log.enabled && g_log.speculative_decoding) {
    auto& stream = Log("speculative_decoding");
    DumpSpan(stream, logits.GetCPU());
    stream << std::endl;
  }
  search_->SetLogits(logits);
  computed_logits_ = true;

  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);
  first_run_in_assist_ = false;
}

void AssistantGenerator::GenerateNextToken() {
  Generator::GenerateNextToken();
  candidate_length_++;
}

void AssistantGenerator::AcceptCandidateTokens(RoamingArray<int32_t> next_tokens) {
  search_->DropLastTokens(candidate_length_);
  search_->SetNextTokens(next_tokens);
  candidate_length_ = 0;
  if (g_log.enabled && g_log.speculative_decoding) {
    auto& stream = Log("speculative_decoding");
    stream << SGR::Fg_Green << "assistant sequence: " << SGR::Reset << std::endl;
    DumpSpan(stream, search_->GetSequence(0).GetCPU());
    stream << std::endl
           << "length: " << search_->GetSequenceLength() << std::endl;
  }
  first_run_in_assist_ = true;
}

SpeculativeDecodingGenerator::SpeculativeDecodingGenerator(const Model& model, const Model& assistant_model, const GeneratorParams& params)
    : assistant_generator_{CreateAssistantGenerator(assistant_model, params)},
      model_{model.shared_from_this()} {
  if (params.search.max_length == 0)
    throw std::runtime_error("search max_length is 0");
  if (params.search.max_length > model.config_->model.context_length)
    throw std::runtime_error("max_length (" + std::to_string(params.search.max_length) + ") cannot be greater than model context_length (" + std::to_string(model.config_->model.context_length) + ")");
  if (params.batch_size != 1)
    throw std::runtime_error("batch_size must be 1, is " + std::to_string(params.batch_size));
  if (params.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.vocab_size));
  if (params.sequence_length >= params.search.max_length)
    throw std::runtime_error("input sequence_length (" + std::to_string(params.sequence_length) + ") is >= max_length (" + std::to_string(params.search.max_length) + ")");
  if (params.input_ids.empty() || params.input_ids.data() == nullptr)
    throw std::runtime_error("input_ids not set in GeneratorParams");

  if (model.config_->model.type != "llama" &&
      model.config_->model.type != "gemma" &&
      model.config_->model.type != "gemma2" &&
      model.config_->model.type != "mistral" &&
      model.config_->model.type != "phi" &&
      model.config_->model.type != "phi3" &&
      model.config_->model.type != "phi3small" &&
      model.config_->model.type != "qwen2")
    throw std::runtime_error("Speculative decoding is not supported for this model type " + model.config_->model.type);

  search_ = CreateSpeculativeSearch(params);
  state_ = std::make_unique<SpeculativeDecodingDecoderOnly_State>(
      *std::dynamic_pointer_cast<const DecoderOnly_Model>(model_), search_->GetSequenceLengths(), params);
}

void SpeculativeDecodingGenerator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling GenerateNextToken first");

  candidate_length_ = 0;
  while (!assistant_generator_->IsDone() && candidate_length_ < max_candidate_length_) {
    assistant_generator_->ComputeLogits();
    assistant_generator_->GenerateNextToken();
    candidate_length_++;
  }

  auto candidate_sequence = assistant_generator_->search_->GetSequence(0);
  if (g_log.enabled && g_log.speculative_decoding) {
    auto& stream = Log("speculative_decoding");
    stream << SGR::Fg_Green << "candidates from assistant model: " << SGR::Reset << std::endl;
    stream << SGR::Fg_Green << "candidate count: " << SGR::Reset << candidate_length_ << std::endl;
    DumpSpan(stream, candidate_sequence.GetCPU());
  }

  auto logits = state_->Run(candidate_sequence, candidate_length_ + 1, search_->GetSequenceLength() - 1, candidate_length_ + 1);
  if (g_log.enabled && g_log.speculative_decoding) {
    auto& stream = Log("speculative_decoding");
    stream << SGR::Fg_Green << "produced logits from main model: " << SGR::Reset << std::endl;
  }

  search_->SetLogits(logits);
  computed_logits_ = true;
}

void SpeculativeDecodingGenerator::GenerateNextToken() {
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

  if (search.do_sample)
    throw std::runtime_error("Not implemented");
  if (search.top_k != 1)
    throw std::runtime_error("Not implemented");
  if (search.top_p != 1.0f)
    throw std::runtime_error("Not implemented");
  if (search.temperature != 1.0f)
    throw std::runtime_error("Not implemented");

  auto candidate_sequence = assistant_generator_->search_->GetSequence(0);

  // Compare with logits one by one to determine the accepted tokens.
  // total new token count is accepted token count + 1.
  auto next_tokens = search_->CheckCandidates(candidate_sequence, candidate_length_);
  // Update sequence to drop tokens of size candidate_length_,
  // and append next tokens.
  assistant_generator_->AcceptCandidateTokens(next_tokens);
  if (g_log.enabled && g_log.speculative_decoding) {
    auto& stream = Log("speculative_decoding");
    stream << SGR::Fg_Green << "candidate count: " << SGR::Reset << candidate_length_ << std::endl;
    stream << SGR::Fg_Green << "next tokens: " << SGR::Reset;
    DumpSpan(stream, next_tokens.GetCPU());
    stream << std::endl;
  }
}

bool SpeculativeDecodingGenerator::IsDone() const {
  if (computed_logits_)
    throw std::runtime_error("IsDone() can't be called in the middle of processing logits");

  return search_->IsDone();
}

TokenSequences Generate(const Model& model, const GeneratorParams& params) {
  auto generator = CreateGenerator(model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  TokenSequences result;
  for (int i = 0; i < params.batch_size * params.search.num_return_sequences; i++) {
    auto sequence = generator->search_->GetSequence(i);
    auto sequence_cpu = sequence.GetCPU();

    auto& v = result.emplace_back();
    v.assign(sequence_cpu.begin(), sequence_cpu.end());
  }
  return result;
}

TokenSequences Generate(const Model& model, const Model& assistant_model, const GeneratorParams& params) {
  auto generator = CreateSpeculativeDecodingGenerator(model, assistant_model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  // Supports only single batch size, single sequence.
  TokenSequences result = {{}};
  auto sequence_cpu = generator->search_->GetSequence(0).GetCPU();
  result[0].assign(sequence_cpu.begin(), sequence_cpu.end());
  return result;
}

}  // namespace Generators
