// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt.h"
#include "engine/request.h"
#include "search.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <set>

namespace Generators {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error("nemotron_bolt: " + message);
}

void RequireTensor(const SessionInfo& info, const std::string& name, bool input,
                   ONNXTensorElementDataType type, const std::vector<int64_t>& shape) {
  Require(input ? info.HasInput(name) : info.HasOutput(name), "missing tensor " + name);
  Require((input ? info.HasInput(name) : info.HasOutput(name)) &&
              (input ? info.GetInputDataType(name) : info.GetOutputDataType(name)) == type &&
              (input ? info.GetInputShape(name) : info.GetOutputShape(name)) == shape,
          "incompatible tensor " + name);
}

std::vector<std::string> ToSymbolicStrings(const std::vector<const char*>& raw) {
  std::vector<std::string> out;
  out.reserve(raw.size());
  for (const char* s : raw) out.emplace_back(s ? s : "");
  return out;
}

template <typename T>
void Upload(Tensor& tensor, std::span<const T> values) {
  tensor.GetDeviceSpan<T>().CopyFromCpu(values);
}

struct BoltStrategy : DecodingStrategy {
  explicit BoltStrategy(Generator& generator)
      : state_{dynamic_cast<NemotronBoltState&>(*generator.state_)} {}
  void Step(Generator& generator) override {
    Require(!generator.search_->IsDone(), "generation is already done");
    generator.search_->CommitToken(state_.NextToken(generator.search_->GetSequenceLength()));
    generator.computed_logits_ = false;
  }
  void PrepareForAppend(Generator& generator) override {
    Require(generator.search_->GetSequenceLength() == 0,
            "continuation AppendTokens is unsupported; reset with RewindToLength(0)");
  }
  bool TryGetExternalLogits(Generator&, DeviceSpan<float>&) override {
    throw std::runtime_error("nemotron_bolt: GetLogits is unsupported for buffered block decoding");
  }
  void PrepareForSetLogits(Generator&) override {
    throw std::runtime_error("nemotron_bolt: SetLogits is unsupported");
  }
  SpeculativeStats GetStats() const override { return state_.Stats(); }
  NemotronBoltState& state_;
};

struct BoltSearch : GreedySearch_Cpu {
  using GreedySearch_Cpu::GreedySearch_Cpu;
  void AppendTokens(DeviceSpan<int32_t>& tokens) override {
    GreedySearch_Cpu::AppendTokens(tokens);
    done_ = GetSequenceLength() >= params_->search.max_length;
  }
  void CommitToken(int32_t token) override {
    Require(!IsDone(), "generation is already done");
    SetNextToken(0, token);
    // Unlike the general search tail, Bolt exposes the terminal EOS too. No GPU scoring is
    // necessary: the target's actual argmax ids already determined every committed token.
    AppendNextTokensToSequences();
  }
};

}  // namespace

bool IsNemotronBoltR16PrefillDisabledValue(const char* env_value) {
  return env_value != nullptr && std::string_view{env_value} == "1";
}

bool ResolveNemotronBoltR16Enabled(bool model_supports_r16_prefill) {
  return model_supports_r16_prefill &&
         !IsNemotronBoltR16PrefillDisabledValue(std::getenv("BOLT_DISABLE_R16_PREFILL"));
}

void ValidateNemotronBoltSearch(const GeneratorParams& params) {
  const auto& s = params.search;
  Require(!s.do_sample && s.batch_size == 1 && s.num_beams == 1 &&
              s.num_return_sequences == 1,
          "only batch-one greedy decoding without beams is supported");
  Require(s.min_length == 0 && s.repetition_penalty == 1.0f &&
              s.no_repeat_ngram_size == 0 && s.diversity_penalty == 0.0f &&
              s.length_penalty == 1.0f && s.blank_penalty == 0.0f,
          "logit/length penalties and min_length are unsupported");
  Require(params.guidance_type.empty() && params.guidance_data.empty() &&
              !params.guidance_ff_tokens_enabled,
          "guidance is unsupported");
  Require(params.speculative.ngram_size == 0 && !params.speculative.ngram_chained_lookup &&
              params.speculative.min_adaptive_k == 0 && !params.speculative.cooldown &&
              params.speculative.max_draft_tokens == 7,
          "speculative.max_draft_tokens must be 7; external/adaptive speculative strategies are unsupported");
  Require(!s.chunk_size && !params.config.model.decoder.sliding_window,
          "external prefill chunking and decoder sliding_window are unsupported");
  Require(params.use_graph_capture, "CUDA graph capture must be enabled");
}

size_t NemotronBoltAcceptedPrefix(std::span<const int32_t> drafts,
                                  std::span<const int64_t> predictions) {
  Require(drafts.size() == kBoltVerifyRows - 1 && predictions.size() == kBoltVerifyRows,
          "verification requires seven drafts and eight predictions");
  size_t accepted = 0;
  while (accepted < drafts.size() && drafts[accepted] == predictions[accepted])
    ++accepted;
  return accepted;
}

std::vector<int32_t> NemotronBoltRoundTokens(std::span<const int32_t> drafts,
                                             std::span<const int64_t> predictions,
                                             size_t remaining, std::span<const int> eos) {
  const size_t accepted = NemotronBoltAcceptedPrefix(drafts, predictions);
  std::vector<int32_t> result;
  for (size_t row = 0; row <= accepted && result.size() < remaining; ++row) {
    const int32_t token = row < accepted ? drafts[row] : static_cast<int32_t>(predictions[row]);
    result.push_back(token);
    if (std::find(eos.begin(), eos.end(), token) != eos.end())
      break;
  }
  return result;
}

NemotronBoltTargetRowContract ValidateNemotronBoltTargetRowContract(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<std::string>& input_ids_symbolic,
    const std::vector<int64_t>& pred_ids_shape,
    const std::vector<std::string>& pred_ids_symbolic,
    const std::vector<int64_t>& logits_shape,
    const std::vector<std::string>& logits_symbolic,
    const std::vector<int64_t>& aux_shape,
    const std::vector<std::string>& aux_symbolic,
    int64_t expected_vocab_size) {
  Require(expected_vocab_size > 0, "vocab_size must be positive");
  Require(input_ids_shape.size() == 2 && input_ids_shape[0] == 1,
          "expected input_ids rank 2 with batch 1");
  Require(pred_ids_shape.size() == 2 && pred_ids_shape[0] == 1,
          "expected pred_ids rank 2 with batch 1");
  Require(logits_shape.size() == 3 && logits_shape[0] == 1 && logits_shape[2] == expected_vocab_size,
          "expected logits[1,R,vocab_size]");
  Require(aux_shape.size() == 3 && aux_shape[0] == 1 && aux_shape[2] > 0,
          "expected aux_hidden_states[1,R,width]");

  const int64_t r_in = input_ids_shape[1];
  if (r_in == static_cast<int64_t>(kBoltVerifyRows)) {
    Require(pred_ids_shape[1] == static_cast<int64_t>(kBoltVerifyRows) &&
                logits_shape[1] == static_cast<int64_t>(kBoltVerifyRows) &&
                aux_shape[1] == static_cast<int64_t>(kBoltVerifyRows),
            "mismatched fixed-R8 token output shapes");
    return {false, static_cast<size_t>(aux_shape[2])};
  }

  Require(r_in == -1 && pred_ids_shape[1] == -1 && logits_shape[1] == -1 && aux_shape[1] == -1,
          "target token row dimension must be fixed 8 or symbolic R (-1)");
  Require(input_ids_symbolic.size() == 2 && input_ids_symbolic[1] == "R" &&
              pred_ids_symbolic.size() == 2 && pred_ids_symbolic[1] == "R" &&
              logits_symbolic.size() == 3 && logits_symbolic[1] == "R" &&
              aux_symbolic.size() == 3 && aux_symbolic[1] == "R",
          "symbolic target token row dimension must be named R on input_ids, pred_ids, logits, and aux_hidden_states");
  return {true, static_cast<size_t>(aux_shape[2])};
}

std::vector<NemotronBoltPrefillStep> PlanNemotronBoltPrefillChunks(
    size_t prompt_rows, size_t capacity, bool allow_r16,
    size_t initial_length, int32_t initial_state_row,
    size_t prefix_r8_rows) {
  Require(prompt_rows > 0, "prompt_rows must be positive");
  Require(initial_state_row >= 0 && initial_state_row < static_cast<int32_t>(kBoltVerifyRows),
          "initial_state_row must be in [0,7]");
  Require(prefix_r8_rows < prompt_rows, "prefix_r8_rows must be strictly less than prompt_rows");
  std::vector<NemotronBoltPrefillStep> steps;
  size_t pos = 0;
  size_t length = initial_length;
  int32_t state_row = initial_state_row;

  while (pos < prefix_r8_rows) {
    const size_t valid = std::min(kBoltVerifyRows, prefix_r8_rows - pos);
    Require(length + kBoltVerifyRows <= capacity, "verify block exceeds fixed capacity");
    const int32_t seqlens = static_cast<int32_t>(length + kBoltVerifyRows - 1);
    const int32_t next_state_row = static_cast<int32_t>(valid - 1);
    steps.push_back(NemotronBoltPrefillStep{
        pos, valid, kBoltVerifyRows, seqlens, state_row, next_state_row, length + valid});
    pos += valid;
    length += valid;
    state_row = next_state_row;
  }

  while (allow_r16 && (prompt_rows - pos >= kBoltPrefillRows16)) {
    Require(length + kBoltPrefillRows16 <= capacity, "R16 prefill block exceeds fixed capacity");
    const int32_t seqlens = static_cast<int32_t>(length + kBoltPrefillRows16 - 1);
    const int32_t next_state_row = static_cast<int32_t>(kBoltVerifyRows - 1);
    steps.push_back(NemotronBoltPrefillStep{
        pos, kBoltPrefillRows16, kBoltPrefillRows16, seqlens, state_row, next_state_row,
        length + kBoltPrefillRows16});
    pos += kBoltPrefillRows16;
    length += kBoltPrefillRows16;
    state_row = next_state_row;
  }

  while (pos < prompt_rows) {
    const size_t valid = std::min(kBoltVerifyRows, prompt_rows - pos);
    Require(length + kBoltVerifyRows <= capacity, "verify block exceeds fixed capacity");
    const int32_t seqlens = static_cast<int32_t>(length + kBoltVerifyRows - 1);
    const int32_t next_state_row = static_cast<int32_t>(valid - 1);
    steps.push_back(NemotronBoltPrefillStep{
        pos, valid, kBoltVerifyRows, seqlens, state_row, next_state_row, length + valid});
    pos += valid;
    length += valid;
    state_row = next_state_row;
  }

  return steps;
}

NemotronBoltModel::NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env)
    : Model{std::move(config)} {
#if ORT_API_VERSION < 27
  // Each generator captures fresh graph ids in model-owned sessions; releasing them needs ORT 27.
  Require(false, "ONNX Runtime API 27 or later is required to release captured graphs");
#endif
  Require(p_device_->GetType() == DeviceType::CUDA, "a CUDA execution provider is required");
  p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);
  Require(IsGraphCaptureEnabled(config_->model.decoder.session_options),
          "enable_cuda_graph must be 1");
  Require(!config_->engine.dynamic_batching && !config_->engine.static_batching &&
              !config_->model.draft && !config_->model.mtp.IsEnabled() &&
              !config_->model.decoder.sliding_window,
          "Engine batching, decoder draft, MTP, and decoder sliding_window are unsupported");
  session_ = CreateSession(env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_);
  const auto& info = session_info_;
  Require(info.HasInput("input_ids") &&
              info.GetInputDataType("input_ids") == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
          "incompatible tensor input_ids");
  RequireTensor(info, "seqlens_k", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  RequireTensor(info, "state_row", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  RequireTensor(info, "total_sequence_length", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {});
  Require(info.HasOutput("pred_ids") &&
              info.GetOutputDataType("pred_ids") == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
          "incompatible tensor pred_ids");
  Require(info.HasOutput("logits") &&
              info.GetOutputDataType("logits") == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
          "incompatible tensor logits");
  Require(info.HasOutput("aux_hidden_states") &&
              info.GetOutputDataType("aux_hidden_states") == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
          "incompatible tensor aux_hidden_states");
  const auto row_contract = ValidateNemotronBoltTargetRowContract(
      info.GetInputShape("input_ids"), ToSymbolicStrings(info.GetInputSymbolicShape("input_ids")),
      info.GetOutputShape("pred_ids"), ToSymbolicStrings(info.GetOutputSymbolicShape("pred_ids")),
      info.GetOutputShape("logits"), ToSymbolicStrings(info.GetOutputSymbolicShape("logits")),
      info.GetOutputShape("aux_hidden_states"),
      ToSymbolicStrings(info.GetOutputSymbolicShape("aux_hidden_states")),
      config_->model.vocab_size);
  supports_r16_prefill_ = row_contract.dynamic_r;
  aux_width_ = row_contract.aux_width;
  std::set<std::string> outputs{"logits", "pred_ids", "aux_hidden_states"};
  size_t conv_count = 0, ssm_count = 0, key_count = 0, value_count = 0;
  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids" || name == "seqlens_k" || name == "state_row" ||
        name == "total_sequence_length")
      continue;
    const auto shape = info.GetInputShape(name);
    const auto type = info.GetInputDataType(name);
    if (name.rfind("past_key.", 0) == 0 || name.rfind("past_value.", 0) == 0) {
      Require(shape.size() == 4 && shape[0] == 1 && shape[1] == 2 &&
                  shape[2] > 8 && shape[3] == 128 && type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
              "expected dense bf16 KV[1,2,capacity,128]: " + name);
      if (capacity_ == 0) capacity_ = static_cast<size_t>(shape[2]);
      Require(capacity_ == static_cast<size_t>(shape[2]), "KV capacities differ");
      name.rfind("past_key.", 0) == 0 ? ++key_count : ++value_count;
    } else if (name.rfind("past_conv.", 0) == 0) {
      Require(shape == std::vector<int64_t>({8, 10240, 3}) &&
                  type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
              "invalid recurrent conv " + name);
      ++conv_count;
    } else if (name.rfind("past_ssm.", 0) == 0) {
      Require(shape == std::vector<int64_t>({8, 128, 128, 64}) &&
                  type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
              "invalid recurrent SSM " + name);
      ++ssm_count;
    } else {
      throw std::runtime_error("nemotron_bolt: unsupported input " + name);
    }
    const auto present = "present_" + name.substr(5);
    RequireTensor(info, present, false, type, shape);
    outputs.insert(present);
  }
  Require(conv_count > 0 && conv_count == ssm_count && key_count == 8 && value_count == 8,
          "expected matching recurrent states and eight dense attention layers");
  Require(session_->GetOutputNames().size() == outputs.size(), "unexpected target outputs");
  Require(config_->model.context_length > 0 &&
              capacity_ <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
              static_cast<size_t>(config_->model.context_length) <= capacity_ - kBoltVerifyRows,
          "context_length must leave eight scratch slots in the fixed KV capacity");
  const auto& d = config_->model.dflash2;
  Require(d.block_size == 8 && d.num_draft_tokens == 7 && !d.is_dspark &&
              !d.independent_sampling && !d.inputs.anchor_row_index.empty() &&
              d.main_aux_hidden_states == "aux_hidden_states",
          "a greedy LiLiCorr DFlash2 block8 drafter with anchor_row_index is required");
  auto drafter_config = CreateDflash2Config(*config_);
  drafter_config->model.decoder.session_options.config_entries.emplace_back(
      "optimization.disable_specified_optimizers", "GatherSliceToSplitFusion");
  drafter_model_ = std::make_shared<Dflash2Model>(std::move(drafter_config), env);
  // The shared validator expects the Engine's packed 2-D auxiliary output. Present an equivalent
  // metadata view without changing the actual fixed-R8 target graph or its buffers.
  struct PackedMetadata : ModelStateMetadata {
    explicit PackedMetadata(const SessionInfo& info) : info_{info} {}
    bool HasInput(const std::string& n) const override { return info_.HasInput(n); }
    bool HasOutput(const std::string& n) const override { return info_.HasOutput(n); }
    ONNXTensorElementDataType GetInputDataType(const std::string& n) const override { return info_.GetInputDataType(n); }
    ONNXTensorElementDataType GetOutputDataType(const std::string& n) const override { return info_.GetOutputDataType(n); }
    std::vector<int64_t> GetInputShape(const std::string& n) const override { return info_.GetInputShape(n); }
    std::vector<int64_t> GetOutputShape(const std::string& n) const override {
      auto shape = info_.GetOutputShape(n);
      if (n == "aux_hidden_states") return {-1, shape.back()};
      return shape;
    }
    const SessionInfo& info_;
  } packed{info};
  ValidateDflash2ModelCompatibility(*config_, packed, drafter_model_->session_info_, 256);
}

std::unique_ptr<State> NemotronBoltModel::CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const {
  ValidateNemotronBoltSearch(params);
  return std::make_unique<NemotronBoltState>(*this, params);
}

Tensor& NemotronBoltState::AddTensor(const std::string& name, bool input, DeviceInterface* device,
                                     ONNXTensorElementDataType type, const std::vector<int64_t>& shape) {
  auto tensor = std::make_unique<Tensor>(device, type);
  tensor->CreateTensor(shape, true);
  auto* value = tensor.get();
  tensors_.emplace(name, std::move(tensor));
  if (input) {
    input_storage_.push_back(name);
    inputs_.push_back(value->GetOrtTensor());
  } else {
    output_storage_.push_back(name);
    outputs_.push_back(value->GetOrtTensor());
  }
  return *value;
}

Tensor& NemotronBoltState::AddTensor16(const std::string& name, DeviceInterface* device,
                                       ONNXTensorElementDataType type, const std::vector<int64_t>& shape) {
  auto tensor = std::make_unique<Tensor>(device, type);
  tensor->CreateTensor(shape, true);
  auto* value = tensor.get();
  tensors_16_.emplace(name, std::move(tensor));
  return *value;
}

NemotronBoltState::NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params)
    : State{params, model},
      bolt_{model},
      r16_enabled_{ResolveNemotronBoltR16Enabled(model.supports_r16_prefill_)},
      request_{std::make_unique<Request>(model, params.search.max_length)} {
  const auto& info = model.session_info_;
  const int64_t r8 = static_cast<int64_t>(kBoltVerifyRows);
  const int64_t vocab_size = static_cast<int64_t>(model.config_->model.vocab_size);
  const int64_t aux_width = static_cast<int64_t>(model.aux_width_);
  const size_t max_input_rows = r16_enabled_ ? kBoltPrefillRows16 : kBoltVerifyRows;
  verify_inputs_span_ = model.p_device_->Allocate<int32_t>(2 * max_input_rows + 8);
  verify_inputs_span_.CpuSpan();
  int32_t* verify_in_base = verify_inputs_span_.Span().data();
  const auto& dev_mem_info = model.p_device_->GetAllocator().GetInfo();
  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids" || name == "seqlens_k" || name == "state_row") {
      void* ptr = name == "input_ids"   ? static_cast<void*>(verify_in_base)
                  : name == "seqlens_k" ? static_cast<void*>(verify_in_base + 2 * max_input_rows)
                                        : static_cast<void*>(verify_in_base + 2 * max_input_rows + 4);
      const size_t bytes = name == "input_ids" ? kBoltVerifyRows * sizeof(int64_t) : sizeof(int32_t);
      const std::vector<int64_t> shape =
          name == "input_ids" ? std::vector<int64_t>{1, r8} : info.GetInputShape(name);
      auto tensor = std::make_unique<Tensor>(
          OrtValue::CreateTensor(dev_mem_info, ptr, bytes, shape, info.GetInputDataType(name)));
      tensor->p_device_ = model.p_device_;
      input_storage_.push_back(name);
      inputs_.push_back(tensor->GetOrtTensor());
      tensors_.emplace(name, std::move(tensor));
      continue;
    }
    auto* device = name == "total_sequence_length" ? GetDeviceInterface(DeviceType::CPU) : model.p_device_;
    auto& tensor = AddTensor(name, true, device, info.GetInputDataType(name), info.GetInputShape(name));
    if (name.rfind("past_", 0) == 0) {
      caches_.push_back(&tensor);
      output_storage_.push_back("present_" + name.substr(5));
      outputs_.push_back(tensor.GetOrtTensor());
    }
  }
  AddTensor("logits", false, model.p_device_, info.GetOutputDataType("logits"), {1, r8, vocab_size});
  AddTensor("pred_ids", false, model.p_device_, info.GetOutputDataType("pred_ids"), {1, r8});
  AddTensor("aux_hidden_states", false, model.p_device_, info.GetOutputDataType("aux_hidden_states"),
            {1, r8, aux_width});
  pred_ids_span_ = tensors_.at("pred_ids")->GetDeviceSpan<int64_t>();
  pred_ids_span_.CpuSpan();
  for (auto& name : input_storage_) input_names_.push_back(name.c_str());
  for (auto& name : output_storage_) output_names_.push_back(name.c_str());
  inputs_8_ = inputs_;
  outputs_8_ = outputs_;
  const int32_t capacity = static_cast<int32_t>(model.capacity_);
  Upload(*tensors_.at("total_sequence_length"), std::span<const int32_t>{&capacity, 1});
  if (model.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model.config_->model.decoder.run_options.value());
  }
  run_options_->AddConfigEntry("disable_synchronize_execution_providers", "1");

  // The view is zero-copy: DFlash2 packs only the committed rows, never the padded/rejected ones.
  auto& aux = *tensors_.at("aux_hidden_states");
  const std::array<int64_t, 2> shape{r8, aux_width};
  aux_view_ = std::make_unique<Tensor>(OrtValue::CreateTensor(
      dev_mem_info, aux.GetMutableRawData(), aux.GetByteSpan().size(),
      shape, aux.GetType()));
  aux_view_->p_device_ = model.p_device_;
  if (r16_enabled_) {
    const int64_t r16 = static_cast<int64_t>(kBoltPrefillRows16);
    const std::array<int64_t, 2> in16_shape{1, r16};
    auto in16 = std::make_unique<Tensor>(OrtValue::CreateTensor(
        dev_mem_info, static_cast<void*>(verify_in_base), kBoltPrefillRows16 * sizeof(int64_t),
        in16_shape, info.GetInputDataType("input_ids")));
    in16->p_device_ = model.p_device_;
    tensors_16_.emplace("input_ids", std::move(in16));
    AddTensor16("logits", model.p_device_, info.GetOutputDataType("logits"), {1, r16, vocab_size});
    AddTensor16("pred_ids", model.p_device_, info.GetOutputDataType("pred_ids"), {1, r16});
    auto& aux16 = AddTensor16("aux_hidden_states", model.p_device_,
                              info.GetOutputDataType("aux_hidden_states"), {1, r16, aux_width});
    pred_ids_16_span_ = tensors_16_.at("pred_ids")->GetDeviceSpan<int64_t>();
    pred_ids_16_span_.CpuSpan();
    const std::array<int64_t, 2> shape16{r16, aux_width};
    aux_view_16_ = std::make_unique<Tensor>(OrtValue::CreateTensor(
        dev_mem_info, aux16.GetMutableRawData(), aux16.GetByteSpan().size(),
        shape16, aux16.GetType()));
    aux_view_16_->p_device_ = model.p_device_;

    inputs_16_ = inputs_8_;
    for (size_t i = 0; i < input_storage_.size(); ++i) {
      if (input_storage_[i] == "input_ids") {
        inputs_16_[i] = tensors_16_.at("input_ids")->GetOrtTensor();
      }
    }
    outputs_16_ = outputs_8_;
    for (size_t i = 0; i < output_storage_.size(); ++i) {
      auto it16 = tensors_16_.find(output_storage_[i]);
      if (it16 != tensors_16_.end()) {
        outputs_16_[i] = it16->second->GetOrtTensor();
      }
    }
  }
  size_t blocks = Dflash2Drafter::PoolBlocks(*model.config_, 256, 1);
  if (blocks == 0) blocks = (model.capacity_ + 255) / 256 + 1;
  drafter_ = std::make_unique<Dflash2Drafter>(model.drafter_model_, 256, blocks, 1, true);
  const size_t max_prefill_rows =
      PlanNemotronBoltPrefillAuxWindow(1, drafter_->ContextWindow(), kBoltVerifyRows).max_buffer_rows;
  const std::array<int64_t, 2> prefill_shape{static_cast<int64_t>(max_prefill_rows),
                                             static_cast<int64_t>(model.aux_width_)};
  prefill_aux_ = std::make_unique<Tensor>(model.p_device_, aux.GetType());
  prefill_aux_->CreateTensor(prefill_shape, true);
  Reset();
  std::vector<int32_t> dummy(kBoltVerifyRows, 1);
  // Initial graph-enabled executions are not trustworthy. Capture on dummy zero state and discard
  // everything, including the recurrent row selection, before any real prompt is processed.
  for (int i = 0; i < 4; ++i) Verify(dummy, i == 3);
  if (r16_enabled_) {
    Reset();
    std::vector<int32_t> dummy16(kBoltPrefillRows16, 1);
    for (int i = 0; i < 4; ++i) PrefillChunk16(dummy16, i == 3);
  }
  for (int i = 0; i < 4; ++i) {
    drafter_->ReleaseAll();
    Propose(8, 0, 1);
  }
  Reset();
  in_warmup_ = false;
}

NemotronBoltState::~NemotronBoltState() = default;

DeviceSpan<float> NemotronBoltState::Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) {
  throw std::runtime_error("nemotron_bolt uses native block decoding, not the logits search path");
}

void NemotronBoltState::Reset() {
  for (auto* cache : caches_) cache->GetByteSpan().Zero();
  model_.p_device_->Synchronize();
  drafter_->ReleaseAll();
  length_ = 0;
  state_row_ = 0;
  anchor_ = 0;
  drafts_.clear();
  pending_.clear();
  prefilled_ = false;
  prefill_r8_calls_ = 0;
  prefill_r16_calls_ = 0;
  verify_r8_calls_ = 0;
  last_prefill_draft_start_ = 0;
  last_prefill_draft_rows_ = 0;
  stats_ = {};
}

void NemotronBoltState::RewindTo(size_t length) {
  Require(length == 0, "only reset-to-zero rewind is supported");
  Reset();
}

void NemotronBoltState::SetExtraInputs(const std::vector<ExtraInput>& inputs) {
  Require(inputs.empty(), "extra inputs are unsupported");
}

void NemotronBoltState::SetActiveAdapter(Adapters*, const std::string&) {
  // Adapters would select a new graph identity that is never warmed on disposable state.
  throw std::runtime_error("nemotron_bolt: adapters are unsupported");
}

std::vector<int64_t> NemotronBoltState::Verify(std::span<const int32_t> tokens, bool want_predictions) {
  Require(tokens.size() == kBoltVerifyRows && length_ + kBoltVerifyRows <= bolt_.capacity_,
          "verify block exceeds fixed capacity");
  const size_t max_input_rows = r16_enabled_ ? kBoltPrefillRows16 : kBoltVerifyRows;
  auto cpu_in = verify_inputs_span_.CpuSpan();
  auto* ids_out = reinterpret_cast<int64_t*>(cpu_in.data());
  for (size_t i = 0; i < kBoltVerifyRows; ++i) ids_out[i] = tokens[i];
  cpu_in[2 * max_input_rows] = static_cast<int32_t>(length_ + kBoltVerifyRows - 1);
  cpu_in[2 * max_input_rows + 4] = state_row_;
  verify_inputs_span_.CopyCpuToDevice();
  inputs_ = inputs_8_;
  outputs_ = outputs_8_;
  State::Run(*bolt_.session_, true, 8, 0, nullptr);
  ++stats_.target_forward_passes;
  if (in_warmup_) {
    ++warmup_r8_calls_;
  } else if (!prefilled_) {
    ++prefill_r8_calls_;
  } else {
    ++verify_r8_calls_;
  }
  if (!want_predictions) return {};
  auto cpu = pred_ids_span_.CopyDeviceToCpu();
  return {cpu.begin(), cpu.end()};
}

std::vector<int64_t> NemotronBoltState::PrefillChunk16(std::span<const int32_t> tokens, bool want_predictions) {
  Require(r16_enabled_ && !inputs_16_.empty(),
          "R16 target prefill profile is unavailable or disabled for this generator");
  Require(tokens.size() == kBoltPrefillRows16 && length_ + kBoltPrefillRows16 <= bolt_.capacity_,
          "R16 prefill block exceeds fixed capacity");
  auto cpu_in = verify_inputs_span_.CpuSpan();
  auto* ids_out = reinterpret_cast<int64_t*>(cpu_in.data());
  for (size_t i = 0; i < kBoltPrefillRows16; ++i) ids_out[i] = tokens[i];
  cpu_in[2 * kBoltPrefillRows16] = static_cast<int32_t>(length_ + kBoltPrefillRows16 - 1);
  cpu_in[2 * kBoltPrefillRows16 + 4] = state_row_;
  verify_inputs_span_.CopyCpuToDevice();
  inputs_ = inputs_16_;
  outputs_ = outputs_16_;
  State::Run(*bolt_.session_, true, static_cast<int>(kBoltPrefillRows16), 0, nullptr);
  ++stats_.target_forward_passes;
  if (in_warmup_) {
    ++warmup_r16_calls_;
  } else {
    ++prefill_r16_calls_;
  }
  if (!want_predictions) return {};
  auto cpu = pred_ids_16_span_.CopyDeviceToCpu();
  return {cpu.begin(), cpu.end()};
}

NemotronBoltPrefillAuxWindow PlanNemotronBoltPrefillAuxWindow(
    size_t prompt_rows, size_t context_window, size_t block_rows) {
  Require(prompt_rows > 0 && block_rows > 0, "prefill rows and block size must be positive");
  if (context_window != 0) {
    Require(context_window <= std::numeric_limits<size_t>::max() - 2 * block_rows,
            "prefill context window overflow");
  }
  const size_t max_buffer_rows =
      ((std::max(context_window, block_rows) + block_rows - 1) / block_rows + 1) * block_rows;
  const size_t draft_start =
      (context_window != 0 && prompt_rows > context_window)
          ? ((prompt_rows - context_window) / block_rows) * block_rows
          : 0;
  const size_t draft_rows = prompt_rows - draft_start;
  const size_t ingest_offset =
      (context_window != 0 && draft_rows > context_window) ? (draft_rows - context_window) : 0;
  const size_t ingest_rows = draft_rows - ingest_offset;
  const size_t ingest_first_position = draft_start + ingest_offset;
  return NemotronBoltPrefillAuxWindow{
      .max_buffer_rows = max_buffer_rows,
      .draft_start = draft_start,
      .draft_rows = draft_rows,
      .ingest_offset = ingest_offset,
      .ingest_rows = ingest_rows,
      .ingest_first_position = ingest_first_position};
}

void CopyNemotronBoltPrefillAuxRows(Tensor& destination, Tensor& source,
                                    size_t chunk_position, size_t valid_rows,
                                    size_t physical_prefix_rows) {
  const auto src_shape = source.GetShape();
  const auto dst_shape = destination.GetShape();
  Require(src_shape.size() == 2 && dst_shape.size() == 2 && src_shape[1] == dst_shape[1] &&
              src_shape[0] > 0 && dst_shape[0] > 0 && src_shape[1] > 0 &&
              source.GetType() == destination.GetType() && valid_rows > 0 &&
              valid_rows <= static_cast<size_t>(src_shape[0]) &&
              physical_prefix_rows <= std::numeric_limits<size_t>::max() - static_cast<size_t>(dst_shape[0]),
          "invalid prefill auxiliary row copy");
  const size_t logical_rows = physical_prefix_rows + static_cast<size_t>(dst_shape[0]);
  Require(chunk_position <= logical_rows && valid_rows <= logical_rows - chunk_position,
          "prefill auxiliary chunk exceeds logical prompt");
  if (chunk_position + valid_rows <= physical_prefix_rows)
    return;  // This prefix is not read by the existing windowed drafter ingestion.
  const size_t source_row = chunk_position < physical_prefix_rows ? physical_prefix_rows - chunk_position : 0;
  const size_t destination_row = chunk_position + source_row - physical_prefix_rows;
  const size_t copied_rows = valid_rows - source_row;
  const size_t row_bytes = static_cast<size_t>(src_shape[1]) * Ort::SizeOf(source.GetType());
  destination.GetByteSpan().subspan(destination_row * row_bytes, copied_rows * row_bytes).CopyFrom(source.GetByteSpan().subspan(source_row * row_bytes, copied_rows * row_bytes));
}

void NemotronBoltState::Propose(size_t context_rows, size_t first_position, int32_t anchor, Tensor* aux_override) {
  Dflash2Drafter::Feed feed{
      .request = request_.get(),
      .aux_row_begin = 0,
      .aux_row_count = context_rows,
      .first_position = first_position,
      .anchor_token = anchor,
      .draft_eligible = true,
      .wants_drafts = true};
  std::vector<std::vector<int32_t>> proposals;
  Tensor& aux_source = aux_override ? *aux_override : *aux_view_;
  Require(drafter_->Propose(aux_source, {&feed, 1}, proposals) &&
              proposals.size() == 1 && proposals[0].size() == 7,
          "LiLiCorr drafter did not produce seven tokens");
  drafts_ = std::move(proposals[0]);
  ++stats_.draft_forward_passes;
}

void NemotronBoltState::Prefill(std::span<const int32_t> tokens) {
  Require(!prefilled_ && !tokens.empty(), "AppendTokens requires a fresh or reset generator");
  for (const auto token : tokens)
    Require(token >= 0 && token < model_.config_->model.vocab_size, "token id outside vocabulary");
  // A failure after the first chunk leaves advanced caches and a native cursor that the public
  // sequence (still empty) cannot rewind, so discard all partial state before propagating.
  try {
    const auto aux_window =
        PlanNemotronBoltPrefillAuxWindow(tokens.size(), drafter_->ContextWindow(), kBoltVerifyRows);
    const size_t draft_start = aux_window.draft_start;
    const size_t draft_rows = aux_window.draft_rows;
    last_prefill_draft_start_ = draft_start;
    last_prefill_draft_rows_ = draft_rows;
    const auto plan = PlanNemotronBoltPrefillChunks(
        tokens.size(), bolt_.capacity_, r16_enabled_, length_, state_row_);
    const bool single_chunk = (draft_start == 0 && plan.size() == 1);
    for (size_t i = 0; i < plan.size(); ++i) {
      const auto& step = plan[i];
      const bool last_chunk = (i + 1 == plan.size());
      if (step.profile_rows == kBoltPrefillRows16) {
        const auto predictions = PrefillChunk16(tokens.subspan(step.pos, kBoltPrefillRows16), last_chunk);
        if (!single_chunk) {
          CopyNemotronBoltPrefillAuxRows(*prefill_aux_, *aux_view_16_, step.pos, kBoltPrefillRows16, draft_start);
        }
        if (last_chunk) {
          anchor_ = static_cast<int32_t>(predictions[kBoltPrefillRows16 - 1]);
        }
      } else {
        std::vector<int32_t> block(kBoltVerifyRows, tokens[step.pos + step.valid_rows - 1]);
        std::copy_n(tokens.begin() + step.pos, step.valid_rows, block.begin());
        const auto predictions = Verify(block, last_chunk);
        if (!single_chunk) {
          CopyNemotronBoltPrefillAuxRows(*prefill_aux_, *aux_view_, step.pos, step.valid_rows, draft_start);
        }
        if (last_chunk) {
          anchor_ = static_cast<int32_t>(predictions[step.valid_rows - 1]);
        }
      }
      length_ = step.length_after;
      state_row_ = step.state_row_out;
    }
    Tensor* single_aux =
        (plan[0].profile_rows == kBoltPrefillRows16) ? aux_view_16_.get() : aux_view_.get();
    Propose(draft_rows, draft_start, anchor_, single_chunk ? single_aux : prefill_aux_.get());
    if (tokens.size() < static_cast<size_t>(params_->search.max_length)) {
      pending_.push_back(anchor_);
      stats_.tokens_queued = 1;
    }
    stats_.effective_k = 7;
    prefilled_ = true;
  } catch (...) {
    try {
      Reset();
    } catch (...) {
    }
    throw;
  }
}

int32_t NemotronBoltState::NextToken(size_t public_length) {
  Require(prefilled_, "AppendTokens must precede GenerateNextToken");
  if (pending_.empty()) {
    std::vector<int32_t> block{anchor_};
    block.insert(block.end(), drafts_.begin(), drafts_.end());
    const auto predictions = Verify(block);
    const size_t accepted = NemotronBoltAcceptedPrefix(drafts_, predictions);
    if (g_log.enabled && g_log.model_logits) {
      auto& log = Log("model_logits");
      log << "Bolt verify length=" << length_ << " state_row=" << state_row_ << " anchor=" << anchor_
          << " accepted=" << accepted << " drafts=";
      for (auto token : drafts_) log << token << ',';
      log << " predictions=";
      for (auto token : predictions) log << token << ',';
      log << std::endl;
    }
    const auto tokens = NemotronBoltRoundTokens(
        drafts_, predictions, static_cast<size_t>(params_->search.max_length) - public_length,
        model_.config_->model.eos_token_id);
    pending_.insert(pending_.end(), tokens.begin(), tokens.end());
    ++stats_.rounds;
    ++stats_.completed_rounds;
    ++stats_.target_verify_forward_passes;
    stats_.draft_tokens_proposed += 7;
    stats_.draft_tokens_evaluated += 7;
    stats_.draft_tokens_accepted += accepted;
    ++stats_.acceptance_length_histogram[accepted];
    stats_.tokens_queued += tokens.size();
    if (tokens.size() == accepted + 1) {
      accepted == 7 ? ++stats_.bonus_tokens : ++stats_.correction_tokens;
    }
    accepted == 7 ? ++stats_.full_accept_rounds : ++stats_.partial_accept_rounds;
    if (accepted == 0) ++stats_.zero_accept_rounds;
    anchor_ = static_cast<int32_t>(predictions[accepted]);
    const size_t first_position = length_;
    length_ += accepted + 1;
    state_row_ = static_cast<int32_t>(accepted);
    const bool finishes = public_length + tokens.size() >= static_cast<size_t>(params_->search.max_length) ||
                          (!tokens.empty() && std::find(model_.config_->model.eos_token_id.begin(),
                                                        model_.config_->model.eos_token_id.end(), tokens.back()) !=
                                                  model_.config_->model.eos_token_id.end());
    if (!finishes) Propose(accepted + 1, first_position, anchor_);
  }
  Require(!pending_.empty(), "no token available");
  const int32_t token = pending_.front();
  pending_.pop_front();
  ++stats_.tokens_emitted;
  return token;
}

SpeculativeStats NemotronBoltState::Stats() const {
  auto stats = stats_;
  stats.tokens_buffered = pending_.size();
  if (stats.draft_tokens_evaluated != 0)
    stats.acceptance_rate = static_cast<float>(stats.draft_tokens_accepted) / stats.draft_tokens_evaluated;
  if (stats.rounds != 0) {
    stats.avg_draft_tokens_per_round = 7.0f;
    stats.mean_emitted_tokens_per_round = static_cast<float>(stats.tokens_emitted - 1) / stats.rounds;
  }
  return stats;
}

std::unique_ptr<DecodingStrategy> MakeNemotronBoltStrategy(Generator& generator) {
  return std::make_unique<BoltStrategy>(generator);
}

std::unique_ptr<Search> CreateNemotronBoltSearch(const GeneratorParams& params) {
  Require(!params.config.model.decoder.sliding_window, "decoder sliding_window is unsupported");
  return std::make_unique<BoltSearch>(params);
}

}  // namespace Generators
