// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt.h"
#include "engine/request.h"
#include "search.h"

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
  Require((input ? info.GetInputDataType(name) : info.GetOutputDataType(name)) == type &&
              (input ? info.GetInputShape(name) : info.GetOutputShape(name)) == shape,
          "incompatible tensor " + name);
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
  Require(!s.chunk_size && !s.adaptive_chunking, "external prefill chunking is unsupported");
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

NemotronBoltModel::NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env)
    : Model{std::move(config)} {
  Require(p_device_->GetType() == DeviceType::CUDA, "a CUDA execution provider is required");
  p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);
  Require(IsGraphCaptureEnabled(config_->model.decoder.session_options),
          "enable_cuda_graph must be 1");
  Require(!config_->engine.dynamic_batching && !config_->engine.static_batching &&
              !config_->model.draft && !config_->model.mtp.IsEnabled(),
          "Engine batching, decoder draft, and MTP are unsupported");
  session_ = CreateSession(env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_);
  const auto& info = session_info_;
  RequireTensor(info, "input_ids", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, 8});
  RequireTensor(info, "seqlens_k", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  RequireTensor(info, "state_row", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  RequireTensor(info, "total_sequence_length", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {});
  RequireTensor(info, "pred_ids", false, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, 8});
  RequireTensor(info, "logits", false, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                {1, 8, config_->model.vocab_size});
  const auto aux_shape = info.GetOutputShape("aux_hidden_states");
  Require(aux_shape.size() == 3 && aux_shape[0] == 1 && aux_shape[1] == 8 && aux_shape[2] > 0 &&
              info.GetOutputDataType("aux_hidden_states") == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
          "expected bf16 aux_hidden_states[1,8,width]");
  aux_width_ = static_cast<size_t>(aux_shape[2]);
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

NemotronBoltState::NemotronBoltState(const NemotronBoltModel& model, const GeneratorParams& params)
    : State{params, model}, bolt_{model}, request_{std::make_unique<Request>(model, params.search.max_length)} {
  const auto& info = model.session_info_;
  for (const auto& name : info.GetInputNames()) {
    auto* device = name == "total_sequence_length" ? GetDeviceInterface(DeviceType::CPU) : model.p_device_;
    auto& tensor = AddTensor(name, true, device, info.GetInputDataType(name), info.GetInputShape(name));
    if (name.rfind("past_", 0) == 0) {
      caches_.push_back(&tensor);
      output_storage_.push_back("present_" + name.substr(5));
      outputs_.push_back(tensor.GetOrtTensor());
    }
  }
  for (const auto* name : {"logits", "pred_ids", "aux_hidden_states"})
    AddTensor(name, false, model.p_device_, info.GetOutputDataType(name), info.GetOutputShape(name));
  for (auto& name : input_storage_) input_names_.push_back(name.c_str());
  for (auto& name : output_storage_) output_names_.push_back(name.c_str());
  binding_ = OrtIoBinding::Create(*model.session_);
  for (size_t i = 0; i < input_names_.size(); ++i)
    binding_->BindInput(input_names_[i], *inputs_[i]);
  for (size_t i = 0; i < output_names_.size(); ++i)
    binding_->BindOutput(output_names_[i], *outputs_[i]);
  const int32_t capacity = static_cast<int32_t>(model.capacity_);
  Upload(*tensors_.at("total_sequence_length"), std::span<const int32_t>{&capacity, 1});

  // The view is zero-copy: DFlash2 packs only the committed rows, never the padded/rejected ones.
  auto& aux = *tensors_.at("aux_hidden_states");
  const int64_t shape[]{8, static_cast<int64_t>(model.aux_width_)};
  aux_view_ = std::make_unique<Tensor>(OrtValue::CreateTensor(
      *model.p_device_->GetMemoryInfo(), aux.GetMutableRawData(), aux.GetByteSpan().size(),
      shape, aux.GetType()));
  aux_view_->p_device_ = model.p_device_;
  size_t blocks = Dflash2Drafter::PoolBlocks(*model.config_, 256, 1);
  if (blocks == 0) blocks = (model.capacity_ + 255) / 256 + 1;
  drafter_ = std::make_unique<Dflash2Drafter>(model.drafter_model_, 256, blocks, 1, true);
  Reset();
  std::vector<int32_t> dummy(kBoltVerifyRows, 1);
  // Initial graph-enabled executions are not trustworthy. Capture on dummy zero state and discard
  // everything, including the recurrent row selection, before any real prompt is processed.
  for (int i = 0; i < 4; ++i) Verify(dummy);
  for (int i = 0; i < 4; ++i) {
    drafter_->ReleaseAll();
    Propose(8, 0, 1);
  }
  Reset();
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
  stats_ = {};
}

void NemotronBoltState::RewindTo(size_t length) {
  Require(length == 0, "only reset-to-zero rewind is supported");
  Reset();
}

void NemotronBoltState::SetExtraInputs(const std::vector<ExtraInput>& inputs) {
  Require(inputs.empty(), "extra inputs are unsupported");
}

std::vector<int64_t> NemotronBoltState::Verify(std::span<const int32_t> tokens) {
  Require(tokens.size() == kBoltVerifyRows && length_ + kBoltVerifyRows <= bolt_.capacity_,
          "verify block exceeds fixed capacity");
  std::vector<int64_t> ids(tokens.begin(), tokens.end());
  Upload(*tensors_.at("input_ids"), std::span<const int64_t>{ids});
  const int32_t seqlens = static_cast<int32_t>(length_ + kBoltVerifyRows - 1);
  Upload(*tensors_.at("seqlens_k"), std::span<const int32_t>{&seqlens, 1});
  Upload(*tensors_.at("state_row"), std::span<const int32_t>{&state_row_, 1});
  State::Run(*bolt_.session_, true, 8, 0, binding_.get());
  auto span = tensors_.at("pred_ids")->GetDeviceSpan<int64_t>();
  auto cpu = span.CopyDeviceToCpu();
  ++stats_.target_forward_passes;
  return {cpu.begin(), cpu.end()};
}

void NemotronBoltState::Propose(size_t context_rows, size_t first_position, int32_t anchor) {
  Dflash2Drafter::Feed feed{request_.get(), 0, context_rows, first_position, anchor, true, true, false};
  std::vector<std::vector<int32_t>> proposals;
  Require(drafter_->Propose(*aux_view_, {&feed, 1}, proposals) &&
              proposals.size() == 1 && proposals[0].size() == 7,
          "LiLiCorr drafter did not produce seven tokens");
  drafts_ = std::move(proposals[0]);
  ++stats_.draft_forward_passes;
}

void NemotronBoltState::Prefill(std::span<const int32_t> tokens) {
  Require(!prefilled_ && !tokens.empty(), "AppendTokens requires a fresh or reset generator");
  for (const auto token : tokens)
    Require(token >= 0 && token < model_.config_->model.vocab_size, "token id outside vocabulary");
  for (size_t pos = 0; pos < tokens.size(); pos += kBoltVerifyRows) {
    const size_t valid = std::min(kBoltVerifyRows, tokens.size() - pos);
    std::vector<int32_t> block(kBoltVerifyRows, tokens[pos + valid - 1]);
    std::copy_n(tokens.begin() + pos, valid, block.begin());
    const auto predictions = Verify(block);
    anchor_ = static_cast<int32_t>(predictions[valid - 1]);
    Propose(valid, length_, anchor_);
    length_ += valid;
    state_row_ = static_cast<int32_t>(valid - 1);
  }
  if (tokens.size() < static_cast<size_t>(params_->search.max_length)) {
    pending_.push_back(anchor_);
    stats_.tokens_queued = 1;
  }
  stats_.effective_k = 7;
  prefilled_ = true;
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
  return std::make_unique<BoltSearch>(params);
}

}  // namespace Generators
