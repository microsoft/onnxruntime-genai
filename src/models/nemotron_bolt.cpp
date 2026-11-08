// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt.h"
#include "engine/request.h"
#include "search.h"

#include <array>
#include <limits>
#include <set>

namespace Generators {
namespace {

constexpr int kCaptureWarmupRuns = 4;

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
  Require(params.IsGreedySampling() && s.batch_size == 1 && s.num_beams == 1 &&
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
              (params.speculative.max_draft_tokens == 0 || params.speculative.max_draft_tokens == 7),
          "speculative.max_draft_tokens must be 0 or 7; external/adaptive speculative strategies are unsupported");
  Require(!s.chunk_size && !params.config.model.decoder.sliding_window,
          "external prefill chunking and decoder sliding_window are unsupported");
  Require(params.speculative.max_draft_tokens == 0 || params.use_graph_capture,
          "CUDA graph capture must be enabled for speculative decoding");
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

size_t NemotronBoltVerifyPrefixRows(size_t position) {
  // The pinned H128 Flash split-K kernel partitions live K tiles of 128 keys.
  // Keep each committed query's padded R8 endpoint in the same tile as row zero,
  // so split widths and BF16 softmax/PV arithmetic match target-only decoding.
  constexpr size_t key_tile = 128;
  return std::min(kBoltVerifyRows, key_tile - ((position + kBoltVerifyRows - 1) % key_tile));
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

std::unique_ptr<Config> CreateNemotronBoltDrafterConfig(const Config& config) {
  auto drafter = CreateDflash2Config(config);
  // Supported external runtimes do not all retain Split's captured host buffers.
  // Exclude only this fusion until the minimum runtime guarantees that fix;
  // preserve caller exclusions without lowering the graph optimization level.
  auto& entries = drafter->model.decoder.session_options.config_entries;
  const auto entry = std::find_if(entries.rbegin(), entries.rend(), [](const auto& option) {
    return option.first == "optimization.disable_specified_optimizers";
  });
  if (entry == entries.rend()) {
    entries.emplace_back("optimization.disable_specified_optimizers", "GatherSliceToSplitFusion");
  } else if ((";" + entry->second + ";").find(";GatherSliceToSplitFusion;") == std::string::npos) {
    if (!entry->second.empty() && entry->second.back() != ';') entry->second += ';';
    entry->second += "GatherSliceToSplitFusion";
  }
  if (Ort::runtime_api_version >= 32) {
    // The native drafter validates its own paged metadata. Preserve logical row counts so
    // partially accepted blocks can use padded, fixed-shape CUDA graphs without changing KV state.
    constexpr auto length_policy = "session.paged_attention.skip_input_sanitizer";
    if (std::none_of(entries.begin(), entries.end(), [](const auto& option) {
          return option.first == length_policy;
        })) {
      entries.emplace_back(length_policy, "1");
    }
  }
  return drafter;
}

void ValidateNemotronBoltGraphCapture(const Config::SessionOptions& options, bool speculative) {
  const bool capture = IsGraphCaptureEnabled(options);
#if ORT_API_VERSION < 27
  // Each generator captures fresh graph ids in model-owned sessions; releasing them needs ORT 27.
  Require(!capture,
          "ONNX Runtime API 27 or later is required to release captured graphs");
#else
  Require(!capture || Ort::api_version >= 27,
          "ONNX Runtime API 27 or later is required to release captured graphs");
#endif
  Require(!speculative || capture, "enable_cuda_graph must be 1 for speculative decoding");
}

NemotronBoltModel::NemotronBoltModel(std::unique_ptr<Config> config, OrtEnv& env)
    : Model{std::move(config)} {
  Require(config_->speculative.max_draft_tokens == 0 ||
              config_->speculative.max_draft_tokens == 7,
          "speculative.max_draft_tokens must be 0 or 7");
  ValidateNemotronBoltGraphCapture(config_->model.decoder.session_options,
                                   config_->speculative.max_draft_tokens != 0);
  Require(p_device_->GetType() == DeviceType::CUDA, "a CUDA execution provider is required");
  p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);
  Require(!config_->engine.dynamic_batching && !config_->engine.static_batching &&
              !config_->model.draft && !config_->model.mtp.IsEnabled() &&
              !config_->model.decoder.sliding_window,
          "Engine batching, decoder draft, MTP, and decoder sliding_window are unsupported");
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
  if (config_->speculative.max_draft_tokens == 0)
    return;
  const auto& d = config_->model.dflash2;
  Require(d.block_size == 8 && d.num_draft_tokens == 7 && !d.is_dspark &&
              !d.independent_sampling && !d.inputs.anchor_row_index.empty() &&
              d.main_aux_hidden_states == "aux_hidden_states",
          "a greedy LiLiCorr DFlash2 block8 drafter with anchor_row_index is required");
  auto drafter_config = CreateNemotronBoltDrafterConfig(*config_);
  ValidateNemotronBoltGraphCapture(drafter_config->model.decoder.session_options, true);
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
  Require(!params.use_graph_capture || IsGraphCaptureEnabled(config_->model.decoder.session_options),
          "graph capture is disabled in the target session");
  Require(params.speculative.max_draft_tokens == 0 || drafter_model_,
          "speculative decoding requires a model configured with a drafter");
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
  verify_inputs_span_ = model.p_device_->Allocate<int32_t>(2 * kBoltVerifyRows + 8);
  verify_inputs_span_.CpuSpan();
  int32_t* verify_in_base = verify_inputs_span_.Span().data();
  const auto& dev_mem_info = model.p_device_->GetAllocator().GetInfo();
  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids" || name == "seqlens_k" || name == "state_row") {
      void* ptr = name == "input_ids"   ? static_cast<void*>(verify_in_base)
                  : name == "seqlens_k" ? static_cast<void*>(verify_in_base + 2 * kBoltVerifyRows)
                                        : static_cast<void*>(verify_in_base + 2 * kBoltVerifyRows + 4);
      const size_t bytes = name == "input_ids" ? kBoltVerifyRows * sizeof(int64_t) : sizeof(int32_t);
      const auto shape = info.GetInputShape(name);
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
  for (const auto* name : {"logits", "pred_ids", "aux_hidden_states"})
    AddTensor(name, false, model.p_device_, info.GetOutputDataType(name), info.GetOutputShape(name));
  pred_ids_span_ = tensors_.at("pred_ids")->GetDeviceSpan<int64_t>();
  pred_ids_span_.CpuSpan();
  for (auto& name : input_storage_) input_names_.push_back(name.c_str());
  for (auto& name : output_storage_) output_names_.push_back(name.c_str());
  const int32_t capacity = static_cast<int32_t>(model.capacity_);
  Upload(*tensors_.at("total_sequence_length"), std::span<const int32_t>{&capacity, 1});
  if (model.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model.config_->model.decoder.run_options.value());
  }
  run_options_->AddConfigEntry("disable_synchronize_execution_providers", "1");
  if (!params.use_graph_capture)
    run_options_->AddConfigEntry("gpu_graph_id", "-1");

  if (params.speculative.max_draft_tokens != 0) {
    // DFlash2 packs only committed rows from this zero-copy view.
    auto& aux = *tensors_.at("aux_hidden_states");
    const std::array<int64_t, 2> shape{8, static_cast<int64_t>(model.aux_width_)};
    aux_view_ = std::make_unique<Tensor>(OrtValue::CreateTensor(
        *model.p_device_->GetMemoryInfo(), aux.GetMutableRawData(), aux.GetByteSpan().size(),
        shape, aux.GetType()));
    aux_view_->p_device_ = model.p_device_;
    size_t blocks = Dflash2Drafter::PoolBlocks(*model.config_, 256, 1);
    if (blocks == 0) blocks = (model.capacity_ + 255) / 256 + 1;
    drafter_ = std::make_unique<Dflash2Drafter>(model.drafter_model_, 256, blocks, 1, true);
    const size_t max_prefill_rows =
        ((std::max(drafter_->ContextWindow(), kBoltVerifyRows) + kBoltVerifyRows - 1) / kBoltVerifyRows + 1) *
        kBoltVerifyRows;
    const std::array<int64_t, 2> prefill_shape{static_cast<int64_t>(max_prefill_rows),
                                               static_cast<int64_t>(model.aux_width_)};
    prefill_aux_ = std::make_unique<Tensor>(model.p_device_, aux.GetType());
    prefill_aux_->CreateTensor(prefill_shape, true);
  }
  Reset();
  std::vector<int32_t> dummy(kBoltVerifyRows, 1);
  // The legacy two-run capture delay needs four calls through the first replay.
  // Newer ORT can warm up and capture internally within a single Run().
  // Use disposable state: warmup may re-execute in-place aliases before Reset().
  if (params.use_graph_capture) {
    for (int i = 0; i < kCaptureWarmupRuns; ++i)
      Verify(dummy, i == kCaptureWarmupRuns - 1);
  }
  if (drafter_) {
    for (int i = 0; i < kCaptureWarmupRuns; ++i) {
      drafter_->ReleaseAll();
      Propose(8, 0, 1);
    }
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
  if (drafter_) drafter_->ReleaseAll();
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

void NemotronBoltState::SetActiveAdapter(Adapters*, const std::string&) {
  // Adapters would select a new graph identity that is never warmed on disposable state.
  throw std::runtime_error("nemotron_bolt: adapters are unsupported");
}

std::vector<int64_t> NemotronBoltState::Verify(std::span<const int32_t> tokens, bool want_predictions) {
  const size_t prefix = NemotronBoltVerifyPrefixRows(length_);
  if (!prefilled_ || !drafter_ || prefix == kBoltVerifyRows)
    return VerifyBlock(tokens, want_predictions);
  if (length_ + prefix + kBoltVerifyRows > bolt_.capacity_) {
    Require(static_cast<size_t>(params_->search.max_length) - length_ - 1 <= prefix,
            "arithmetic boundary exceeds verification scratch capacity");
    return VerifyBlock(tokens, want_predictions);
  }
  const size_t position = length_;
  const int32_t selected = state_row_;
  try {
    auto predictions = VerifyBlock(tokens, want_predictions);
    const auto preserve = [&](const std::string& name, size_t source_row, size_t destination_row, size_t rows) {
      auto& tensor = *tensors_.at(name);
      auto& backup = verify_prefix_backup_[name];
      if (!backup) {
        backup = std::make_unique<Tensor>(model_.p_device_, tensor.GetType());
        backup->CreateTensor(tensor.GetShape(), true);
      }
      const size_t row_bytes = tensor.GetByteSpan().size() / kBoltVerifyRows;
      backup->GetByteSpan().subspan(destination_row * row_bytes, rows * row_bytes).CopyFrom(tensor.GetByteSpan().subspan(source_row * row_bytes, rows * row_bytes));
    };
    for (const auto& [name, tensor] : tensors_) {
      if (name.rfind("past_conv.", 0) == 0 || name.rfind("past_ssm.", 0) == 0 ||
          name == "logits" || name == "aux_hidden_states" || name == "pred_ids")
        preserve(name, 0, 0, prefix);
    }
    // Reuse only this verified candidate input prefix, not any ordinary/reference state.
    // The second native R8 run is anchored at the arithmetic tile boundary.
    length_ += prefix;
    state_row_ = static_cast<int32_t>(prefix - 1);
    std::vector<int32_t> remainder(kBoltVerifyRows, tokens.back());
    std::copy(tokens.begin() + prefix, tokens.end(), remainder.begin());
    const auto tail_predictions = VerifyBlock(remainder, want_predictions);
    if (want_predictions)
      std::copy_n(tail_predictions.begin(), kBoltVerifyRows - prefix, predictions.begin() + prefix);
    for (const auto& [name, backup] : verify_prefix_backup_) {
      preserve(name, 0, prefix, kBoltVerifyRows - prefix);
      tensors_.at(name)->GetByteSpan().CopyFrom(backup->GetByteSpan());
    }
    length_ = position;
    state_row_ = selected;
    return predictions;
  } catch (...) {
    try {
      Reset();
    } catch (...) {
    }
    throw;
  }
}

std::vector<int64_t> NemotronBoltState::VerifyBlock(std::span<const int32_t> tokens, bool want_predictions) {
  Require(tokens.size() == kBoltVerifyRows && length_ + kBoltVerifyRows <= bolt_.capacity_,
          "verify block exceeds fixed capacity");
  auto cpu_in = verify_inputs_span_.CpuSpan();
  auto* ids_out = reinterpret_cast<int64_t*>(cpu_in.data());
  for (size_t i = 0; i < kBoltVerifyRows; ++i) ids_out[i] = tokens[i];
  cpu_in[2 * kBoltVerifyRows] = static_cast<int32_t>(length_ + kBoltVerifyRows - 1);
  cpu_in[2 * kBoltVerifyRows + 4] = state_row_;
  verify_inputs_span_.CopyCpuToDevice();
  State::Run(*bolt_.session_, params_->use_graph_capture, 8, 0, nullptr);
  ++stats_.target_forward_passes;
  if (!want_predictions) return {};
  auto cpu = pred_ids_span_.CopyDeviceToCpu();
  return {cpu.begin(), cpu.end()};
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
    const size_t window = drafter_ ? drafter_->ContextWindow() : 0;
    const size_t draft_start =
        (window != 0 && tokens.size() > window)
            ? ((tokens.size() - window) / kBoltVerifyRows) * kBoltVerifyRows
            : 0;
    const size_t draft_rows = tokens.size() - draft_start;
    const bool single_chunk = draft_rows <= kBoltVerifyRows;
    const size_t aux_row_bytes = bolt_.aux_width_ * sizeof(uint16_t);
    for (size_t pos = 0; pos < tokens.size(); pos += kBoltVerifyRows) {
      const size_t valid = std::min(kBoltVerifyRows, tokens.size() - pos);
      const bool last_chunk = pos + kBoltVerifyRows >= tokens.size();
      std::vector<int32_t> block(kBoltVerifyRows, tokens[pos + valid - 1]);
      std::copy_n(tokens.begin() + pos, valid, block.begin());
      const auto predictions = Verify(block, last_chunk);
      if (drafter_ && !single_chunk && pos >= draft_start) {
        auto aux_src = aux_view_->GetByteSpan();
        auto prefill_dst = prefill_aux_->GetByteSpan();
        prefill_dst.subspan((pos - draft_start) * aux_row_bytes, valid * aux_row_bytes)
            .CopyFrom(aux_src.subspan(0, valid * aux_row_bytes));
      }
      if (last_chunk) {
        anchor_ = static_cast<int32_t>(predictions[valid - 1]);
      }
      length_ += valid;
      state_row_ = static_cast<int32_t>(valid - 1);
    }
    if (drafter_)
      Propose(draft_rows, draft_start, anchor_, single_chunk ? aux_view_.get() : prefill_aux_.get());
    if (tokens.size() < static_cast<size_t>(params_->search.max_length)) {
      pending_.push_back(anchor_);
      stats_.tokens_queued = 1;
    }
    stats_.effective_k = drafter_ ? 7 : 0;
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
  if (pending_.empty() && !drafter_) {
    // Only row zero is committed; later rows are causal scratch and are overwritten.
    const auto predictions = Verify(std::vector<int32_t>(kBoltVerifyRows, anchor_));
    anchor_ = static_cast<int32_t>(predictions[0]);
    ++length_;
    state_row_ = 0;
    pending_.push_back(anchor_);
    ++stats_.tokens_queued;
  }
  if (pending_.empty()) {
    std::vector<int32_t> block{anchor_};
    block.insert(block.end(), drafts_.begin(), drafts_.end());
    const auto target_passes = stats_.target_forward_passes;
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
    stats_.target_verify_forward_passes += stats_.target_forward_passes - target_passes;
    stats_.draft_tokens_proposed += 7;
    stats_.draft_tokens_evaluated += 7;
    stats_.draft_tokens_accepted += accepted;
    ++stats_.acceptance_length_histogram[accepted];
    stats_.tokens_queued += tokens.size();
    if (tokens.size() == accepted + 1) {
      accepted == 7 ? ++stats_.bonus_tokens : ++stats_.correction_tokens;
    }
    if (accepted == 0)
      ++stats_.zero_accept_rounds;
    else if (accepted == 7)
      ++stats_.full_accept_rounds;
    else
      ++stats_.partial_accept_rounds;
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
