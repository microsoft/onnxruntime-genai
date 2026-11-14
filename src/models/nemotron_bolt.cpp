// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt.h"
#include "engine/paged_key_value_cache.h"
#include "engine/request.h"
#include "search.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <set>

namespace Generators {
namespace {

constexpr int kCaptureWarmupRuns = 4;

std::string PresentCacheName(const std::string& past_name) {
  constexpr std::string_view kPagedPrefix = "past_key_values.";
  if (past_name.rfind(kPagedPrefix.data(), 0) == 0)
    return "present." + past_name.substr(kPagedPrefix.size());
  return "present_" + past_name.substr(5);
}

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
  Require(!params.config.model.decoder.sliding_window, "decoder sliding_window is unsupported");
  Require(params.speculative.max_draft_tokens == 0 || params.use_graph_capture,
          "CUDA graph capture must be enabled for speculative decoding");
}

size_t NemotronBoltPrefillChunkRows(std::optional<size_t> configured, size_t capacity) {
  const size_t requested = configured.value_or(0) != 0 ? *configured : kBoltDefaultPrefillChunkRows;
  const size_t limit = capacity > kBoltVerifyRows ? capacity - kBoltVerifyRows : 0;
  return std::max(kBoltVerifyRows, std::min(requested, limit));
}

size_t NemotronBoltAcceptedPrefix(std::span<const int32_t> drafts,
                                  std::span<const int64_t> predictions, size_t max_accepted) {
  Require(drafts.size() == kBoltVerifyRows - 1 && predictions.size() == kBoltVerifyRows &&
              max_accepted < kBoltVerifyRows,
          "verification requires seven drafts and eight predictions");
  size_t accepted = 0;
  while (accepted < max_accepted && drafts[accepted] == predictions[accepted])
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

size_t NemotronBoltVerifyUsableRows(size_t position, size_t capacity, int max_length) {
  Require(max_length > 0 && position < static_cast<size_t>(max_length) - 1,
          "verification requires a positive remaining output budget");
  Require(capacity >= kBoltVerifyRows && position <= capacity - kBoltVerifyRows,
          "verify block exceeds fixed capacity");
  const size_t prefix = NemotronBoltVerifyPrefixRows(position);
  if (prefix == kBoltVerifyRows || prefix <= capacity - position - kBoltVerifyRows)
    return kBoltVerifyRows;
  Require(static_cast<size_t>(max_length) <= position + 1 + prefix,
          "arithmetic boundary exceeds verification scratch capacity");
  return prefix;
}

NemotronBoltRoundStats GetNemotronBoltRoundStats(std::span<const int32_t> drafts,
                                                 std::span<const int32_t> emitted_tokens) {
  Require(drafts.size() == kBoltVerifyRows - 1 && !emitted_tokens.empty() &&
              emitted_tokens.size() <= kBoltVerifyRows,
          "round statistics require seven drafts and one to eight emitted tokens");
  NemotronBoltRoundStats stats;
  const size_t proposal_count = std::min(drafts.size(), emitted_tokens.size());
  for (size_t i = 0; i < proposal_count; ++i) {
    ++stats.evaluated;
    if (drafts[i] != emitted_tokens[i])
      break;
    ++stats.accepted;
  }
  stats.correction = stats.evaluated > stats.accepted;
  stats.bonus = stats.accepted == drafts.size() && emitted_tokens.size() > drafts.size();
  stats.full = stats.accepted == drafts.size();
  return stats;
}

std::vector<int32_t> NemotronBoltRoundTokens(std::span<const int32_t> drafts,
                                             std::span<const int64_t> predictions,
                                             size_t remaining, std::span<const int> eos,
                                             size_t usable_rows) {
  Require(usable_rows > 0 && usable_rows <= kBoltVerifyRows, "invalid usable verification rows");
  const size_t limit = std::min(remaining, usable_rows);
  // Reserve the last delivered row for its prediction/anchor, not an uncommitted draft.
  const size_t accepted = NemotronBoltAcceptedPrefix(drafts, predictions, limit ? limit - 1 : 0);
  std::vector<int32_t> result;
  for (size_t row = 0; row <= accepted && result.size() < limit; ++row) {
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
  const auto configured_batching = config_->engine.dynamic_batching;
  if (configured_batching) {
    Require(configured_batching->max_batch_size >= 1 &&
                configured_batching->max_batch_size <= kBoltMaxBatchSize,
            "native Bolt engine.dynamic_batching.max_batch_size must be in [1, 64]");
    Require(!configured_batching->prefix_caching.value_or(false) ||
                configured_batching->max_snapshots == 0,
            "prefix_caching is unsupported");
    engine_max_batch_size_ = configured_batching->max_batch_size;
    if (configured_batching->max_scheduled_tokens > 0) {
      engine_max_scheduled_tokens_ = configured_batching->max_scheduled_tokens;
    }
    config_->engine.dynamic_batching.reset();
  } else if (config_->engine.static_batching) {
    Require(config_->engine.static_batching->max_batch_size >= 1 &&
                config_->engine.static_batching->max_batch_size <= kBoltMaxBatchSize,
            "native Bolt engine.static_batching.max_batch_size must be in [1, 64]");
    engine_max_batch_size_ = config_->engine.static_batching->max_batch_size;
  }
  Require(!config_->model.draft && !config_->model.mtp.IsEnabled() &&
              !config_->model.decoder.sliding_window,
          "decoder draft, MTP, and decoder sliding_window are unsupported");
  config_->engine.static_batching = Config::Engine::StaticBatching{engine_max_batch_size_};

  session_ = CreateSession(env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_);
  const auto& info = session_info_;
  const auto ids_shape = info.GetInputShape("input_ids");
  const bool has_query_rows = info.HasInput("query_rows");
  const bool has_state_slots = info.HasInput("state_slots");
  has_logit_rows_ = info.HasInput("logit_rows");
  has_aux_rows_ = info.HasInput("aux_rows");
  paged_attention_ = info.HasInput("block_table") && info.HasInput("cumulative_sequence_lengths") &&
                     info.HasInput("past_sequence_lengths") && info.HasInput("attention_metadata");
  single_token_decode_ = ids_shape == std::vector<int64_t>({1, -1}) && has_query_rows;
  dynamic_prefill_ = ids_shape == std::vector<int64_t>({1, -1}) && !has_query_rows;
  Require(ids_shape == std::vector<int64_t>({1, 8}) || dynamic_prefill_ ||
              ((ids_shape == std::vector<int64_t>({1, 1}) || single_token_decode_) &&
               config_->speculative.max_draft_tokens == 0),
          "input_ids must be [1,8], dynamic [1,R], or an opt-in target-only row contract");
  target_rows_ = (single_token_decode_ || dynamic_prefill_) ? kBoltVerifyRows : static_cast<size_t>(ids_shape[1]);
  const auto rows = ids_shape[1];
  if (single_token_decode_)
    RequireTensor(info, "query_rows", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
  if (dynamic_prefill_ && has_logit_rows_)
    RequireTensor(info, "logit_rows", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
  if (dynamic_prefill_ && has_aux_rows_)
    RequireTensor(info, "aux_rows", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
  RequireTensor(info, "input_ids", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, rows});
  Require(info.HasInput("state_row") &&
              info.GetInputDataType("state_row") == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
          "missing or invalid state_row input");
  const auto state_row_shape = info.GetInputShape("state_row");
  Require(state_row_shape == std::vector<int64_t>({1}) ||
              (has_state_slots && state_row_shape == std::vector<int64_t>({-1})),
          "incompatible tensor state_row");
  if (has_state_slots) {
    Require(paged_attention_ &&
                info.GetInputDataType("state_slots") == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
                info.GetInputShape("state_slots") == state_row_shape,
            "state_slots requires PagedAttention and matching int32 batch geometry with state_row");
  }
  if (paged_attention_) {
    RequireTensor(info, "cumulative_sequence_lengths", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
    RequireTensor(info, "past_sequence_lengths", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
    RequireTensor(info, "block_table", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, -1});
    RequireTensor(info, "attention_metadata", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {3});
  } else {
    Require(!configured_batching || !configured_batching->num_blocks,
            "num_blocks requires a PagedAttention Bolt target graph");
    RequireTensor(info, "seqlens_k", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
    RequireTensor(info, "total_sequence_length", true, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {});
  }
  RequireTensor(info, "pred_ids", false, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, rows});
  RequireTensor(info, "logits", false, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                {1, rows, config_->model.vocab_size});
  const auto aux_shape = info.GetOutputShape("aux_hidden_states");
  Require(aux_shape.size() == 3 && aux_shape[0] == 1 && aux_shape[1] == rows && aux_shape[2] > 0 &&
              info.GetOutputDataType("aux_hidden_states") == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
          "expected bf16 aux_hidden_states[1,target_rows,width]");
  aux_width_ = static_cast<size_t>(aux_shape[2]);
  std::set<std::string> outputs{"logits", "pred_ids", "aux_hidden_states"};
  size_t conv_count = 0, ssm_count = 0, key_count = 0, value_count = 0;
  size_t recurrent_bank_bytes = 0;
  std::optional<int64_t> recurrent_bank_dim;
  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids" || name == "seqlens_k" || name == "state_row" || name == "state_slots" ||
        name == "total_sequence_length" || name == "cumulative_sequence_lengths" ||
        name == "past_sequence_lengths" || name == "block_table" || name == "attention_metadata" ||
        (single_token_decode_ && name == "query_rows") ||
        (dynamic_prefill_ && (name == "logit_rows" || name == "aux_rows")))
      continue;
    const auto shape = info.GetInputShape(name);
    const auto type = info.GetInputDataType(name);
    if (!paged_attention_ && (name.rfind("past_key.", 0) == 0 || name.rfind("past_value.", 0) == 0)) {
      Require(shape.size() == 4 && shape[0] == 1 && shape[1] == 2 &&
                  shape[2] > 8 && shape[3] == 128 && type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
              "expected dense bf16 KV[1,2,capacity,128]: " + name);
      if (capacity_ == 0) capacity_ = static_cast<size_t>(shape[2]);
      Require(capacity_ == static_cast<size_t>(shape[2]), "KV capacities differ");
      name.rfind("past_key.", 0) == 0 ? ++key_count : ++value_count;
    } else if (paged_attention_ && name.rfind("past_key_values.", 0) == 0 &&
               (name.size() > 4 && (name.substr(name.size() - 4) == ".key" ||
                                    (name.size() > 6 && name.substr(name.size() - 6) == ".value")))) {
      Require(shape.size() == 4 && shape[0] < 0 && shape[1] == static_cast<int64_t>(kBoltPagedBlockSize) &&
                  shape[2] == 2 && shape[3] == 128 && type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
              "expected paged bf16 KV[num_blocks,256,2,128]: " + name);
      name.substr(name.size() - 4) == ".key" ? ++key_count : ++value_count;
    } else if (name.rfind("past_conv.", 0) == 0) {
      if (has_state_slots && shape.size() == 4) {
        Require((shape[0] < 0 || shape[0] >= static_cast<int64_t>(engine_max_batch_size_)) &&
                    shape[1] == 8 && shape[2] == 10240 && shape[3] == 3 &&
                    type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                "invalid banked recurrent conv " + name);
        if (!recurrent_bank_dim) recurrent_bank_dim = shape[0];
        Require(*recurrent_bank_dim == shape[0], "recurrent bank dimensions differ");
        batched_recurrent_ = true;
      } else {
        Require(!has_state_slots && shape == std::vector<int64_t>({8, 10240, 3}) &&
                    type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                "invalid recurrent conv " + name);
      }
      recurrent_bank_bytes += 8ULL * 10240 * 3 * sizeof(uint16_t);
      ++conv_count;
    } else if (name.rfind("past_ssm.", 0) == 0) {
      if (has_state_slots && shape.size() == 5) {
        Require((shape[0] < 0 || shape[0] >= static_cast<int64_t>(engine_max_batch_size_)) &&
                    shape[1] == 8 && shape[2] == 128 && shape[3] == 128 && shape[4] == 64 &&
                    type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "invalid banked recurrent SSM " + name);
        if (!recurrent_bank_dim) recurrent_bank_dim = shape[0];
        Require(*recurrent_bank_dim == shape[0], "recurrent bank dimensions differ");
        batched_recurrent_ = true;
      } else {
        Require(!has_state_slots && shape == std::vector<int64_t>({8, 128, 128, 64}) &&
                    type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "invalid recurrent SSM " + name);
      }
      recurrent_bank_bytes += 8ULL * 128 * 128 * 64 * sizeof(float);
      ++ssm_count;
    } else {
      throw std::runtime_error("nemotron_bolt: unsupported input " + name);
    }
    const auto present = PresentCacheName(name);
    RequireTensor(info, present, false, type, shape);
    outputs.insert(present);
  }
  Require(conv_count > 0 && conv_count == ssm_count && key_count == 8 && value_count == 8 &&
              batched_recurrent_ == has_state_slots,
          "expected matching recurrent states and eight attention layers");
  Require(session_->GetOutputNames().size() == outputs.size(), "unexpected target outputs");
  if (engine_max_batch_size_ > 1) {
    Require(paged_attention_ && batched_recurrent_ && dynamic_prefill_ && has_logit_rows_ && has_aux_rows_ &&
                state_row_shape == std::vector<int64_t>({-1}),
            "max_batch_size > 1 requires a PagedAttention dynamic-prefill Bolt target with banked recurrent state");
  }

  if (paged_attention_) {
    paged_block_size_ = configured_batching ? configured_batching->block_size : kBoltPagedBlockSize;
    Require(paged_block_size_ == kBoltPagedBlockSize, "native Bolt PagedAttention requires block_size=256");
    constexpr size_t kPrimaryBytesPerBlock = 8ULL * 2 * kBoltPagedBlockSize * 2 * 128 * sizeof(uint16_t);
    size_t drafter_reserved_bytes = 0;
    if (config_->speculative.max_draft_tokens != 0 && !config_->model.dflash2.filename.empty()) {
      const size_t df_blocks =
          Dflash2Drafter::PoolBlocks(*config_, kBoltPagedBlockSize, engine_max_batch_size_);
      if (df_blocks != 0) {
        drafter_reserved_bytes = Dflash2Drafter::PoolBytes(
            *config_, kBoltPagedBlockSize, df_blocks, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      }
      drafter_reserved_bytes += Dflash2Drafter::EmbeddingReservedBytes(
          engine_max_batch_size_, static_cast<size_t>(std::max(config_->model.dflash2.block_size, 8)),
          static_cast<size_t>(std::max(config_->model.decoder.hidden_size, 4096)),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    }
    if (configured_batching && configured_batching->num_blocks.has_value()) {
      paged_num_blocks_ = ResolveConfiguredPagedBlockCount(
          *configured_batching->num_blocks, kPrimaryBytesPerBlock, 0, drafter_reserved_bytes);
    } else if (configured_batching && (configured_batching->gpu_utilization_factor.has_value() ||
                                       configured_batching->device_memory_budget_bytes.has_value())) {
      size_t free_bytes = 0, total_bytes = 0;
      p_device_kvcache_->GetAvailableMemory(free_bytes, total_bytes);
      const size_t state_reserve =
          engine_max_batch_size_ * recurrent_bank_bytes + drafter_reserved_bytes + (256ULL << 20);
      if (configured_batching->device_memory_budget_bytes.has_value()) {
        Require(device_memory_budget_initial_free_bytes_.has_value(),
                "device_memory_budget_bytes must be configured before loading the model");
        const size_t avail = DeviceMemoryBudgetAvailableBytes(
            *configured_batching->device_memory_budget_bytes,
            *device_memory_budget_initial_free_bytes_, free_bytes,
            configured_batching->device_memory_headroom_bytes);
        paged_num_blocks_ = ComputePagedBlockCapacityFromBudget(avail, state_reserve, kPrimaryBytesPerBlock, 0);
      } else {
        paged_num_blocks_ = ComputePagedBlockCapacityFromBytes(
            free_bytes, *configured_batching->gpu_utilization_factor, state_reserve, kPrimaryBytesPerBlock, 0);
      }
    } else {
      Require(config_->model.context_length > 0, "context_length must be positive");
      paged_num_blocks_ =
          (static_cast<size_t>(config_->model.context_length) + kBoltVerifyRows + paged_block_size_ - 1) /
          paged_block_size_;
    }
    Require(paged_num_blocks_ >= engine_max_batch_size_,
            "PagedAttention requires at least max_batch_size KV blocks");
    capacity_ = paged_num_blocks_ * paged_block_size_;
    Require(capacity_ > kBoltVerifyRows && capacity_ <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "paged KV capacity out of int32 range");
    const size_t max_ctx = capacity_ - kBoltVerifyRows;
    if (config_->model.context_length <= 0 || static_cast<size_t>(config_->model.context_length) > max_ctx)
      config_->model.context_length = static_cast<int>(max_ctx);
    if (config_->search.max_length <= 0 || config_->search.max_length > config_->model.context_length)
      config_->search.max_length = config_->model.context_length;
  } else {
    Require(config_->model.context_length > 0 &&
                capacity_ <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
                static_cast<size_t>(config_->model.context_length) <= capacity_ - kBoltVerifyRows,
            "context_length must leave eight scratch slots in the fixed KV capacity");
  }
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
  Require(dynamic_prefill_ || params.search.chunk_size.value_or(0) <= kBoltVerifyRows,
          "search.chunk_size > 8 requires a dynamic-prefill target (input_ids [1, target_rows])");
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
  std::lock_guard lock{model.execution_mutex_};
  if (model.dynamic_prefill_) {
    prefill_chunk_rows_ = NemotronBoltPrefillChunkRows(params.search.chunk_size, model.capacity_);
  }
  const auto& info = model.session_info_;
  verify_inputs_span_ = model.p_device_->Allocate<int32_t>(2 * kBoltVerifyRows + 8);
  verify_inputs_span_.CpuSpan();
  int32_t* verify_in_base = verify_inputs_span_.Span().data();
  const auto& dev_mem_info = model.p_device_->GetAllocator().GetInfo();
  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids" || name == "seqlens_k" || name == "past_sequence_lengths" ||
        name == "cumulative_sequence_lengths" || name == "state_row") {
      void* ptr = name == "input_ids"
                      ? static_cast<void*>(verify_in_base)
                  : (name == "seqlens_k" || name == "past_sequence_lengths")
                      ? static_cast<void*>(verify_in_base + 2 * kBoltVerifyRows)
                  : name == "cumulative_sequence_lengths"
                      ? static_cast<void*>(verify_in_base + 2 * kBoltVerifyRows + 2)
                      : static_cast<void*>(verify_in_base + 2 * kBoltVerifyRows + 4);
      const size_t bytes = name == "input_ids"
                               ? kBoltVerifyRows * sizeof(int64_t)
                           : name == "cumulative_sequence_lengths"
                               ? 2 * sizeof(int32_t)
                               : sizeof(int32_t);
      auto shape = info.GetInputShape(name);
      if ((model.single_token_decode_ || model.dynamic_prefill_) && name == "input_ids")
        shape[1] = kBoltVerifyRows;
      if (name == "past_sequence_lengths")
        shape = {1};
      if (name == "cumulative_sequence_lengths")
        shape = {2};
      if (name == "state_row")
        shape = {1};
      auto tensor = std::make_unique<Tensor>(
          OrtValue::CreateTensor(dev_mem_info, ptr, bytes, shape, info.GetInputDataType(name)));
      tensor->p_device_ = model.p_device_;
      input_storage_.push_back(name);
      inputs_.push_back(tensor->GetOrtTensor());
      tensors_.emplace(name, std::move(tensor));
      continue;
    }
    auto* device = (name == "total_sequence_length" || name == "attention_metadata")
                       ? GetDeviceInterface(DeviceType::CPU)
                       : model.p_device_;
    auto shape = info.GetInputShape(name);
    if ((model.single_token_decode_ && name == "query_rows") ||
        (model.dynamic_prefill_ && (name == "logit_rows" || name == "aux_rows")))
      shape[0] = kBoltVerifyRows;
    if (name == "state_slots")
      shape = {1};
    if (name == "block_table")
      shape = {1, static_cast<int64_t>(model.paged_num_blocks_)};
    if (model.paged_attention_ && name.rfind("past_key_values.", 0) == 0)
      shape[0] = static_cast<int64_t>(model.paged_num_blocks_);
    if (model.batched_recurrent_ && (name.rfind("past_conv.", 0) == 0 || name.rfind("past_ssm.", 0) == 0) &&
        shape[0] < 0)
      shape[0] = 1;
    auto& tensor = AddTensor(name, true, device, info.GetInputDataType(name), shape);
    if (name == "query_rows" || name == "logit_rows" || name == "aux_rows") {
      const std::array<int64_t, 8> indices{0, 1, 2, 3, 4, 5, 6, 7};
      Upload(tensor, std::span<const int64_t>{indices});
    } else if (name == "state_slots") {
      const std::array<int32_t, 1> slot_zero{0};
      Upload(tensor, std::span<const int32_t>{slot_zero});
    } else if (name == "block_table") {
      std::vector<int32_t> table(model.paged_num_blocks_);
      std::iota(table.begin(), table.end(), 0);
      Upload(tensor, std::span<const int32_t>{table});
    } else if (name == "attention_metadata") {
      const std::array<int32_t, 3> meta{
          static_cast<int32_t>(model.target_rows_), static_cast<int32_t>(model.capacity_), 1};
      Upload(tensor, std::span<const int32_t>{meta});
    }
    if (name.rfind("past_", 0) == 0) {
      caches_.push_back(&tensor);
      if (name.rfind("past_key.", 0) == 0 || name.rfind("past_value.", 0) == 0 ||
          name.rfind("past_key_values.", 0) == 0)
        kv_caches_.push_back(&tensor);
      else
        recurrent_caches_.push_back(&tensor);
      output_storage_.push_back(PresentCacheName(name));
      outputs_.push_back(tensor.GetOrtTensor());
    }
  }
  for (const auto* name : {"logits", "pred_ids", "aux_hidden_states"}) {
    auto shape = info.GetOutputShape(name);
    if (model.single_token_decode_ || model.dynamic_prefill_)
      shape[1] = kBoltVerifyRows;
    AddTensor(name, false, model.p_device_, info.GetOutputDataType(name), shape);
  }
  for (size_t i = 0; i < input_storage_.size(); ++i) {
    if (input_storage_[i] == "input_ids") input_idx_input_ids_ = i;
    if (input_storage_[i] == "logit_rows") input_idx_logit_rows_ = i;
    if (input_storage_[i] == "aux_rows") input_idx_aux_rows_ = i;
    if (input_storage_[i] == "attention_metadata") input_idx_attn_meta_ = i;
  }
  for (size_t i = 0; i < output_storage_.size(); ++i) {
    if (output_storage_[i] == "logits") output_idx_logits_ = i;
    if (output_storage_[i] == "pred_ids") output_idx_pred_ids_ = i;
    if (output_storage_[i] == "aux_hidden_states") output_idx_aux_ = i;
  }
  if (model.single_token_decode_) {
    for (const auto* name : {"input_ids", "query_rows", "logits", "pred_ids", "aux_hidden_states"}) {
      auto& full = *tensors_.at(name);
      auto shape = full.GetShape();
      shape[std::string_view{name} == "query_rows" ? 0 : 1] = 1;
      auto view = std::make_unique<Tensor>(OrtValue::CreateTensor(
          dev_mem_info, full.GetMutableRawData(), full.GetByteSpan().size(), shape, full.GetType()));
      view->p_device_ = model.p_device_;
      single_token_tensors_.emplace(name, std::move(view));
    }
    if (model.paged_attention_) {
      const std::array<int64_t, 1> meta_shape{3};
      auto meta_one = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU),
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      meta_one->CreateTensor(meta_shape, true);
      const std::array<int32_t, 3> meta{1, static_cast<int32_t>(model.capacity_), 1};
      Upload(*meta_one, std::span<const int32_t>{meta});
      single_token_tensors_.emplace("attention_metadata", std::move(meta_one));
    }
    for (size_t i = 0; i < input_storage_.size(); ++i)
      if (single_token_tensors_.find(input_storage_[i]) != single_token_tensors_.end())
        single_token_input_indices_.push_back(i);
    for (size_t i = 0; i < output_storage_.size(); ++i)
      if (single_token_tensors_.find(output_storage_[i]) != single_token_tensors_.end())
        single_token_output_indices_.push_back(i);
  }
  pred_ids_span_ = tensors_.at("pred_ids")->GetDeviceSpan<int64_t>();
  pred_ids_span_.CpuSpan();
  for (auto& name : input_storage_) input_names_.push_back(name.c_str());
  for (auto& name : output_storage_) output_names_.push_back(name.c_str());
  if (!model.paged_attention_) {
    const int32_t capacity = static_cast<int32_t>(model.capacity_);
    Upload(*tensors_.at("total_sequence_length"), std::span<const int32_t>{&capacity, 1});
  }
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
  if (model.dynamic_prefill_ && prefill_chunk_rows_ > kBoltVerifyRows) {
    prefill_ids_span_ = model.p_device_->Allocate<int64_t>(prefill_chunk_rows_);
    prefill_ids_span_.CpuSpan();
    if (model.has_logit_rows_) {
      prefill_logit_rows_span_ = model.p_device_->Allocate<int64_t>(1);
      prefill_logit_rows_span_.CpuSpan();
      const std::array<int64_t, 3> one_logit_shape{1, 1, model.config_->model.vocab_size};
      const std::array<int64_t, 2> one_pred_shape{1, 1};
      prefill_logits_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      prefill_logits_->CreateTensor(one_logit_shape, true);
      prefill_pred_ids_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      prefill_pred_ids_->CreateTensor(one_pred_shape, true);
      prefill_pred_span_ = prefill_pred_ids_->GetDeviceSpan<int64_t>();
      prefill_pred_span_.CpuSpan();
    }
    if (model.has_aux_rows_) {
      const size_t max_aux_rows =
          prefill_aux_ ? static_cast<size_t>(prefill_aux_->GetShape()[0]) : size_t{1};
      prefill_aux_rows_span_ = model.p_device_->Allocate<int64_t>(std::max<size_t>(max_aux_rows, 1));
      prefill_aux_rows_span_.CpuSpan();
    }
  }
  Reset();
  std::vector<int32_t> dummy(model.target_rows_, 1);
  // The legacy two-run capture delay needs four calls through the first replay.
  // Newer ORT can warm up and capture internally within a single Run().
  // Use disposable state: warmup may re-execute in-place aliases before Reset().
  if (params.use_graph_capture) {
    for (int i = 0; i < kCaptureWarmupRuns; ++i)
      Verify(dummy, i == kCaptureWarmupRuns - 1);
    if (model.single_token_decode_) {
      const std::vector<int32_t> one_token(1, 1);
      for (int i = 0; i < kCaptureWarmupRuns; ++i)
        Verify(one_token, i == kCaptureWarmupRuns - 1);
    }
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

bool NemotronBoltState::IsEosToken(int32_t token) const {
  const auto& eos = model_.config_->model.eos_token_id;
  return std::find(eos.begin(), eos.end(), token) != eos.end();
}

void NemotronBoltState::Reset() {
  if (bolt_.paged_attention_) {
    for (auto* cache : recurrent_caches_) cache->GetByteSpan().Zero();
    caches_zeroed_ = true;
  } else if (!caches_zeroed_) {
    for (auto* cache : caches_) cache->GetByteSpan().Zero();
    caches_zeroed_ = true;
  } else {
    for (auto* cache : recurrent_caches_) cache->GetByteSpan().Zero();
    if (touched_kv_rows_ > 0) {
      const size_t rows_to_zero = std::min(touched_kv_rows_, bolt_.capacity_);
      for (auto* kv : kv_caches_) {
        const auto shape = kv->GetShape();
        auto span = kv->GetByteSpan();
        const size_t head_bytes = span.size() / static_cast<size_t>(shape[1]);
        const size_t clear_bytes = rows_to_zero * static_cast<size_t>(shape[3]) * sizeof(uint16_t);
        for (size_t head = 0; head < static_cast<size_t>(shape[1]); ++head) {
          bolt_.p_device_->WrapMemory(span.subspan(head * head_bytes, clear_bytes).Span()).Zero();
        }
      }
    }
  }
  touched_kv_rows_ = 0;
  model_.p_device_->Synchronize();
  if (drafter_) drafter_->ReleaseAll();
  length_ = 0;
  verified_rows_ = 0;
  state_row_ = 0;
  anchor_ = 0;
  drafts_.clear();
  pending_.clear();
  has_drafts_ = false;
  prefilled_ = false;
  stats_ = {};
}

void NemotronBoltState::RewindTo(size_t length) {
  std::lock_guard lock{bolt_.execution_mutex_};
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

std::unique_ptr<Tensor> NemotronBoltState::SaveVerifyPrefix(Tensor& tensor, size_t prefix_rows) {
  const std::array<int64_t, 2> shape{
      static_cast<int64_t>(prefix_rows),
      static_cast<int64_t>(tensor.GetElementCount() / kBoltVerifyRows)};
  auto prefix = std::make_unique<Tensor>(tensor.p_device_, tensor.GetType());
  prefix->CreateTensor(shape, true);
  auto saved = prefix->GetByteSpan();
  saved.CopyFrom(tensor.GetByteSpan().subspan(0, saved.size()));
  return prefix;
}

void NemotronBoltState::RestoreVerifyPrefix(Tensor& tensor, Tensor& prefix) {
  const size_t prefix_rows = static_cast<size_t>(prefix.GetShape()[0]);
  auto target = tensor.GetByteSpan();
  const size_t row_bytes = target.size() / kBoltVerifyRows;
  // Move backward one disjoint row at a time: CUDA copies do not support overlapping ranges.
  for (size_t row = kBoltVerifyRows - prefix_rows; row > 0; --row) {
    target.subspan((prefix_rows + row - 1) * row_bytes, row_bytes)
        .CopyFrom(target.subspan((row - 1) * row_bytes, row_bytes));
  }
  target.subspan(0, prefix_rows * row_bytes).CopyFrom(prefix.GetByteSpan());
}

std::vector<int64_t> NemotronBoltState::Verify(std::span<const int32_t> tokens, bool want_predictions) {
  verified_rows_ = kBoltVerifyRows;
  if (bolt_.paged_attention_)
    return VerifyBlock(tokens, want_predictions);
  const size_t prefix = NemotronBoltVerifyPrefixRows(length_);
  if (!prefilled_ || !drafter_ || prefix == kBoltVerifyRows)
    return VerifyBlock(tokens, want_predictions);
  verified_rows_ = NemotronBoltVerifyUsableRows(length_, bolt_.capacity_, params_->search.max_length);
  if (verified_rows_ < kBoltVerifyRows)
    return VerifyBlock(tokens, want_predictions);
  const size_t position = length_;
  const int32_t selected = state_row_;
  // Keep compact prefixes alive through stream completion, including failure/reset cleanup.
  std::vector<std::pair<Tensor*, std::unique_ptr<Tensor>>> prefixes;
  try {
    prefixes.reserve(tensors_.size());
    VerifyBlock(tokens, false);
    for (const auto& [name, tensor] : tensors_) {
      if (name.rfind("past_conv.", 0) == 0 || name.rfind("past_ssm.", 0) == 0 ||
          name == "logits" || name == "aux_hidden_states" || name == "pred_ids")
        prefixes.emplace_back(tensor.get(), SaveVerifyPrefix(*tensor, prefix));
    }
    // Reuse only this verified candidate input prefix, not any ordinary/reference state.
    // The second native R8 run is anchored at the arithmetic tile boundary.
    length_ += prefix;
    state_row_ = static_cast<int32_t>(prefix - 1);
    std::vector<int32_t> remainder(kBoltVerifyRows, tokens.back());
    std::copy(tokens.begin() + prefix, tokens.end(), remainder.begin());
    VerifyBlock(remainder, false);
    for (const auto& [tensor, saved] : prefixes)
      RestoreVerifyPrefix(*tensor, *saved);
    std::vector<int64_t> predictions;
    if (want_predictions) {
      auto cpu = pred_ids_span_.CopyDeviceToCpu();
      predictions.assign(cpu.begin(), cpu.end());
    } else {
      // Even discarded predictions must finish using the temporary prefix buffers before release.
      model_.p_device_->Synchronize();
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
  const auto rows = tokens.size();
  Require((rows == bolt_.target_rows_ || (bolt_.single_token_decode_ && rows == 1)) &&
              bolt_.capacity_ >= kBoltVerifyRows && length_ <= bolt_.capacity_ - kBoltVerifyRows,
          "verify block exceeds fixed capacity");
  if (bolt_.single_token_decode_) {
    const auto& selected = rows == 1 ? single_token_tensors_ : tensors_;
    for (const auto i : single_token_input_indices_)
      inputs_[i] = selected.at(input_storage_[i])->GetOrtTensor();
    for (const auto i : single_token_output_indices_)
      outputs_[i] = selected.at(output_storage_[i])->GetOrtTensor();
  }
  auto cpu_in = verify_inputs_span_.CpuSpan();
  auto* ids_out = reinterpret_cast<int64_t*>(cpu_in.data());
  for (size_t i = 0; i < rows; ++i) ids_out[i] = tokens[i];
  if (bolt_.paged_attention_) {
    cpu_in[2 * kBoltVerifyRows] = static_cast<int32_t>(length_);
    cpu_in[2 * kBoltVerifyRows + 2] = 0;
    cpu_in[2 * kBoltVerifyRows + 3] = static_cast<int32_t>(rows);
  } else {
    // The R1 package pads only attention to R8, preserving its position and tile arithmetic.
    cpu_in[2 * kBoltVerifyRows] = static_cast<int32_t>(length_ + kBoltVerifyRows - 1);
  }
  cpu_in[2 * kBoltVerifyRows + 4] = state_row_;
  verify_inputs_span_.CopyCpuToDevice();
  State::Run(*bolt_.session_, params_->use_graph_capture,
             static_cast<int>(rows), 0, nullptr);
  touched_kv_rows_ = std::max(touched_kv_rows_, length_ + kBoltVerifyRows);
  ++stats_.target_forward_passes;
  if (!want_predictions) return {};
  auto cpu = pred_ids_span_.CopyDeviceToCpu();
  return {cpu.begin(), cpu.begin() + rows};
}

int64_t NemotronBoltState::PrefillChunk(
    std::span<const int32_t> tokens, size_t pos, size_t draft_start, bool want_prediction) {
  const size_t rows = tokens.size();
  Require(bolt_.dynamic_prefill_ && rows > kBoltVerifyRows && length_ + rows <= bolt_.capacity_,
          "prefill chunk exceeds capacity or requires dynamic prefill contract");
  const auto& dev_mem_info = bolt_.p_device_->GetAllocator().GetInfo();
  const size_t aux_row_bytes = bolt_.aux_width_ * sizeof(uint16_t);

  OrtValue* orig_ids = inputs_[input_idx_input_ids_];
  OrtValue* orig_logit_rows = bolt_.has_logit_rows_ ? inputs_[input_idx_logit_rows_] : nullptr;
  OrtValue* orig_aux_rows = bolt_.has_aux_rows_ ? inputs_[input_idx_aux_rows_] : nullptr;
  OrtValue* orig_attn_meta = bolt_.paged_attention_ ? inputs_[input_idx_attn_meta_] : nullptr;
  OrtValue* orig_logits = outputs_[output_idx_logits_];
  OrtValue* orig_preds = outputs_[output_idx_pred_ids_];
  OrtValue* orig_aux = outputs_[output_idx_aux_];
  struct RestoreBindings {
    NemotronBoltState& s;
    OrtValue *ids, *lrows, *arows, *meta, *logits, *preds, *aux;
    ~RestoreBindings() {
      s.inputs_[s.input_idx_input_ids_] = ids;
      if (s.bolt_.has_logit_rows_) s.inputs_[s.input_idx_logit_rows_] = lrows;
      if (s.bolt_.has_aux_rows_) s.inputs_[s.input_idx_aux_rows_] = arows;
      if (s.bolt_.paged_attention_) s.inputs_[s.input_idx_attn_meta_] = meta;
      s.outputs_[s.output_idx_logits_] = logits;
      s.outputs_[s.output_idx_pred_ids_] = preds;
      s.outputs_[s.output_idx_aux_] = aux;
    }
  } restore{*this, orig_ids, orig_logit_rows, orig_aux_rows, orig_attn_meta, orig_logits, orig_preds, orig_aux};

  auto ids_sub = prefill_ids_span_.subspan(0, rows);
  auto ids_cpu = ids_sub.CpuSpan();
  for (size_t i = 0; i < rows; ++i) ids_cpu[i] = tokens[i];
  ids_sub.CopyCpuRangeToDevice();
  const std::array<int64_t, 2> ids_shape{1, static_cast<int64_t>(rows)};
  auto ids_ort = OrtValue::CreateTensor(
      dev_mem_info, ids_sub.Span().data(), rows * sizeof(int64_t), ids_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  inputs_[input_idx_input_ids_] = ids_ort.get();

  auto cpu_in = verify_inputs_span_.CpuSpan();
  if (bolt_.paged_attention_) {
    cpu_in[2 * kBoltVerifyRows] = static_cast<int32_t>(length_);
    cpu_in[2 * kBoltVerifyRows + 2] = 0;
    cpu_in[2 * kBoltVerifyRows + 3] = static_cast<int32_t>(rows);
  } else {
    cpu_in[2 * kBoltVerifyRows] = static_cast<int32_t>(length_ + rows - 1);
  }
  cpu_in[2 * kBoltVerifyRows + 4] = state_row_;
  verify_inputs_span_.subspan(2 * kBoltVerifyRows, 8).CopyCpuRangeToDevice();

  std::array<int32_t, 3> prefill_meta_host{
      static_cast<int32_t>(rows), static_cast<int32_t>(bolt_.capacity_), 1};
  std::unique_ptr<OrtValue> prefill_meta_ort;
  if (bolt_.paged_attention_) {
    const auto& cpu_mem_info = GetDeviceInterface(DeviceType::CPU)->GetAllocator().GetInfo();
    const std::array<int64_t, 1> meta_shape{3};
    prefill_meta_ort = OrtValue::CreateTensor(
        cpu_mem_info, prefill_meta_host.data(), sizeof(prefill_meta_host), meta_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    inputs_[input_idx_attn_meta_] = prefill_meta_ort.get();
  }

  std::unique_ptr<OrtValue> logit_rows_ort, logits_out_ort, aux_rows_ort, aux_out_ort;
  std::unique_ptr<Tensor> temp_logits, temp_preds, temp_aux;
  const size_t logit_row_bytes =
      static_cast<size_t>(bolt_.config_->model.vocab_size) * sizeof(uint16_t);
  if (bolt_.has_logit_rows_) {
    auto lr_cpu = prefill_logit_rows_span_.CpuSpan();
    lr_cpu[0] = static_cast<int64_t>(rows - 1);
    prefill_logit_rows_span_.CopyCpuRangeToDevice();
    const std::array<int64_t, 1> lr_shape{1};
    logit_rows_ort = OrtValue::CreateTensor(
        dev_mem_info, prefill_logit_rows_span_.Span().data(), sizeof(int64_t), lr_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    inputs_[input_idx_logit_rows_] = logit_rows_ort.get();
    if (want_prediction) {
      const std::array<int64_t, 3> one_logit_shape{1, 1, bolt_.config_->model.vocab_size};
      uint8_t* logits_dst =
          reinterpret_cast<uint8_t*>(tensors_.at("logits")->GetMutableRawData()) +
          (kBoltVerifyRows - 1) * logit_row_bytes;
      logits_out_ort = OrtValue::CreateTensor(
          dev_mem_info, logits_dst, logit_row_bytes, one_logit_shape,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      outputs_[output_idx_logits_] = logits_out_ort.get();
    } else {
      outputs_[output_idx_logits_] = prefill_logits_->GetOrtTensor();
    }
    outputs_[output_idx_pred_ids_] = prefill_pred_ids_->GetOrtTensor();
  } else {
    const std::array<int64_t, 3> logits_shape{
        1, static_cast<int64_t>(rows), bolt_.config_->model.vocab_size};
    const std::array<int64_t, 2> preds_shape{1, static_cast<int64_t>(rows)};
    temp_logits = std::make_unique<Tensor>(bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    temp_logits->CreateTensor(logits_shape, true);
    temp_preds = std::make_unique<Tensor>(bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    temp_preds->CreateTensor(preds_shape, true);
    outputs_[output_idx_logits_] = temp_logits->GetOrtTensor();
    outputs_[output_idx_pred_ids_] = temp_preds->GetOrtTensor();
  }

  const bool need_aux = drafter_ != nullptr && (pos + rows > draft_start);
  const size_t overlap_start = need_aux ? std::max(pos, draft_start) : pos;
  const size_t aux_begin_in_chunk = overlap_start - pos;
  const size_t aux_count = need_aux ? (rows - aux_begin_in_chunk) : 1;
  if (bolt_.has_aux_rows_) {
    auto ar_sub = prefill_aux_rows_span_.subspan(0, aux_count);
    auto ar_cpu = ar_sub.CpuSpan();
    for (size_t i = 0; i < aux_count; ++i) {
      ar_cpu[i] = need_aux ? static_cast<int64_t>(aux_begin_in_chunk + i) : int64_t{0};
    }
    ar_sub.CopyCpuRangeToDevice();
    const std::array<int64_t, 1> ar_shape{static_cast<int64_t>(aux_count)};
    aux_rows_ort = OrtValue::CreateTensor(
        dev_mem_info, ar_sub.Span().data(), aux_count * sizeof(int64_t), ar_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    inputs_[input_idx_aux_rows_] = aux_rows_ort.get();
    const std::array<int64_t, 3> aux_out_shape{
        1, static_cast<int64_t>(aux_count), static_cast<int64_t>(bolt_.aux_width_)};
    uint8_t* aux_dst_ptr =
        need_aux ? (reinterpret_cast<uint8_t*>(prefill_aux_->GetMutableRawData()) +
                    (overlap_start - draft_start) * aux_row_bytes)
                 : reinterpret_cast<uint8_t*>(tensors_.at("aux_hidden_states")->GetMutableRawData());
    aux_out_ort = OrtValue::CreateTensor(
        dev_mem_info, aux_dst_ptr, aux_count * aux_row_bytes, aux_out_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    outputs_[output_idx_aux_] = aux_out_ort.get();
  } else {
    const std::array<int64_t, 3> full_aux_shape{
        1, static_cast<int64_t>(rows), static_cast<int64_t>(bolt_.aux_width_)};
    temp_aux = std::make_unique<Tensor>(bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    temp_aux->CreateTensor(full_aux_shape, true);
    outputs_[output_idx_aux_] = temp_aux->GetOrtTensor();
  }

  State::Run(*bolt_.session_, false, 0, 0, nullptr);
  touched_kv_rows_ = std::max(touched_kv_rows_, length_ + rows);
  ++stats_.target_forward_passes;

  if (need_aux && !bolt_.has_aux_rows_) {
    auto aux_src = temp_aux->GetByteSpan();
    auto prefill_dst = prefill_aux_->GetByteSpan();
    prefill_dst.subspan((overlap_start - draft_start) * aux_row_bytes, aux_count * aux_row_bytes)
        .CopyFrom(aux_src.subspan(aux_begin_in_chunk * aux_row_bytes, aux_count * aux_row_bytes));
  }
  if (!want_prediction) {
    // Non-final chunks do not read predictions back on host, so synchronize before reusing pinned
    // input mirrors or releasing temporary fallback output buffers on the next chunk.
    bolt_.p_device_->Synchronize();
    return 0;
  }
  if (bolt_.has_logit_rows_) {
    auto cpu = prefill_pred_span_.CopyDeviceToCpu();
    return cpu[0];
  }
  tensors_.at("logits")
      ->GetByteSpan()
      .subspan((kBoltVerifyRows - 1) * logit_row_bytes, logit_row_bytes)
      .CopyFrom(temp_logits->GetByteSpan().subspan((rows - 1) * logit_row_bytes, logit_row_bytes));
  auto pred_span = temp_preds->GetDeviceSpan<int64_t>();
  auto cpu = pred_span.CopyDeviceToCpu();
  return cpu[rows - 1];
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
  has_drafts_ = true;
  ++stats_.draft_forward_passes;
}

void NemotronBoltState::Prefill(std::span<const int32_t> tokens) {
  std::lock_guard lock{bolt_.execution_mutex_};
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
    const bool can_dynamic_prefill =
        bolt_.dynamic_prefill_ && prefill_chunk_rows_ > kBoltVerifyRows;
    const bool single_chunk =
        draft_rows <= kBoltVerifyRows && (!can_dynamic_prefill || tokens.size() <= kBoltVerifyRows);
    const size_t aux_row_bytes = bolt_.aux_width_ * sizeof(uint16_t);
    for (size_t pos = 0; pos < tokens.size();) {
      const size_t remaining = tokens.size() - pos;
      if (can_dynamic_prefill && remaining > kBoltVerifyRows) {
        const size_t valid = std::min(prefill_chunk_rows_, remaining);
        const bool last_chunk = pos + valid >= tokens.size();
        const int64_t pred = PrefillChunk(tokens.subspan(pos, valid), pos, draft_start, last_chunk);
        if (last_chunk) {
          anchor_ = static_cast<int32_t>(pred);
        }
        length_ += valid;
        state_row_ = static_cast<int32_t>(kBoltVerifyRows - 1);
        pos += valid;
        continue;
      }
      const size_t valid = std::min(bolt_.target_rows_, remaining);
      const bool last_chunk = pos + bolt_.target_rows_ >= tokens.size();
      std::vector<int32_t> block(bolt_.target_rows_, tokens[pos + valid - 1]);
      std::copy_n(tokens.begin() + pos, valid, block.begin());
      const auto predictions = Verify(block, last_chunk);
      if (drafter_ && !single_chunk && pos + valid > draft_start) {
        const size_t overlap_start = std::max(pos, draft_start);
        const size_t src_row = overlap_start - pos;
        const size_t count = valid - src_row;
        auto aux_src = aux_view_->GetByteSpan();
        auto prefill_dst = prefill_aux_->GetByteSpan();
        prefill_dst.subspan((overlap_start - draft_start) * aux_row_bytes, count * aux_row_bytes)
            .CopyFrom(aux_src.subspan(src_row * aux_row_bytes, count * aux_row_bytes));
      }
      if (last_chunk) {
        anchor_ = static_cast<int32_t>(predictions[valid - 1]);
      }
      length_ += valid;
      state_row_ = static_cast<int32_t>(valid - 1);
      pos += bolt_.target_rows_;
    }
    const size_t max_context = static_cast<size_t>(bolt_.config_->model.context_length);
    if (drafter_ && tokens.size() + kBoltVerifyRows <= max_context) {
      Propose(draft_rows, draft_start, anchor_, single_chunk ? aux_view_.get() : prefill_aux_.get());
    } else {
      has_drafts_ = false;
      drafts_.clear();
    }
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
  std::lock_guard lock{bolt_.execution_mutex_};
  Require(prefilled_, "AppendTokens must precede GenerateNextToken");
  Require(params_->search.max_length > 0 &&
              public_length < static_cast<size_t>(params_->search.max_length),
          "generation is already at its output budget");
  if (pending_.empty() && (!drafter_ || !has_drafts_)) {
    // Only row zero is committed; later rows are causal scratch and are overwritten.
    const auto rows = bolt_.single_token_decode_ ? 1 : bolt_.target_rows_;
    const auto predictions = Verify(std::vector<int32_t>(rows, anchor_));
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
    const auto tokens = NemotronBoltRoundTokens(
        drafts_, predictions, static_cast<size_t>(params_->search.max_length) - public_length,
        model_.config_->model.eos_token_id, verified_rows_);
    const NemotronBoltRoundStats round_stats = GetNemotronBoltRoundStats(drafts_, tokens);
    // The final queued token is the new anchor; state includes only its preceding input rows.
    const size_t committed_rows = tokens.size() - 1;
    if (g_log.enabled && g_log.model_logits) {
      auto& log = Log("model_logits");
      log << "Bolt verify length=" << length_ << " state_row=" << state_row_ << " anchor=" << anchor_
          << " accepted=" << round_stats.accepted << " committed_rows=" << committed_rows << " drafts=";
      for (auto token : drafts_) log << token << ',';
      log << " predictions=";
      for (auto token : predictions) log << token << ',';
      log << std::endl;
    }
    pending_.insert(pending_.end(), tokens.begin(), tokens.end());
    ++stats_.rounds;
    ++stats_.completed_rounds;
    stats_.target_verify_forward_passes += stats_.target_forward_passes - target_passes;
    stats_.draft_tokens_proposed += 7;
    stats_.draft_tokens_evaluated += round_stats.evaluated;
    stats_.draft_tokens_accepted += round_stats.accepted;
    ++stats_.acceptance_length_histogram[round_stats.accepted];
    stats_.tokens_queued += tokens.size();
    if (round_stats.bonus) ++stats_.bonus_tokens;
    if (round_stats.correction) ++stats_.correction_tokens;
    if (round_stats.accepted == 0)
      ++stats_.zero_accept_rounds;
    else if (round_stats.full)
      ++stats_.full_accept_rounds;
    else
      ++stats_.partial_accept_rounds;
    anchor_ = static_cast<int32_t>(predictions[committed_rows]);
    const size_t first_position = length_;
    length_ += committed_rows + 1;
    state_row_ = static_cast<int32_t>(committed_rows);
    const bool finishes = public_length + tokens.size() >= static_cast<size_t>(params_->search.max_length) ||
                          (!tokens.empty() && std::find(model_.config_->model.eos_token_id.begin(),
                                                        model_.config_->model.eos_token_id.end(), tokens.back()) !=
                                                  model_.config_->model.eos_token_id.end());
    const size_t max_context = static_cast<size_t>(bolt_.config_->model.context_length);
    if (!finishes && length_ + kBoltVerifyRows <= max_context) {
      Propose(committed_rows + 1, first_position, anchor_);
    } else {
      has_drafts_ = false;
      drafts_.clear();
    }
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
    const size_t decode_emitted = stats.tokens_emitted > 0 ? stats.tokens_emitted - 1 : 0;
    stats.mean_emitted_tokens_per_round = static_cast<float>(decode_emitted) / stats.rounds;
  }
  return stats;
}

Tensor& NemotronBoltBatchState::AddBaseTensor(const std::string& name, bool input, DeviceInterface* device,
                                              ONNXTensorElementDataType type,
                                              const std::vector<int64_t>& shape) {
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

NemotronBoltBatchState::NemotronBoltBatchState(const NemotronBoltModel& model,
                                               const GeneratorParams& params,
                                               size_t max_slots)
    : State{params, model}, bolt_{model}, max_slots_{std::max<size_t>(1, max_slots)} {
  std::lock_guard lock{model.execution_mutex_};
  Require(model.paged_attention_ && model.batched_recurrent_ && model.dynamic_prefill_ &&
              model.has_logit_rows_ && model.has_aux_rows_ && max_slots_ <= kBoltMaxBatchSize &&
              model.paged_num_blocks_ >= max_slots_,
          "NemotronBoltBatchState requires a PagedAttention dynamic-prefill Bolt target with banked recurrent state");
  const size_t max_seq_tokens =
      static_cast<size_t>(std::max(model.config_->model.context_length, 1)) + kBoltVerifyRows;
  max_blocks_per_seq_ = std::max<size_t>(
      1, std::min(model.paged_num_blocks_,
                  (max_seq_tokens + kBoltPagedBlockSize - 1) / kBoltPagedBlockSize));
  prefill_chunk_rows_ =
      NemotronBoltPrefillChunkRows(params.search.chunk_size, max_blocks_per_seq_ * kBoltPagedBlockSize);

  const auto& info = model.session_info_;
  const auto& dev_mem_info = model.p_device_->GetAllocator().GetInfo();
  auto* cpu_device = GetDeviceInterface(DeviceType::CPU);

  const size_t max_decode_rows = max_slots_ * kBoltVerifyRows;
  decode_ids_span_ = model.p_device_->Allocate<int64_t>(max_decode_rows);
  decode_ids_span_.CpuSpan();
  // Packed int32 layout:
  // [0 .. max_slots_): past_sequence_lengths
  // [max_slots_ .. 2*max_slots_ + 1): cumulative_sequence_lengths (0, 8, 16, ...)
  // [2*max_slots_ + 1 .. 3*max_slots_ + 1): state_row
  // [3*max_slots_ + 1 .. 4*max_slots_ + 1): state_slots
  decode_meta_span_ = model.p_device_->Allocate<int32_t>(4 * max_slots_ + 4);
  {
    auto cpu_meta = decode_meta_span_.CpuSpan();
    std::fill(cpu_meta.begin(), cpu_meta.end(), 0);
    for (size_t i = 0; i <= max_slots_; ++i) {
      cpu_meta[max_slots_ + i] = static_cast<int32_t>(i * kBoltVerifyRows);
    }
    decode_meta_span_.CopyCpuToDevice();
  }
  decode_block_table_span_ = model.p_device_->Allocate<int32_t>(max_slots_ * max_blocks_per_seq_);
  decode_block_table_span_.CpuSpan();

  decode_row_indices_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const std::array<int64_t, 1> max_decode_rows_1d{static_cast<int64_t>(max_decode_rows)};
  decode_row_indices_->CreateTensor(max_decode_rows_1d, true);
  {
    std::vector<int64_t> indices(max_decode_rows);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    Upload(*decode_row_indices_, std::span<const int64_t>{indices});
  }

  decode_attn_meta_ = std::make_unique<Tensor>(cpu_device, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  const std::array<int64_t, 1> meta_3_shape{3};
  decode_attn_meta_->CreateTensor(meta_3_shape, true);
  {
    const std::array<int32_t, 3> meta{
        static_cast<int32_t>(kBoltVerifyRows),
        static_cast<int32_t>(max_blocks_per_seq_ * kBoltPagedBlockSize),
        1};
    Upload(*decode_attn_meta_, std::span<const int32_t>{meta});
  }

  decode_logits_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  const std::array<int64_t, 3> max_logits_shape{
      1, static_cast<int64_t>(max_decode_rows), model.config_->model.vocab_size};
  decode_logits_->CreateTensor(max_logits_shape, true);
  decode_pred_ids_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const std::array<int64_t, 2> max_preds_shape{1, static_cast<int64_t>(max_decode_rows)};
  decode_pred_ids_->CreateTensor(max_preds_shape, true);
  decode_pred_span_ = decode_pred_ids_->GetDeviceSpan<int64_t>();
  decode_pred_span_.CpuSpan();
  decode_aux_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  const std::array<int64_t, 3> max_aux_shape{
      1, static_cast<int64_t>(max_decode_rows), static_cast<int64_t>(model.aux_width_)};
  decode_aux_->CreateTensor(max_aux_shape, true);

  for (const auto& name : info.GetInputNames()) {
    if (name == "input_ids") {
      input_idx_input_ids_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "past_sequence_lengths") {
      input_idx_past_seqlens_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "cumulative_sequence_lengths") {
      input_idx_cu_seqlens_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "state_row") {
      input_idx_state_row_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "state_slots") {
      input_idx_state_slots_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "block_table") {
      input_idx_block_table_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "logit_rows") {
      input_idx_logit_rows_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "aux_rows") {
      input_idx_aux_rows_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(nullptr);
    } else if (name == "attention_metadata") {
      input_idx_attn_meta_ = input_storage_.size();
      input_storage_.push_back(name);
      inputs_.push_back(decode_attn_meta_->GetOrtTensor());
    } else if (name.rfind("past_", 0) == 0) {
      auto shape = info.GetInputShape(name);
      if (name.rfind("past_key_values.", 0) == 0) {
        shape[0] = static_cast<int64_t>(model.paged_num_blocks_);
      } else if (shape[0] < 0) {
        shape[0] = static_cast<int64_t>(max_slots_);
      }
      auto& tensor = AddBaseTensor(name, true, model.p_device_, info.GetInputDataType(name), shape);
      if (name.rfind("past_key_values.", 0) == 0) {
        kv_caches_.push_back(&tensor);
      } else {
        recurrent_caches_.push_back(&tensor);
      }
      output_storage_.push_back(PresentCacheName(name));
      outputs_.push_back(tensor.GetOrtTensor());
    } else {
      throw std::runtime_error("nemotron_bolt: unsupported batched input " + name);
    }
  }

  output_idx_logits_ = output_storage_.size();
  output_storage_.push_back("logits");
  outputs_.push_back(nullptr);
  output_idx_pred_ids_ = output_storage_.size();
  output_storage_.push_back("pred_ids");
  outputs_.push_back(nullptr);
  output_idx_aux_ = output_storage_.size();
  output_storage_.push_back("aux_hidden_states");
  outputs_.push_back(nullptr);

  for (auto& name : input_storage_) input_names_.push_back(name.c_str());
  for (auto& name : output_storage_) output_names_.push_back(name.c_str());

  if (model.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model.config_->model.decoder.run_options.value());
  }
  run_options_->AddConfigEntry("disable_synchronize_execution_providers", "1");
  if (!params.use_graph_capture) {
    run_options_->AddConfigEntry("gpu_graph_id", "-1");
  }

  // Build static-address OrtValue bindings for every active decode batch size b in [1..max_slots_].
  batch_bindings_.resize(max_slots_ + 1);
  int64_t* ids_dev = decode_ids_span_.Span().data();
  int32_t* meta_dev = decode_meta_span_.Span().data();
  int32_t* bt_dev = decode_block_table_span_.Span().data();
  void* idx_dev = decode_row_indices_->GetMutableRawData();
  void* logits_dev = decode_logits_->GetMutableRawData();
  void* preds_dev = decode_pred_ids_->GetMutableRawData();
  void* aux_dev = decode_aux_->GetMutableRawData();

  for (size_t b = 1; b <= max_slots_; ++b) {
    auto& bind = batch_bindings_[b];
    bind.inputs = inputs_;
    bind.outputs = outputs_;
    const size_t rows_b = b * kBoltVerifyRows;
    auto add_in_view = [&](size_t idx, void* ptr, size_t bytes, std::span<const int64_t> shape,
                           ONNXTensorElementDataType dtype) {
      auto val = OrtValue::CreateTensor(dev_mem_info, ptr, bytes, shape, dtype);
      bind.inputs[idx] = val.get();
      bind.owned_inputs.push_back(std::move(val));
    };
    auto add_out_view = [&](size_t idx, void* ptr, size_t bytes, std::span<const int64_t> shape,
                            ONNXTensorElementDataType dtype) {
      auto val = OrtValue::CreateTensor(dev_mem_info, ptr, bytes, shape, dtype);
      bind.outputs[idx] = val.get();
      bind.owned_outputs.push_back(std::move(val));
    };

    const std::array<int64_t, 2> ids_shape{1, static_cast<int64_t>(rows_b)};
    const std::array<int64_t, 1> b_shape{static_cast<int64_t>(b)};
    const std::array<int64_t, 1> bp1_shape{static_cast<int64_t>(b + 1)};
    const std::array<int64_t, 2> bt_shape{
        static_cast<int64_t>(b), static_cast<int64_t>(max_blocks_per_seq_)};
    const std::array<int64_t, 1> rows_1d{static_cast<int64_t>(rows_b)};
    const std::array<int64_t, 3> logits_shape{
        1, static_cast<int64_t>(rows_b), model.config_->model.vocab_size};
    const std::array<int64_t, 3> aux_3d{
        1, static_cast<int64_t>(rows_b), static_cast<int64_t>(model.aux_width_)};
    const std::array<int64_t, 2> aux_2d{
        static_cast<int64_t>(rows_b), static_cast<int64_t>(model.aux_width_)};

    add_in_view(input_idx_input_ids_, ids_dev, rows_b * sizeof(int64_t), ids_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    add_in_view(input_idx_past_seqlens_, meta_dev, b * sizeof(int32_t), b_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    add_in_view(input_idx_cu_seqlens_, meta_dev + max_slots_, (b + 1) * sizeof(int32_t), bp1_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    add_in_view(input_idx_state_row_, meta_dev + 2 * max_slots_ + 1, b * sizeof(int32_t), b_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    add_in_view(input_idx_state_slots_, meta_dev + 3 * max_slots_ + 1, b * sizeof(int32_t), b_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    add_in_view(input_idx_block_table_, bt_dev,
                b * max_blocks_per_seq_ * sizeof(int32_t), bt_shape,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    add_in_view(input_idx_logit_rows_, idx_dev, rows_b * sizeof(int64_t), rows_1d,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    add_in_view(input_idx_aux_rows_, idx_dev, rows_b * sizeof(int64_t), rows_1d,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    add_out_view(output_idx_logits_, logits_dev,
                 rows_b * static_cast<size_t>(model.config_->model.vocab_size) * sizeof(uint16_t),
                 logits_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    add_out_view(output_idx_pred_ids_, preds_dev, rows_b * sizeof(int64_t), ids_shape,
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    add_out_view(output_idx_aux_, aux_dev, rows_b * model.aux_width_ * sizeof(uint16_t), aux_3d,
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

    bind.aux_view = std::make_unique<Tensor>(OrtValue::CreateTensor(
        *model.p_device_->GetMemoryInfo(), aux_dev, rows_b * model.aux_width_ * sizeof(uint16_t),
        aux_2d, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
    bind.aux_view->p_device_ = model.p_device_;
  }
  inputs_ = batch_bindings_[1].inputs;
  outputs_ = batch_bindings_[1].outputs;

  size_t max_single_prefill_aux_rows = kBoltVerifyRows;
  if (params.speculative.max_draft_tokens != 0) {
    size_t blocks = Dflash2Drafter::PoolBlocks(*model.config_, kBoltPagedBlockSize, max_slots_);
    if (blocks == 0) {
      blocks = max_slots_ * ((model.capacity_ + kBoltPagedBlockSize - 1) / kBoltPagedBlockSize + 1);
    }
    drafter_ = std::make_unique<Dflash2Drafter>(
        model.drafter_model_, kBoltPagedBlockSize, blocks, max_slots_, /*fixed_single_request=*/false);
    drafter_->SetPinnedBlockTableWidth(true);
    drafter_->SetPaddedDecodeSteps(true);
    max_single_prefill_aux_rows =
        ((std::max(drafter_->ContextWindow(), kBoltVerifyRows) + kBoltVerifyRows - 1) / kBoltVerifyRows + 1) *
        kBoltVerifyRows;
  }

  const size_t max_mixed_rows = max_slots_ * kBoltVerifyRows + prefill_chunk_rows_;
  mixed_ids_span_ = model.p_device_->Allocate<int64_t>(max_mixed_rows);
  mixed_ids_span_.CpuSpan();
  mixed_meta_span_ = model.p_device_->Allocate<int32_t>(4 * max_slots_ + 4);
  mixed_meta_span_.CpuSpan();
  mixed_block_table_span_ = model.p_device_->Allocate<int32_t>(max_slots_ * max_blocks_per_seq_);
  mixed_block_table_span_.CpuSpan();
  mixed_logit_rows_span_ = model.p_device_->Allocate<int64_t>(max_slots_ * kBoltVerifyRows);
  mixed_logit_rows_span_.CpuSpan();
  mixed_aux_rows_span_ = model.p_device_->Allocate<int64_t>(max_slots_ * max_single_prefill_aux_rows);
  mixed_aux_rows_span_.CpuSpan();
  if (drafter_) {
    mixed_propose_aux_ = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    const std::array<int64_t, 2> max_propose_aux_shape{
        static_cast<int64_t>(max_slots_ * max_single_prefill_aux_rows),
        static_cast<int64_t>(model.aux_width_)};
    mixed_propose_aux_->CreateTensor(max_propose_aux_shape, true);
  }

  slot_blocks_.resize(max_slots_);
  slots_.resize(max_slots_);
  const std::array<int64_t, 2> slot_prefill_aux_shape{
      static_cast<int64_t>(max_single_prefill_aux_rows), static_cast<int64_t>(model.aux_width_)};
  for (size_t s = 0; s < max_slots_; ++s) {
    slots_[s].drafter_key = std::make_unique<Request>(model, params.search.max_length);
    if (drafter_) {
      slots_[s].prefill_aux = std::make_unique<Tensor>(model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      slots_[s].prefill_aux->CreateTensor(slot_prefill_aux_shape, true);
    }
  }

  ResetAll();
  if (params.use_graph_capture) {
    for (size_t b = 1; b <= max_slots_; ++b) {
      auto ids_cpu = decode_ids_span_.CpuSpan();
      std::fill_n(ids_cpu.begin(), b * kBoltVerifyRows, int64_t{1});
      decode_ids_span_.subspan(0, b * kBoltVerifyRows).CopyCpuRangeToDevice();

      auto meta_cpu = decode_meta_span_.CpuSpan();
      for (size_t i = 0; i < b; ++i) {
        meta_cpu[i] = 0;
        meta_cpu[2 * max_slots_ + 1 + i] = 0;
        meta_cpu[3 * max_slots_ + 1 + i] = static_cast<int32_t>(i);
      }
      for (size_t i = 0; i <= b; ++i) {
        meta_cpu[max_slots_ + i] = static_cast<int32_t>(i * kBoltVerifyRows);
      }
      decode_meta_span_.CopyCpuToDevice();

      auto bt_cpu = decode_block_table_span_.CpuSpan();
      std::fill_n(bt_cpu.begin(), b * max_blocks_per_seq_, int32_t{-1});
      for (size_t i = 0; i < b; ++i) {
        bt_cpu[i * max_blocks_per_seq_] = static_cast<int32_t>(i);
      }
      decode_block_table_span_.subspan(0, b * max_blocks_per_seq_).CopyCpuRangeToDevice();

      inputs_ = batch_bindings_[b].inputs;
      outputs_ = batch_bindings_[b].outputs;
      for (int run = 0; run < kCaptureWarmupRuns; ++run) {
        State::Run(*bolt_.session_, true, static_cast<int>(b * kBoltVerifyRows), static_cast<int>(b), nullptr);
        if (run == kCaptureWarmupRuns - 1) {
          (void)decode_pred_span_.subspan(0, b * kBoltVerifyRows).CopyDeviceToCpu();
        }
      }
    }
    if (drafter_) {
      for (size_t b = 1; b <= max_slots_; ++b) {
        std::vector<Dflash2Drafter::Feed> feeds(b);
        for (size_t i = 0; i < b; ++i) {
          feeds[i] = Dflash2Drafter::Feed{
              .request = slots_[i].drafter_key.get(),
              .aux_row_begin = i * kBoltVerifyRows,
              .aux_row_count = kBoltVerifyRows,
              .first_position = 0,
              .anchor_token = 1,
              .draft_eligible = true,
              .wants_drafts = true,
          };
        }
        std::vector<std::vector<int32_t>> proposals;
        for (int run = 0; run < kCaptureWarmupRuns; ++run) {
          drafter_->ReleaseAll();
          Require(drafter_->Propose(*batch_bindings_[b].aux_view, feeds, proposals) &&
                      proposals.size() == b,
                  "failed to warm batched DFlash2 drafter");
        }
      }
    }
    ResetAll();
  }
}

NemotronBoltBatchState::~NemotronBoltBatchState() = default;

DeviceSpan<float> NemotronBoltBatchState::Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) {
  throw std::runtime_error("nemotron_bolt uses native block decoding, not the logits search path");
}

bool NemotronBoltBatchState::IsEosToken(int32_t token) const {
  const auto& eos = model_.config_->model.eos_token_id;
  return std::find(eos.begin(), eos.end(), token) != eos.end();
}

size_t NemotronBoltBatchState::BlocksNeededForSlot(size_t slot, size_t total_seq_len) const {
  const size_t needed = (total_seq_len + kBoltPagedBlockSize - 1) / kBoltPagedBlockSize;
  const size_t current = slot_blocks_.at(slot).size();
  return needed > current ? needed - current : 0;
}

void NemotronBoltBatchState::EnsureSlotBlocks(size_t slot, size_t total_seq_len) {
  auto& blocks = slot_blocks_.at(slot);
  const size_t needed = (total_seq_len + kBoltPagedBlockSize - 1) / kBoltPagedBlockSize;
  Require(needed <= max_blocks_per_seq_, "sequence exceeds per-request block-table capacity");
  while (blocks.size() < needed) {
    Require(!free_blocks_.empty(), "PagedAttention block pool exhausted");
    blocks.push_back(free_blocks_.back());
    free_blocks_.pop_back();
  }
}

void NemotronBoltBatchState::ResetAll() {
  for (auto* cache : recurrent_caches_) {
    cache->GetByteSpan().Zero();
  }
  model_.p_device_->Synchronize();
  if (drafter_) drafter_->ReleaseAll();
  free_blocks_.resize(bolt_.paged_num_blocks_);
  std::iota(free_blocks_.rbegin(), free_blocks_.rend(), int32_t{0});
  for (auto& blocks : slot_blocks_) blocks.clear();
  for (auto& s : slots_) {
    s.length = 0;
    s.state_row = 0;
    s.anchor = 0;
    s.drafts.clear();
    s.has_drafts = false;
    s.prefilled = false;
    s.dirty = false;
  }
  last_decode_slots_.clear();
  last_decode_block_counts_.clear();
  stats_ = {};
}

void NemotronBoltBatchState::ResetSlot(size_t slot) {
  std::lock_guard lock{bolt_.execution_mutex_};
  Require(slot < max_slots_, "slot index out of range");
  auto& s = slots_[slot];
  auto& blocks = slot_blocks_[slot];
  if (!blocks.empty()) {
    free_blocks_.insert(free_blocks_.end(), blocks.rbegin(), blocks.rend());
    blocks.clear();
  }
  last_decode_slots_.clear();
  last_decode_block_counts_.clear();
  if (drafter_) {
    drafter_->Release(s.drafter_key.get());
  }
  if (s.dirty) {
    for (auto* cache : recurrent_caches_) {
      auto span = cache->GetByteSpan();
      const size_t bank_bytes = span.size() / max_slots_;
      bolt_.p_device_->WrapMemory(span.subspan(slot * bank_bytes, bank_bytes).Span()).Zero();
    }
    model_.p_device_->Synchronize();
    s.dirty = false;
  }
  s.length = 0;
  s.state_row = 0;
  s.anchor = 0;
  s.drafts.clear();
  s.has_drafts = false;
  s.prefilled = false;
}

void NemotronBoltBatchState::StepBatch(std::span<const NemotronBoltSlotTask> tasks,
                                       std::span<NemotronBoltEngineStep> results) {
  std::lock_guard lock{bolt_.execution_mutex_};
  Require(!tasks.empty() && tasks.size() <= max_slots_ && results.size() == tasks.size(),
          "invalid NemotronBoltBatchState step batch");
  for (size_t i = 0; i < tasks.size(); ++i) {
    results[i] = NemotronBoltEngineStep{};
    const auto& task = tasks[i];
    Require(task.slot < max_slots_ && task.remaining_budget > 0, "invalid slot or zero budget in StepBatch");
    const auto& s = slots_[task.slot];
    if (task.is_decode) {
      Require(s.prefilled, "StepBatch decode task requires a prefilled slot");
    } else {
      Require(!s.prefilled && !task.full_prompt.empty() && task.prefill_tokens > 0 &&
                  task.prefill_offset == s.length &&
                  task.prefill_offset + task.prefill_tokens <= task.full_prompt.size(),
              "invalid StepBatch prefill cursor");
      if (task.prefill_offset == 0) {
        for (const auto token : task.full_prompt) {
          Require(token >= 0 && token < model_.config_->model.vocab_size, "token id outside vocabulary");
        }
      }
    }
  }
  const bool all_decode = std::all_of(tasks.begin(), tasks.end(), [](const auto& t) { return t.is_decode; });
  if (all_decode) {
    StepPureDecode(tasks, results);
  } else {
    StepMixedOrPrefill(tasks, results);
  }
}

void NemotronBoltBatchState::StepPureDecode(std::span<const NemotronBoltSlotTask> tasks,
                                            std::span<NemotronBoltEngineStep> results) {
  const size_t b = tasks.size();
  const size_t rows = b * kBoltVerifyRows;
  auto ids_cpu = decode_ids_span_.CpuSpan();
  auto meta_cpu = decode_meta_span_.CpuSpan();

  bool same_block_table = (last_decode_slots_.size() == b);
  for (size_t i = 0; i < b; ++i) {
    const size_t s_idx = tasks[i].slot;
    auto& s = slots_[s_idx];
    s.dirty = true;
    EnsureSlotBlocks(s_idx, s.length + kBoltVerifyRows);
    ids_cpu[i * kBoltVerifyRows] = s.anchor;
    if (drafter_ && s.has_drafts) {
      Require(s.drafts.size() == kBoltVerifyRows - 1, "missing draft block for speculative slot");
      for (size_t j = 0; j < kBoltVerifyRows - 1; ++j) {
        ids_cpu[i * kBoltVerifyRows + 1 + j] = s.drafts[j];
      }
    } else {
      for (size_t j = 1; j < kBoltVerifyRows; ++j) {
        ids_cpu[i * kBoltVerifyRows + j] = s.anchor;
      }
    }
    meta_cpu[i] = static_cast<int32_t>(s.length);
    meta_cpu[max_slots_ + i] = static_cast<int32_t>(i * kBoltVerifyRows);
    meta_cpu[2 * max_slots_ + 1 + i] = s.state_row;
    meta_cpu[3 * max_slots_ + 1 + i] = static_cast<int32_t>(s_idx);
    if (!same_block_table || last_decode_slots_[i] != s_idx ||
        last_decode_block_counts_[i] != slot_blocks_[s_idx].size()) {
      same_block_table = false;
    }
  }
  meta_cpu[max_slots_ + b] = static_cast<int32_t>(rows);

  decode_ids_span_.subspan(0, rows).CopyCpuRangeToDevice();
  decode_meta_span_.CopyCpuToDevice();

  if (!same_block_table) {
    auto bt_cpu = decode_block_table_span_.CpuSpan();
    const size_t cols = max_blocks_per_seq_;
    for (size_t i = 0; i < b; ++i) {
      const auto& blocks = slot_blocks_[tasks[i].slot];
      auto row = bt_cpu.subspan(i * cols, cols);
      std::copy(blocks.begin(), blocks.end(), row.begin());
      std::fill(row.begin() + blocks.size(), row.end(), int32_t{-1});
    }
    decode_block_table_span_.subspan(0, b * cols).CopyCpuRangeToDevice();
    last_decode_slots_.resize(b);
    last_decode_block_counts_.resize(b);
    for (size_t i = 0; i < b; ++i) {
      last_decode_slots_[i] = tasks[i].slot;
      last_decode_block_counts_[i] = slot_blocks_[tasks[i].slot].size();
    }
  }

  inputs_ = batch_bindings_[b].inputs;
  outputs_ = batch_bindings_[b].outputs;
  State::Run(*bolt_.session_, params_->use_graph_capture, static_cast<int>(rows), static_cast<int>(b), nullptr);
  ++stats_.target_forward_passes;

  auto preds_cpu = decode_pred_span_.subspan(0, rows).CopyDeviceToCpu();
  const size_t max_context = static_cast<size_t>(bolt_.config_->model.context_length);
  std::vector<Dflash2Drafter::Feed> feeds;
  std::vector<size_t> feed_slots;
  if (drafter_) {
    feeds.reserve(b);
    feed_slots.reserve(b);
  }
  bool counted_verify_pass = false;

  for (size_t i = 0; i < b; ++i) {
    const size_t s_idx = tasks[i].slot;
    auto& s = slots_[s_idx];
    auto& out = results[i];
    std::span<const int64_t> preds_i{preds_cpu.data() + i * kBoltVerifyRows, kBoltVerifyRows};
    if (!drafter_ || !s.has_drafts) {
      s.anchor = static_cast<int32_t>(preds_i[0]);
      ++s.length;
      s.state_row = 0;
      s.has_drafts = false;
      s.drafts.clear();
      out.tokens[0] = s.anchor;
      out.count = 1;
      out.speculation = SpeculativeRowMetrics{1, 1};
      continue;
    }

    const auto tokens = NemotronBoltRoundTokens(
        s.drafts, preds_i, tasks[i].remaining_budget, model_.config_->model.eos_token_id, kBoltVerifyRows);
    const NemotronBoltRoundStats round_stats = GetNemotronBoltRoundStats(s.drafts, tokens);
    const size_t committed_rows = tokens.size() - 1;

    std::copy_n(s.drafts.begin(), kBoltVerifyRows - 1, out.drafts.begin());
    for (size_t j = 0; j < tokens.size(); ++j) out.tokens[j] = tokens[j];
    out.count = tokens.size();
    out.speculative_round = true;
    out.verify_forward_passes = counted_verify_pass ? 0 : 1;
    counted_verify_pass = true;
    out.speculation = SpeculativeRowMetrics{7, round_stats.accepted};

    const size_t first_position = s.length;
    s.anchor = static_cast<int32_t>(preds_i[committed_rows]);
    s.length += committed_rows + 1;
    s.state_row = static_cast<int32_t>(committed_rows);

    const bool finishes =
        tokens.size() >= tasks[i].remaining_budget || (!tokens.empty() && IsEosToken(tokens.back()));
    if (!finishes && s.length + kBoltVerifyRows <= max_context) {
      feeds.push_back(Dflash2Drafter::Feed{
          .request = s.drafter_key.get(),
          .aux_row_begin = i * kBoltVerifyRows,
          .aux_row_count = committed_rows + 1,
          .first_position = first_position,
          .anchor_token = s.anchor,
          .draft_eligible = true,
          .wants_drafts = true,
      });
      feed_slots.push_back(s_idx);
    } else {
      s.has_drafts = false;
      s.drafts.clear();
    }
  }

  if (!feeds.empty()) {
    std::vector<std::vector<int32_t>> proposals;
    Require(drafter_->Propose(*batch_bindings_[b].aux_view, feeds, proposals) &&
                proposals.size() == feeds.size(),
            "batched DFlash2 drafter proposal failed");
    for (size_t f = 0; f < feeds.size(); ++f) {
      Require(proposals[f].size() == kBoltVerifyRows - 1, "DFlash2 drafter did not produce seven tokens");
      slots_[feed_slots[f]].drafts = std::move(proposals[f]);
      slots_[feed_slots[f]].has_drafts = true;
    }
    ++stats_.draft_forward_passes;
  }
}

void NemotronBoltBatchState::StepMixedOrPrefill(std::span<const NemotronBoltSlotTask> tasks,
                                                std::span<NemotronBoltEngineStep> results) {
  last_decode_slots_.clear();
  last_decode_block_counts_.clear();

  const size_t b = tasks.size();
  const size_t window = drafter_ ? drafter_->ContextWindow() : 0;
  const size_t aux_row_bytes = bolt_.aux_width_ * sizeof(uint16_t);

  struct TaskLayout {
    size_t row_offset{};
    size_t row_count{};
    size_t logit_offset{};
    size_t aux_offset{};
    size_t aux_count{};
    size_t draft_start{};
    size_t draft_rows{};
    size_t overlap_start{};
    bool last_chunk{};
    bool need_aux{};
    bool single_aux_chunk{};
  };
  std::vector<TaskLayout> layouts(b);

  size_t total_rows = 0;
  int32_t max_q_len = 1;
  int32_t max_kv_len = 1;
  std::vector<int64_t> logit_rows;
  std::vector<int64_t> aux_rows;
  bool want_any_preds = false;

  for (size_t i = 0; i < b; ++i) {
    const auto& task = tasks[i];
    auto& s = slots_[task.slot];
    auto& lay = layouts[i];
    s.dirty = true;
    lay.row_offset = total_rows;
    if (task.is_decode) {
      lay.row_count = kBoltVerifyRows;
      EnsureSlotBlocks(task.slot, s.length + kBoltVerifyRows);
      lay.logit_offset = logit_rows.size();
      for (size_t r = 0; r < kBoltVerifyRows; ++r) {
        logit_rows.push_back(static_cast<int64_t>(lay.row_offset + r));
      }
      want_any_preds = true;
      if (drafter_) {
        lay.need_aux = true;
        lay.single_aux_chunk = true;
        lay.aux_offset = aux_rows.size();
        lay.aux_count = kBoltVerifyRows;
        for (size_t r = 0; r < kBoltVerifyRows; ++r) {
          aux_rows.push_back(static_cast<int64_t>(lay.row_offset + r));
        }
      }
      max_q_len = std::max(max_q_len, static_cast<int32_t>(kBoltVerifyRows));
      max_kv_len = std::max(max_kv_len, static_cast<int32_t>(s.length + kBoltVerifyRows));
    } else {
      lay.row_count = task.prefill_tokens;
      lay.last_chunk = (task.prefill_offset + task.prefill_tokens >= task.full_prompt.size());
      EnsureSlotBlocks(task.slot, s.length + lay.row_count);
      if (lay.last_chunk) {
        lay.logit_offset = logit_rows.size();
        logit_rows.push_back(static_cast<int64_t>(lay.row_offset + lay.row_count - 1));
        want_any_preds = true;
      }
      if (drafter_) {
        lay.draft_start =
            (window != 0 && task.full_prompt.size() > window)
                ? ((task.full_prompt.size() - window) / kBoltVerifyRows) * kBoltVerifyRows
                : 0;
        lay.draft_rows = task.full_prompt.size() - lay.draft_start;
        if (task.prefill_offset + lay.row_count > lay.draft_start) {
          lay.need_aux = true;
          lay.overlap_start = std::max(task.prefill_offset, lay.draft_start);
          const size_t begin_in_chunk = lay.overlap_start - task.prefill_offset;
          lay.aux_count = lay.row_count - begin_in_chunk;
          lay.aux_offset = aux_rows.size();
          lay.single_aux_chunk = (lay.last_chunk && lay.overlap_start == lay.draft_start);
          for (size_t r = 0; r < lay.aux_count; ++r) {
            aux_rows.push_back(static_cast<int64_t>(lay.row_offset + begin_in_chunk + r));
          }
        }
      }
      max_q_len = std::max(max_q_len, static_cast<int32_t>(lay.row_count));
      max_kv_len = std::max(max_kv_len, static_cast<int32_t>(s.length + lay.row_count));
    }
    total_rows += lay.row_count;
  }

  if (logit_rows.empty()) logit_rows.push_back(0);
  if (aux_rows.empty()) aux_rows.push_back(0);

  if (total_rows > mixed_ids_span_.size()) {
    mixed_ids_span_ = bolt_.p_device_->Allocate<int64_t>(total_rows);
    mixed_ids_span_.CpuSpan();
  }
  if (logit_rows.size() > mixed_logit_rows_span_.size()) {
    mixed_logit_rows_span_ = bolt_.p_device_->Allocate<int64_t>(logit_rows.size());
    mixed_logit_rows_span_.CpuSpan();
  }
  if (aux_rows.size() > mixed_aux_rows_span_.size()) {
    mixed_aux_rows_span_ = bolt_.p_device_->Allocate<int64_t>(aux_rows.size());
    mixed_aux_rows_span_.CpuSpan();
  }

  {
    auto ids_sub = mixed_ids_span_.subspan(0, total_rows);
    auto ids_cpu = ids_sub.CpuSpan();
    auto meta_cpu = mixed_meta_span_.CpuSpan();
    auto bt_cpu = mixed_block_table_span_.CpuSpan();
    const size_t cols = max_blocks_per_seq_;

    meta_cpu[max_slots_] = 0;
    for (size_t i = 0; i < b; ++i) {
      const auto& task = tasks[i];
      const auto& s = slots_[task.slot];
      const auto& lay = layouts[i];
      if (task.is_decode) {
        ids_cpu[lay.row_offset] = s.anchor;
        if (drafter_ && s.has_drafts) {
          for (size_t j = 0; j < kBoltVerifyRows - 1; ++j) {
            ids_cpu[lay.row_offset + 1 + j] = s.drafts[j];
          }
        } else {
          for (size_t j = 1; j < kBoltVerifyRows; ++j) {
            ids_cpu[lay.row_offset + j] = s.anchor;
          }
        }
      } else {
        for (size_t r = 0; r < lay.row_count; ++r) {
          ids_cpu[lay.row_offset + r] = task.full_prompt[task.prefill_offset + r];
        }
      }
      meta_cpu[i] = static_cast<int32_t>(s.length);
      meta_cpu[max_slots_ + i + 1] = static_cast<int32_t>(lay.row_offset + lay.row_count);
      meta_cpu[2 * max_slots_ + 1 + i] = s.state_row;
      meta_cpu[3 * max_slots_ + 1 + i] = static_cast<int32_t>(task.slot);

      const auto& blocks = slot_blocks_[task.slot];
      auto bt_row = bt_cpu.subspan(i * cols, cols);
      std::copy(blocks.begin(), blocks.end(), bt_row.begin());
      std::fill(bt_row.begin() + blocks.size(), bt_row.end(), int32_t{-1});
    }
    ids_sub.CopyCpuRangeToDevice();
    mixed_meta_span_.CopyCpuToDevice();
    mixed_block_table_span_.subspan(0, b * cols).CopyCpuRangeToDevice();
  }

  {
    auto lr_sub = mixed_logit_rows_span_.subspan(0, logit_rows.size());
    std::copy(logit_rows.begin(), logit_rows.end(), lr_sub.CpuSpan().begin());
    lr_sub.CopyCpuRangeToDevice();

    auto ar_sub = mixed_aux_rows_span_.subspan(0, aux_rows.size());
    std::copy(aux_rows.begin(), aux_rows.end(), ar_sub.CpuSpan().begin());
    ar_sub.CopyCpuRangeToDevice();
  }

  const auto& dev_mem_info = bolt_.p_device_->GetAllocator().GetInfo();
  const auto& cpu_mem_info = GetDeviceInterface(DeviceType::CPU)->GetAllocator().GetInfo();
  int32_t* meta_dev = mixed_meta_span_.Span().data();

  const std::array<int64_t, 2> ids_shape{1, static_cast<int64_t>(total_rows)};
  const std::array<int64_t, 1> b_shape{static_cast<int64_t>(b)};
  const std::array<int64_t, 1> bp1_shape{static_cast<int64_t>(b + 1)};
  const std::array<int64_t, 2> bt_shape{
      static_cast<int64_t>(b), static_cast<int64_t>(max_blocks_per_seq_)};
  const std::array<int64_t, 1> lr_shape{static_cast<int64_t>(logit_rows.size())};
  const std::array<int64_t, 1> ar_shape{static_cast<int64_t>(aux_rows.size())};
  const std::array<int64_t, 1> attn_meta_shape{3};
  std::array<int32_t, 3> attn_meta_host{max_q_len, max_kv_len, 1};

  auto ids_ort = OrtValue::CreateTensor(
      dev_mem_info, mixed_ids_span_.Span().data(), total_rows * sizeof(int64_t), ids_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto past_ort = OrtValue::CreateTensor(
      dev_mem_info, meta_dev, b * sizeof(int32_t), b_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto cu_ort = OrtValue::CreateTensor(
      dev_mem_info, meta_dev + max_slots_, (b + 1) * sizeof(int32_t), bp1_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto srow_ort = OrtValue::CreateTensor(
      dev_mem_info, meta_dev + 2 * max_slots_ + 1, b * sizeof(int32_t), b_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto sslot_ort = OrtValue::CreateTensor(
      dev_mem_info, meta_dev + 3 * max_slots_ + 1, b * sizeof(int32_t), b_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto bt_ort = OrtValue::CreateTensor(
      dev_mem_info, mixed_block_table_span_.Span().data(),
      b * max_blocks_per_seq_ * sizeof(int32_t), bt_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto lr_ort = OrtValue::CreateTensor(
      dev_mem_info, mixed_logit_rows_span_.Span().data(), logit_rows.size() * sizeof(int64_t),
      lr_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto ar_ort = OrtValue::CreateTensor(
      dev_mem_info, mixed_aux_rows_span_.Span().data(), aux_rows.size() * sizeof(int64_t),
      ar_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto attn_meta_ort = OrtValue::CreateTensor(
      cpu_mem_info, attn_meta_host.data(), sizeof(attn_meta_host), attn_meta_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  Tensor temp_logits{bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16};
  const std::array<int64_t, 3> temp_logits_shape{
      1, static_cast<int64_t>(logit_rows.size()), bolt_.config_->model.vocab_size};
  temp_logits.CreateTensor(temp_logits_shape, true);
  Tensor temp_preds{bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  const std::array<int64_t, 2> temp_preds_shape{1, static_cast<int64_t>(logit_rows.size())};
  temp_preds.CreateTensor(temp_preds_shape, true);
  Tensor temp_aux{bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16};
  const std::array<int64_t, 3> temp_aux_shape{
      1, static_cast<int64_t>(aux_rows.size()), static_cast<int64_t>(bolt_.aux_width_)};
  temp_aux.CreateTensor(temp_aux_shape, true);

  inputs_ = batch_bindings_[1].inputs;
  outputs_ = batch_bindings_[1].outputs;
  inputs_[input_idx_input_ids_] = ids_ort.get();
  inputs_[input_idx_past_seqlens_] = past_ort.get();
  inputs_[input_idx_cu_seqlens_] = cu_ort.get();
  inputs_[input_idx_state_row_] = srow_ort.get();
  inputs_[input_idx_state_slots_] = sslot_ort.get();
  inputs_[input_idx_block_table_] = bt_ort.get();
  inputs_[input_idx_logit_rows_] = lr_ort.get();
  inputs_[input_idx_aux_rows_] = ar_ort.get();
  inputs_[input_idx_attn_meta_] = attn_meta_ort.get();
  outputs_[output_idx_logits_] = temp_logits.GetOrtTensor();
  outputs_[output_idx_pred_ids_] = temp_preds.GetOrtTensor();
  outputs_[output_idx_aux_] = temp_aux.GetOrtTensor();

  try {
    State::Run(*bolt_.session_, false, 0, 0, nullptr);
    ++stats_.target_forward_passes;

    std::vector<int64_t> preds_host;
    if (want_any_preds) {
      auto pred_span = temp_preds.GetDeviceSpan<int64_t>();
      auto cpu = pred_span.CopyDeviceToCpu();
      preds_host.assign(cpu.begin(), cpu.end());
    }

    auto temp_aux_bytes = temp_aux.GetByteSpan();
    for (size_t i = 0; i < b; ++i) {
      const auto& task = tasks[i];
      const auto& lay = layouts[i];
      if (!task.is_decode && lay.need_aux && !lay.single_aux_chunk) {
        auto dst = slots_[task.slot].prefill_aux->GetByteSpan();
        dst.subspan((lay.overlap_start - lay.draft_start) * aux_row_bytes, lay.aux_count * aux_row_bytes)
            .CopyFrom(temp_aux_bytes.subspan(lay.aux_offset * aux_row_bytes, lay.aux_count * aux_row_bytes));
      }
    }

    const size_t max_context = static_cast<size_t>(bolt_.config_->model.context_length);
    std::vector<Dflash2Drafter::Feed> feeds;
    std::vector<size_t> feed_slots;
    std::vector<size_t> feed_task_indices;
    bool needs_staged_propose_aux = false;
    bool counted_verify_pass = false;

    for (size_t i = 0; i < b; ++i) {
      const auto& task = tasks[i];
      const auto& lay = layouts[i];
      auto& s = slots_[task.slot];
      auto& out = results[i];

      if (task.is_decode) {
        std::span<const int64_t> preds_i{preds_host.data() + lay.logit_offset, kBoltVerifyRows};
        if (!drafter_ || !s.has_drafts) {
          s.anchor = static_cast<int32_t>(preds_i[0]);
          ++s.length;
          s.state_row = 0;
          s.has_drafts = false;
          s.drafts.clear();
          out.tokens[0] = s.anchor;
          out.count = 1;
          out.speculation = SpeculativeRowMetrics{1, 1};
          continue;
        }
        const auto tokens = NemotronBoltRoundTokens(
            s.drafts, preds_i, task.remaining_budget, model_.config_->model.eos_token_id, kBoltVerifyRows);
        const NemotronBoltRoundStats round_stats = GetNemotronBoltRoundStats(s.drafts, tokens);
        const size_t committed_rows = tokens.size() - 1;
        std::copy_n(s.drafts.begin(), kBoltVerifyRows - 1, out.drafts.begin());
        for (size_t j = 0; j < tokens.size(); ++j) out.tokens[j] = tokens[j];
        out.count = tokens.size();
        out.speculative_round = true;
        out.verify_forward_passes = counted_verify_pass ? 0 : 1;
        counted_verify_pass = true;
        out.speculation = SpeculativeRowMetrics{7, round_stats.accepted};

        const size_t first_position = s.length;
        s.anchor = static_cast<int32_t>(preds_i[committed_rows]);
        s.length += committed_rows + 1;
        s.state_row = static_cast<int32_t>(committed_rows);

        const bool finishes =
            tokens.size() >= task.remaining_budget || (!tokens.empty() && IsEosToken(tokens.back()));
        if (!finishes && s.length + kBoltVerifyRows <= max_context) {
          feeds.push_back(Dflash2Drafter::Feed{
              .request = s.drafter_key.get(),
              .aux_row_begin = lay.aux_offset,
              .aux_row_count = committed_rows + 1,
              .first_position = first_position,
              .anchor_token = s.anchor,
              .draft_eligible = true,
              .wants_drafts = true,
          });
          feed_slots.push_back(task.slot);
          feed_task_indices.push_back(i);
        } else {
          s.has_drafts = false;
          s.drafts.clear();
        }
      } else {
        s.length += lay.row_count;
        s.state_row = static_cast<int32_t>(std::min(lay.row_count, kBoltVerifyRows) - 1);
        out.prompt_tokens_processed = lay.row_count;
        out.prompt_complete = lay.last_chunk;
        if (lay.last_chunk) {
          s.anchor = static_cast<int32_t>(preds_host[lay.logit_offset]);
          s.prefilled = true;
          stats_.effective_k = drafter_ ? 7 : 0;
          if (task.remaining_budget > 0) {
            out.tokens[0] = s.anchor;
            out.count = 1;
            out.speculation = SpeculativeRowMetrics{1, 1};
            if (drafter_ && task.remaining_budget > 1 && !IsEosToken(s.anchor) &&
                s.length + kBoltVerifyRows <= max_context) {
              if (!lay.single_aux_chunk) needs_staged_propose_aux = true;
              feeds.push_back(Dflash2Drafter::Feed{
                  .request = s.drafter_key.get(),
                  .aux_row_begin = lay.aux_offset,
                  .aux_row_count = lay.draft_rows,
                  .first_position = lay.draft_start,
                  .anchor_token = s.anchor,
                  .draft_eligible = true,
                  .wants_drafts = true,
              });
              feed_slots.push_back(task.slot);
              feed_task_indices.push_back(i);
            } else {
              s.has_drafts = false;
              s.drafts.clear();
            }
          }
        }
      }
    }

    if (!feeds.empty()) {
      std::unique_ptr<Tensor> direct_aux_2d;
      Tensor* propose_aux_source = nullptr;
      if (!needs_staged_propose_aux) {
        const std::array<int64_t, 2> aux_2d_shape{
            static_cast<int64_t>(aux_rows.size()), static_cast<int64_t>(bolt_.aux_width_)};
        direct_aux_2d = std::make_unique<Tensor>(OrtValue::CreateTensor(
            *bolt_.p_device_->GetMemoryInfo(), temp_aux.GetMutableRawData(),
            temp_aux.GetByteSpan().size(), aux_2d_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
        direct_aux_2d->p_device_ = bolt_.p_device_;
        propose_aux_source = direct_aux_2d.get();
      } else {
        size_t staged_rows = 0;
        for (const auto& f : feeds) staged_rows += f.aux_row_count;
        if (staged_rows * aux_row_bytes > mixed_propose_aux_->GetByteSpan().size()) {
          mixed_propose_aux_ = std::make_unique<Tensor>(bolt_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
          const std::array<int64_t, 2> staged_aux_shape{
              static_cast<int64_t>(staged_rows), static_cast<int64_t>(bolt_.aux_width_)};
          mixed_propose_aux_->CreateTensor(staged_aux_shape, true);
        }
        auto dst = mixed_propose_aux_->GetByteSpan();
        size_t cursor = 0;
        for (size_t f = 0; f < feeds.size(); ++f) {
          const size_t ti = feed_task_indices[f];
          const auto& lay = layouts[ti];
          const size_t count = feeds[f].aux_row_count;
          if (tasks[ti].is_decode || lay.single_aux_chunk) {
            dst.subspan(cursor * aux_row_bytes, count * aux_row_bytes)
                .CopyFrom(temp_aux_bytes.subspan(lay.aux_offset * aux_row_bytes, count * aux_row_bytes));
          } else {
            auto src = slots_[feed_slots[f]].prefill_aux->GetByteSpan();
            dst.subspan(cursor * aux_row_bytes, count * aux_row_bytes)
                .CopyFrom(src.subspan(0, count * aux_row_bytes));
          }
          feeds[f].aux_row_begin = cursor;
          cursor += count;
        }
        propose_aux_source = mixed_propose_aux_.get();
      }
      std::vector<std::vector<int32_t>> proposals;
      Require(drafter_->Propose(*propose_aux_source, feeds, proposals) &&
                  proposals.size() == feeds.size(),
              "mixed-step DFlash2 drafter proposal failed");
      for (size_t f = 0; f < feeds.size(); ++f) {
        Require(proposals[f].size() == kBoltVerifyRows - 1, "DFlash2 drafter did not produce seven tokens");
        slots_[feed_slots[f]].drafts = std::move(proposals[f]);
        slots_[feed_slots[f]].has_drafts = true;
      }
      ++stats_.draft_forward_passes;
    } else if (!want_any_preds) {
      bolt_.p_device_->Synchronize();
    }
  } catch (...) {
    try {
      bolt_.p_device_->Synchronize();
    } catch (...) {
    }
    inputs_ = batch_bindings_[1].inputs;
    outputs_ = batch_bindings_[1].outputs;
    throw;
  }
  inputs_ = batch_bindings_[1].inputs;
  outputs_ = batch_bindings_[1].outputs;
}

std::unique_ptr<DecodingStrategy> MakeNemotronBoltStrategy(Generator& generator) {
  return std::make_unique<BoltStrategy>(generator);
}

std::unique_ptr<Search> CreateNemotronBoltSearch(const GeneratorParams& params) {
  Require(!params.config.model.decoder.sliding_window, "decoder sliding_window is unsupported");
  return std::make_unique<BoltSearch>(params);
}

}  // namespace Generators
