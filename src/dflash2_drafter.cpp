// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "dflash2_drafter.h"
#include "engine/request.h"
#include "models/io/kv_cache.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

namespace Generators {

namespace {

// The drafter's packed stream is [context rows of request 0, its block rows, context rows of
// request 1, ...]. PagedAttention derives each row's position from past_sequence_lengths plus its
// offset within the request, so the block rows land at the anchor's position and the ones after it.
struct PackedLayout {
  std::vector<int32_t> q_row_map;
  std::vector<int32_t> qkv_row_map;
  std::vector<int32_t> block_row_index;
  std::vector<int32_t> cumulative_sequence_lengths{0};
  std::vector<int32_t> past_sequence_lengths;
  int32_t max_query_len{};
  int32_t max_kv_len{};
  int32_t min_kv_len{std::numeric_limits<int32_t>::max()};
};

size_t CheckedAdd(size_t left, size_t right, std::string_view description) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error(std::string{description} + " exceeds the supported size range.");
  }
  return left + right;
}

size_t CheckedMultiply(size_t left, size_t right, std::string_view description) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(std::string{description} + " exceeds the supported size range.");
  }
  return left * right;
}

int32_t CheckedMetadataValue(size_t value, std::string_view description) {
  if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        std::string{description} + " exceeds the int32 attention metadata range.");
  }
  return static_cast<int32_t>(value);
}

void InheritProviderOptions(const Config::SessionOptions& parent,
                            Config::SessionOptions& child) {
  if (child.providers.empty()) {
    child.providers = parent.providers;
  }
  for (const auto& parent_provider : parent.provider_options) {
    auto child_provider = std::find_if(
        child.provider_options.begin(), child.provider_options.end(),
        [&parent_provider](const Config::ProviderOptions& provider) {
          return provider.name == parent_provider.name;
        });
    if (child_provider == child.provider_options.end()) {
      child.provider_options.push_back(parent_provider);
      continue;
    }
    if (!child_provider->device_filtering_options) {
      child_provider->device_filtering_options = parent_provider.device_filtering_options;
    }
    for (const auto& parent_option : parent_provider.options) {
      const bool overridden = std::any_of(
          child_provider->options.begin(), child_provider->options.end(),
          [&parent_option](const Config::NamedString& option) {
            return option.first == parent_option.first;
          });
      if (!overridden) {
        child_provider->options.push_back(parent_option);
      }
    }
  }
}

void RequireTensor(const ModelStateMetadata& metadata, const std::string& name,
                   bool input, ONNXTensorElementDataType type, size_t rank) {
  const bool present = input ? metadata.HasInput(name) : metadata.HasOutput(name);
  if (name.empty() || !present) {
    throw std::runtime_error(
        "model.dflash2 " + std::string{input ? "input '" : "output '"} + name + "' is missing.");
  }
  const auto actual_type = input ? metadata.GetInputDataType(name) : metadata.GetOutputDataType(name);
  const auto shape = input ? metadata.GetInputShape(name) : metadata.GetOutputShape(name);
  if (actual_type != type || shape.size() != rank) {
    throw std::runtime_error(
        "model.dflash2 " + std::string{input ? "input '" : "output '"} + name +
        "' has an incompatible type or rank.");
  }
}

bool DimensionMatches(int64_t actual, int expected) {
  return actual < 0 || actual == expected;
}

}  // namespace

size_t Dflash2DraftWidth(size_t capability_limit, size_t configured_limit,
                         size_t sequence_length_after_step, size_t sequence_limit,
                         size_t remaining_turn_tokens_after_step) {
  const size_t sequence_capacity =
      sequence_length_after_step < sequence_limit &&
              sequence_limit - sequence_length_after_step > 1
          ? sequence_limit - sequence_length_after_step - 1
          : 0;
  const size_t turn_capacity =
      remaining_turn_tokens_after_step > 1 ? remaining_turn_tokens_after_step - 1 : 0;
  return std::min({capability_limit, configured_limit, sequence_capacity, turn_capacity});
}

std::unique_ptr<Config> CreateDflash2Config(const Config& config) {
  const auto& dflash2 = config.model.dflash2;
  if (!config.model.mtp.filename.empty()) {
    throw std::runtime_error(
        "An Engine model cannot configure both model.mtp and model.dflash2.");
  }
  if (dflash2.filename.empty()) {
    throw std::runtime_error("model.dflash2.filename is required to create a DFlash 2 drafter.");
  }
  if (dflash2.num_hidden_layers <= 0 || dflash2.num_key_value_heads <= 0 || dflash2.head_size <= 0 ||
      dflash2.block_size <= 1 || dflash2.num_draft_tokens <= 0 || dflash2.selector_top_k <= 0) {
    throw std::runtime_error("model.dflash2 geometry must be positive and describe a block of >1 token.");
  }
  if (dflash2.num_draft_tokens != dflash2.block_size - 1) {
    throw std::runtime_error("model.dflash2.num_draft_tokens must be block_size - 1.");
  }
  if (dflash2.run_options) {
    for (const auto& [name, value] : *dflash2.run_options) {
      if (name == "disable_synchronize_execution_providers" && value == "1") {
        throw std::runtime_error(
            "model.dflash2.run_options cannot disable execution-provider synchronization.");
      }
    }
  }

  auto projected = std::make_unique<Config>(config);
  auto& decoder = projected->model.decoder;
  decoder.filename = dflash2.filename;
  if (dflash2.session_options) {
    decoder.session_options = *dflash2.session_options;
    InheritProviderOptions(config.model.decoder.session_options,
                           decoder.session_options);
  }
  decoder.run_options = dflash2.run_options;
  decoder.shared_initializers = dflash2.shared_initializers;
  decoder.num_hidden_layers = dflash2.num_hidden_layers;
  decoder.num_key_value_heads = dflash2.num_key_value_heads;
  decoder.head_size = dflash2.head_size;
  decoder.state_groups.reset();
  decoder.sliding_window.reset();
  return projected;
}

void ValidateDflash2ModelCompatibility(const Config& config,
                                       const ModelStateMetadata& target_metadata,
                                       const ModelStateMetadata& drafter_metadata) {
  const auto& dflash2 = config.model.dflash2;
  const auto& target_aux_output = dflash2.main_aux_hidden_states;
  if (target_aux_output.empty() || !target_metadata.HasOutput(target_aux_output)) {
    throw std::runtime_error(
        "model.dflash2.main_aux_hidden_states must name a main-model output.");
  }

  const auto& drafter_aux_input = dflash2.inputs.aux_hidden_states;
  if (drafter_aux_input.empty() || !drafter_metadata.HasInput(drafter_aux_input)) {
    throw std::runtime_error(
        "model.dflash2.inputs.aux_hidden_states must name a drafter-model input.");
  }

  const auto target_aux_shape = target_metadata.GetOutputShape(target_aux_output);
  const auto drafter_aux_shape = drafter_metadata.GetInputShape(drafter_aux_input);
  if (target_aux_shape.size() != 2 || target_aux_shape[1] <= 0 ||
      drafter_aux_shape.size() != 2 || drafter_aux_shape[1] <= 0 ||
      target_aux_shape[1] != drafter_aux_shape[1]) {
    throw std::runtime_error(
        "DFlash 2 requires matching 2-D auxiliary hidden-state tensors with a static width.");
  }
  if (target_metadata.GetOutputDataType(target_aux_output) !=
      drafter_metadata.GetInputDataType(drafter_aux_input)) {
    throw std::runtime_error(
        "DFlash 2 requires matching auxiliary hidden-state tensor types.");
  }

  const auto& inputs = dflash2.inputs;
  for (const auto* name : {&inputs.q_row_map, &inputs.qkv_row_map,
                           &inputs.block_row_index, &inputs.cumulative_sequence_lengths,
                           &inputs.past_sequence_lengths}) {
    RequireTensor(drafter_metadata, *name, true, Ort::TypeToTensorType<int32_t>, 1);
  }
  RequireTensor(drafter_metadata, inputs.input_ids, true,
                Ort::TypeToTensorType<int64_t>, 1);
  RequireTensor(drafter_metadata, inputs.block_table, true,
                Ort::TypeToTensorType<int32_t>, 2);
  RequireTensor(drafter_metadata, inputs.attention_metadata, true,
                Ort::TypeToTensorType<int32_t>, 1);
  const auto metadata_shape = drafter_metadata.GetInputShape(inputs.attention_metadata);
  if (!DimensionMatches(metadata_shape[0], 3)) {
    throw std::runtime_error("model.dflash2 attention_metadata must contain three values.");
  }

  RequireTensor(drafter_metadata, dflash2.outputs.candidate_ids, false,
                Ort::TypeToTensorType<int32_t>, 3);
  RequireTensor(drafter_metadata, dflash2.outputs.scores, false,
                Ort::TypeToTensorType<float>, 4);
  const auto candidate_shape = drafter_metadata.GetOutputShape(dflash2.outputs.candidate_ids);
  const auto scores_shape = drafter_metadata.GetOutputShape(dflash2.outputs.scores);
  if (!DimensionMatches(candidate_shape[1], dflash2.num_draft_tokens) ||
      !DimensionMatches(candidate_shape[2], dflash2.selector_top_k) ||
      !DimensionMatches(scores_shape[1], dflash2.num_draft_tokens) ||
      !DimensionMatches(scores_shape[2], dflash2.selector_top_k) ||
      !DimensionMatches(scores_shape[3], dflash2.selector_top_k)) {
    throw std::runtime_error("model.dflash2 selector outputs do not match the configured geometry.");
  }

  ONNXTensorElementDataType cache_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (int layer = 0; layer < dflash2.num_hidden_layers; ++layer) {
    for (const auto [input_pattern, output_pattern] : {
             std::pair{&inputs.past_key_names, &dflash2.outputs.present_key_names},
             std::pair{&inputs.past_value_names, &dflash2.outputs.present_value_names}}) {
      const auto input_name = ComposeKeyValueName(*input_pattern, layer);
      const auto output_name = ComposeKeyValueName(*output_pattern, layer);
      if (input_name.empty() || !drafter_metadata.HasInput(input_name) ||
          output_name.empty() || !drafter_metadata.HasOutput(output_name)) {
        throw std::runtime_error("model.dflash2 cache input or output is missing.");
      }
      const auto input_type = drafter_metadata.GetInputDataType(input_name);
      const auto output_type = drafter_metadata.GetOutputDataType(output_name);
      const auto input_shape = drafter_metadata.GetInputShape(input_name);
      const auto output_shape = drafter_metadata.GetOutputShape(output_name);
      if (input_shape.size() != 4 || output_shape.size() != 4 || input_type != output_type ||
          (cache_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED && input_type != cache_type) ||
          !DimensionMatches(input_shape[2], dflash2.num_key_value_heads) ||
          !DimensionMatches(input_shape[3], dflash2.head_size) ||
          !DimensionMatches(output_shape[2], dflash2.num_key_value_heads) ||
          !DimensionMatches(output_shape[3], dflash2.head_size)) {
        throw std::runtime_error("model.dflash2 cache tensors do not match the configured geometry.");
      }
      cache_type = input_type;
    }
  }
}

Dflash2Model::Dflash2Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_);
}

std::unique_ptr<State> Dflash2Model::CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const {
  throw std::logic_error("The DFlash 2 drafter is driven by the Engine and has no State.");
}

size_t Dflash2Drafter::PoolBlocks(const Config& config, size_t paged_block_size,
                                  size_t max_batch_size) {
  const auto& dflash2 = config.model.dflash2;
  if (dflash2.filename.empty()) {
    return 0;
  }
  if (dflash2.sliding_window <= 0) {
    throw std::runtime_error(
        "The Engine-hosted DFlash 2 drafter requires a sliding window; a full-attention drafter "
        "would need a cache as large as the target's.");
  }
  if (paged_block_size == 0) {
    throw std::runtime_error("The DFlash 2 paged cache block size must be positive.");
  }
  // The window bounds what a query block can ever read, so a request only needs a ring long enough
  // to hold that window plus the block itself, whatever its context length.
  const size_t positions = CheckedAdd(
      static_cast<size_t>(dflash2.sliding_window),
      CheckedMultiply(size_t{2}, static_cast<size_t>(dflash2.block_size),
                      "DFlash 2 cache ring positions"),
      "DFlash 2 cache ring positions");
  const size_t rounded_blocks = positions / paged_block_size +
                                static_cast<size_t>(positions % paged_block_size != 0);
  const size_t ring = CheckedAdd(rounded_blocks, size_t{1}, "DFlash 2 cache ring blocks");
  return CheckedMultiply(std::max(max_batch_size, size_t{1}), ring,
                         "DFlash 2 cache block count");
}

Dflash2Drafter::Dflash2Drafter(std::shared_ptr<Dflash2Model> model, size_t paged_block_size,
                               size_t num_blocks)
    : model_{std::move(model)},
      config_{model_->config_->model.dflash2},
      paged_block_size_{paged_block_size},
      num_blocks_{num_blocks} {
  if (paged_block_size_ == 0 || num_blocks_ == 0) {
    throw std::runtime_error("The DFlash 2 drafter needs a non-empty paged cache pool.");
  }
  if (config_.sliding_window > 0) {
    // Positions older than this are masked out of every query row, so they are never ingested and
    // the ring may alias them.
    context_window_ = static_cast<size_t>(config_.sliding_window) +
                      static_cast<size_t>(config_.block_size);
    ring_blocks_ =
        (context_window_ + static_cast<size_t>(config_.block_size) + paged_block_size_ - 1) /
            paged_block_size_ +
        1;
  }

  const auto& inputs = config_.inputs;
  aux_type_ = model_->session_info_.GetInputDataType(inputs.aux_hidden_states);
  const auto aux_shape = model_->session_info_.GetInputShape(inputs.aux_hidden_states);
  if (aux_shape.size() != 2 || aux_shape[1] <= 0) {
    throw std::runtime_error("model.dflash2 expects a 2-D aux_hidden_states input with a static width.");
  }
  aux_hidden_size_ = static_cast<size_t>(aux_shape[1]);

  AllocateCache();

  free_blocks_.resize(num_blocks_);
  std::iota(free_blocks_.rbegin(), free_blocks_.rend(), int32_t{0});

  run_options_ = OrtRunOptions::Create();
  if (model_->config_->model.decoder.run_options) {
    for (const auto& entry : *model_->config_->model.decoder.run_options) {
      if (entry.first != "gpu_graph_id") {
        run_options_->AddConfigEntry(entry.first.c_str(), entry.second.c_str());
      }
    }
  }
  // Proposal tensors are rebuilt for every packed batch, so their addresses are not capturable.
  run_options_->AddConfigEntry("gpu_graph_id", "-1");
}

void Dflash2Drafter::AllocateCache() {
  const size_t layers = static_cast<size_t>(config_.num_hidden_layers);
  const std::vector<int64_t> shape{static_cast<int64_t>(num_blocks_),
                                   static_cast<int64_t>(paged_block_size_),
                                   config_.num_key_value_heads,
                                   config_.head_size};
  cache_type_ = model_->session_info_.GetInputDataType(
      ComposeKeyValueName(config_.inputs.past_key_names, 0));
  for (size_t layer = 0; layer < layers; ++layer) {
    for (const auto* pattern : {&config_.inputs.past_key_names, &config_.inputs.past_value_names}) {
      cache_input_names_.push_back(ComposeKeyValueName(*pattern, static_cast<int>(layer)));
      auto cache = std::make_unique<Tensor>(model_->p_device_kvcache_, cache_type_);
      cache->CreateTensor(shape);
      cache->GetByteSpan().Zero();
      caches_.push_back(std::move(cache));
    }
    cache_output_names_.push_back(ComposeKeyValueName(config_.outputs.present_key_names, static_cast<int>(layer)));
    cache_output_names_.push_back(ComposeKeyValueName(config_.outputs.present_value_names, static_cast<int>(layer)));
  }
}

bool Dflash2Drafter::Admit(const Feed& feed) {
  if (requests_.find(feed.request) != requests_.end()) {
    return true;
  }
  // The drafter cannot backfill K/V for context whose auxiliary hidden states have already been
  // consumed, so a request can only join from the start of its sequence. Requests that arrive when
  // the ring pool is full decode without DFlash 2 drafts instead of taking the whole drafter down.
  if (feed.first_position != 0) {
    return false;
  }
  if (ring_blocks_ != 0) {
    if (free_blocks_.size() < ring_blocks_) {
      return false;
    }
    // Claim the whole ring now so a second new request in the same step sees the smaller pool.
    EnsureBlocks(requests_[feed.request], 0);
    return true;
  }
  if (free_blocks_.empty()) {
    return false;
  }
  requests_.emplace(feed.request, RequestState{});
  return true;
}

void Dflash2Drafter::EnsureBlocks(RequestState& state, size_t positions) {
  const size_t needed =
      ring_blocks_ != 0 ? ring_blocks_ : (positions + paged_block_size_ - 1) / paged_block_size_;
  while (state.blocks.size() < needed) {
    if (free_blocks_.empty()) {
      throw std::runtime_error("The DFlash 2 drafter's paged cache pool is exhausted.");
    }
    state.blocks.push_back(free_blocks_.back());
    free_blocks_.pop_back();
  }
}

void Dflash2Drafter::Release(const Request* request) {
  auto entry = requests_.find(request);
  if (entry == requests_.end()) {
    return;
  }
  free_blocks_.insert(free_blocks_.end(), entry->second.blocks.begin(), entry->second.blocks.end());
  requests_.erase(entry);
}

void Dflash2Drafter::Propose(Tensor& aux_hidden_states, std::span<const Feed> feeds,
                             std::vector<std::vector<int32_t>>& drafts) {
  drafts.assign(feeds.size(), {});
  if (feeds.empty()) {
    return;
  }

  const size_t block_size = static_cast<size_t>(config_.block_size);
  const size_t num_spec = static_cast<size_t>(config_.num_draft_tokens);
  const size_t top_k = static_cast<size_t>(config_.selector_top_k);

  // Only requests the ring pool can hold take part; the rest keep decoding without drafts.
  std::vector<size_t> served;
  served.reserve(feeds.size());
  for (size_t i = 0; i < feeds.size(); ++i) {
    if (Admit(feeds[i])) {
      served.push_back(i);
    }
  }
  if (served.empty()) {
    return;
  }

  // Batch layout. Every served feed contributes its context rows so the drafter cache never
  // develops a hole; only the feeds that asked also contribute a query block.
  std::vector<size_t> block_feed_indices;
  for (const size_t i : served) {
    if (feeds[i].wants_drafts) {
      block_feed_indices.push_back(i);
    }
  }
  // The graph reshapes the query rows to [batch, block_size, hidden], so it needs at least one
  // block. When nothing is eligible, borrow the first served feed's slot and drop its lattice: the
  // block rows only write scratch K/V at positions a later step overwrites.
  const bool drafts_wanted = !block_feed_indices.empty();
  if (!drafts_wanted) {
    block_feed_indices.push_back(served.front());
  }
  std::vector<size_t> block_slot_of_feed(feeds.size(), block_feed_indices.size());
  for (size_t slot = 0; slot < block_feed_indices.size(); ++slot) {
    block_slot_of_feed[block_feed_indices[slot]] = slot;
  }

  const size_t num_block_rows =
      CheckedMultiply(block_feed_indices.size(), block_size, "DFlash 2 block rows");
  CheckedMetadataValue(num_block_rows, "DFlash 2 block rows");
  size_t num_ctx_rows = 0;
  for (const size_t i : served) {
    num_ctx_rows = CheckedAdd(num_ctx_rows, feeds[i].aux_row_count, "DFlash 2 context rows");
  }

  PackedLayout layout;
  const size_t reserved_rows =
      CheckedAdd(num_ctx_rows, num_block_rows, "DFlash 2 packed rows");
  CheckedMetadataValue(reserved_rows, "DFlash 2 packed rows");
  layout.q_row_map.reserve(reserved_rows);
  layout.qkv_row_map.reserve(reserved_rows);
  layout.block_row_index.reserve(num_block_rows);
  layout.cumulative_sequence_lengths.reserve(served.size() + 1);
  layout.past_sequence_lengths.reserve(served.size());

  // Rows this step actually ingests, after a windowed drafter drops the ones its query block can
  // never read.
  std::vector<size_t> ingest_begin(feeds.size());
  std::vector<size_t> ingest_count(feeds.size());
  size_t ctx_row = 0;
  size_t max_blocks = 0;
  num_ctx_rows = 0;
  for (const size_t i : served) {
    const auto& feed = feeds[i];
    auto& state = requests_[feed.request];
    if (state.cached_positions != feed.first_position) {
      throw std::logic_error(
          "The DFlash 2 drafter's cached context is not contiguous with the target's step.");
    }

    size_t dropped = 0;
    if (context_window_ != 0 && feed.aux_row_count > context_window_) {
      dropped = feed.aux_row_count - context_window_;
    }
    ingest_begin[i] = CheckedAdd(feed.aux_row_begin, dropped, "DFlash 2 auxiliary row offset");
    ingest_count[i] = feed.aux_row_count - dropped;
    num_ctx_rows = CheckedAdd(num_ctx_rows, ingest_count[i], "DFlash 2 context rows");
  }
  CheckedMetadataValue(
      CheckedAdd(num_ctx_rows, num_block_rows, "DFlash 2 packed rows"),
      "DFlash 2 packed rows");

  for (const size_t i : served) {
    const auto& feed = feeds[i];
    auto& state = requests_[feed.request];
    const size_t first_position = CheckedAdd(
        feed.first_position, feed.aux_row_count - ingest_count[i],
        "DFlash 2 first position");

    const size_t slot = block_slot_of_feed[i];
    const bool has_block = slot < block_feed_indices.size();
    const size_t block_rows = has_block ? block_size : 0;
    const size_t query_len =
        CheckedAdd(ingest_count[i], block_rows, "DFlash 2 query length");
    if (query_len == 0) {
      throw std::logic_error("A DFlash 2 feed carries neither context nor a query block.");
    }
    const size_t total_positions =
        CheckedAdd(first_position, query_len, "DFlash 2 KV length");
    EnsureBlocks(state, total_positions);
    max_blocks = std::max(
        max_blocks, (total_positions - 1) / paged_block_size_ + 1);

    // Context rows borrow the block's first query row; their attention output is discarded.
    const size_t first_block_row =
        CheckedMultiply(slot, block_size, "DFlash 2 block row index");
    const int32_t borrowed_q_row = has_block
                                       ? CheckedMetadataValue(first_block_row, "DFlash 2 query row")
                                       : 0;
    for (size_t row = 0; row < ingest_count[i]; ++row) {
      layout.q_row_map.push_back(borrowed_q_row);
      layout.qkv_row_map.push_back(CheckedMetadataValue(
          CheckedAdd(num_block_rows, CheckedAdd(ctx_row, row, "DFlash 2 context row"),
                     "DFlash 2 QKV row"),
          "DFlash 2 QKV row"));
    }
    for (size_t row = 0; row < block_rows; ++row) {
      layout.block_row_index.push_back(
          CheckedMetadataValue(layout.q_row_map.size(), "DFlash 2 packed row"));
      const int32_t block_row = CheckedMetadataValue(
          CheckedAdd(first_block_row, row, "DFlash 2 block row"), "DFlash 2 block row");
      layout.q_row_map.push_back(block_row);
      layout.qkv_row_map.push_back(block_row);
    }
    ctx_row = CheckedAdd(ctx_row, ingest_count[i], "DFlash 2 context row");

    layout.cumulative_sequence_lengths.push_back(
        CheckedMetadataValue(layout.q_row_map.size(), "DFlash 2 cumulative sequence length"));
    layout.past_sequence_lengths.push_back(
        CheckedMetadataValue(first_position, "DFlash 2 past sequence length"));
    const int32_t query_length = CheckedMetadataValue(query_len, "DFlash 2 query length");
    const int32_t kv_length = CheckedMetadataValue(total_positions, "DFlash 2 KV length");
    layout.max_query_len = std::max(layout.max_query_len, query_length);
    layout.max_kv_len = std::max(layout.max_kv_len, kv_length);
    layout.min_kv_len = std::min(layout.min_kv_len, kv_length);

    state.cached_positions = CheckedAdd(
        feed.first_position, feed.aux_row_count, "DFlash 2 cached positions");
  }

  const size_t num_tokens = layout.q_row_map.size();
  auto device = model_->p_device_inputs_;

  auto make = [&](ONNXTensorElementDataType type, std::vector<int64_t> shape) {
    auto tensor = std::make_unique<Tensor>(device, type);
    tensor->CreateTensor(shape);
    return tensor;
  };
  auto fill_int32 = [](Tensor& tensor, const std::vector<int32_t>& values) {
    auto span = tensor.GetDeviceSpan<int32_t>();
    std::copy(values.begin(), values.end(), span.CpuSpan().begin());
    span.CopyCpuToDevice();
  };

  auto packed_aux = make(aux_type_, {static_cast<int64_t>(num_ctx_rows),
                                     static_cast<int64_t>(aux_hidden_size_)});
  const size_t aux_row_bytes =
      CheckedMultiply(aux_hidden_size_, Ort::SizeOf(aux_type_), "DFlash 2 auxiliary row bytes");
  auto source_bytes = aux_hidden_states.GetByteSpan();
  auto destination_bytes = packed_aux->GetByteSpan();
  size_t destination_row = 0;
  for (const size_t i : served) {
    if (ingest_count[i] == 0) {
      continue;
    }
    destination_bytes.subspan(destination_row * aux_row_bytes, ingest_count[i] * aux_row_bytes)
        .CopyFrom(source_bytes.subspan(ingest_begin[i] * aux_row_bytes,
                                       ingest_count[i] * aux_row_bytes));
    destination_row += ingest_count[i];
  }

  auto input_ids = make(Ort::TypeToTensorType<int64_t>, {static_cast<int64_t>(num_block_rows)});
  {
    auto span = input_ids->GetDeviceSpan<int64_t>();
    auto cpu = span.CpuSpan();
    for (size_t slot = 0; slot < block_feed_indices.size(); ++slot) {
      const auto& feed = feeds[block_feed_indices[slot]];
      cpu[slot * block_size] = feed.wants_drafts ? feed.anchor_token : config_.mask_token_id;
      for (size_t row = 1; row < block_size; ++row) {
        cpu[slot * block_size + row] = config_.mask_token_id;
      }
    }
    span.CopyCpuToDevice();
  }

  auto q_row_map = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(num_tokens)});
  fill_int32(*q_row_map, layout.q_row_map);
  auto qkv_row_map = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(num_tokens)});
  fill_int32(*qkv_row_map, layout.qkv_row_map);
  auto block_row_index = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(num_block_rows)});
  fill_int32(*block_row_index, layout.block_row_index);
  auto cumulative = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(served.size() + 1)});
  fill_int32(*cumulative, layout.cumulative_sequence_lengths);
  auto past_lengths = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(served.size())});
  fill_int32(*past_lengths, layout.past_sequence_lengths);

  auto block_table = make(Ort::TypeToTensorType<int32_t>,
                          {static_cast<int64_t>(served.size()), static_cast<int64_t>(max_blocks)});
  {
    auto span = block_table->GetDeviceSpan<int32_t>();
    auto cpu = span.CpuSpan();
    std::fill(cpu.begin(), cpu.end(), int32_t{-1});
    for (size_t row = 0; row < served.size(); ++row) {
      const auto& blocks = requests_[feeds[served[row]].request].blocks;
      if (ring_blocks_ == 0) {
        std::copy(blocks.begin(), blocks.end(), cpu.begin() + row * max_blocks);
        continue;
      }
      // A windowed drafter repeats its ring across every column: column j holds the block that
      // owns position j * block_size, which is ring[j % ring_blocks].
      for (size_t column = 0; column < max_blocks; ++column) {
        cpu[row * max_blocks + column] = blocks[column % blocks.size()];
      }
    }
    span.CopyCpuToDevice();
  }

  auto metadata = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU),
                                           Ort::TypeToTensorType<int32_t>);
  metadata->CreateTensor(std::vector<int64_t>{3});
  {
    auto span = metadata->GetDeviceSpan<int32_t>();
    auto cpu = span.CpuSpan();
    cpu[0] = layout.max_query_len;
    cpu[1] = layout.max_kv_len;
    cpu[2] = layout.min_kv_len;
  }

  const size_t batch = block_feed_indices.size();
  auto candidate_ids = make(Ort::TypeToTensorType<int32_t>,
                            {static_cast<int64_t>(batch), static_cast<int64_t>(num_spec),
                             static_cast<int64_t>(top_k)});
  auto scores = make(Ort::TypeToTensorType<float>,
                     {static_cast<int64_t>(batch), static_cast<int64_t>(num_spec),
                      static_cast<int64_t>(top_k), static_cast<int64_t>(top_k)});

  std::vector<const char*> input_names{
      config_.inputs.aux_hidden_states.c_str(), config_.inputs.input_ids.c_str(),
      config_.inputs.q_row_map.c_str(), config_.inputs.qkv_row_map.c_str(),
      config_.inputs.block_row_index.c_str(), config_.inputs.cumulative_sequence_lengths.c_str(),
      config_.inputs.past_sequence_lengths.c_str(), config_.inputs.block_table.c_str(),
      config_.inputs.attention_metadata.c_str()};
  std::vector<OrtValue*> inputs{packed_aux->GetOrtTensor(), input_ids->GetOrtTensor(),
                                q_row_map->GetOrtTensor(), qkv_row_map->GetOrtTensor(),
                                block_row_index->GetOrtTensor(), cumulative->GetOrtTensor(),
                                past_lengths->GetOrtTensor(), block_table->GetOrtTensor(),
                                metadata->GetOrtTensor()};
  std::vector<const char*> output_names{config_.outputs.candidate_ids.c_str(),
                                        config_.outputs.scores.c_str()};
  std::vector<OrtValue*> outputs{candidate_ids->GetOrtTensor(), scores->GetOrtTensor()};
  for (size_t i = 0; i < caches_.size(); ++i) {
    input_names.push_back(cache_input_names_[i].c_str());
    inputs.push_back(caches_[i]->GetOrtTensor());
    output_names.push_back(cache_output_names_[i].c_str());
    outputs.push_back(caches_[i]->GetOrtTensor());
  }

  model_->session_->Run(run_options_.get(), input_names.data(), inputs.data(), input_names.size(),
                        output_names.data(), outputs.data(), output_names.size());

  if (!drafts_wanted) {
    return;
  }

  // The spans own the host mirrors these point into, so they must outlive the reads below.
  auto candidate_span = candidate_ids->GetDeviceSpan<int32_t>();
  auto scores_span = scores->GetDeviceSpan<float>();
  auto candidate_cpu = candidate_span.CopyDeviceToCpu();
  auto scores_cpu = scores_span.CopyDeviceToCpu();
  for (size_t slot = 0; slot < block_feed_indices.size(); ++slot) {
    // Greedy walk of the lattice: slot l's chosen candidate index selects the row of slot l+1's
    // score matrix, so the drafted block is one coherent path rather than seven independent argmaxes.
    auto& out = drafts[block_feed_indices[slot]];
    out.reserve(num_spec);
    size_t previous = 0;
    for (size_t step = 0; step < num_spec; ++step) {
      const float* row = scores_cpu.data() + ((slot * num_spec + step) * top_k + previous) * top_k;
      const size_t best = static_cast<size_t>(std::max_element(row, row + top_k) - row);
      out.push_back(candidate_cpu[(slot * num_spec + step) * top_k + best]);
      previous = best;
    }
  }
}

}  // namespace Generators
