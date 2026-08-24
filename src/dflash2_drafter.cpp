// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "dflash2_drafter.h"
#include "engine/request.h"
#include "models/kv_cache.h"

#include <algorithm>
#include <limits>
#include <numeric>

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

}  // namespace

std::unique_ptr<Config> CreateDflash2Config(const Config& config) {
  const auto& dflash2 = config.model.dflash2;
  if (dflash2.filename.empty()) {
    throw std::runtime_error("model.dflash2.filename is required to create a DFlash 2 drafter.");
  }
  if (dflash2.num_hidden_layers <= 0 || dflash2.num_key_value_heads <= 0 || dflash2.head_size <= 0 ||
      dflash2.block_size <= 1 || dflash2.num_draft_tokens <= 0) {
    throw std::runtime_error("model.dflash2 geometry must be positive and describe a block of >1 token.");
  }
  if (dflash2.num_draft_tokens > dflash2.block_size || dflash2.num_draft_tokens < dflash2.block_size - 1) {
    // DFlash 2's row 0 carries the anchor and predicts nothing, so its block is one row wider
    // than its draft count; every DSpark row predicts, so the two are equal.
    throw std::runtime_error("model.dflash2.num_draft_tokens must be block_size or block_size - 1.");
  }

  auto projected = std::make_unique<Config>(config);
  auto& decoder = projected->model.decoder;
  decoder.filename = dflash2.filename;
  if (dflash2.session_options) {
    decoder.session_options = *dflash2.session_options;
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

Dflash2Model::Dflash2Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_);
}

std::unique_ptr<State> Dflash2Model::CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const {
  throw std::logic_error("The DFlash 2 drafter is driven by the Engine and has no State.");
}

size_t Dflash2Drafter::BytesPerBlock(const Config& config, size_t paged_block_size) {
  const auto& dflash2 = config.model.dflash2;
  if (dflash2.filename.empty()) {
    return 0;
  }
  // K and V, for every layer, for every slot in a block. The cache element width follows the
  // drafter body, which is bfloat16.
  return size_t{2} * static_cast<size_t>(dflash2.num_hidden_layers) * paged_block_size *
         static_cast<size_t>(dflash2.num_key_value_heads) * static_cast<size_t>(dflash2.head_size) *
         sizeof(uint16_t);
}

size_t Dflash2Drafter::PoolBlocks(const Config& config, size_t paged_block_size,
                                 size_t max_batch_size) {
  const auto& dflash2 = config.model.dflash2;
  if (dflash2.filename.empty() || dflash2.sliding_window <= 0) {
    // A full-attention drafter keeps the whole sequence, so its pool is sized against the main
    // cache's block count instead (engine.cpp budgets it through auxiliary_bytes_per_block).
    return 0;
  }
  // The window bounds what a query block can ever read, so a request only needs a ring long enough
  // to hold that window plus the block itself, whatever its context length.
  const size_t positions = static_cast<size_t>(dflash2.sliding_window) +
                           2 * static_cast<size_t>(dflash2.block_size);
  const size_t ring = (positions + paged_block_size - 1) / paged_block_size + 1;
  return std::max(max_batch_size, size_t{1}) * ring;
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
      run_options_->AddConfigEntry(entry.first.c_str(), entry.second.c_str());
    }
  }
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
      try {
        cache->CreateTensor(shape);
      } catch (const std::exception& error) {
        // A full-attention drafter mirrors the main pool block for block, so this is the usual
        // symptom of a gpu_utilization_factor that leaves no room for the second cache.
        throw std::runtime_error(
            std::string("The block drafter could not allocate its paged cache (") +
            std::to_string(num_blocks_) + " blocks x " +
            std::to_string(BytesPerBlock(*model_->config_, paged_block_size_) / config_.num_hidden_layers / 2) +
            " bytes per layer-cache): " + error.what() +
            ". Lower engine.dynamic_batching.gpu_utilization_factor.");
      }
      cache->GetByteSpan().Zero();
      caches_.push_back(std::move(cache));
    }
    cache_output_names_.push_back(ComposeKeyValueName(config_.outputs.present_key_names, static_cast<int>(layer)));
    cache_output_names_.push_back(ComposeKeyValueName(config_.outputs.present_value_names, static_cast<int>(layer)));
  }
}

Dflash2Drafter::RequestState& Dflash2Drafter::StateFor(const Request* request) {
  return requests_[request];
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

  // Batch layout. Every feed contributes its context rows so the drafter cache never develops a
  // hole; only the feeds that asked also contribute a query block.
  std::vector<size_t> block_feed_indices;
  for (size_t i = 0; i < feeds.size(); ++i) {
    if (feeds[i].wants_drafts) {
      block_feed_indices.push_back(i);
    }
  }
  // The graph reshapes the query rows to [batch, block_size, hidden], so it needs at least one
  // block. When nothing is eligible, borrow the first feed's slot and drop its lattice: the block
  // rows only write scratch K/V at positions a later step overwrites.
  const bool drafts_wanted = !block_feed_indices.empty();
  if (!drafts_wanted) {
    block_feed_indices.push_back(0);
  }
  std::vector<size_t> block_slot_of_feed(feeds.size(), block_feed_indices.size());
  for (size_t slot = 0; slot < block_feed_indices.size(); ++slot) {
    block_slot_of_feed[block_feed_indices[slot]] = slot;
  }

  const size_t num_block_rows = block_feed_indices.size() * block_size;
  size_t num_ctx_rows = 0;
  for (const auto& feed : feeds) {
    num_ctx_rows += feed.aux_row_count;
  }

  PackedLayout layout;
  layout.q_row_map.reserve(num_ctx_rows + num_block_rows);
  layout.qkv_row_map.reserve(num_ctx_rows + num_block_rows);
  layout.block_row_index.reserve(num_block_rows);
  layout.cumulative_sequence_lengths.reserve(feeds.size() + 1);
  layout.past_sequence_lengths.reserve(feeds.size());

  // Rows this step actually ingests, after a windowed drafter drops the ones its query block can
  // never read.
  std::vector<size_t> ingest_begin(feeds.size());
  std::vector<size_t> ingest_count(feeds.size());
  size_t ctx_row = 0;
  size_t max_blocks = 0;
  num_ctx_rows = 0;
  for (size_t i = 0; i < feeds.size(); ++i) {
    const auto& feed = feeds[i];
    auto& state = StateFor(feed.request);
    if (state.cached_positions != feed.first_position) {
      throw std::logic_error(
          "The DFlash 2 drafter's cached context is not contiguous with the target's step.");
    }

    size_t dropped = 0;
    if (context_window_ != 0 && feed.aux_row_count > context_window_) {
      dropped = feed.aux_row_count - context_window_;
    }
    ingest_begin[i] = feed.aux_row_begin + dropped;
    ingest_count[i] = feed.aux_row_count - dropped;
    num_ctx_rows += ingest_count[i];
  }

  for (size_t i = 0; i < feeds.size(); ++i) {
    const auto& feed = feeds[i];
    auto& state = requests_[feed.request];
    const size_t first_position = feed.first_position + (feed.aux_row_count - ingest_count[i]);

    const size_t slot = block_slot_of_feed[i];
    const bool has_block = slot < block_feed_indices.size();
    const size_t block_rows = has_block ? block_size : 0;
    const size_t query_len = ingest_count[i] + block_rows;
    if (query_len == 0) {
      throw std::logic_error("A DFlash 2 feed carries neither context nor a query block.");
    }
    const size_t total_positions = first_position + query_len;
    EnsureBlocks(state, total_positions);
    max_blocks = std::max(max_blocks,
                          (total_positions + paged_block_size_ - 1) / paged_block_size_);

    // Context rows borrow the block's first query row; their attention output is discarded.
    const int32_t borrowed_q_row = has_block ? static_cast<int32_t>(slot * block_size) : 0;
    for (size_t row = 0; row < ingest_count[i]; ++row) {
      layout.q_row_map.push_back(borrowed_q_row);
      layout.qkv_row_map.push_back(static_cast<int32_t>(num_block_rows + ctx_row + row));
    }
    for (size_t row = 0; row < block_rows; ++row) {
      layout.block_row_index.push_back(static_cast<int32_t>(layout.q_row_map.size()));
      layout.q_row_map.push_back(static_cast<int32_t>(slot * block_size + row));
      layout.qkv_row_map.push_back(static_cast<int32_t>(slot * block_size + row));
    }
    ctx_row += ingest_count[i];

    layout.cumulative_sequence_lengths.push_back(static_cast<int32_t>(layout.q_row_map.size()));
    layout.past_sequence_lengths.push_back(static_cast<int32_t>(first_position));
    layout.max_query_len = std::max(layout.max_query_len, static_cast<int32_t>(query_len));
    layout.max_kv_len = std::max(layout.max_kv_len, static_cast<int32_t>(total_positions));
    layout.min_kv_len = std::min(layout.min_kv_len, static_cast<int32_t>(total_positions));

    state.cached_positions = feed.first_position + feed.aux_row_count;
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
  const size_t aux_row_bytes = aux_hidden_size_ * Ort::SizeOf(aux_type_);
  auto source_bytes = aux_hidden_states.GetByteSpan();
  auto destination_bytes = packed_aux->GetByteSpan();
  size_t destination_row = 0;
  for (size_t i = 0; i < feeds.size(); ++i) {
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
  auto cumulative = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(feeds.size() + 1)});
  fill_int32(*cumulative, layout.cumulative_sequence_lengths);
  auto past_lengths = make(Ort::TypeToTensorType<int32_t>, {static_cast<int64_t>(feeds.size())});
  fill_int32(*past_lengths, layout.past_sequence_lengths);

  auto block_table = make(Ort::TypeToTensorType<int32_t>,
                          {static_cast<int64_t>(feeds.size()), static_cast<int64_t>(max_blocks)});
  {
    auto span = block_table->GetDeviceSpan<int32_t>();
    auto cpu = span.CpuSpan();
    std::fill(cpu.begin(), cpu.end(), int32_t{-1});
    for (size_t i = 0; i < feeds.size(); ++i) {
      const auto& blocks = requests_[feeds[i].request].blocks;
      if (ring_blocks_ == 0) {
        std::copy(blocks.begin(), blocks.end(), cpu.begin() + i * max_blocks);
        continue;
      }
      // A windowed drafter repeats its ring across every column: column j holds the block that
      // owns position j * block_size, which is ring[j % ring_blocks].
      for (size_t column = 0; column < max_blocks; ++column) {
        cpu[i * max_blocks + column] = blocks[column % blocks.size()];
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
      config_.inputs.q_row_map.c_str(),         config_.inputs.qkv_row_map.c_str(),
      config_.inputs.block_row_index.c_str(),   config_.inputs.cumulative_sequence_lengths.c_str(),
      config_.inputs.past_sequence_lengths.c_str(), config_.inputs.block_table.c_str(),
      config_.inputs.attention_metadata.c_str()};
  std::vector<OrtValue*> inputs{packed_aux->GetOrtTensor(),  input_ids->GetOrtTensor(),
                                q_row_map->GetOrtTensor(),   qkv_row_map->GetOrtTensor(),
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
