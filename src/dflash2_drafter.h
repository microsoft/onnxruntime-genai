// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "models/model.h"

namespace Generators {

struct Request;

size_t Dflash2DraftWidth(size_t capability_limit, size_t configured_limit,
                         size_t sequence_length_after_step, size_t sequence_limit,
                         size_t remaining_turn_tokens_after_step);

/**
 * @brief Hosts the ``dflash2.onnx`` session.
 *
 * The drafter graph is not decoder-shaped, so this only borrows Model for its session options,
 * shared initializers and device interfaces. It never produces a State.
 */
struct Dflash2Model : Model {
  Dflash2Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override;

  std::unique_ptr<OrtSession> session_;
};

// Decoder-shaped view of model.dflash2, so Model's session-option and shared-initializer plumbing
// applies to the drafter session unchanged.
std::unique_ptr<Config> CreateDflash2Config(const Config& config);

// Validates the auxiliary hidden-state tensor passed from the target to the drafter.
void ValidateDflash2ModelCompatibility(const Config& config,
                                       const ModelStateMetadata& target_metadata,
                                       const ModelStateMetadata& drafter_metadata);

/**
 * @brief Runs the DFlash 2 block drafter for one engine step.
 *
 * DFlash 2 never re-runs the target's layers. Each step it turns the target's auxiliary hidden
 * states into per-layer K/V for the tokens the target just committed, writes them into its own
 * paged cache, and then attends a block of `block_size` query rows (the committed token plus
 * `num_draft_tokens` mask tokens) over that cache. The block rows produce a lattice -- top-k
 * candidates per slot plus pairwise edge scores -- which is walked greedily on the host.
 *
 * Context rows and query rows travel through one packed PagedAttention call: `qkv_row_map` picks
 * each packed row's K/V out of `concat(query, context)` and the context rows' attention output is
 * discarded. That is also what fills the cache, so there is no separate cache-store op.
 */
struct Dflash2Drafter {
  /**
   * @brief One request's contribution to a step.
   *
   * Rows [aux_row_begin, aux_row_begin + aux_row_count) of the target's packed auxiliary hidden
   * states hold positions [first_position, first_position + aux_row_count). Rejected draft rows
   * are excluded by the caller, so every row named here is committed context.
   */
  struct Feed {
    Request* request{};
    size_t aux_row_begin{};
    size_t aux_row_count{};
    size_t first_position{};
    int32_t anchor_token{};
    bool wants_drafts{};
  };

  Dflash2Drafter(std::shared_ptr<Dflash2Model> model, size_t paged_block_size, size_t num_blocks);

  // Blocks the pool needs for `max_batch_size` concurrent requests. The drafter is windowed, so a
  // request only needs a fixed ring however long its context grows.
  static size_t PoolBlocks(const Config& config, size_t paged_block_size, size_t max_batch_size);

  size_t NumDraftTokens() const { return static_cast<size_t>(config_.num_draft_tokens); }
  size_t AdmissionMisses() const { return admission_misses_; }

  /**
   * @brief Ingests every served feed's context and drafts for the ones that asked.
   * @param aux_hidden_states The target's packed [token_count, aux_hidden_size] output.
   * @param drafts Resized to feeds.size(); entry i is empty unless feeds[i] was served and asked.
   *
   * A request joins the drafter on its first feed, which must start at position zero, and keeps its
   * ring until Release. Requests that arrive once the ring pool is full are skipped for good rather
   * than failing the step, so they decode without DFlash 2 drafts.
   */
  void Propose(Tensor& aux_hidden_states, std::span<const Feed> feeds,
               std::vector<std::vector<int32_t>>& drafts);

  // Returns a request's blocks to the pool. Safe for requests the drafter never saw.
  void Release(const Request* request);

  // Drops every tracked request's cache state and returns its blocks. Used after a recoverable
  // proposal failure: the drafter's cached context is no longer contiguous with the target for any
  // in-flight request, and a request can only rejoin from position zero, so those requests finish
  // without drafts while requests admitted later still get them.
  void ReleaseAll();

 private:
  struct RequestState {
    std::vector<int32_t> blocks;
    size_t cached_positions{};  // Positions [0, cached_positions) hold committed context K/V.
  };

  // Whether the drafter can carry this feed's request, admitting it to the pool when it can.
  bool Admit(const Feed& feed);
  // Grows a request's block list so positions [0, positions) are addressable. A windowed drafter
  // gets a fixed ring instead, which its block table repeats across every column.
  void EnsureBlocks(RequestState& state, size_t positions);
  void AllocateCache();
  // Reshapes a reused proposal tensor, replacing its buffer only when a step needs more room.
  static Tensor& StepTensor(std::unique_ptr<Tensor>& slot, DeviceInterface* device,
                            ONNXTensorElementDataType type, const std::vector<int64_t>& shape);

  std::shared_ptr<Dflash2Model> model_;
  const Config::Model::Dflash2& config_;
  size_t paged_block_size_{};
  size_t num_blocks_{};
  // Context positions the drafter must keep behind the query block. Zero when it is not windowed,
  // in which case the whole sequence stays resident.
  size_t context_window_{};
  size_t ring_blocks_{};
  size_t aux_hidden_size_{};
  ONNXTensorElementDataType aux_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  ONNXTensorElementDataType cache_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};

  std::vector<std::unique_ptr<Tensor>> caches_;  // 2 * num_hidden_layers, key then value per layer
  std::vector<std::string> cache_input_names_, cache_output_names_;
  // Proposal tensors, reused across steps and grown to the high-water mark of the batches served
  // so far, so a steady-state decode step allocates none of them.
  struct StepTensors {
    std::unique_ptr<Tensor> packed_aux;
    std::unique_ptr<Tensor> input_ids;
    std::unique_ptr<Tensor> q_row_map;
    std::unique_ptr<Tensor> qkv_row_map;
    std::unique_ptr<Tensor> block_row_index;
    std::unique_ptr<Tensor> cumulative_sequence_lengths;
    std::unique_ptr<Tensor> past_sequence_lengths;
    std::unique_ptr<Tensor> block_table;
    std::unique_ptr<Tensor> attention_metadata;
    std::unique_ptr<Tensor> candidate_ids;
    std::unique_ptr<Tensor> scores;
  } step_tensors_;
  std::vector<int32_t> free_blocks_;
  std::unordered_map<const Request*, RequestState> requests_;
  size_t admission_misses_{};

  std::unique_ptr<OrtRunOptions> run_options_;
};

}  // namespace Generators
