// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>

#include "decoder.h"
#include "../step_plan.h"
#include "../../models/decoder_only.h"

namespace Generators {

struct AttentionMetadataValues {
  int32_t max_query_len_bound{};
  int32_t max_kv_len_bound{};
  int32_t max_kv_len_lower_bound{};
};

inline constexpr size_t kAttentionMetadataElementCount = 3;

void ValidatePackedPositionIdsInput(
    ONNXTensorElementDataType data_type,
    std::span<const int64_t> shape,
    std::span<const char* const> symbolic_shape = {});

AttentionMetadataValues GetAttentionMetadataForPlan(const StepPlan& plan);
AttentionMetadataValues GetAttentionMetadataForGraph(size_t block_table_columns, size_t block_size);
AttentionMetadataValues GetAttentionMetadataForGraphStep(
    const StepPlan& plan, size_t block_table_columns, size_t block_size);
std::array<int32_t, kAttentionMetadataElementCount> PackAttentionMetadata(
    const AttentionMetadataValues& metadata);

/**
 * @struct VarlenGraphBuffers
 * @brief Fixed-address input and output buffers for capturable decode steps.
 *
 * CUDA graph replay re-issues the launches recorded at capture time together with the device
 * pointers they were given, and it never re-runs the operators' host-side code. Every buffer the
 * captured step reads or writes must therefore stay at the same address for the life of the graph,
 * which the per-step tensors of an ordinary decode step do not. This holder outlives the per-step
 * VarlenDecoderIO and hands it the same allocations every time.
 *
 * The buffers are sized once for the largest batch the engine will schedule; a given step views a
 * smaller prefix of them. Steps that differ in shape are captured under different annotation ids.
 */
struct VarlenGraphBuffers {
  VarlenGraphBuffers(DecoderOnly_Model& model);

  // Annotation id for a decode step of this shape. Distinct (batch, block table columns) pairs need
  // distinct graphs because the captured launches bake in grid dimensions derived from both.
  static int GraphId(size_t batch_size, size_t block_table_columns);

  bool Fits(size_t batch_size) const { return batch_size <= max_batch_size; }

  std::unique_ptr<Tensor> input_ids;
  std::unique_ptr<Tensor> cumulative_sequence_lengths;
  std::unique_ptr<Tensor> past_sequence_lengths;
  std::unique_ptr<Tensor> logits;
  // Null unless the model consumes hidden_states (for example, an MTP head).
  std::unique_ptr<Tensor> hidden_states_input;
  // Null unless the model was exported with include_hidden_states.
  std::unique_ptr<Tensor> hidden_states;
  size_t max_batch_size{};
};

/**
 * @class VarlenDecoderIO
 * @brief Prepares and manages the inputs and outputs for a variable-length decoder model.
 *
 * A variable-length decoder model is one that can handle input sequences of varying lengths
 * within the same batch. This class handles the preparation of the following input and output tensors:
 * Inputs:
 * - Input IDs - int64[total_num_tokens]
 * - Cumulative Sequence Lengths - int32[batch_size + 1]
 * - Past Sequence Lengths - int32[batch_size]
 * - Attention Metadata - int32[3] (CPU), optional
 * Outputs:
 * - Logits - float16/float32[batch_size, vocab_size] for one row per request, or
 *   float16/float32[total_num_tokens, vocab_size] for one row per packed token.
 * - Hidden States - float16/float32[total_num_tokens, hidden_size], only when the model was
 *   exported with include_hidden_states. This is what an MTP draft head consumes.
 *
 * The inputs prepared by this class are compatible with models that use the
 * PagedAttention operator.
 */
struct VarlenDecoderIO : DecoderIO {
  VarlenDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                  ScheduledRequests& scheduled_requests,
                  std::shared_ptr<CacheManager> cache_manager,
                  const ExecutionContext* execution_context = nullptr,
                  VarlenGraphBuffers* graph_buffers = nullptr,
                  size_t position_planes = 0);

  std::vector<DeviceSpan<float>> ProcessLogits() override;

  // The step's packed [total_num_tokens, hidden_size] hidden states, or null when the model does
  // not expose them. Row i corresponds to packed token i; logits have the same ordering only when
  // the model emits one logits row per packed token.
  Tensor* HiddenStates() const override { return active_hidden_states_; }

  // The step's packed [total_num_tokens, aux_hidden_size] auxiliary hidden states, or null when
  // the model was not exported with aux_hidden_state_layers. This is what a DFlash 2 drafter reads.
  Tensor* AuxHiddenStates() const override { return aux_hidden_states_.get(); }

 private:
  void PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareAttentionMetadata(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareHiddenStatesInput(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareHiddenStates(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareAuxHiddenStates(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);

  // Number of packed token rows in this step, which is what both the logits and the hidden states
  // are indexed by.
  size_t TokenCount(ScheduledRequests& scheduled_requests) const;

  // Non-null when this step is being captured or replayed, in which case the tensors below are
  // borrowed from the holder instead of being allocated fresh.
  VarlenGraphBuffers* graph_buffers_{};
  const StepPlan* plan_{};
  size_t block_table_columns_{};
  // Borrowed spans/pointers whose backing storage is owned through synchronous model execution.
  DeviceSpan<int32_t> input_ids_;
  OrtValue* hidden_states_input_{};
  size_t position_planes_{};
  std::vector<std::unique_ptr<Tensor>> owned_inputs_;
  std::unique_ptr<Tensor> logits_;
  Tensor* active_logits_{};
  std::unique_ptr<Tensor> logits_fp32_;
  std::unique_ptr<Tensor> hidden_states_;
  Tensor* active_hidden_states_{};
  std::unique_ptr<Tensor> aux_hidden_states_;
  bool logits_are_per_token_{true};
};

}  // namespace Generators
