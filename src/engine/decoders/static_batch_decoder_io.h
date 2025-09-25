// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "decoder.h"
#include "../../models/decoder_only.h"

namespace Generators {

/**
 * @class StaticBatchDecoderIO
 * @brief Prepares and manages the inputs and outputs for a static batch decoder model.
 *
 * A static batch decoder model is one that expects the first dimension of the input tensors
 * to be the batch size, and all sequences in the batch to have the same length (padded as necessary).
 * This class handles the preparation of the following input and output tensors:
 * Inputs:
 * - Input IDs - int64[batch_size, sequence_length]
 * - Attention Mask - int64[batch_size, sequence_length]
 * - Position IDs - int64[batch_size, sequence_length]
 * Outputs:
 * - Logits - float16/float32[batch_size, sequence_length, vocab_size]
 *
 * The inputs prepared by this class are compatible with models that use GroupQueryAttention,
 * or MultiHeadAttention operators.
 */
struct StaticBatchDecoderIO : DecoderIO {
  StaticBatchDecoderIO(std::shared_ptr<DecoderOnly_Model> model,
                       ScheduledRequests& scheduled_requests,
                       std::shared_ptr<CacheManager> cache_manager);

  std::vector<DeviceSpan<float>> ProcessLogits() override;

 private:
  void PrepareInputIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareAttentionMask(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PreparePositionIds(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);
  void PrepareLogits(std::shared_ptr<DecoderOnly_Model> model, ScheduledRequests& scheduled_requests);

  std::vector<std::unique_ptr<Tensor>> owned_inputs_;
  std::unique_ptr<Tensor> logits_;
  std::unique_ptr<Tensor> logits_fp32_;
};

}  // namespace Generators
